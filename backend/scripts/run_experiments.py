"""PhishGuard AI – Robustness experiments."""
import os, sys, json, re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib

SEED = 42
np.random.seed(SEED)
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

sys.path.insert(0, str(BASE))
from normalizer import normalize_text
from data_loader import load_full_dataset

# --- Transformations ---
def char_substitution(text):
    subs = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$'}
    out = list(text)
    for i, c in enumerate(out):
        if c.lower() in subs and np.random.random() < 0.3:
            out[i] = subs[c.lower()]
    return ''.join(out)

def insert_spaces(text):
    words = text.split()
    out = []
    for w in words:
        if len(w) > 4 and np.random.random() < 0.3:
            pos = np.random.randint(1, len(w))
            w = w[:pos] + ' ' + w[pos:]
        out.append(w)
    return ' '.join(out)

def unicode_lookalikes(text):
    subs = {'a': '\u0430', 'e': '\u0435', 'o': '\u043e', 'c': '\u0441', 'p': '\u0440'}
    out = list(text)
    for i, c in enumerate(out):
        if c in subs and np.random.random() < 0.3:
            out[i] = subs[c]
    return ''.join(out)

def url_obfuscation(text):
    text = re.sub(r'https?://([^\s]+)', lambda m: f'http://192.168.1.{np.random.randint(1,255)}/{m.group(1)[:10]}', text)
    return text

def brand_in_domain(text):
    brands = ['paypal', 'apple', 'google', 'microsoft', 'amazon']
    brand = np.random.choice(brands)
    if 'http' not in text.lower():
        text += f' Visit http://{brand}.security-check-{np.random.randint(100,999)}.com/verify'
    return text

def urgency_rephrase(text):
    additions = [" Act immediately or face consequences!", " This is extremely urgent - respond NOW!",
                 " Time is running out! Last chance!", " URGENT: Immediate action required!"]
    return text + np.random.choice(additions)

TRANSFORMATIONS = {
    "char_substitution": char_substitution,
    "inserted_spaces": insert_spaces,
    "unicode_lookalikes": unicode_lookalikes,
    "url_obfuscation": url_obfuscation,
    "brand_in_domain": brand_in_domain,
    "urgency_rephrase": urgency_rephrase,
}

def run_experiments():
    # Load models
    tfidf_word = joblib.load(MODELS_DIR / "tfidf_word.joblib")
    tfidf_char = joblib.load(MODELS_DIR / "tfidf_char.joblib")
    lr = joblib.load(MODELS_DIR / "lr_model.joblib")
    cal_lr = joblib.load(MODELS_DIR / "cal_lr_model.joblib")

    # Load test data (re-split with same seed as train_model.py)
    df = load_full_dataset()
    df['text'] = df['text'].apply(normalize_text)
    train_val_df, test_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=SEED)

    # Take a subset for robustness (max 500 for speed)
    if len(test_df) > 500:
        test_df = test_df.sample(n=500, random_state=SEED)

    def predict(texts):
        X_w = tfidf_word.transform(texts)
        X_c = tfidf_char.transform(texts)
        X = hstack([X_w, X_c])
        return cal_lr.predict(X), cal_lr.predict_proba(X)[:, 1]

    # Original metrics
    y_test = test_df['label'].values
    texts_orig = test_df['text'].values
    preds_orig, probs_orig = predict(texts_orig)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    orig = {
        "accuracy": float(accuracy_score(y_test, preds_orig)),
        "f1": float(f1_score(y_test, preds_orig, zero_division=0)),
        "precision": float(precision_score(y_test, preds_orig, zero_division=0)),
        "recall": float(recall_score(y_test, preds_orig, zero_division=0)),
    }
    cm_orig = confusion_matrix(y_test, preds_orig)
    orig["fpr"] = float(cm_orig[0][1] / (cm_orig[0][1] + cm_orig[0][0])) if (cm_orig[0][1] + cm_orig[0][0]) > 0 else 0.0

    results = {"original": orig, "transformations": {}}

    for name, transform_fn in TRANSFORMATIONS.items():
        np.random.seed(SEED)
        perturbed = [transform_fn(t) for t in texts_orig]

        # Without defense
        preds_p, probs_p = predict(perturbed)
        cm_p = confusion_matrix(y_test, preds_p)
        fpr_p = float(cm_p[0][1] / (cm_p[0][1] + cm_p[0][0])) if (cm_p[0][1] + cm_p[0][0]) > 0 else 0.0
        no_defense = {
            "accuracy": float(accuracy_score(y_test, preds_p)),
            "f1": float(f1_score(y_test, preds_p, zero_division=0)),
            "precision": float(precision_score(y_test, preds_p, zero_division=0)),
            "recall": float(recall_score(y_test, preds_p, zero_division=0)),
            "fpr": fpr_p,
        }

        # With defense (normalize)
        defended = [normalize_text(t) for t in perturbed]
        preds_d, probs_d = predict(defended)
        cm_d = confusion_matrix(y_test, preds_d)
        fpr_d = float(cm_d[0][1] / (cm_d[0][1] + cm_d[0][0])) if (cm_d[0][1] + cm_d[0][0]) > 0 else 0.0
        with_defense = {
            "accuracy": float(accuracy_score(y_test, preds_d)),
            "f1": float(f1_score(y_test, preds_d, zero_division=0)),
            "precision": float(precision_score(y_test, preds_d, zero_division=0)),
            "recall": float(recall_score(y_test, preds_d, zero_division=0)),
            "fpr": fpr_d,
        }

        # Flipped examples
        flipped = int((preds_orig != preds_p).sum())
        flipped_defended = int((preds_orig != preds_d).sum())

        # Example
        flip_idx = np.where(preds_orig != preds_p)[0]
        examples = []
        for idx in flip_idx[:3]:
            examples.append({
                "original": texts_orig[idx][:200],
                "perturbed": perturbed[idx][:200],
                "original_pred": int(preds_orig[idx]),
                "perturbed_pred": int(preds_p[idx]),
            })

        results["transformations"][name] = {
            "no_defense": no_defense,
            "with_defense": with_defense,
            "delta_f1_no_defense": round(no_defense["f1"] - orig["f1"], 4),
            "delta_f1_with_defense": round(with_defense["f1"] - orig["f1"], 4),
            "delta_fpr_no_defense": round(no_defense["fpr"] - orig["fpr"], 4),
            "delta_fpr_with_defense": round(with_defense["fpr"] - orig["fpr"], 4),
            "flipped_predictions": flipped,
            "flipped_after_defense": flipped_defended,
            "examples": examples,
        }
        print(f"{name}: F1 {orig['f1']:.4f} -> {no_defense['f1']:.4f} (no def) / {with_defense['f1']:.4f} (def), flipped={flipped}/{flipped_defended}")

    with open(RESULTS_DIR / "robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Robustness results saved.")


if __name__ == "__main__":
    run_experiments()
