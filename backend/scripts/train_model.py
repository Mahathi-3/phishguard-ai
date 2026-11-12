"""
PhishGuard AI – High-Performance Model Training Pipeline.
Trained on the combined 53,973-email corpus from:
  - Dataset 1: Phishing_Email1.csv (Chakraborty, 2023)
  - Dataset 2: spam2.csv (Qureshi, 2023)
  - Dataset 3: Enron.csv (Islam, 2023)
Corpus matches Chakravorty et al. (arXiv:2603.27326v1) with 31,938 safe and 22,035 phishing emails.
Implements:
  - Dual-granularity Word (1-2) + Character (2-4) TF-IDF feature space (60,000 features)
  - Calibrated Logistic Regression + Multinomial Naive Bayes
  - Validation-partition Platt scaling calibration (eliminates test-set leakage)
  - Full metrics export to backend/results/metrics.json and calibration.json
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, brier_score_loss)
from scipy.sparse import hstack
import joblib

SEED = 42
np.random.seed(SEED)
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE))
from data_loader import load_full_dataset
from normalizer import normalize_text

def train():
    print("=" * 65)
    print("  PhishGuard AI – Training on 53,973-Email Combined Corpus")
    print("=" * 65)

    t0 = time.time()
    df = load_full_dataset()
    print(f"Loaded {len(df)} emails in {time.time() - t0:.2f}s.")
    print(f"  Legitimate / Safe (0): {(df['label'] == 0).sum()} ({((df['label'] == 0).mean()*100):.1f}%)")
    print(f"  Phishing / Malicious (1): {(df['label'] == 1).sum()} ({((df['label'] == 1).mean()*100):.1f}%)")

    # 1. Stratified Partition: 70% Train, 10% Validation, 20% Held-Out Test
    # Test partition has exactly 10,794 emails (matching paper's 20% evaluation partition)
    train_val_df, test_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=SEED)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, stratify=train_val_df['label'], random_state=SEED)
    # 0.8 * 0.125 = 0.10 of total -> Train: 70%, Val: 10%, Test: 20%
    print(f"Dataset partitions:")
    print(f"  Train partition (70%): {len(train_df)} samples")
    print(f"  Validation partition (10%): {len(val_df)} samples")
    print(f"  Test partition (20%): {len(test_df)} samples")

    # 2. Dual-Granularity Feature Extraction
    print("\nExtracting Word TF-IDF n-grams (1-2)...")
    t0 = time.time()
    tfidf_word = TfidfVectorizer(
        max_features=35000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode'
    )
    X_train_word = tfidf_word.fit_transform(train_df['text'])
    X_val_word = tfidf_word.transform(val_df['text'])
    X_test_word = tfidf_word.transform(test_df['text'])
    print(f"  Word TF-IDF fitted in {time.time() - t0:.2f}s. Vocabulary size: {X_train_word.shape[1]}")

    print("Extracting Character TF-IDF n-grams (2-4, within-word)...")
    t0 = time.time()
    tfidf_char = TfidfVectorizer(
        max_features=25000,
        ngram_range=(2, 4),
        analyzer='char_wb',
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents='unicode'
    )
    X_train_char = tfidf_char.fit_transform(train_df['text'])
    X_val_char = tfidf_char.transform(val_df['text'])
    X_test_char = tfidf_char.transform(test_df['text'])
    print(f"  Char TF-IDF fitted in {time.time() - t0:.2f}s. Vocabulary size: {X_train_char.shape[1]}")

    X_train = hstack([X_train_word, X_train_char])
    X_val = hstack([X_val_word, X_val_char])
    X_test = hstack([X_test_word, X_test_char])
    print(f"  Combined sparse feature dimension: {X_train.shape[1]}")

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # 3. Model Training
    print("\nTraining Logistic Regression (C=5.0, L2 penalty)...")
    t0 = time.time()
    lr = LogisticRegression(C=5.0, max_iter=1000, class_weight='balanced', random_state=SEED)
    lr.fit(X_train, y_train)
    print(f"  Logistic Regression trained in {time.time() - t0:.2f}s.")

    print("Training Multinomial Naive Bayes baseline...")
    t0 = time.time()
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train, y_train)
    print(f"  Naive Bayes trained in {time.time() - t0:.2f}s.")

    # 4. Platt Scaling Probability Calibration (fitted strictly on Validation set)
    print("\nFitting Platt scaling calibration on validation set...")
    t0 = time.time()
    cal_lr = CalibratedClassifierCV(FrozenEstimator(lr), method='sigmoid', cv=2)
    cal_lr.fit(X_val, y_val)
    print(f"  Calibration fitted in {time.time() - t0:.2f}s.")

    # 5. Comprehensive Metric Evaluation on Held-Out Test Set (n=10,794)
    print("\nEvaluating on Held-Out Test Set (n=%d)..." % len(test_df))

    def evaluate_model(model, X, y, name, calibrated_model=None):
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        cm = confusion_matrix(y, y_pred).tolist()
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        brier_uncal = float(brier_score_loss(y, y_prob))

        if calibrated_model:
            y_prob_cal = calibrated_model.predict_proba(X)[:, 1]
            y_pred_cal = calibrated_model.predict(X)
            brier_cal = float(brier_score_loss(y, y_prob_cal))
        else:
            y_prob_cal = y_prob
            brier_cal = brier_uncal

        # Compute Expected Calibration Error (ECE) across 10 deciles
        bins = np.linspace(0, 1, 11)
        reliability = []
        ece_sum = 0.0
        probs = y_prob_cal if calibrated_model else y_prob
        for i in range(10):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            count = int(mask.sum())
            if count > 0:
                mean_p = float(probs[mask].mean())
                mean_y = float(y[mask].mean())
                ece_sum += abs(mean_p - mean_y) * (count / len(y))
                reliability.append({
                    "bin_start": float(bins[i]),
                    "bin_end": float(bins[i+1]),
                    "mean_predicted": mean_p,
                    "mean_actual": mean_y,
                    "count": count
                })

        report = classification_report(y, y_pred, target_names=['Legitimate', 'Phishing'], output_dict=True)

        return {
            "model": name,
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "fpr": float(fpr),
            "ece": float(ece_sum),
            "confusion_matrix": cm,
            "classification_report": {k: v for k, v in report.items() if k != 'accuracy'},
            "brier_uncalibrated": brier_uncal,
            "brier_calibrated": brier_cal,
            "reliability": reliability,
        }

    lr_metrics = evaluate_model(lr, X_test, y_test, "LogisticRegression", cal_lr)
    nb_metrics = evaluate_model(nb, X_test, y_test, "MultinomialNB")

    print("\n" + "=" * 65)
    print("  FINAL TEST RESULTS (Comparison Against Chakravorty et al. Paper)")
    print("=" * 65)
    print(f"Paper Baseline LR:  Acc=95.41%, Prec=94.99%, Rec=93.69%, F1=94.33%")
    print(f"PhishGuard AI LR:   Acc={lr_metrics['accuracy']*100:.2f}%, Prec={lr_metrics['precision']*100:.2f}%, Rec={lr_metrics['recall']*100:.2f}%, F1={lr_metrics['f1']*100:.2f}%, FPR={lr_metrics['fpr']*100:.2f}%")
    print(f"PhishGuard AI NB:   Acc={nb_metrics['accuracy']*100:.2f}%, Prec={nb_metrics['precision']*100:.2f}%, Rec={nb_metrics['recall']*100:.2f}%, F1={nb_metrics['f1']*100:.2f}%, FPR={nb_metrics['fpr']*100:.2f}%")
    print(f"Brier Calibration:  {lr_metrics['brier_uncalibrated']:.4f} -> {lr_metrics['brier_calibrated']:.4f} (ECE: {lr_metrics['ece']*100:.2f}%)")

    # 6. Save Model Artifacts
    print("\nSaving model artifacts to backend/models/...")
    joblib.dump(tfidf_word, MODELS_DIR / "tfidf_word.joblib")
    joblib.dump(tfidf_char, MODELS_DIR / "tfidf_char.joblib")
    joblib.dump(lr, MODELS_DIR / "lr_model.joblib")
    joblib.dump(cal_lr, MODELS_DIR / "cal_lr_model.joblib")
    joblib.dump(nb, MODELS_DIR / "nb_model.joblib")
    print("  Saved: tfidf_word.joblib, tfidf_char.joblib, lr_model.joblib, cal_lr_model.joblib, nb_model.joblib")

    # 7. Save JSON Metrics
    meta = {
        "dataset": "Combined 53,973 Phishing Corpus (Chakravorty et al. Benchmark)",
        "total_rows": len(df),
        "phishing_count": int(df['label'].sum()),
        "legitimate_count": int((df['label'] == 0).sum()),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "split_method": "stratified 70/10/20, seed=42",
        "positive_class": "phishing (label=1)",
        "date": datetime.now().isoformat(),
        "seed": SEED,
        "paper_baseline": {
            "logistic_regression": {
                "accuracy": 0.9541,
                "precision": 0.9499,
                "recall": 0.9369,
                "f1": 0.9433
            },
            "naive_bayes": {
                "accuracy": 0.9386,
                "precision": 0.9180,
                "recall": 0.9328,
                "f1": 0.9254
            }
        }
    }
    results = {
        "metadata": meta,
        "logistic_regression": lr_metrics,
        "multinomial_nb": nb_metrics
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    cal_results = {
        "model": "LogisticRegression (calibrated, sigmoid, fitted strictly on validation set)",
        "brier_before": lr_metrics["brier_uncalibrated"],
        "brier_after": lr_metrics["brier_calibrated"],
        "brier_reduction_pct": round((1.0 - lr_metrics["brier_calibrated"] / lr_metrics["brier_uncalibrated"]) * 100, 2),
        "ece": round(lr_metrics["ece"], 4),
        "reliability": lr_metrics["reliability"],
        "note": "Calibration fitted on validation set only. Evaluated on unseen 10,794-sample test set."
    }
    with open(RESULTS_DIR / "calibration.json", "w") as f:
        json.dump(cal_results, f, indent=2)

    print(f"Results successfully saved to {RESULTS_DIR}")
    return lr_metrics

if __name__ == "__main__":
    train()
