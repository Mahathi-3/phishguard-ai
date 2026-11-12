"""
PhishGuard AI – Training Pipeline on External Phishing Dataset (82,486 / 82,077 samples).
Dataset: phishing_email_external.csv (Subhajournal / Kaggle, arXiv:2405.11619).
Combines Enron, Ling, CEAS, Nazario, Nigerian fraud, and SpamAssassin corpora.

Implements:
  1. Stratified 70/10/20 train/val/test split (57,453 train, 8,208 val, 16,416 test)
  2. Dual-granularity TF-IDF (35,000 word n-grams 1-2 + 25,000 char n-grams 2-4)
  3. Logistic Regression + Platt Scaling (Val set only) + Multinomial Naive Bayes
  4. Full metrics evaluation on held-out test partition (n=16,416)
  5. Cross-dataset generalization evaluation between Benchmark (53k) and External (82k)
  6. Artifact export to backend/models/ and backend/results/
"""

import os, sys, json, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
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
from data_loader import load_full_dataset, load_external_dataset
from normalizer import normalize_text


def evaluate_model(model, X, y, name, calibrated_model=None):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, y_pred).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    brier_uncal = float(brier_score_loss(y, y_prob))

    if calibrated_model:
        y_prob_cal = calibrated_model.predict_proba(X)[:, 1]
        brier_cal = float(brier_score_loss(y, y_prob_cal))
        probs = y_prob_cal
    else:
        probs = y_prob
        brier_cal = brier_uncal

    # Expected Calibration Error (ECE) across 10 deciles
    bins = np.linspace(0, 1, 11)
    reliability = []
    ece_sum = 0.0
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


def train_external():
    print("=" * 70)
    print("  PhishGuard AI – Training on External Phishing Dataset (82,077 emails)")
    print("=" * 70)

    t0 = time.time()
    df_ext = load_external_dataset(deduplicate=True)
    print(f"Loaded {len(df_ext)} deduplicated emails in {time.time() - t0:.2f}s.")
    print(f"  Legitimate (0): {(df_ext['label'] == 0).sum()} ({((df_ext['label'] == 0).mean()*100):.1f}%)")
    print(f"  Phishing (1):   {(df_ext['label'] == 1).sum()} ({((df_ext['label'] == 1).mean()*100):.1f}%)")

    # 1. Stratified Partition: 70% Train, 10% Val, 20% Test
    train_val_df, test_df = train_test_split(df_ext, test_size=0.20, stratify=df_ext['label'], random_state=SEED)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, stratify=train_val_df['label'], random_state=SEED)

    print(f"\nPartitions:")
    print(f"  Train (70%):      {len(train_df)} samples")
    print(f"  Validation (10%): {len(val_df)} samples")
    print(f"  Held-out Test (20%): {len(test_df)} samples")

    # 2. Dual-Granularity Feature Extraction
    print("\nFitting Word TF-IDF n-grams (1-2, max 30k)...", flush=True)
    t0 = time.time()
    tfidf_word = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents='unicode'
    )
    X_train_word = tfidf_word.fit_transform(train_df['text'])
    X_val_word = tfidf_word.transform(val_df['text'])
    X_test_word = tfidf_word.transform(test_df['text'])
    print(f"  Fitted Word TF-IDF in {time.time() - t0:.2f}s. Vocabulary size: {X_train_word.shape[1]}", flush=True)

    print("Fitting Character TF-IDF n-grams (2-4, within-word, max 20k)...", flush=True)
    t0 = time.time()
    tfidf_char = TfidfVectorizer(
        max_features=20000,
        ngram_range=(2, 4),
        analyzer='char_wb',
        sublinear_tf=True,
        min_df=5,
        max_df=0.95,
        strip_accents='unicode'
    )
    X_train_char = tfidf_char.fit_transform(train_df['text'])
    X_val_char = tfidf_char.transform(val_df['text'])
    X_test_char = tfidf_char.transform(test_df['text'])
    print(f"  Fitted Char TF-IDF in {time.time() - t0:.2f}s. Vocabulary size: {X_train_char.shape[1]}", flush=True)

    X_train = hstack([X_train_word, X_train_char])
    X_val = hstack([X_val_word, X_val_char])
    X_test = hstack([X_test_word, X_test_char])
    print(f"  Combined sparse feature shape: {X_train.shape}", flush=True)

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # 3. Model Training
    print("\nTraining Logistic Regression (C=5.0, L2, balanced)...", flush=True)
    t0 = time.time()
    lr = LogisticRegression(C=5.0, max_iter=1000, class_weight='balanced', random_state=SEED)
    lr.fit(X_train, y_train)
    print(f"  Trained LR in {time.time() - t0:.2f}s.", flush=True)

    print("Training Multinomial Naive Bayes baseline (alpha=0.1)...", flush=True)
    t0 = time.time()
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train, y_train)
    print(f"  Trained NB in {time.time() - t0:.2f}s.", flush=True)

    # 4. Platt Scaling Calibration
    print("Fitting Platt Scaling Calibration strictly on validation set...", flush=True)
    t0 = time.time()
    cal_lr = CalibratedClassifierCV(FrozenEstimator(lr), method='sigmoid', cv=2)
    cal_lr.fit(X_val, y_val)
    print(f"  Fitted Calibration in {time.time() - t0:.2f}s.", flush=True)

    # 5. Evaluation on External Test Partition
    print(f"\nEvaluating on Held-Out External Test Partition (n={len(test_df)})...", flush=True)
    lr_metrics = evaluate_model(lr, X_test, y_test, "LogisticRegression", cal_lr)
    nb_metrics = evaluate_model(nb, X_test, y_test, "MultinomialNB")

    print("\n" + "=" * 70)
    print("  EXTERNAL DATASET TEST RESULTS (n=16,416)")
    print("=" * 70)
    print(f"PhishGuard AI LR:  Acc={lr_metrics['accuracy']*100:.2f}%, Prec={lr_metrics['precision']*100:.2f}%, Rec={lr_metrics['recall']*100:.2f}%, F1={lr_metrics['f1']*100:.2f}%, FPR={lr_metrics['fpr']*100:.2f}%")
    print(f"PhishGuard AI NB:  Acc={nb_metrics['accuracy']*100:.2f}%, Prec={nb_metrics['precision']*100:.2f}%, Rec={nb_metrics['recall']*100:.2f}%, F1={nb_metrics['f1']*100:.2f}%, FPR={nb_metrics['fpr']*100:.2f}%")
    print(f"Platt Calibration: Brier {lr_metrics['brier_uncalibrated']:.4f} -> {lr_metrics['brier_calibrated']:.4f} (ECE: {lr_metrics['ece']*100:.2f}%)")

    # 6. Save Artifacts
    print("\nSaving model artifacts...")
    joblib.dump(tfidf_word, MODELS_DIR / "external_tfidf_word.joblib")
    joblib.dump(tfidf_char, MODELS_DIR / "external_tfidf_char.joblib")
    joblib.dump(lr, MODELS_DIR / "external_lr_model.joblib")
    joblib.dump(cal_lr, MODELS_DIR / "external_cal_lr_model.joblib")
    joblib.dump(nb, MODELS_DIR / "external_nb_model.joblib")
    print("  Saved external models to backend/models/")

    # 7. Save JSON Metrics
    meta = {
        "dataset": "External Kaggle Phishing Email Dataset (Subhajournal / arXiv:2405.11619)",
        "source_file": "phishing_email_external.csv",
        "total_rows_raw": 82486,
        "total_rows_dedup": len(df_ext),
        "phishing_count": int(df_ext['label'].sum()),
        "legitimate_count": int((df_ext['label'] == 0).sum()),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "split_method": "stratified 70/10/20, seed=42",
        "positive_class": "phishing (label=1)",
        "date": datetime.now().isoformat(),
        "seed": SEED
    }
    results = {
        "metadata": meta,
        "logistic_regression": lr_metrics,
        "multinomial_nb": nb_metrics
    }
    with open(RESULTS_DIR / "metrics_external.json", "w") as f:
        json.dump(results, f, indent=2)

    cal_results = {
        "model": "LogisticRegression (External dataset, calibrated sigmoid)",
        "brier_before": lr_metrics["brier_uncalibrated"],
        "brier_after": lr_metrics["brier_calibrated"],
        "brier_reduction_pct": round((1.0 - lr_metrics["brier_calibrated"] / lr_metrics["brier_uncalibrated"]) * 100, 2),
        "ece": round(lr_metrics["ece"], 4),
        "reliability": lr_metrics["reliability"],
        "note": f"Evaluated on {len(test_df)} unseen external test samples."
    }
    with open(RESULTS_DIR / "calibration_external.json", "w") as f:
        json.dump(cal_results, f, indent=2)

    # 8. Cross-Dataset Generalization Evaluation
    print("\n" + "=" * 70)
    print("  CROSS-DATASET GENERALIZATION BENCHMARK")
    print("=" * 70)
    cross_results = {}
    try:
        # Load 53k benchmark test set
        df_bench = load_full_dataset()
        _, bench_test_df = train_test_split(df_bench, test_size=0.20, stratify=df_bench['label'], random_state=SEED)
        
        # A: External Model evaluated on Benchmark 53k Test Set
        X_bench_w = tfidf_word.transform(bench_test_df['text'])
        X_bench_c = tfidf_char.transform(bench_test_df['text'])
        X_bench = hstack([X_bench_w, X_bench_c])
        y_bench = bench_test_df['label'].values
        
        ext_on_bench_pred = lr.predict(X_bench)
        cross_results["external_model_on_benchmark_test"] = {
            "test_corpus": "Chakravorty et al. Benchmark (53,970 corpus)",
            "test_samples": len(bench_test_df),
            "accuracy": float(accuracy_score(y_bench, ext_on_bench_pred)),
            "precision": float(precision_score(y_bench, ext_on_bench_pred, zero_division=0)),
            "recall": float(recall_score(y_bench, ext_on_bench_pred, zero_division=0)),
            "f1": float(f1_score(y_bench, ext_on_bench_pred, zero_division=0)),
            "fpr": float(confusion_matrix(y_bench, ext_on_bench_pred)[0][1] / (confusion_matrix(y_bench, ext_on_bench_pred)[0][0] + confusion_matrix(y_bench, ext_on_bench_pred)[0][1]))
        }
        print(f"External Model on Benchmark Test (n={len(bench_test_df)}):")
        print(f"  Acc: {cross_results['external_model_on_benchmark_test']['accuracy']*100:.2f}%, F1: {cross_results['external_model_on_benchmark_test']['f1']*100:.2f}%, FPR: {cross_results['external_model_on_benchmark_test']['fpr']*100:.2f}%")

        # B: Benchmark Model evaluated on External 82k Test Set
        bench_tfidf_word = joblib.load(MODELS_DIR / "tfidf_word.joblib")
        bench_tfidf_char = joblib.load(MODELS_DIR / "tfidf_char.joblib")
        bench_lr = joblib.load(MODELS_DIR / "lr_model.joblib")

        X_ext_w = bench_tfidf_word.transform(test_df['text'])
        X_ext_c = bench_tfidf_char.transform(test_df['text'])
        X_ext = hstack([X_ext_w, X_ext_c])
        
        bench_on_ext_pred = bench_lr.predict(X_ext)
        cross_results["benchmark_model_on_external_test"] = {
            "test_corpus": "External Kaggle Phishing Dataset (82,486 corpus)",
            "test_samples": len(test_df),
            "accuracy": float(accuracy_score(y_test, bench_on_ext_pred)),
            "precision": float(precision_score(y_test, bench_on_ext_pred, zero_division=0)),
            "recall": float(recall_score(y_test, bench_on_ext_pred, zero_division=0)),
            "f1": float(f1_score(y_test, bench_on_ext_pred, zero_division=0)),
            "fpr": float(confusion_matrix(y_test, bench_on_ext_pred)[0][1] / (confusion_matrix(y_test, bench_on_ext_pred)[0][0] + confusion_matrix(y_test, bench_on_ext_pred)[0][1]))
        }
        print(f"Benchmark Model on External Test (n={len(test_df)}):")
        print(f"  Acc: {cross_results['benchmark_model_on_external_test']['accuracy']*100:.2f}%, F1: {cross_results['benchmark_model_on_external_test']['f1']*100:.2f}%, FPR: {cross_results['benchmark_model_on_external_test']['fpr']*100:.2f}%")

        with open(RESULTS_DIR / "cross_dataset_evaluation.json", "w") as f:
            json.dump(cross_results, f, indent=2)
        print(f"Cross-dataset evaluation saved to {RESULTS_DIR / 'cross_dataset_evaluation.json'}")
    except Exception as e:
        print(f"Cross-dataset evaluation warning: {e}")

    print(f"\nAll artifacts saved successfully to {RESULTS_DIR}")
    return lr_metrics


if __name__ == "__main__":
    train_external()
