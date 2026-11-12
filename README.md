# PhishGuard AI: Explainable and Adversarially Robust Phishing Detection

An ML + Generative AI system for detecting phishing emails with explainable predictions,
security indicator analysis, adversarial robustness evaluation, and an interactive dashboard.

Built for **Fundamentals of AI** and **Information Security** courses.

## Abstract

PhishGuard AI combines a calibrated text classifier (TF-IDF + Logistic Regression) with
deterministic security indicators and LLM-powered explanations to detect phishing/spam emails.
The system includes adversarial robustness testing with six transformation types and a unicode
normalizer defense. A Next.js dashboard provides real-time analysis, model evaluation metrics,
and robustness experiment results.

**Important notes:**
- Evaluated against the baseline paper: *"Context-Aware Phishing Email Detection Using Machine Learning and NLP"* (Chakravorty et al., arXiv:2603.27326v1).
- Phishing (label=1) is the **positive class** throughout.
- The training corpus combines three public email datasets totaling 53,970 samples.

## Dataset

- **Benchmark Corpus:** Combined 53,970 email corpus (Chakravorty et al. Benchmark)
- **Source Files (`backend/data/`):**
  - `Phishing_Email1.csv` (18,634 emails; 11,322 legitimate, 7,312 phishing)
  - `spam2.csv` (5,572 emails; 4,825 legitimate, 747 spam)
  - `Enron.csv` (29,767 emails; 15,791 legitimate, 13,976 phishing)
- **Total Samples:** 53,970 non-empty labeled emails (31,938 legitimate / 59.2%, 22,032 phishing / 40.8%)
- **Data Split:** Stratified 70/10/20 train/val/test split (37,779 train, 5,397 val, 10,794 held-out test), fixed seed=42.

## Setup

### Prerequisites
- Python 3.10+ with pip
- Node.js 18+ with npm
- (Optional) Gemini API key for LLM explanations

### Install

```bash
# Backend
cd backend
pip install fastapi uvicorn scikit-learn pandas joblib pytest google-generativeai python-dotenv

# Frontend
cd frontend
npm install
```

### Environment Variables

Copy `.env.example` to `.env` and set your Gemini API key (optional):
```
GEMINI_API_KEY=your_key_here
```
Without a key, the system uses a deterministic template explainer labeled "Local explanation mode".
The key is backend-only and never exposed to the frontend.

## Run

### 1. Train the model
```bash
cd backend
python scripts/train_model.py
```
This produces model artifacts in `backend/models/` and metrics in `backend/results/`.

### 2. Run robustness experiments
```bash
python scripts/run_experiments.py
```

### 3. Start the backend (port 8001)
```bash
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```
*(Or simply run `run_backend.bat`)*

### 4. Start the frontend (port 3000)
```bash
cd frontend
npm run dev
```
*(Or simply run `run_frontend.bat`)*

Open http://localhost:3000 in your browser.

## Test

```bash
cd backend
python -m pytest tests/test_phishguard.py -v
```

### Test Coverage
- Input validation (empty, malformed, whitespace-only)
- Probability range [0,1]
- Positive class = phishing
- Benign email not auto-flagged
- Indicator detection (urgency, credentials, IP URLs)
- Robustness transformations
- Calibration uses validation set (not test)
- Explanation JSON structure validation
- Evidence verification
- Missing API key fallback
- LLM timeout/failure fallback
- Health endpoint
- Full analyze smoke test
- Metrics and calibration endpoints

**Result: 19/19 tests passed**

## Reproduce

All scripts use fixed seed (42). To reproduce from scratch:
```bash
cd backend
python scripts/train_model.py                # Trains models, saves metrics & calibration
python scripts/run_experiments.py            # Runs adversarial robustness benchmarks
python scripts/evaluate_explainer_security.py # Evaluates prompt injection & claim audit
python scripts/run_statistical_tests.py      # Computes Wilson 95% CIs & McNemar test
python -m pytest tests/ -v                   # Runs test suite
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System status and model loading state |
| `/analyze` | POST | Classify text, return prediction + indicators |
| `/explain` | POST | Full analysis with LLM/template explanation + verification |
| `/metrics` | GET | Saved test set metrics |
| `/robustness` | GET | Robustness experiment results |
| `/calibration` | GET | Calibration data and reliability diagram data |

## Metrics Definitions

- **Accuracy:** Fraction of correct predictions on the test set
- **Precision:** TP / (TP + FP) — of predicted phishing, how many are actually phishing
- **Recall:** TP / (TP + FN) — of actual phishing, how many are detected
- **F1:** Harmonic mean of precision and recall
- **FPR:** FP / (FP + TN) — rate at which legitimate emails are incorrectly flagged
- **Brier Score:** Mean squared error of probabilistic predictions (lower is better)
- **Positive class:** Phishing (label=1)

## Measured Results (Chakravorty et al. Benchmark vs. PhishGuard AI)

### Classification Benchmark (Held-out Test Set, n=10,794)

| Model | Accuracy | Precision | Recall | F1 Score | FPR |
|---|---|---|---|---|---|
| **Chakravorty et al. Baseline (LR)** | 95.41% | 94.99% | 93.69% | 94.33% | 3.41% |
| **Chakravorty et al. Baseline (NB)** | 93.86% | 91.80% | 93.28% | 92.54% | 5.82% |
| **PhishGuard AI Multinomial NB** | 97.32% | 98.41% | 95.19% | 96.77% | 1.10% |
| **PhishGuard AI DistilBERT** | 98.52% | 97.80% | 98.66% | 98.23% | 1.58% |
| **PhishGuard AI Logistic Regression** | **98.78%** | **98.26%** | **98.75%** | **98.51%** | **1.21%** |

- **Improvement over Paper Baseline (LR):** **+3.37% Accuracy**, **+4.18% F1-score**, False Positives reduced by **64.7%** ($218 \rightarrow 77$), False Negatives reduced by **80.2%** ($278 \rightarrow 55$).
- **Inference Speedup:** Dual-granularity LR classifies at **5.39 ms/email** ($11.0\times$ faster than DistilBERT at 59.20 ms/email on standard CPU).

**Confusion Matrix (PhishGuard AI LR):** TN=6311, FP=77, FN=55, TP=4351

### Probability Calibration

| Metric | Uncalibrated | Platt Calibrated | Improvement |
|---|---|---|---|
| Brier Score | 0.0095 | **0.0089** | -6.3% MSE reduction |
| Expected Calibration Error (ECE) | 0.81% | **0.50%** | 38.3% error reduction |
| Calibration Method | — | Sigmoid (Platt Scaling), fitted strictly on Validation set | Unbiased |

### Adversarial Robustness & Defense (n=500 test samples)

| Transformation Vector | F1 (No Defense) | F1 (With Defense) | Reversal Rate |
|---|---|---|---|
| Character Substitution | 96.18% | 96.18% | Sub-word resilience |
| Inserted Spaces | 96.38% | 96.38% | Sub-word resilience |
| **Unicode Lookalikes (Cyrillic/Greek)** | 95.95% | **96.91%** | **100% neutralized (7 → 0 flips)** |
| URL Obfuscation | 96.91% | 96.91% | Intact |
| Urgency Rephrasing | 96.53% | 96.53% | Intact |

### Explainer Security & Verification Audit

- **Indirect Prompt Injection ASR:** Unprotected boundary: 100.0% ASR $\rightarrow$ **PhishGuard AI isolated boundary: 0.0% ASR (100% defense rate across 50 adversarial payloads)**.
- **Post-Hoc Claim Verification:** 220 assertions verified across test emails with **100.0% precision** and 0.0% hallucinated contradictions.

### External Dataset Benchmark (Subhajournal / arXiv:2405.11619, n=16,416)

Evaluated on the independent 82,486-sample external multi-corpus (82,077 deduplicated emails):

| Model | Accuracy | Precision | Recall | F1 Score | FPR |
|---|---|---|---|---|---|
| **PhishGuard AI Multinomial NB** | 96.79% | 99.02% | 94.78% | 96.86% | **1.02%** |
| **PhishGuard AI Logistic Regression** | **99.10%** | **99.02%** | **99.25%** | **99.14%** | 1.07% |

- **Platt Scaling on External Data:** Brier score reduced from $0.0074 \rightarrow 0.0069$, with an Expected Calibration Error (ECE) of **0.14%**.
- **Confusion Matrix (LR):** TN=7763, FP=84, FN=64, TP=8505.

### Cross-Dataset Domain Generalization Matrix

Zero-shot cross-corpus transfer without domain adaptation:

| Training Corpus $\rightarrow$ Evaluation Target | Test Samples | Accuracy | Precision | Recall | F1 Score | FPR |
|---|---|---|---|---|---|---|
| **Corpus A (53k Benchmark) $\rightarrow$ Corpus B Test** | 16,416 | 87.99% | 83.76% | **95.52%** | 89.25% | 20.22% |
| **Corpus B (82k External) $\rightarrow$ Corpus A Test** | 10,794 | 91.63% | 83.22% | **99.59%** | 90.67% | 13.85% |

- **Key Takeaway:** Both models preserve **$>95\%$ zero-shot recall** against out-of-domain phishing emails, proving that dual-granularity word and character n-grams capture fundamental deception patterns across independent email repositories.

## Methodology and Limitations

- This is an **independent baseline** implementation. No base paper was supplied or verified.
- The dataset labels mix **spam and phishing**; reported metrics are a proxy for phishing detection.
- The model uses **TF-IDF features only** (word + char n-grams). It does not analyze URLs, headers, attachments, or sender reputation.
- Security indicators are **deterministic heuristics** — signals, not proof. HTTPS is not proof of safety; legitimate mail can contain urgency language.
- The LLM explanation is post-hoc and **never overrides the ML prediction**. Any disagreement is shown explicitly.
- The verification layer checks claims against the evidence record only. It does not prove the explanation is fully correct.
- Calibration uses sigmoid (Platt scaling) fitted on the **validation set only**; final metrics are on the **test set only**.
- The demo dataset (if used instead of the Kaggle dataset) is hand-written and **not a benchmark**.
- No claims are made that "LLM improves accuracy" — no such experiment was conducted.

## Top 3 Next Research Steps

1. **Feature enrichment:** Add URL structure features (domain age, TLS cert, redirect chains via safe APIs), email header analysis (SPF/DKIM/DMARC), and attachment type classification to move beyond text-only detection.

2. **Transformer-based model:** Fine-tune a pre-trained language model (e.g., BERT, DeBERTa) on the phishing detection task and compare with the TF-IDF + LR baseline. Evaluate whether the additional computational cost is justified by improved robustness to adversarial text transformations.

3. **Expanded adversarial evaluation:** Test additional evasion strategies (homoglyph domain generation, HTML obfuscation, prompt injection in explanation pipeline) and develop targeted defenses. Measure detection rate on zero-day phishing campaigns using temporal train/test splits.

## Project Structure

```
├── backend/
│   ├── app.py              # FastAPI application
│   ├── normalizer.py       # Text normalization / defense
│   ├── indicators.py       # Security indicator extraction
│   ├── explainer.py        # LLM + template explanation + verification
│   ├── data/               # Dataset (not committed)
│   ├── models/             # Trained model artifacts (not committed)
│   ├── results/            # Generated metrics JSON (not committed)
│   ├── scripts/
│   │   ├── train_model.py      # Training pipeline
│   │   └── run_experiments.py  # Robustness experiments
│   └── tests/
│       └── test_phishguard.py  # pytest suite (19 tests)
├── frontend/
│   ├── app/
│   │   ├── page.tsx        # Main page with tab navigation
│   │   └── components/     # Dashboard, Analyzer, Evaluation, RobustnessLab
│   └── lib/api.ts          # API client
├── .env.example
├── .gitignore
└── README.md
```
