"""
PhishGuard AI – Unified Data Loader for the 53,973-email corpus.
Combines:
  1. Dataset 1: Phishing_Email1.csv (18,634 emails; 11,322 safe, 7,312 phishing)
  2. Dataset 2: spam2.csv (5,572 emails; 4,825 ham, 747 spam)
  3. Dataset 3: Enron.csv (29,767 emails; 15,791 safe, 13,976 phishing)
Total: 53,973 labeled emails (31,938 legitimate / safe = 0, 22,035 phishing / spam = 1).
Matches the exact corpus evaluated in Chakravorty et al. (arXiv:2603.27326v1).
"""
from pathlib import Path
import pandas as pd

def load_full_dataset(data_dir: Path = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"

    d1_path = data_dir / "Phishing_Email1.csv"
    d2_path = data_dir / "spam2.csv"
    d3_path = data_dir / "Enron.csv"

    if not (d1_path.exists() and d2_path.exists() and d3_path.exists()):
        raise FileNotFoundError(f"One or more required dataset files not found in {data_dir}")

    # Dataset 1
    df1 = pd.read_csv(d1_path).dropna(subset=['Email Text'])
    df1['text'] = df1['Email Text'].astype(str)
    df1['label'] = (df1['Email Type'].str.strip() == 'Phishing Email').astype(int)

    # Dataset 2
    df2 = pd.read_csv(d2_path, encoding='latin1')
    df2['text'] = df2['Message'].astype(str)
    df2['label'] = (df2['Category'].str.strip().str.lower() == 'spam').astype(int)

    # Dataset 3
    df3 = pd.read_csv(d3_path)
    df3['text'] = (df3['subject'].fillna('') + ' ' + df3['body'].fillna('')).astype(str)
    df3['label'] = df3['label'].astype(int)

    combined = pd.concat([
        df1[['text', 'label']],
        df2[['text', 'label']],
        df3[['text', 'label']]
    ], ignore_index=True)

    # Ensure no empty texts
    combined = combined.dropna(subset=['text'])
    combined['text'] = combined['text'].str.strip()
    combined = combined[combined['text'].str.len() > 0]

    return combined

def load_external_dataset(data_dir: Path = None, deduplicate: bool = True) -> pd.DataFrame:
    """
    Loads the external 82,486-sample phishing email dataset (Kaggle / Subhajournal, arXiv:2405.11619).
    Combines Enron, Ling, CEAS, Nazario, Nigerian fraud, and SpamAssassin corpora.
    Columns: ['text', 'label'] where 0 = Legitimate / Safe, 1 = Phishing / Spam.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"
    ext_path = data_dir / "phishing_email_external.csv"
    if not ext_path.exists():
        raise FileNotFoundError(f"External dataset not found: {ext_path}")
    df = pd.read_csv(ext_path)
    df = df.dropna(subset=['text_combined'])
    df['text'] = df['text_combined'].astype(str).str.strip()
    df = df[df['text'].str.len() > 0]
    df['label'] = df['label'].astype(int)
    if deduplicate:
        df = df.drop_duplicates(subset=['text'])
    return df[['text', 'label']].reset_index(drop=True)

if __name__ == "__main__":
    df = load_full_dataset()
    print(f"Loaded {len(df)} emails (Paper Benchmark): {df['label'].value_counts().to_dict()}")
    df_ext = load_external_dataset()
    print(f"Loaded {len(df_ext)} emails (External Dataset): {df_ext['label'].value_counts().to_dict()}")
