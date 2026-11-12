"""PhishGuard AI – Text normalizer for robustness defense."""
import re, unicodedata

UNICODE_LOOKALIKES = str.maketrans({
    # Cyrillic confusables
    '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p', '\u0441': 'c',
    '\u0443': 'y', '\u0456': 'i', '\u0458': 'j', '\u04bb': 'h', '\u0455': 's',
    '\u0445': 'x', '\u0501': 'd', '\u051b': 'q', '\u051d': 'w',
    # Greek confusables (UTS #39)
    '\u03b1': 'a', '\u03b5': 'e', '\u03b9': 'i', '\u03ba': 'k', '\u03bd': 'v',
    '\u03bf': 'o', '\u03c1': 'p', '\u03c4': 't', '\u03c7': 'x', '\u03c9': 'w',
    '\u0391': 'A', '\u0392': 'B', '\u0395': 'E', '\u0397': 'H', '\u0399': 'I',
    '\u039a': 'K', '\u039c': 'M', '\u039d': 'N', '\u039f': 'O', '\u03a1': 'P',
    '\u03a4': 'T', '\u03a7': 'X',
    # Fullwidth Latin
    '\uff41': 'a', '\uff42': 'b', '\uff43': 'c', '\uff44': 'd', '\uff45': 'e',
    '\uff46': 'f', '\uff47': 'g', '\uff48': 'h', '\uff49': 'i', '\uff4a': 'j',
    '\uff4b': 'k', '\uff4c': 'l', '\uff4d': 'm', '\uff4e': 'n', '\uff4f': 'o',
    '\uff50': 'p', '\uff51': 'q', '\uff52': 'r', '\uff53': 's', '\uff54': 't',
    '\uff55': 'u', '\uff56': 'v', '\uff57': 'w', '\uff58': 'x', '\uff59': 'y', '\uff5a': 'z',
    # Cyrillic Uppercase
    '\u0410': 'A', '\u0412': 'B', '\u0415': 'E', '\u041a': 'K', '\u041c': 'M',
    '\u041d': 'H', '\u041e': 'O', '\u0420': 'P', '\u0421': 'C', '\u0422': 'T',
    '\u0425': 'X', '\u0417': '3',
    '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
    '\u2013': '-', '\u2014': '-', '\u2026': '...',
    '\u00a0': ' ', '\u200b': '', '\u200c': '', '\u200d': '', '\ufeff': '',
})

def normalize_text(text: str) -> str:
    """Normalize text: unicode NFKC, lookalike replacement, whitespace cleanup."""
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(UNICODE_LOOKALIKES)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
