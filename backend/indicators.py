"""PhishGuard AI – Security indicator extraction (deterministic, no URL fetching)."""
import re
from typing import Any

def _snip(text: str, start: int, end: int, ctx: int = 30) -> str:
    s = max(0, start - ctx)
    e = min(len(text), end + ctx)
    return text[s:e]

def detect_indicators(text: str) -> list[dict[str, Any]]:
    indicators = []
    lower = text.lower()

    # 1. Urgency / threats
    urgency = re.findall(
        r'(urgent|immediate(ly)?|act now|expire[sd]?|suspend|terminat|lock(ed)?|'
        r'within \d+ hours?|last chance|final warning|account.{0,20}(clos|restrict|limit))',
        lower
    )
    if urgency:
        m = re.search(r'(?i)(urgent|immediately|act now|expire|suspend|terminat|locked|within \d+ hours?|last chance|final warning)', text)
        indicators.append({
            "name": "Urgency/Threat Language",
            "category": "social_engineering",
            "severity": "medium",
            "evidence_snippet": _snip(text, m.start(), m.end()) if m else urgency[0][0],
            "explanation": "Message uses urgency or threat language to pressure quick action."
        })

    # 2. Credential / OTP requests
    cred = re.findall(
        r'(password|passcode|pin|otp|one.time|verification code|social security|ssn|'
        r'credit card|cvv|login credential|confirm your (identity|account|email))',
        lower
    )
    if cred:
        m = re.search(r'(?i)(password|passcode|pin|otp|verification code|ssn|credit card|cvv|confirm your)', text)
        indicators.append({
            "name": "Credential/OTP Request",
            "category": "credential_harvesting",
            "severity": "high",
            "evidence_snippet": _snip(text, m.start(), m.end()) if m else cred[0][0],
            "explanation": "Message requests sensitive credentials or one-time codes."
        })

    # 3. Payment / verification requests
    pay = re.findall(
        r'(wire transfer|western union|bitcoin|crypto|gift card|money order|pay(ment)? (immediate|now|today)|'
        r'verify your (payment|billing|bank))',
        lower
    )
    if pay:
        m = re.search(r'(?i)(wire transfer|western union|bitcoin|gift card|money order|verify your (payment|billing|bank))', text)
        indicators.append({
            "name": "Payment/Verification Request",
            "category": "financial",
            "severity": "high",
            "evidence_snippet": _snip(text, m.start(), m.end()) if m else pay[0][0],
            "explanation": "Message requests payment or financial verification."
        })

    # 4-8. URL analysis
    urls = re.findall(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', text, re.IGNORECASE)
    for url in urls[:10]:
        url_lower = url.lower()
        # Bad URL schemes
        if re.match(r'(data|javascript|vbscript):', url_lower):
            indicators.append({
                "name": "Suspicious URL Scheme",
                "category": "url_manipulation",
                "severity": "high",
                "evidence_snippet": url[:80],
                "explanation": "URL uses a potentially dangerous scheme (data/javascript/vbscript)."
            })
        # IP-based URLs
        if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_lower):
            indicators.append({
                "name": "IP-Based URL",
                "category": "url_manipulation",
                "severity": "high",
                "evidence_snippet": url[:80],
                "explanation": "URL uses a raw IP address instead of a domain name."
            })
        # Shortened URL services (masking destinations)
        shortener_patterns = ['bit.ly', 'tinyurl.com', 't.co', 'ow.ly', 'is.gd', 'buff.ly', 'rebrand.ly', 'cutt.ly', 'goo.gl', 'shorte.st']
        for shortener in shortener_patterns:
            if shortener in url_lower:
                indicators.append({
                    "name": "Shortened/Masked URL",
                    "category": "url_manipulation",
                    "severity": "medium",
                    "evidence_snippet": url[:80],
                    "explanation": f"URL uses a link shortening service ({shortener}) that masks the real destination without active network fetches."
                })
                break
        # Odd URL encoding
        if re.search(r'%[0-9a-f]{2}.*%[0-9a-f]{2}.*%[0-9a-f]{2}', url_lower):
            indicators.append({
                "name": "Excessive URL Encoding",
                "category": "url_manipulation",
                "severity": "medium",
                "evidence_snippet": url[:80],
                "explanation": "URL contains heavy percent-encoding which may hide the true destination."
            })
        # Deceptive subdomains
        brands = ['paypal', 'apple', 'google', 'microsoft', 'amazon', 'netflix', 'facebook', 'instagram',
                   'bank', 'chase', 'wellsfargo', 'citibank', 'usps', 'fedex', 'dhl', 'irs']
        domain_match = re.match(r'https?://([^/]+)', url_lower)
        if domain_match:
            domain = domain_match.group(1)
            parts = domain.split('.')
            # Brand in subdomain but not in main domain
            if len(parts) > 2:
                subdomain = '.'.join(parts[:-2])
                main = '.'.join(parts[-2:])
                for brand in brands:
                    if brand in subdomain and brand not in main:
                        indicators.append({
                            "name": "Brand-Domain Mismatch",
                            "category": "impersonation",
                            "severity": "high",
                            "evidence_snippet": url[:80],
                            "explanation": f"URL subdomain contains '{brand}' but main domain is '{main}', suggesting impersonation."
                        })
                        break

    # 9. Character substitutions / lookalikes in body
    lookalike_patterns = [
        (r'[0oO]\s*[0oO]\s*[0oO]', 'repeated zero/O substitution'),
        (r'p\s*[@aA]\s*y\s*p\s*[@aA]\s*l', 'PayPal lookalike'),
        (r'[аеіоруАВЕКМНОРСТХ]', 'Cyrillic lookalike characters'),
        (r'[ａ-ｚＡ-Ｚ]', 'fullwidth character substitution'),
    ]
    for pat, desc in lookalike_patterns:
        m = re.search(pat, text)
        if m:
            indicators.append({
                "name": "Character Substitution/Lookalike",
                "category": "obfuscation",
                "severity": "medium",
                "evidence_snippet": _snip(text, m.start(), m.end(), 20),
                "explanation": f"Detected {desc} which may be used to evade text filters."
            })
            break

    return indicators
