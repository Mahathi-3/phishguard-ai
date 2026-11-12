"""PhishGuard AI – pytest test suite."""
import sys, os, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import pytest
from fastapi.testclient import TestClient

# Lazy app import
@pytest.fixture(scope="module")
def client():
    from dotenv import load_dotenv
    load_dotenv()
    # Force template mode in sandbox (no network for Gemini)
    os.environ["GEMINI_API_KEY"] = ""
    from app import app
    yield TestClient(app)

# --- Input validation ---
def test_empty_input_rejected(client):
    r = client.post("/analyze", json={"text": ""})
    assert r.status_code in (400, 422)

def test_malformed_input_rejected(client):
    r = client.post("/analyze", json={})
    assert r.status_code == 422

def test_whitespace_only_rejected(client):
    r = client.post("/analyze", json={"text": "   "})
    assert r.status_code in (400, 422)

# --- Prediction quality ---
def test_probability_in_range(client):
    r = client.post("/analyze", json={"text": "Hello, this is a normal email about the meeting tomorrow."})
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["probability"] <= 1
    assert 0 <= data["calibrated_probability"] <= 1

def test_positive_class_is_phishing(client):
    r = client.post("/analyze", json={"text": "URGENT: Click here to verify your account http://192.168.1.1/verify password"})
    data = r.json()
    assert data["prediction"] in (0, 1)
    if data["prediction"] == 1:
        assert data["label"] == "Phishing/Spam"

def test_benign_email_not_auto_flagged(client):
    benign = "Hi team, the quarterly report is attached. Please review before Friday's meeting. Thanks, Sarah."
    r = client.post("/analyze", json={"text": benign})
    data = r.json()
    # Benign email should have low phishing probability
    assert data["calibrated_probability"] < 0.8, f"Benign email flagged with high probability: {data['calibrated_probability']}"

# --- Indicators ---
def test_indicator_urgency(client):
    r = client.post("/analyze", json={"text": "URGENT: Your account will be suspended immediately unless you act now!"})
    data = r.json()
    names = [i["name"] for i in data["indicators"]]
    assert any("Urgency" in n or "urgency" in n.lower() for n in names)

def test_indicator_credential(client):
    r = client.post("/analyze", json={"text": "Please enter your password and OTP to verify your identity."})
    data = r.json()
    names = [i["name"] for i in data["indicators"]]
    assert any("Credential" in n or "OTP" in n for n in names)

def test_indicator_ip_url(client):
    r = client.post("/analyze", json={"text": "Visit http://192.168.1.100/login to update your account."})
    data = r.json()
    names = [i["name"] for i in data["indicators"]]
    assert any("IP" in n for n in names)

# --- Robustness transformations ---
def test_transformations_work():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from scripts.run_experiments import char_substitution, insert_spaces, unicode_lookalikes
    text = "This is a test email about your account"
    assert char_substitution(text) != "" 
    assert insert_spaces(text) != ""
    assert unicode_lookalikes(text) != ""

# --- Calibration uses val not test ---
def test_calibration_note():
    cal_path = Path("results/calibration.json")
    if cal_path.exists():
        data = json.loads(cal_path.read_text())
        assert "val" in data.get("note", "").lower() or "validation" in data.get("model", "").lower()

# --- Explanation ---
def test_explanation_json_structure(client):
    r = client.post("/explain", json={"text": "URGENT: Verify your PayPal account at http://paypal.fake.com/verify"})
    assert r.status_code == 200
    data = r.json()
    exp = data["explanation"]
    assert "summary" in exp
    assert "risk_interpretation" in exp
    assert "evidence_based_findings" in exp
    assert "recommended_actions" in exp
    assert "uncertainty" in exp
    assert "explanation_mode" in exp
    assert "verification" in exp

def test_evidence_verification(client):
    r = client.post("/explain", json={"text": "URGENT: Click here immediately to avoid account closure!"})
    data = r.json()
    verification = data["explanation"]["verification"]
    assert isinstance(verification, list)
    assert len(verification) > 0
    for v in verification:
        assert "claim" in v
        assert "status" in v
        assert v["status"] in ("supported", "unsupported", "unverified")

# --- Missing key fallback ---
def test_missing_key_fallback():
    old_key = os.environ.get("GEMINI_API_KEY", "")
    os.environ["GEMINI_API_KEY"] = ""
    try:
        from explainer import get_explanation
        result = get_explanation({"prediction": 1, "probability": 0.9, "indicators": []})
        assert result["explanation_mode"] == "local_template"
    finally:
        if old_key:
            os.environ["GEMINI_API_KEY"] = old_key

def test_llm_timeout_fallback():
    """Simulate LLM failure by clearing key."""
    old_key = os.environ.get("GEMINI_API_KEY", "")
    os.environ["GEMINI_API_KEY"] = ""
    try:
        from explainer import get_explanation
        result = get_explanation({"prediction": 0, "probability": 0.1, "indicators": []})
        assert "summary" in result
        assert result["explanation_mode"] == "local_template"
    finally:
        if old_key:
            os.environ["GEMINI_API_KEY"] = old_key

# --- API integration ---
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_full_analyze_smoke(client):
    """End-to-end smoke test."""
    r = client.post("/analyze", json={
        "text": "Dear customer, your Amazon account has been locked. Verify at http://amazon.fake-verify.com/login immediately or it will be permanently deleted."
    })
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "indicators" in data
    assert "warning" in data
    assert data["warning"]  # Non-empty warning

def test_metrics_endpoint(client):
    r = client.get("/metrics")
    if r.status_code == 200:
        data = r.json()
        assert "metadata" in data
        assert "logistic_regression" in data

def test_calibration_endpoint(client):
    r = client.get("/calibration")
    if r.status_code == 200:
        data = r.json()
        assert "brier_before" in data
