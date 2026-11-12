"""Quick E2E verification."""
import urllib.request, json

def test(name, url, method="GET", data=None):
    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
        r = urllib.request.urlopen(req, timeout=30)
        result = json.loads(r.read())
        print(f"  PASS {name}: {r.status}")
        return result
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        return None

print("=== E2E Verification ===")
test("Health", "http://localhost:8000/health")
test("Metrics", "http://localhost:8000/metrics")
test("Calibration", "http://localhost:8000/calibration")
test("Robustness", "http://localhost:8000/robustness")

r = test("Analyze phishing", "http://localhost:8000/analyze", "POST",
    json.dumps({"text": "URGENT: Verify PayPal at http://paypal.fake.com password OTP"}).encode())
if r:
    print(f"    Pred={r['label']}, Prob={r['calibrated_probability']}, Risk={r['risk_category']}, Indicators={len(r['indicators'])}")

r = test("Analyze legit", "http://localhost:8000/analyze", "POST",
    json.dumps({"text": "Hi team, reminder about our meeting on Friday at 2 PM in Conference Room B."}).encode())
if r:
    print(f"    Pred={r['label']}, Prob={r['calibrated_probability']}, Risk={r['risk_category']}")

r = test("Explain", "http://localhost:8000/explain", "POST",
    json.dumps({"text": "Click here to verify your account http://192.168.1.1/login urgently"}).encode())
if r:
    exp = r.get("explanation", {})
    print(f"    Mode={exp.get('explanation_mode')}, Verification={len(exp.get('verification', []))} claims")

# Reject empty
try:
    req = urllib.request.Request("http://localhost:8000/analyze",
        data=json.dumps({"text": ""}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req)
    print("  FAIL Empty input: should have been rejected")
except urllib.error.HTTPError as e:
    print(f"  PASS Empty input rejected: {e.code}")

print("=== Done ===")
