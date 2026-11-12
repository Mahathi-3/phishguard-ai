import json
m = json.load(open("results/metrics.json"))
lr = m["logistic_regression"]; nb = m["multinomial_nb"]; md = m["metadata"]
c = json.load(open("results/calibration.json"))
r = json.load(open("results/robustness.json"))
print("=== METRICS ===")
print(f"Dataset: {md['dataset']}, rows={md['total_rows']}, phish={md['phishing_count']}, legit={md['legitimate_count']}")
print(f"Split: {md['split_method']}, train={md['train_size']}, val={md['val_size']}, test={md['test_size']}")
print(f"LR: acc={lr['accuracy']:.4f} prec={lr['precision']:.4f} rec={lr['recall']:.4f} f1={lr['f1']:.4f} fpr={lr['fpr']:.4f}")
print(f"NB: acc={nb['accuracy']:.4f} prec={nb['precision']:.4f} rec={nb['recall']:.4f} f1={nb['f1']:.4f} fpr={nb['fpr']:.4f}")
print(f"Brier before={c['brier_before']:.4f} after={c['brier_after']:.4f}")
print(f"CM: {lr['confusion_matrix']}")
print("=== ROBUSTNESS ===")
orig = r["original"]
print(f"Original: acc={orig['accuracy']:.4f} f1={orig['f1']:.4f} fpr={orig['fpr']:.4f}")
for name, t in r["transformations"].items():
    nd = t["no_defense"]; wd = t["with_defense"]
    print(f"{name}: F1_noDef={nd['f1']:.4f} F1_wDef={wd['f1']:.4f} dF1={t['delta_f1_no_defense']:.4f} flipped={t['flipped_predictions']}/{t['flipped_after_defense']}")
