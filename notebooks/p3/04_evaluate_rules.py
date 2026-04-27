# %%
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, accuracy_score,
)

from gp.rules.decision_set import GreedyDecisionSet

# %%
FEATURES_PATH = Path("results/p3_rule_extraction/window_features.npz")
RULES_JSON = Path("results/p3_rule_extraction/rules/decision_set_rules.json")
PSEUDO_LABELS_CSV = Path("results/p3_rule_extraction/pseudo_labels.csv")
OUTPUT_DIR = Path("results/p3_rule_extraction")

# %%
data = np.load(FEATURES_PATH, allow_pickle=True)
X = data["X"]
y_gt = data["y"]
feature_names = data["feature_names"].tolist()
filenames = data["filenames"]

with open(RULES_JSON) as f:
    rules_data = json.load(f)

ds = GreedyDecisionSet.from_dict(rules_data, feature_names=feature_names)
print(f"Loaded {len(ds.rules)} rules")

# %%
y_pred = ds.predict(X)
y_proba = ds.predict_proba(X)

# %%
# 1. Window-level vs ground truth
print("=" * 60)
print("WINDOW-LEVEL vs GROUND TRUTH")
print("=" * 60)
auroc = roc_auc_score(y_gt, y_proba)
f1 = f1_score(y_gt, y_pred, zero_division=0)
prec = precision_score(y_gt, y_pred, zero_division=0)
rec = recall_score(y_gt, y_pred, zero_division=0)
acc = accuracy_score(y_gt, y_pred)
tn, fp, fn, tp = confusion_matrix(y_gt, y_pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

print(f"AUROC:      {auroc:.4f}")
print(f"Accuracy:   {acc:.4f}")
print(f"Precision:  {prec:.4f}")
print(f"Recall:     {rec:.4f}")
print(f"F1:         {f1:.4f}")
print(f"FPR:        {fpr:.4f}  ({fp}/{fp+tn})")
print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")

# %%
# 2. Fidelity vs Teacher pseudo-labels
pseudo_df = pd.read_csv(PSEUDO_LABELS_CSV)
rec_to_pseudo = dict(zip(pseudo_df["filename"], pseudo_df["pseudo_label"]))
window_pseudo = np.array([rec_to_pseudo.get(str(fn), "discard") for fn in filenames])
keep = (window_pseudo == "attack") | (window_pseudo == "normal")
y_pseudo = (window_pseudo[keep] == "attack").astype(int)
y_pred_filtered = y_pred[keep]

fidelity = accuracy_score(y_pseudo, y_pred_filtered)
print(f"\nFidelity vs Teacher pseudo-labels: {fidelity:.4f}")

# %%
# 3. Recording-level aggregation (max over windows)
rec_results: dict = {}
for i, fname in enumerate(filenames):
    rec_name = str(fname)
    if rec_name not in rec_results:
        rec_results[rec_name] = {"preds": [], "gt": 1 if "_exploit_" in rec_name else 0}
    rec_results[rec_name]["preds"].append(int(y_pred[i]))

rec_pred = np.array([int(max(v["preds"])) for v in rec_results.values()])
rec_gt = np.array([v["gt"] for v in rec_results.values()])

rec_auroc = roc_auc_score(rec_gt, rec_pred)
rec_f1 = f1_score(rec_gt, rec_pred, zero_division=0)
rec_prec = precision_score(rec_gt, rec_pred, zero_division=0)
rec_rec = recall_score(rec_gt, rec_pred, zero_division=0)

print("\n" + "=" * 60)
print("RECORDING-LEVEL")
print("=" * 60)
print(f"AUROC:      {rec_auroc:.4f}")
print(f"Precision:  {rec_prec:.4f}")
print(f"Recall:     {rec_rec:.4f}")
print(f"F1:         {rec_f1:.4f}")

# %%
results = {
    "window_auroc": float(auroc),
    "window_f1": float(f1),
    "window_precision": float(prec),
    "window_recall": float(rec),
    "window_fpr": float(fpr),
    "window_fidelity_vs_teacher": float(fidelity),
    "recording_auroc": float(rec_auroc),
    "recording_f1": float(rec_f1),
    "recording_precision": float(rec_prec),
    "recording_recall": float(rec_rec),
    "n_rules": len(ds.rules),
    "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
}

with open(OUTPUT_DIR / "rule_evaluation.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved evaluation to: {OUTPUT_DIR / 'rule_evaluation.json'}")

# %%
print("\n" + "=" * 60)
print(f"RULE SET ({len(ds.rules)} rules)")
print("=" * 60)
for i, rule in enumerate(ds.rules, 1):
    print(f"{i:2d}. {rule.to_human_readable()}")
