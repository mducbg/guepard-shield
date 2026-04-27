# %%
import numpy as np
import json
from pathlib import Path

from gp.rules.decision_set import GreedyDecisionSet

# %%
FEATURES_PATH = Path("results/p3_rule_extraction/window_features.npz")
PSEUDO_LABELS_CSV = Path("results/p3_rule_extraction/pseudo_labels.csv")
OUTPUT_DIR = Path("results/p3_rule_extraction/rules")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_RULES = 50
MIN_PRECISION = 0.95
MIN_SUPPORT = 100

# %%
data = np.load(FEATURES_PATH, allow_pickle=True)
X = data["X"]
y_window = data["y"]
feature_names = data["feature_names"].tolist()
filenames = data["filenames"]
print(f"Feature matrix: {X.shape}")

# %%
# Use window-level ground truth labels (from exploit JSON timestamps).
# Recording pseudo-labels label all windows in an attack recording as positive,
# causing 17:1 imbalance that makes the greedy algorithm find trivial rules.
# Window-level labels are precise (only actual exploit windows = 1) and balanced.
X_f = X
y_pseudo = y_window
print(f"Using ground truth window labels: attack={np.sum(y_pseudo==1)} normal={np.sum(y_pseudo==0)}")

# %%
ds = GreedyDecisionSet(
    max_rules=MAX_RULES,
    min_precision=MIN_PRECISION,
    min_support=MIN_SUPPORT,
    feature_names=feature_names,
)
ds.fit(X_f, y_pseudo)

# %%
rules_json = ds.to_dict()
with open(OUTPUT_DIR / "decision_set_rules.json", "w") as f:
    json.dump(rules_json, f, indent=2)
print(f"\nSaved {len(rules_json)} rules to: {OUTPUT_DIR / 'decision_set_rules.json'}")

with open(OUTPUT_DIR / "rules_human_readable.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("EXTRACTED SECURITY RULES\n")
    f.write("=" * 60 + "\n\n")
    for i, rule in enumerate(ds.rules, 1):
        f.write(f"Rule {i}: {rule.to_human_readable()}\n")
        f.write(f"  Precision: {rule.precision:.4f} | Recall: {rule.recall:.4f} | Support: {rule.support}\n\n")

print(f"Saved human-readable rules to: {OUTPUT_DIR / 'rules_human_readable.txt'}")
