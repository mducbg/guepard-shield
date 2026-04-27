# %%
import numpy as np
import json
from pathlib import Path

from gp.rules.mitre_mapper import LIDDS2021MITREMapper
from gp.rules.decision_set import GreedyDecisionSet

# %%
FEATURES_PATH = Path("results/p3_rule_extraction/window_features.npz")
RULES_JSON = Path("results/p3_rule_extraction/rules/decision_set_rules.json")
OUTPUT_DIR = Path("results/p3_rule_extraction")

# %%
data = np.load(FEATURES_PATH, allow_pickle=True)
X = data["X"]
filenames = data["filenames"]
feature_names = data["feature_names"].tolist()

with open(RULES_JSON) as f:
    rules_data = json.load(f)

ds = GreedyDecisionSet.from_dict(rules_data, feature_names=feature_names)
mapper = LIDDS2021MITREMapper()

# %%
print("=" * 60)
print("MITRE ATT&CK MAPPING")
print("=" * 60)

analyses = []
for i, rule in enumerate(ds.rules, 1):
    fired_idx = np.where(rule.evaluate(X))[0]
    fired_recs = list({str(filenames[j]).replace("_windows.npy", "") for j in fired_idx})
    analysis = mapper.analyze_rule_coverage(i, fired_recs)
    analyses.append(analysis)

    print(f"\nRule {i}: {rule.to_human_readable()}")
    print(f"  Fires on {len(fired_recs)} unique recordings")
    top3_t = list(analysis["mitre_techniques"].items())[:3]
    for tid, info in top3_t:
        print(f"  {tid} ({info['description']}): {info['count']} recordings")

# %%
all_techniques = {t for a in analyses for t in a["mitre_techniques"]}
print("\n" + "=" * 60)
print(f"OVERALL MITRE COVERAGE: {len(all_techniques)} techniques")
print("=" * 60)
for tid in sorted(all_techniques):
    print(f"  {tid}: {mapper.MITRE_DESCRIPTIONS.get(tid, 'Unknown')}")

# %%
out = OUTPUT_DIR / "mitre_mapping.json"
with open(out, "w") as f:
    json.dump(analyses, f, indent=2)
print(f"\nSaved MITRE mapping to: {out}")
