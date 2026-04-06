# %% [markdown]
# # P2 / 04 — Checkpoint Validation
#
# Evaluate three P2 pass/fail criteria from sweep results.
#
# **Requires:** `teacher_comparison.json`, `temperature_sweep.json` from p02/p03.
#
# **Outputs → `results/p2/`:**
# - `p2_checkpoint.json`

# %%
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINT_PATH, COMPARISON_PATH, SWEEP_PATH
from guepard.training.temperature_sweep import evaluate_p2_checkpoint

# %%
with open(COMPARISON_PATH) as f:
    comparison = json.load(f)
with open(SWEEP_PATH) as f:
    sweep_data = json.load(f)

T_calib = sweep_data["T_calib"]
T_star = sweep_data["T_star"]
sweep_results = sweep_data["sweep"]
winner_name = comparison["winner"]

# %%
checkpoint = evaluate_p2_checkpoint(
    sweep_results, T_calib, T_star, comparison, winner_name
)

with open(CHECKPOINT_PATH, "w") as f:
    json.dump(checkpoint, f, indent=2)

print("\n" + "=" * 60)
print("P2 CHECKPOINT")
print("=" * 60)
print(f"  Winner:   {winner_name} (val F1 = {checkpoint['winner_val_f1']:.4f})")
print(f"  T_calib:  {T_calib:.3f}    T*: {T_star}")
print()
for key, val in checkpoint["criteria"].items():
    print(f"  [{'PASS' if val['pass'] else 'FAIL'}] {key}")
print()
print(
    "  ✓ ALL CRITERIA PASSED — proceed to P3"
    if checkpoint["all_pass"]
    else "  ✗ SOME CRITERIA FAILED — investigate before proceeding"
)
print("=" * 60)
