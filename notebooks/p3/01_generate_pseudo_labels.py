# %%
import pandas as pd
import numpy as np
from pathlib import Path

# %%
TEACHER_CSV = Path("results/evaluation/transformer/recording_predictions.csv")
OUTPUT_DIR = Path("results/p3_rule_extraction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POS_THR = 0.74  # attack:  1% normal contamination, 93.9% attack coverage
NEG_THR = 0.50  # normal: 81% normal coverage,  2.1% attack contamination

# %%
df = pd.read_csv(TEACHER_CSV)
print(f"Loaded {len(df)} recording predictions")
print(f"Columns: {df.columns.tolist()}")
print(f"Labels: {df['label'].value_counts().to_dict()}")

# %%
def assign_pseudo_label(score: float) -> str:
    if score >= POS_THR:
        return "attack"
    elif score <= NEG_THR:
        return "normal"
    return "discard"

df["pseudo_label"] = df["score"].apply(assign_pseudo_label)

# %%
print("\nPseudo-label distribution:")
print(df["pseudo_label"].value_counts())

for lbl in ["attack", "normal", "discard"]:
    sub = df[df["pseudo_label"] == lbl]["score"]
    if len(sub):
        print(f"\n{lbl}: n={len(sub)}, score=[{sub.min():.4f}, {sub.max():.4f}]")

# %%
out = OUTPUT_DIR / "pseudo_labels.csv"
df.to_csv(out, index=False)
print(f"\nSaved to: {out}")
