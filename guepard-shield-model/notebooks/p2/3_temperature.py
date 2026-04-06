# %% [markdown]
# # P2 / 03 — Temperature Sweep
#
# Extract val logits from winner teacher, run Platt scaling and DT temperature sweep.
#
# **Requires:** `vocab.json`, `vectorizer.joblib`, `teacher_comparison.json`,
#               `best_teacher_lidds.ckpt` from p01/p02.
#
# **Outputs → `results/p2/`:**
# - `temperature_sweep.json`
# - `temperature_sweep.png`

# %%
import json
import sys
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BATCH_SIZE,
    COMPARISON_PATH,
    DATA_DIR,
    IN_DIST_SCENARIOS,
    MAX_WINDOWS_PER_SEQ,
    OUTPUT_DIR,
    SEED,
    STRIDE,
    SWEEP_PATH,
    T_SWEEP,
    VECTORIZER_PATH,
    VOCAB_PATH,
    WINDOW_SIZE,
    WINNER_CKPT_PATH,
)
from guepard.config import TeacherConfig, WindowConfig
from guepard.data_loader.lidds_corpus import LiddsCorpus, lidds_label_fn, read_sc_tokens
from guepard.data_loader.splits import make_supervised_splits
from guepard.data_loader.teacher_dataset import TeacherDataset
from guepard.data_loader.vocab import SyscallVocab
from guepard.features.vectorizer import SyscallVectorizer
from guepard.models.teacher import SyscallLSTM, SyscallTransformer
from guepard.training.teacher_module import TeacherLightningModule
from guepard.training.temperature_sweep import (
    extract_val_data,
    platt_scaling,
    run_temperature_sweep,
)

L.seed_everything(SEED, workers=True)

# %% [markdown]
# ## 1. Reload Artifacts

# %%
with open(COMPARISON_PATH) as f:
    comparison = json.load(f)

winner_name = comparison["winner"]
vocab_size = comparison.get("vocab_size") or len(SyscallVocab.load(VOCAB_PATH))
print(f"Winner: {winner_name}  |  Vocab size: {vocab_size}")

vocab = SyscallVocab.load(VOCAB_PATH)
vectorizer = SyscallVectorizer.load(VECTORIZER_PATH)
window_config = WindowConfig(window_size=WINDOW_SIZE, stride=STRIDE)

# Rebuild teacher model architecture (class_weights not needed for inference)
teacher_config = TeacherConfig(vocab_size=vocab_size)
if winner_name == "transformer":
    model_cls, model_kwargs = (
        SyscallTransformer,
        {"config": teacher_config, "window_size": WINDOW_SIZE},
    )
else:
    model_cls, model_kwargs = SyscallLSTM, {"config": teacher_config}

winner_module = TeacherLightningModule.load_from_checkpoint(
    WINNER_CKPT_PATH, model=model_cls(**model_kwargs), config=teacher_config
)
winner_module.eval()
print(f"Loaded checkpoint from {WINNER_CKPT_PATH}")

# %% [markdown]
# ## 2. Build Val Dataset

# %%
corpus = LiddsCorpus(DATA_DIR, scenarios=IN_DIST_SCENARIOS)
make_supervised_splits(corpus, seed=SEED)  # assigns seq_class labels

val_ds = TeacherDataset(
    corpus=corpus,
    vocab=vocab,
    window_config=window_config,
    split_name="p2_val",
    shuffle=False,
    max_windows_per_seq=MAX_WINDOWS_PER_SEQ,
    token_reader=read_sc_tokens,
    window_label_fn=lidds_label_fn,
)
print(f"Val windows: {len(val_ds)}")

# %% [markdown]
# ## 3. Extract Logits + TF-IDF Features

# %%
print("Extracting logits and TF-IDF features...")
val_logits, val_labels, X_val, y_val = extract_val_data(
    winner_module.model, val_ds, vectorizer, BATCH_SIZE
)
print(
    f"  Windows: {len(val_logits)}  (normal={int((val_labels == 0).sum())}, attack={int((val_labels == 1).sum())})"
)
print(f"  Feature matrix: {X_val.shape}")

# %% [markdown]
# ## 4. Platt Scaling + Temperature Sweep

# %%
T_calib = platt_scaling(val_logits, val_labels)
print(f"T_calib = {T_calib:.3f}")

sweep_results = run_temperature_sweep(val_logits, X_val, y_val, T_SWEEP, seed=SEED)
for r in sweep_results:
    print(
        f"  T={r['T']:4.1f}  Attack Fid={r['attack_fidelity']:.4f}"
        f"  H(atk)={r['attack_entropy']:.4f}  H(norm)={r['normal_entropy']:.4f}"
    )

T_star = max(sweep_results, key=lambda r: r["attack_fidelity"])["T"]
print(f"\nT* = {T_star}")

with open(SWEEP_PATH, "w") as f:
    json.dump(
        {"T_calib": T_calib, "T_star": T_star, "sweep": sweep_results}, f, indent=2
    )

# %% [markdown]
# ## 5. Plot

# %%
Ts = [r["T"] for r in sweep_results]
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(
    Ts, [r["attack_fidelity"] for r in sweep_results], "b-o", label="Attack Fidelity"
)
ax1.plot(
    Ts, [r["overall_fidelity"] for r in sweep_results], "g--s", label="Overall Fidelity"
)
ax1.axvline(T_star, color="r", linestyle=":", alpha=0.7, label=f"T*={T_star}")
ax1.axvline(
    T_calib, color="orange", linestyle=":", alpha=0.7, label=f"T_calib={T_calib:.2f}"
)
ax1.set_xlabel("Temperature T")
ax1.set_ylabel("Fidelity")
ax1.legend(loc="lower right")
ax1.set_title("Temperature Sweep: DT Fidelity vs T")

ax2 = ax1.twinx()
ax2.plot(
    Ts,
    [r["attack_entropy"] for r in sweep_results],
    "r--^",
    alpha=0.5,
    label="H(attack)",
)
ax2.plot(
    Ts,
    [r["normal_entropy"] for r in sweep_results],
    "m--v",
    alpha=0.5,
    label="H(normal)",
)
ax2.set_ylabel("Entropy")
ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "temperature_sweep.png", dpi=120)
plt.show()
