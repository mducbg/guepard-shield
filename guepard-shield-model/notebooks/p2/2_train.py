# %% [markdown]
# # P2 / 02 — Teacher Training
#
# Train BiLSTM and Transformer teachers, pick winner by val F1.
#
# **Requires:** `vocab.json` from p01.
#
# **Outputs → `results/p2/`:**
# - `teacher_bilstm.ckpt`, `teacher_transformer.ckpt`
# - `best_teacher_lidds.ckpt`
# - `teacher_comparison.json`
# - `architecture_comparison.png`

# %%
import json
import shutil
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
    MAX_EPOCHS,
    MAX_WINDOWS_PER_SEQ,
    NUM_WORKERS,
    OUTPUT_DIR,
    PATIENCE,
    PRECISION,
    SEED,
    STRIDE,
    VOCAB_PATH,
    WINDOW_SIZE,
    WINNER_CKPT_PATH,
)
from guepard.config import TeacherConfig, WindowConfig
from guepard.data_loader.datamodule import TeacherDataModule
from guepard.data_loader.lidds_corpus import LiddsCorpus, lidds_label_fn, read_sc_tokens
from guepard.data_loader.splits import make_supervised_splits
from guepard.data_loader.vocab import SyscallVocab
from guepard.models.teacher import SyscallLSTM, SyscallTransformer
from guepard.training.teacher_trainer import compute_class_weights, train_teacher

L.seed_everything(SEED, workers=True)

# %% [markdown]
# ## 1. Reload Corpus & Vocab

# %%
corpus = LiddsCorpus(DATA_DIR, scenarios=IN_DIST_SCENARIOS)
splits = make_supervised_splits(corpus, seed=SEED)
train_metas = splits.train

vocab = SyscallVocab.load(VOCAB_PATH)
print(f"Vocab size: {len(vocab)}")

window_config = WindowConfig(window_size=WINDOW_SIZE, stride=STRIDE)

# %% [markdown]
# ## 2. Create DataModule

# %%
datamodule = TeacherDataModule(
    corpus=corpus,
    vocab=vocab,
    window_config=window_config,
    train_split="p2_train",
    val_split="p2_val",
    batch_size=BATCH_SIZE,
    max_windows_per_seq=MAX_WINDOWS_PER_SEQ,
    seed=SEED,
    token_reader=read_sc_tokens,
    window_label_fn=lidds_label_fn,
    num_workers=NUM_WORKERS,
)
print(
    f"Train windows: {len(datamodule.train_dataset):,}  |  Val windows: {len(datamodule.val_dataset):,}"
)

# %% [markdown]
# ## 3. Train BiLSTM & Transformer

# %%
class_weights = compute_class_weights(train_metas, window_config, MAX_WINDOWS_PER_SEQ)
print(f"Class weights — normal: {class_weights[0]:.2f}  attack: {class_weights[1]:.2f}")

teacher_config = TeacherConfig(
    vocab_size=len(vocab),
    temperature=1.0,
    class_weights=class_weights,
)

bilstm_module, bilstm_history = train_teacher(
    SyscallLSTM,
    {"config": teacher_config},
    teacher_config,
    datamodule,
    OUTPUT_DIR,
    "bilstm",
    MAX_EPOCHS,
    PATIENCE,
    PRECISION,
)
transformer_module, transformer_history = train_teacher(
    SyscallTransformer,
    {"config": teacher_config, "window_size": WINDOW_SIZE},
    teacher_config,
    datamodule,
    OUTPUT_DIR,
    "transformer",
    MAX_EPOCHS,
    PATIENCE,
    PRECISION,
)

# %% [markdown]
# ## 4. Pick Winner & Save Artifacts

# %%
bilstm_f1 = max(bilstm_history.history["val_f1"], default=0)
transformer_f1 = max(transformer_history.history["val_f1"], default=0)
winner_name = "transformer" if transformer_f1 > bilstm_f1 else "bilstm"

comparison = {
    "bilstm": {
        "best_val_f1": bilstm_f1,
        "best_val_accuracy": max(bilstm_history.history["val_accuracy"], default=0),
        "epochs_trained": len(bilstm_history.history["train_loss"]),
    },
    "transformer": {
        "best_val_f1": transformer_f1,
        "best_val_accuracy": max(
            transformer_history.history["val_accuracy"], default=0
        ),
        "epochs_trained": len(transformer_history.history["train_loss"]),
    },
    "winner": winner_name,
    "vocab_size": len(vocab),
}
winner_stats: dict[str, object] = comparison[winner_name]  # type: ignore[assignment]
print(f"\nWINNER: {winner_name} (val F1 = {winner_stats['best_val_f1']:.4f})")

with open(COMPARISON_PATH, "w") as f:
    json.dump(comparison, f, indent=2)

winner_src = OUTPUT_DIR / f"teacher_{winner_name}.ckpt"
if winner_src.exists():
    shutil.copy2(winner_src, WINNER_CKPT_PATH)
    print(f"Saved winner checkpoint → {WINNER_CKPT_PATH}")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for hist, name in [(bilstm_history, "BiLSTM"), (transformer_history, "Transformer")]:
    axes[0].plot(hist.history["val_loss"], label=name)
    axes[1].plot(hist.history["val_f1"], label=name)
for ax, title in zip(axes, ["Validation Loss", "Validation F1"]):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "architecture_comparison.png", dpi=120)
plt.show()
