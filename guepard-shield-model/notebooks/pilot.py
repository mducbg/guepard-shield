import datetime
import json
import warnings
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F
from guepard.config import TeacherConfig, WindowConfig
from guepard.data_loader.corpus import DongTingCorpus
from guepard.data_loader.datamodule import TeacherDataModule
from guepard.data_loader.surrogate_dataset import SurrogateDataset
from guepard.data_loader.teacher_dataset import TeacherDataset
from guepard.data_loader.vocab import SyscallVocab
from guepard.data_loader.windowing import num_sliding_windows
from guepard.features.vectorizer import SyscallVectorizer
from guepard.models.surrogate import SurrogateDT
from guepard.models.teacher import SyscallLSTM
from guepard.training.teacher_module import TeacherLightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.serialization import SourceChangeWarning
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=SourceChangeWarning)
warnings.filterwarnings("ignore", message=".*LeafSpec.*deprecated.*")
torch.set_float32_matmul_precision("medium")

# %%
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 1. Setup Data Paths & Output Directory

# %%
data_dir = "../data/processed/DongTing"
output_dir = "../results/pilot/no-class-weight-torch"

output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)
soft_labels_dir = output_path / "soft_labels"
soft_labels_dir.mkdir(exist_ok=True)

# %% [markdown]
# ## 2. Load Corpus

# %%
print("Loading Corpus...")
corpus = DongTingCorpus(data_dir)

# Note: We limit sequence lengths here for pilot speed.
LIMIT = 500

# Extract subsets robustly to guarantee a validation split exists
all_train = corpus.get_split("DTDS-train")
abnormal_metas = [m for m in all_train if m.label == 1]
normal_metas = [m for m in all_train if m.label == 0]

# Sample evenly
half_limit = LIMIT // 2
train_abnormal = abnormal_metas[: int(half_limit * 0.8)]
val_abnormal = abnormal_metas[int(half_limit * 0.8) : half_limit]

train_normal = normal_metas[: int(half_limit * 0.8)]
val_normal = normal_metas[int(half_limit * 0.8) : half_limit]

train_metas = train_abnormal + train_normal
val_metas = val_abnormal + val_normal

# Override metadata logic for the dataset fetchers
for m in train_metas:
    m.seq_class = "pilot-train"
for m in val_metas:
    m.seq_class = "pilot-evaluation"

corpus.metadata = train_metas + val_metas

# %% [markdown]
# ## 2b. Data Diagnostics

# %%
print("\n=== DATA DIAGNOSTICS ===")

# 1. Label distribution
print(
    f"\nTrain  — normal: {sum(1 for m in train_metas if m.label == 0)}, attack: {sum(1 for m in train_metas if m.label == 1)}"
)
print(
    f"Val    — normal: {sum(1 for m in val_metas if m.label == 0)}, attack: {sum(1 for m in val_metas if m.label == 1)}"
)

# 2. seq_length (metadata) vs actual file token count — sample 10 sequences
print("\nseq_length sanity check (metadata vs file, first 10 train seqs):")
length_diffs = []
for meta in train_metas[:10]:
    try:
        with open(meta.file_path, "r", encoding="utf-8") as f:
            actual = len(f.read().strip().split("|"))
        diff = actual - meta.seq_length
        length_diffs.append(diff)
        print(
            f"  {meta.seq_id}: metadata={meta.seq_length}, file={actual}, diff={diff:+d}"
        )
    except Exception as e:
        print(f"  {meta.seq_id}: ERROR — {e}")

if length_diffs:
    print(
        f"  Mean diff: {np.mean(length_diffs):.1f}, Max |diff|: {max(abs(d) for d in length_diffs)}"
    )

# 3. Window count and class balance after windowing
window_config_diag = WindowConfig(window_size=64, stride=12)
train_win = {0: 0, 1: 0}
for m in train_metas:
    train_win[m.label] += num_sliding_windows(m.seq_length, window_config_diag)
val_win = {0: 0, 1: 0}
for m in val_metas:
    val_win[m.label] += num_sliding_windows(m.seq_length, window_config_diag)

print("\nWindow counts (metadata-based):")
print(
    f"  Train — normal: {train_win[0]}, attack: {train_win[1]}, ratio: {train_win[1] / max(train_win[0], 1):.2f}"
)
print(
    f"  Val   — normal: {val_win[0]}, attack: {val_win[1]}, ratio: {val_win[1] / max(val_win[0], 1):.2f}"
)

# Plot: sequence length distribution by class
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metas, title in [(axes[0], train_metas, "Train"), (axes[1], val_metas, "Val")]:
    normal_lens = [m.seq_length for m in metas if m.label == 0]
    attack_lens = [m.seq_length for m in metas if m.label == 1]
    ax.boxplot([normal_lens, attack_lens], tick_labels=["Normal", "Attack"])
    ax.set_title(f"{title}: Sequence Length by Class")
    ax.set_ylabel("seq_length (metadata)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(output_path / "diag_seq_lengths.png", dpi=120)
plt.show()
print("Saved: diag_seq_lengths.png")

# %% [markdown]
# ## 3. Build Vocab & Feature Vectorizer

# %%
print("Building Vocab & Vectorizer...")
vocab = SyscallVocab()

# Stream directly — do NOT materialise all tokens into a list (OOM for long attack seqs)
n_seqs = len(corpus.metadata)
vocab.build(corpus.iter_corpus(), total=n_seqs)
print(f"Vocab size: {len(vocab)}")

vectorizer = SyscallVectorizer(max_features=1000, ngram_range=(1, 2))
vectorizer.fit(corpus.iter_corpus(), total=n_seqs)

# %% [markdown]
# ## 4. Init DataLoaders

# %%
print("Building Datasets...")
window_config = WindowConfig(window_size=64, stride=12)
MAX_WINDOWS = 5
BATCH_SIZE = 1024

datamodule = TeacherDataModule(
    corpus=corpus,
    vocab=vocab,
    window_config=window_config,
    train_split="pilot-train",
    val_split="pilot-evaluation",
    batch_size=BATCH_SIZE,
    max_windows_per_seq=MAX_WINDOWS,
)

print(
    f"Train samples: {len(datamodule.train_dataset)}, "
    f"Val samples: {len(datamodule.val_dataset)}"
)

# %% [markdown]
# ## 5. Build & Train Teacher Model


# %%
class DatasetReshuffleCallback(L.Callback):
    """Rebuilds sequence-level shuffle order after each training epoch."""

    def __init__(self, datamodule: TeacherDataModule):
        self.datamodule = datamodule

    def on_train_epoch_end(self, trainer, pl_module):
        if self.datamodule.train_dataset is not None:
            self.datamodule.train_dataset.reshuffle()


class MetricsHistory(L.Callback):
    """Collects per-epoch metrics, mirroring Keras history.history."""

    def __init__(self):
        self.history: dict[str, list] = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        self.history["loss"].append(
            float(trainer.callback_metrics.get("train_loss", 0))
        )
        self.history["accuracy"].append(
            float(trainer.callback_metrics.get("train_accuracy", 0))
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        self.history["val_loss"].append(
            float(trainer.callback_metrics.get("val_loss", 0))
        )
        self.history["val_accuracy"].append(
            float(trainer.callback_metrics.get("val_accuracy", 0))
        )


print("Building Teacher Model (BiLSTM)...")
teacher_config = TeacherConfig(vocab_size=len(vocab), temperature=4.0)
teacher_model = SyscallLSTM(teacher_config)
module = TeacherLightningModule(teacher_model, teacher_config)

metrics_history = MetricsHistory()

callbacks = [
    ModelCheckpoint(
        dirpath=str(output_path),
        filename="best_teacher",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    ),
    EarlyStopping(monitor="val_loss", patience=10, mode="min"),
    DatasetReshuffleCallback(datamodule),
    metrics_history,
]

trainer = L.Trainer(
    max_epochs=50,
    callbacks=callbacks,
    enable_progress_bar=True,
    log_every_n_steps=1,
)

print("Training Teacher...")
trainer.fit(module, datamodule)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(metrics_history.history["loss"], label="train")
axes[0].plot(metrics_history.history["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(metrics_history.history["accuracy"], label="train")
axes[1].plot(metrics_history.history["val_accuracy"], label="val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.savefig(output_path / "training_history.png", dpi=120)
plt.show()
print("Saved: training_history.png")

# %% [markdown]
# ## 6. Compute Knowledge Distillation Soft Targets

# %%
print("Extracting predictions & resolving SurrogateDT features...")

extract_train_ds = TeacherDataset(
    corpus=corpus,
    vocab=vocab,
    window_config=window_config,
    split_name="pilot-train",
    batch_size=BATCH_SIZE,
    shuffle=False,
    max_windows_per_seq=MAX_WINDOWS,
)

surrogate_train_ds = SurrogateDataset(
    corpus=corpus,
    vectorizer=vectorizer,
    window_config=window_config,
    split_name="pilot-train",
    label_source="hard",
    batch_size=BATCH_SIZE,
    shuffle=False,
    max_windows_per_seq=MAX_WINDOWS,
)

extract_val_ds = TeacherDataset(
    corpus=corpus,
    vocab=vocab,
    window_config=window_config,
    split_name="pilot-evaluation",
    batch_size=BATCH_SIZE,
    shuffle=False,
    max_windows_per_seq=MAX_WINDOWS,
)

surrogate_val_ds = SurrogateDataset(
    corpus=corpus,
    vectorizer=vectorizer,
    window_config=window_config,
    split_name="pilot-evaluation",
    label_source="hard",
    batch_size=BATCH_SIZE,
    shuffle=False,
    max_windows_per_seq=MAX_WINDOWS,
)


def extract_features_and_soft_labels(
    surrogate_dataset: SurrogateDataset,
    teacher_dataset: TeacherDataset,
    model: nn.Module,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass extraction: TF-IDF features, hard labels, and soft labels."""
    assert len(surrogate_dataset) == len(teacher_dataset), (
        f"Dataset length mismatch: surrogate={len(surrogate_dataset)} teacher={len(teacher_dataset)}"
    )
    device = next(model.parameters()).device
    surrogate_loader = DataLoader(
        surrogate_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    teacher_loader = DataLoader(teacher_dataset, batch_size=BATCH_SIZE, shuffle=False)

    f_list, h_list, logits_list = [], [], []
    model.eval()
    with torch.no_grad():
        for (feats, hard_y), (token_ids, _) in zip(surrogate_loader, teacher_loader):
            f_list.append(feats.numpy())
            h_list.append(hard_y.numpy())
            logits_list.append(model(token_ids.to(device)).cpu())

    logits = torch.cat(logits_list, dim=0)
    soft_y = F.softmax(logits / temperature, dim=-1).numpy()
    return np.vstack(f_list), np.concatenate(h_list), soft_y


model_eval = module.model

print("   Extracting Training features (all batches)...")
X_train, y_hard_train, y_soft_train = extract_features_and_soft_labels(
    surrogate_train_ds, extract_train_ds, model_eval, teacher_config.temperature
)
print(
    f"   Train: {X_train.shape[0]} windows, label balance: "
    f"normal={int((y_hard_train == 0).sum())}, "
    f"attack={int((y_hard_train == 1).sum())}"
)

print("   Extracting Validation features (all batches)...")
X_val, y_hard_val, y_soft_val = extract_features_and_soft_labels(
    surrogate_val_ds, extract_val_ds, model_eval, teacher_config.temperature
)
print(
    f"   Val:   {X_val.shape[0]} windows, label balance: "
    f"normal={int((y_hard_val == 0).sum())}, "
    f"attack={int((y_hard_val == 1).sum())}"
)

# %% [markdown]
# ## 7. Fit & Compare Surrogate Decision Trees

# %%
print("Fitting Direct & Distilled Surrogate DTs...")
surrogate = SurrogateDT(max_depth=3)
# Train using truth labels
surrogate.fit_direct(X_train, y_hard_train)
# Train using teacher probability distributions
surrogate.fit_distilled(X_train, y_soft_train)

# EVALUATE on validation split to expose true metrics instead of overfitted train metrics
metrics = surrogate.evaluate(X_val, y_hard_val, teacher_preds=y_soft_val)
print("\nResults:")
for model_name, m in metrics.items():
    print(f"\n  {model_name.upper()} MODEL")
    print(f"    Accuracy:        {m['accuracy']:.4f}")
    print(
        f"    Macro F1:        {m['f1']:.4f}  (P={m['precision']:.4f}  R={m['recall']:.4f})"
    )
    print(
        f"    Normal  — P={m['normal']['precision']:.4f}  R={m['normal']['recall']:.4f}  "
        f"F1={m['normal']['f1']:.4f}  support={m['normal']['support']}"
    )
    print(
        f"    Attack  — P={m['attack']['precision']:.4f}  R={m['attack']['recall']:.4f}  "
        f"F1={m['attack']['f1']:.4f}  support={m['attack']['support']}"
    )
    if "fidelity_to_teacher" in m:
        print(f"    Fidelity (overall):  {m['fidelity_to_teacher']:.4f}")
    if "attack_fidelity" in m:
        print(f"    Fidelity (attack):   {m['attack_fidelity']:.4f}  ← pivot metric")

# %% [markdown]
# ## 8. Diagnostic Plots

# %%
y_hard_val_np = np.array(y_hard_val)
y_soft_val_np = np.array(y_soft_val)
# teacher_preds already computed during extraction as argmax of soft labels
teacher_preds_val = np.argmax(y_soft_val_np, axis=1)

# Direct DT predictions
direct_preds_val = surrogate.direct_model.predict(np.array(X_val))

# Distilled DT predictions
distilled_soft = surrogate.distilled_model.predict(np.array(X_val))
if distilled_soft.ndim > 1 and distilled_soft.shape[1] > 1:
    distilled_preds_val = np.argmax(distilled_soft, axis=1)
else:
    distilled_preds_val = (distilled_soft > 0.5).astype(int)

# 1. Confusion matrices (3 subplots)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
class_names = ["Normal", "Attack"]
for ax, preds, title in [
    (axes[0], teacher_preds_val, "Teacher (BiLSTM)"),
    (axes[1], direct_preds_val, "Direct DT"),
    (axes[2], distilled_preds_val, "Distilled DT"),
]:
    cm = confusion_matrix(y_hard_val_np[: len(preds)], preds)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
plt.tight_layout()
plt.savefig(output_path / "confusion_matrices.png", dpi=120)
plt.show()
print("Saved: confusion_matrices.png")

# 2. Top-20 n-gram features: mean value by class (heatmap)
X_val_np = np.array(X_val)
feature_names = vectorizer.get_feature_names()
mean_normal = X_val_np[y_hard_val_np == 0].mean(axis=0)
mean_attack = X_val_np[y_hard_val_np == 1].mean(axis=0)
diff = np.abs(mean_attack - mean_normal)
top20_idx = np.argsort(diff)[-20:][::-1]

heatmap_data = np.stack([mean_normal[top20_idx], mean_attack[top20_idx]])
fig, ax = plt.subplots(figsize=(14, 3))
im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Normal", "Attack"])
ax.set_xticks(range(20))
ax.set_xticklabels(feature_names[top20_idx], rotation=45, ha="right", fontsize=8)
ax.set_title("Top-20 Most Discriminative N-gram Features (mean count by class)")
plt.colorbar(im, ax=ax, label="Mean count")
plt.tight_layout()
plt.savefig(output_path / "feature_heatmap.png", dpi=120)
plt.show()
print("Saved: feature_heatmap.png")

# %% [markdown]
# ## 9. Save Results

# %%
results_record = {
    "timestamp": datetime.datetime.now().isoformat(),
    "config": {
        "limit": LIMIT,
        "max_windows_per_seq": MAX_WINDOWS,
        "window_size": window_config.window_size,
        "stride": window_config.stride,
        "batch_size": BATCH_SIZE,
        "temperature": teacher_config.temperature,
        "d_model": teacher_config.d_model,
        "vocab_size": len(vocab),
        "max_features": 1000,
        "ngram_range": [1, 2],
        "surrogate_max_depth": surrogate.direct_model.max_depth,
    },
    "teacher": {
        "val_accuracy": float(max(metrics_history.history["val_accuracy"])),
        "val_loss_final": float(metrics_history.history["val_loss"][-1]),
        "epochs_trained": len(metrics_history.history["loss"]),
    },
    "window_balance": {
        "train_normal": int((y_hard_train == 0).sum()),
        "train_attack": int((y_hard_train == 1).sum()),
        "val_normal": int((y_hard_val == 0).sum()),
        "val_attack": int((y_hard_val == 1).sum()),
    },
    "metrics": metrics,
}

results_path = output_path / "pilot_results.json"
with open(results_path, "w") as f:
    json.dump(results_record, f, indent=2)
print(f"Saved: {results_path}")
