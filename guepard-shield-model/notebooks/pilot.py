# %%
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress XLA/JAX compiler spam
import datetime
import json

import jax
import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from guepard.config import TeacherConfig, WindowConfig
from guepard.data_loader.corpus import DongTingCorpus
from guepard.data_loader.surrogate_dataset import SurrogateDataset
from guepard.data_loader.teacher_dataset import TeacherDataset
from guepard.data_loader.vocab import SyscallVocab
from guepard.data_loader.windowing import num_sliding_windows
from guepard.features.vectorizer import SyscallVectorizer
from guepard.models.surrogate import SurrogateDT
from guepard.models.teacher import SyscallLSTM
from sklearn.metrics import confusion_matrix

# %%
print(jax.devices())
print(f"Default backend: {jax.default_backend()}")

# %% [markdown]
# ## 1. Setup Data Paths & Output Directory

# %%
data_dir = "../data/processed/DongTing"
output_dir = "../results/pilot/no-class-weight"

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
LIMIT = 2000

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

# Override metadata logic for the PyDataset fetchers
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
# Cap windows per sequence to reduce class imbalance (diagnostic: 125:1 → ~2:1)
# P90 of normal class = 21; capping attack sequences at same value balances training signal.
MAX_WINDOWS = (
    5  # set near avg natural windows of normal class (~4.5) to balance window ratio
)

train_ds = TeacherDataset(
    corpus=corpus,
    vocab=vocab,
    window_config=window_config,
    split_name="pilot-train",
    batch_size=1024,
    shuffle=True,
    max_windows_per_seq=MAX_WINDOWS,
)
val_ds = TeacherDataset(
    corpus=corpus,
    vocab=vocab,
    window_config=window_config,
    split_name="pilot-evaluation",
    batch_size=1024,
    shuffle=False,
    max_windows_per_seq=MAX_WINDOWS,
)
print(f"Train batches: {len(train_ds)}, Val batches: {len(val_ds)}")


# %% [markdown]
# ## 5. Build & Train Teacher Model
# %%
print("Building & Compiling Teacher Model (BiLSTM)...")
teacher_config = TeacherConfig(vocab_size=len(vocab), temperature=4.0)
teacher_model = SyscallLSTM(teacher_config)

optimizer = keras.optimizers.AdamW(
    learning_rate=teacher_config.lr, weight_decay=teacher_config.weight_decay
)
teacher_model.compile(
    optimizer=optimizer,
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# Trigger model build with dummy batch
dummy_x, dummy_y = train_ds[0]
teacher_model(dummy_x)
teacher_model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(output_path / "best_teacher.weights.h5"),
        save_best_only=True,
        save_weights_only=True,
    ),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
]

print("Training Teacher...")
history = teacher_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="val")
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
    batch_size=1024,
    shuffle=False,
    max_windows_per_seq=MAX_WINDOWS,
)

surrogate_train_ds = SurrogateDataset(
    corpus=corpus,
    vectorizer=vectorizer,
    window_config=window_config,
    split_name="pilot-train",
    label_source="hard",
    batch_size=1024,
    shuffle=False,
    max_windows_per_seq=MAX_WINDOWS,
)

extract_val_ds = TeacherDataset(
    corpus=corpus,
    vocab=vocab,
    window_config=window_config,
    split_name="pilot-evaluation",
    batch_size=1024,
    shuffle=False,
    max_windows_per_seq=MAX_WINDOWS,
)

surrogate_val_ds = SurrogateDataset(
    corpus=corpus,
    vectorizer=vectorizer,
    window_config=window_config,
    split_name="pilot-evaluation",
    label_source="hard",
    batch_size=1024,
    shuffle=False,
    max_windows_per_seq=MAX_WINDOWS,
)


def extract_features_and_soft_labels(surrogate_dataset, teacher_dataset):
    """Extract all windows — no limit_batches, use full dataset.

    Pass 1: collect TF-IDF features, hard labels, and token ID batches (cheap).
    Pass 2: single model.predict() over all tokens — one JIT compile, full GPU util.
    """
    assert len(surrogate_dataset) == len(teacher_dataset), (
        f"Dataset length mismatch: surrogate={len(surrogate_dataset)} teacher={len(teacher_dataset)}"
    )
    f_list, h_list, token_list = [], [], []
    for i in range(len(surrogate_dataset)):
        feats, hard_y = surrogate_dataset[i]
        token_ids, _ = teacher_dataset[i]
        f_list.append(feats)
        h_list.append(hard_y)
        token_list.append(token_ids)

    all_tokens = np.vstack(token_list)  # (N, window_size) int32
    logits = teacher_model.predict(all_tokens, batch_size=1024, verbose=1)
    soft_y = keras.ops.softmax(logits / teacher_config.temperature)

    return (
        np.vstack(f_list),
        np.concatenate(h_list),
        np.array(soft_y),
    )


print("   Extracting Training features (all batches)...")
X_train, y_hard_train, y_soft_train = extract_features_and_soft_labels(
    surrogate_train_ds, extract_train_ds
)
print(
    f"   Train: {X_train.shape[0]} windows, label balance: "
    f"normal={int((y_hard_train == 0).sum())}, "
    f"attack={int((y_hard_train == 1).sum())}"
)

print("   Extracting Validation features (all batches)...")
X_val, y_hard_val, y_soft_val = extract_features_and_soft_labels(
    surrogate_val_ds, extract_val_ds
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
        "batch_size": train_ds.batch_size,
        "temperature": teacher_config.temperature,
        "d_model": teacher_config.d_model,
        "vocab_size": len(vocab),
        "max_features": 1000,
        "ngram_range": [1, 2],
        "surrogate_max_depth": surrogate.direct_model.max_depth,
    },
    "teacher": {
        "val_accuracy": float(max(history.history["val_accuracy"])),
        "val_loss_final": float(history.history["val_loss"][-1]),
        "epochs_trained": len(history.history["loss"]),
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

# %%
