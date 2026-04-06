# %% [markdown]
# # P1-03 — Knowledge Distillation & Surrogate Evaluation
# Extracts soft labels from the teacher and evaluates Direct vs Distilled DTs.
# Requires: p01 artifacts (vocab, vectorizer) and p02 artifact (teacher checkpoint).

# %%
import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from guepard.config import TeacherConfig, WindowConfig
from guepard.data_loader.corpus import DongTingCorpus
from guepard.data_loader.surrogate_dataset import SurrogateDataset
from guepard.data_loader.teacher_dataset import TeacherDataset
from guepard.data_loader.vocab import SyscallVocab
from guepard.features.vectorizer import SyscallVectorizer
from guepard.models.surrogate import SurrogateDT
from guepard.models.teacher import SyscallLSTM
from guepard.training.distillation import extract_features_and_soft_labels
from guepard.training.teacher_module import TeacherLightningModule

from config import (
    BATCH_SIZE, DATA_DIR, LIMIT, OUTPUT_DIR, RESULTS_PATH,
    STRIDE, TEACHER_CKPT_PATH, TEMPERATURE, VECTORIZER_PATH, VOCAB_PATH, WINDOW_SIZE,
)

torch.set_float32_matmul_precision("medium")

# %%
print("Loading Corpus & Artifacts...")
corpus = DongTingCorpus(DATA_DIR)
all_train = corpus.get_split("DTDS-train")
abnormal_metas = [m for m in all_train if m.label == 1]
normal_metas   = [m for m in all_train if m.label == 0]
half_limit = LIMIT // 2
train_metas = abnormal_metas[: int(half_limit * 0.8)] + normal_metas[: int(half_limit * 0.8)]
val_metas   = abnormal_metas[int(half_limit * 0.8) : half_limit] + normal_metas[int(half_limit * 0.8) : half_limit]
for m in train_metas:
    m.seq_class = "pilot-train"
for m in val_metas:
    m.seq_class = "pilot-evaluation"
corpus.metadata = train_metas + val_metas

vocab = SyscallVocab.load(VOCAB_PATH)
vectorizer = SyscallVectorizer.load(VECTORIZER_PATH)

teacher_config = TeacherConfig(vocab_size=len(vocab), temperature=TEMPERATURE)
module = TeacherLightningModule.load_from_checkpoint(
    TEACHER_CKPT_PATH,
    model=SyscallLSTM(teacher_config),
    config=teacher_config,
)
model_eval = module.model

# %%
print("Building Datasets...")
window_config = WindowConfig(window_size=WINDOW_SIZE, stride=STRIDE)

train_teacher_ds = TeacherDataset(
    corpus=corpus, vocab=vocab, window_config=window_config,
    split_name="pilot-train", batch_size=BATCH_SIZE, shuffle=False,
)
val_teacher_ds = TeacherDataset(
    corpus=corpus, vocab=vocab, window_config=window_config,
    split_name="pilot-evaluation", batch_size=BATCH_SIZE, shuffle=False,
)
train_surrogate_ds = SurrogateDataset(
    corpus=corpus, vectorizer=vectorizer, window_config=window_config,
    split_name="pilot-train", label_source="hard", batch_size=BATCH_SIZE, shuffle=False,
)
val_surrogate_ds = SurrogateDataset(
    corpus=corpus, vectorizer=vectorizer, window_config=window_config,
    split_name="pilot-evaluation", label_source="hard", batch_size=BATCH_SIZE, shuffle=False,
)

# %%
print("Extracting soft labels...")
X_train, y_hard_train, y_soft_train = extract_features_and_soft_labels(
    train_surrogate_ds, train_teacher_ds, model_eval, TEMPERATURE, BATCH_SIZE,
)
X_val, y_hard_val, y_soft_val = extract_features_and_soft_labels(
    val_surrogate_ds, val_teacher_ds, model_eval, TEMPERATURE, BATCH_SIZE,
)
print(f"Train: {X_train.shape[0]} windows | Val: {X_val.shape[0]} windows")

# %%
print("Fitting Surrogate DTs...")
surrogate = SurrogateDT(max_depth=3)
surrogate.fit_direct(X_train, y_hard_train)
surrogate.fit_distilled(X_train, y_soft_train)

metrics = surrogate.evaluate(X_val, y_hard_val, teacher_preds=y_soft_val)
print("\nResults:")
for model_name, m in metrics.items():
    print(f"\n  {model_name.upper()}")
    print(f"    Accuracy: {m['accuracy']:.4f}  |  Macro F1: {m['f1']:.4f}")
    print(f"    Normal  — P={m['normal']['precision']:.4f}  R={m['normal']['recall']:.4f}  F1={m['normal']['f1']:.4f}")
    print(f"    Attack  — P={m['attack']['precision']:.4f}  R={m['attack']['recall']:.4f}  F1={m['attack']['f1']:.4f}")
    if "attack_fidelity" in m:
        print(f"    Fidelity (attack): {m['attack_fidelity']:.4f}")

# %%
# ── Confusion matrices ─────────────────────────────────────────────────────────
teacher_preds_val  = np.argmax(y_soft_val, axis=1)
direct_preds_val   = surrogate.direct_model.predict(X_val)
distilled_soft     = surrogate.distilled_model.predict(X_val)
distilled_preds_val = (
    np.argmax(distilled_soft, axis=1) if distilled_soft.ndim > 1 and distilled_soft.shape[1] > 1
    else (distilled_soft > 0.5).astype(int)
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
class_names = ["Normal", "Attack"]
for ax, preds, title in [
    (axes[0], teacher_preds_val,  "Teacher (BiLSTM)"),
    (axes[1], direct_preds_val,   "Direct DT"),
    (axes[2], distilled_preds_val,"Distilled DT"),
]:
    cm = confusion_matrix(y_hard_val[: len(preds)], preds)
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
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=120)
plt.show()
print("Saved: confusion_matrices.png")

# ── Feature heatmap ────────────────────────────────────────────────────────────
feature_names = vectorizer.get_feature_names()
mean_normal = X_val[y_hard_val == 0].mean(axis=0)
mean_attack = X_val[y_hard_val == 1].mean(axis=0)
top20_idx = np.argsort(np.abs(mean_attack - mean_normal))[-20:][::-1]

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
plt.savefig(OUTPUT_DIR / "feature_heatmap.png", dpi=120)
plt.show()
print("Saved: feature_heatmap.png")

# %%
results_record = {
    "timestamp": datetime.datetime.now().isoformat(),
    "config": {
        "limit": LIMIT,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "vocab_size": len(vocab),
        "max_features": vectorizer.vectorizer.max_features,
    },
    "window_balance": {
        "train_normal": int((y_hard_train == 0).sum()),
        "train_attack": int((y_hard_train == 1).sum()),
        "val_normal":   int((y_hard_val == 0).sum()),
        "val_attack":   int((y_hard_val == 1).sum()),
    },
    "metrics": metrics,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results_record, f, indent=2)
print(f"Saved: {RESULTS_PATH}")
