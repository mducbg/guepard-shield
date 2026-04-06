# %% [markdown]
# # P1-02 — Train Teacher Model
# Builds the TeacherDataModule and trains a BiLSTM teacher.
# Requires: p01_data_prep.py artifacts (vocab, vectorizer).

# %%
import matplotlib.pyplot as plt
import torch

from guepard.config import TeacherConfig, WindowConfig
from guepard.data_loader.corpus import DongTingCorpus
from guepard.data_loader.datamodule import TeacherDataModule
from guepard.data_loader.vocab import SyscallVocab
from guepard.models.teacher import SyscallLSTM
from guepard.training.teacher_trainer import train_teacher

from config import (
    BATCH_SIZE, DATA_DIR, LIMIT, MAX_EPOCHS, MAX_WINDOWS, OUTPUT_DIR,
    PATIENCE, PRECISION, STRIDE, TEMPERATURE, VOCAB_PATH, WINDOW_SIZE,
)

torch.set_float32_matmul_precision("medium")

# %%
print("Loading Corpus...")
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
print(f"Vocab size: {len(vocab)}")

# %%
print("Building DataModule...")
window_config = WindowConfig(window_size=WINDOW_SIZE, stride=STRIDE)
datamodule = TeacherDataModule(
    corpus=corpus,
    vocab=vocab,
    window_config=window_config,
    train_split="pilot-train",
    val_split="pilot-evaluation",
    batch_size=BATCH_SIZE,
    max_windows_per_seq=MAX_WINDOWS,
)
print(f"Train: {len(datamodule.train_dataset)}, Val: {len(datamodule.val_dataset)}")

# %%
print("Training Teacher (BiLSTM)...")
teacher_config = TeacherConfig(vocab_size=len(vocab), temperature=TEMPERATURE)

module, history = train_teacher(
    model_cls=SyscallLSTM,
    model_kwargs={"config": teacher_config},
    config=teacher_config,
    datamodule=datamodule,
    output_dir=OUTPUT_DIR,
    tag="bilstm",
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    precision=PRECISION,
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["train_loss"], label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history.history.get("train_f1", []), label="train")
axes[1].plot(history.history["val_f1"], label="val")
axes[1].set_title("F1")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_history.png", dpi=120)
plt.show()
print("Saved: training_history.png")
print(f"Best val_f1: {max(history.history['val_f1']):.4f}")
