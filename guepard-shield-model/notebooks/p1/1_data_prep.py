# %% [markdown]
# # P1-01 — Data Preparation
# Loads the DongTing corpus (pilot subset), runs diagnostics,
# builds vocabulary and TF-IDF vectorizer, saves both as artifacts.

# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from config import (
    DATA_DIR,
    LIMIT,
    NGRAM_RANGE,
    OUTPUT_DIR,
    STRIDE,
    VECTORIZER_MAX_FEATURES,
    VECTORIZER_PATH,
    VOCAB_PATH,
    WINDOW_SIZE,
)
from guepard.config import WindowConfig
from guepard.data_loader.corpus import DongTingCorpus
from guepard.data_loader.vocab import SyscallVocab
from guepard.data_loader.windowing import num_sliding_windows
from guepard.features.vectorizer import SyscallVectorizer

torch.set_float32_matmul_precision("medium")

# %%
print("Loading Corpus...")
corpus = DongTingCorpus(DATA_DIR)

all_train = corpus.get_split("DTDS-train")
abnormal_metas = [m for m in all_train if m.label == 1]
normal_metas = [m for m in all_train if m.label == 0]

half_limit = LIMIT // 2
train_abnormal = abnormal_metas[: int(half_limit * 0.8)]
val_abnormal = abnormal_metas[int(half_limit * 0.8) : half_limit]
train_normal = normal_metas[: int(half_limit * 0.8)]
val_normal = normal_metas[int(half_limit * 0.8) : half_limit]

train_metas = train_abnormal + train_normal
val_metas = val_abnormal + val_normal

for m in train_metas:
    m.seq_class = "pilot-train"
for m in val_metas:
    m.seq_class = "pilot-evaluation"

corpus.metadata = train_metas + val_metas
print(f"Train: {len(train_metas)}, Val: {len(val_metas)}")

# %%
print("\n=== DATA DIAGNOSTICS ===")
print(
    f"\nTrain  — normal: {sum(1 for m in train_metas if m.label == 0)}, "
    f"attack: {sum(1 for m in train_metas if m.label == 1)}"
)
print(
    f"Val    — normal: {sum(1 for m in val_metas if m.label == 0)}, "
    f"attack: {sum(1 for m in val_metas if m.label == 1)}"
)

window_config = WindowConfig(window_size=WINDOW_SIZE, stride=STRIDE)
train_win = {0: 0, 1: 0}
val_win = {0: 0, 1: 0}
for m in train_metas:
    train_win[m.label] += num_sliding_windows(m.seq_length, window_config)
for m in val_metas:
    val_win[m.label] += num_sliding_windows(m.seq_length, window_config)

print(
    f"\nWindow counts: Train — normal: {train_win[0]}, attack: {train_win[1]}, "
    f"ratio: {train_win[1] / max(train_win[0], 1):.2f}"
)
print(
    f"               Val   — normal: {val_win[0]}, attack: {val_win[1]}, "
    f"ratio: {val_win[1] / max(val_win[0], 1):.2f}"
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metas, title in [(axes[0], train_metas, "Train"), (axes[1], val_metas, "Val")]:
    normal_lens = [m.seq_length for m in metas if m.label == 0]
    attack_lens = [m.seq_length for m in metas if m.label == 1]
    ax.boxplot([normal_lens, attack_lens], tick_labels=["Normal", "Attack"])
    ax.set_title(f"{title}: Sequence Length by Class")
    ax.set_ylabel("seq_length")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "diag_seq_lengths.png", dpi=120)
plt.show()
print("Saved: diag_seq_lengths.png")

# %%
print("Building Vocab & Vectorizer...")
vocab = SyscallVocab()
n_seqs = len(corpus.metadata)
vocab.build(corpus.iter_corpus(), total=n_seqs)
print(f"Vocab size: {len(vocab)}")

vectorizer = SyscallVectorizer(
    max_features=VECTORIZER_MAX_FEATURES, ngram_range=NGRAM_RANGE
)
vectorizer.fit(corpus.iter_corpus(), total=n_seqs)

vocab.save(VOCAB_PATH)
vectorizer.save(VECTORIZER_PATH)
print(f"Saved vocab → {VOCAB_PATH}")
print(f"Saved vectorizer → {VECTORIZER_PATH}")
