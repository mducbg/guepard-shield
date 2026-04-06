# %% [markdown]
# # P2 / 01 — Data Preparation
#
# Load LID-DS corpus, create supervised splits, run diagnostics,
# build vocab and TF-IDF vectorizer, validate phase segmenter.
#
# **Outputs → `results/p2/`:**
# - `vocab.json`
# - `vectorizer.joblib`
# - `diag_seq_lengths.png`

# %%
import random
import sys
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, IN_DIST_SCENARIOS, MAX_WINDOWS_PER_SEQ, NGRAM_RANGE, OUTPUT_DIR,
    SEED, STRIDE, VECTORIZER_MAX_FEATURES, VECTORIZER_PATH, VOCAB_PATH, WINDOW_SIZE,
)

from guepard.data_loader.lidds_corpus import LiddsCorpus
from guepard.data_loader.phase_segmenter import phase_summary, read_sc_timestamps, segment_phases
from guepard.data_loader.splits import make_supervised_splits
from guepard.data_loader.vocab import SyscallVocab
from guepard.data_loader.windowing import num_sliding_windows
from guepard.features.vectorizer import SyscallVectorizer
from guepard.config import WindowConfig

L.seed_everything(SEED, workers=True)

# %% [markdown]
# ## 1. Load Corpus & Create Supervised Splits

# %%
corpus = LiddsCorpus(DATA_DIR, scenarios=IN_DIST_SCENARIOS)
print(f"Total recordings: {len(corpus.metadata)}")
for s in IN_DIST_SCENARIOS:
    metas = [m for m in corpus.metadata if m.scenario == s]
    print(f"  {s}: {len(metas)} recordings ({sum(1 for m in metas if m.label == 1)} attack)")

splits = make_supervised_splits(corpus, seed=SEED)
train_metas, val_metas = splits.train, splits.val

for name, metas in [("Train", train_metas), ("Val", val_metas), ("Test", splits.test)]:
    n_norm = sum(1 for m in metas if m.label == 0)
    n_atk = sum(1 for m in metas if m.label == 1)
    print(f"  {name:5s}: {len(metas):5d} recordings  (normal={n_norm}, attack={n_atk})")

# %% [markdown]
# ## 2. Data Diagnostics

# %%
window_config = WindowConfig(window_size=WINDOW_SIZE, stride=STRIDE)

for split_name, metas in [("Train", train_metas), ("Val", val_metas)]:
    win_counts = {0: 0, 1: 0}
    for m in metas:
        capped = min(num_sliding_windows(m.seq_length, window_config), MAX_WINDOWS_PER_SEQ)
        win_counts[m.label] += capped
    ratio = win_counts[1] / max(win_counts[0], 1)
    print(f"{split_name} windows: normal={win_counts[0]:,}, attack={win_counts[1]:,}, ratio={ratio:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metas, title in [(axes[0], train_metas, "Train"), (axes[1], val_metas, "Val")]:
    ax.boxplot(
        [[m.seq_length for m in metas if m.label == 0],
         [m.seq_length for m in metas if m.label == 1]],
        tick_labels=["Normal", "Attack"], showfliers=False,
    )
    ax.set_title(f"{title}: Sequence Length by Class")
    ax.set_ylabel("# exit events")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "diag_seq_lengths.png", dpi=120)
plt.show()

# %% [markdown]
# ## 3. Build Vocab & Vectorizer

# %%
train_seqs = list(corpus.iter_sequences("p2_train"))

vocab = SyscallVocab()
vocab.build((tokens for _, _, tokens in train_seqs), total=len(train_seqs))
vocab.save(VOCAB_PATH)
print(f"Vocab size: {len(vocab)}  →  saved to {VOCAB_PATH}")

vectorizer = SyscallVectorizer(max_features=VECTORIZER_MAX_FEATURES, ngram_range=NGRAM_RANGE)
vectorizer.fit((tokens for _, _, tokens in train_seqs), total=len(train_seqs))
vectorizer.save(VECTORIZER_PATH)
print(f"Vectorizer features: {len(vectorizer.get_feature_names())}  →  saved to {VECTORIZER_PATH}")

del train_seqs

# %% [markdown]
# ## 4. Phase Segmentation (sample validation)

# %%
phase_stats: dict[str, list[dict]] = {"normal": [], "attack": []}
for meta in random.Random(SEED).sample(
    [m for m in train_metas if m.seq_length > 100], min(20, len(train_metas))
):
    ts = read_sc_timestamps(meta.file_path)
    if len(ts) >= 10:
        phase_stats["attack" if meta.label == 1 else "normal"].append(
            phase_summary(segment_phases(ts))
        )

for cls, stats in phase_stats.items():
    if stats:
        print(f"\n  {cls.upper()} (n={len(stats)}):")
        for phase in ["startup", "active", "idle", "shutdown"]:
            vals = [s.get(phase, 0) for s in stats]
            print(f"    {phase:10s}: mean={np.mean(vals):8.1f}  std={np.std(vals):8.1f}")
