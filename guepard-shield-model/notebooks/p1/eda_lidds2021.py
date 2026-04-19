# %%
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from gp.config import LIDDS_2021_DIR, RESULTS_DIR
from gp.data_loader.lidds_2021 import count_recordings, iter_recordings
from gp.diagnostic.stats import Stats
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

OUT_DIR = RESULTS_DIR / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class _Tee:
    """Write to both stdout and an in-memory buffer for later saving."""

    def __init__(self, stream):
        self._buf: list[str] = []
        self._stream = stream

    def write(self, s: str) -> None:
        self._buf.append(s)
        self._stream.write(s)

    def flush(self) -> None:
        self._stream.flush()

    def getvalue(self) -> str:
        return "".join(self._buf)


_tee = _Tee(sys.stdout)
sys.stdout = _tee  # type: ignore[assignment]

# %% [Stream dataset]
stats = Stats()
for rec in tqdm(
    iter_recordings(LIDDS_2021_DIR), total=count_recordings(LIDDS_2021_DIR)
):
    stats.analyze(rec)

# %% [1. Recording counts — data]

rows = []
for split, counts in stats._recording_counts.items():
    for kind, n in counts.items():
        rows.append({"split": split, "kind": kind, "count": n})
df_counts = pl.DataFrame(rows)
print(df_counts)
df_counts.write_csv(OUT_DIR / "01_recording_counts.csv")

# %% [1. Recording counts — plot]
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(
    data=df_counts,
    x="split",
    y="count",
    hue="kind",
    ax=ax,
    order=["train", "val", "test"],
)
ax.set_title("Recording counts per split")
ax.set_ylabel("# recordings")
plt.tight_layout()
fig.savefig(OUT_DIR / "01_recording_counts.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [2. Sequence length distribution — data]
QUANTILES = [0.50, 0.90, 0.99, 1.00]
labels = {0.50: "p50", 0.90: "p90", 0.99: "p99", 1.00: "max"}
rows_pct = []
for split in ["train", "val", "test"]:
    pct = stats.seq_length_percentiles(split, QUANTILES)
    row = "  ".join(f"{labels[q]}={v:,}" for q, v in pct.items())
    print(f"{split:5s}: {row}")
    rows_pct.append({"split": split, **{labels[q]: v for q, v in pct.items()}})
pl.DataFrame(rows_pct).write_csv(OUT_DIR / "02_seq_length_percentiles.csv")

# %% [2. Sequence length distribution — plot]
rows_len = []
for split, lengths in stats.seq_lengths.items():
    for v in lengths:
        rows_len.append({"split": split, "seq_length": v})
df_len = pl.DataFrame(rows_len)
df_len.write_csv(OUT_DIR / "02_seq_lengths.csv")

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df_len,
    x="split",
    y="seq_length",
    order=["train", "val", "test"],
    ax=ax,
    showfliers=False,
)
ax.set_yscale("log")
ax.set_title("Sequence length distribution per split (log scale, outliers hidden)")
ax.set_ylabel("# syscalls (exit events)")
plt.tight_layout()
fig.savefig(OUT_DIR / "02_seq_length_dist.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [3. Syscall vocabulary — data]
print("Vocab size per split:")
rows_vocab_size = []
for split, size in stats.vocab_size.items():
    print(f"  {split}: {size}")
    rows_vocab_size.append({"split": split, "vocab_size": size})
pl.DataFrame(rows_vocab_size).write_csv(OUT_DIR / "03_vocab_size.csv")

oov = stats.oov_syscalls
print(f"\nOOV syscalls (test \\ train): {len(oov)}")
if oov:
    print(" ", sorted(oov))
(OUT_DIR / "03_oov_syscalls.txt").write_text("\n".join(sorted(oov)))

# %% [3. Syscall vocabulary — plot (top-20 train)]
TOP_N = 20
train_freq = stats._vocab_freq.get("train", {})
top_syscalls = sorted(train_freq, key=lambda k: train_freq[k], reverse=True)[:TOP_N]

rows_vocab = []
for split, counter in stats._vocab_freq.items():
    for name in top_syscalls:
        rows_vocab.append(
            {"split": split, "syscall": name, "count": counter.get(name, 0)}
        )
df_vocab = pl.DataFrame(rows_vocab)
df_vocab.write_csv(OUT_DIR / "03_vocab_freq_top20.csv")

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(
    data=df_vocab.filter(pl.col("split") == "train"), x="syscall", y="count", ax=ax
)
ax.set_title(f"Top-{TOP_N} syscall frequencies (train split)")
ax.set_ylabel("count")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
fig.savefig(OUT_DIR / "03_vocab_top20.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [4. Thread structure — data]
rows_threads = []
for split in ["train", "val", "test"]:
    tc = stats._thread_counts[split]
    if tc:
        med = float(np.median(tc))
        print(
            f"{split:5s}: median threads = {med:.1f}, max = {max(tc)}, min = {min(tc)}"
        )
        rows_threads.append(
            {"split": split, "median": med, "max": max(tc), "min": min(tc)}
        )
pl.DataFrame(rows_threads).write_csv(OUT_DIR / "04_thread_summary.csv")

# %% [4. Thread structure — plots]
rows_tc = []
for split, counts in stats._thread_counts.items():
    for v in counts:
        rows_tc.append({"split": split, "thread_count": v})
df_tc = pl.DataFrame(rows_tc)

rows_ptl = []
for split, lengths in stats._per_thread_lengths.items():
    for v in lengths:
        rows_ptl.append({"split": split, "per_thread_len": v})
df_ptl = pl.DataFrame(rows_ptl)
df_ptl.write_csv(OUT_DIR / "04_per_thread_lengths.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.histplot(
    data=df_tc,
    x="thread_count",
    hue="split",
    multiple="dodge",
    discrete=True,
    ax=axes[0],
)
axes[0].set_title("Thread count per recording")
axes[0].set_xlabel("# unique threads")

sns.histplot(
    data=df_ptl,
    x="per_thread_len",
    hue="split",
    log_scale=(True, True),
    element="step",
    fill=False,
    ax=axes[1],
)
axes[1].set_title("Per-thread syscall count (all recordings)")
axes[1].set_xlabel("syscalls per thread (log scale)")

plt.tight_layout()
fig.savefig(OUT_DIR / "04_thread_structure.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [5. Attack timing — data]
offsets = stats._attack_offsets
fractions = stats._attack_fractions
print(f"Exploit recordings (test): {len(offsets)}")
rows_timing = []
if offsets:
    med_off = float(np.median(offsets))
    p90_off = float(np.percentile(offsets, 90))
    print(
        f"  offset  — median={med_off:.1f}s  "
        f"p90={p90_off:.1f}s  max={max(offsets):.1f}s"
    )
    rows_timing.append(
        {"metric": "offset_s", "median": med_off, "p90": p90_off, "max": max(offsets)}
    )
if fractions:
    med_fr = float(np.median(fractions))
    p90_fr = float(np.percentile(fractions, 90))
    print(
        f"  fraction — median={med_fr:.2f}  p90={p90_fr:.2f}  max={max(fractions):.2f}"
    )
    rows_timing.append(
        {"metric": "fraction", "median": med_fr, "p90": p90_fr, "max": max(fractions)}
    )
if rows_timing:
    pl.DataFrame(rows_timing).write_csv(OUT_DIR / "05_attack_timing.csv")

# %% [5. Attack timing — plots]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

if offsets:
    sns.histplot(offsets, bins=30, ax=axes[0])
    axes[0].set_title("Exploit offset from warmup_end")
    axes[0].set_xlabel("seconds")
else:
    axes[0].text(
        0.5,
        0.5,
        "no exploit recordings",
        ha="center",
        va="center",
        transform=axes[0].transAxes,
    )

if fractions:
    sns.histplot(fractions, bins=30, ax=axes[1])
    axes[1].set_title("Exploit fraction of recording duration")
    axes[1].set_xlabel("fraction elapsed at exploit")
    axes[1].set_xlim(0, 1)
else:
    axes[1].text(
        0.5,
        0.5,
        "no exploit recordings",
        ha="center",
        va="center",
        transform=axes[1].transAxes,
    )

plt.tight_layout()
fig.savefig(OUT_DIR / "05_attack_timing.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [Save summary]
sys.stdout = _tee._stream
(OUT_DIR / "summary.txt").write_text(_tee.getvalue())
print(f"Results saved to {OUT_DIR}")
