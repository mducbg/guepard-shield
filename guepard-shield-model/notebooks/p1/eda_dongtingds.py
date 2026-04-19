# %%
import sys
import warnings

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from gp.config import DONGTING_DIR, RESULTS_DIR
from gp.data_loader.dongting import count_recordings, iter_recordings
from gp.diagnostic.dongtingstats import DongTingStats
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

OUT_DIR = RESULTS_DIR / "eda_dongting"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class _Tee:
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
stats = DongTingStats()
for rec in tqdm(iter_recordings(DONGTING_DIR), total=count_recordings(DONGTING_DIR)):
    stats.analyze(rec)

# %% [1. Recording counts — data]
rows = []
for split in ["train", "val", "test"]:
    for label in ["normal", "abnormal"]:
        rows.append(
            {
                "split": split,
                "label": label,
                "count": stats._recording_counts[split][label],
            }
        )
df_counts = pl.DataFrame(rows)
print(df_counts)
df_counts.write_csv(OUT_DIR / "01_recording_counts.csv")

# %% [1. Recording counts — plot]
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(
    data=df_counts,
    x="split",
    y="count",
    hue="label",
    order=["train", "val", "test"],
    ax=ax,
)
ax.set_title("DongTing — recording counts per split")
ax.set_ylabel("# recordings")
plt.tight_layout()
fig.savefig(OUT_DIR / "01_recording_counts.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [2. Sequence length distribution — data]
QUANTILES = [0.50, 0.90, 0.99, 1.00]
labels_q = {0.50: "p50", 0.90: "p90", 0.99: "p99", 1.00: "max"}
rows_pct = []
for split in ["train", "val", "test"]:
    for label in ["normal", "abnormal"]:
        pct = stats.seq_length_percentiles(split, label, QUANTILES)
        row = "  ".join(f"{labels_q[q]}={v:,}" for q, v in pct.items())
        print(f"{split:5s}/{label:8s}: {row}")
        rows_pct.append(
            {"split": split, "label": label, **{labels_q[q]: v for q, v in pct.items()}}
        )
pl.DataFrame(rows_pct).write_csv(OUT_DIR / "02_seq_length_percentiles.csv")

# %% [2. Sequence length distribution — plot]
rows_len = []
for (split, label), lengths in stats._seq_lengths.items():
    for v in lengths:
        rows_len.append({"split": split, "label": label, "seq_length": v})
df_len = pl.DataFrame(rows_len)
df_len.write_csv(OUT_DIR / "02_seq_lengths.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, split in zip(axes, ["train", "test"]):
    subset = df_len.filter(pl.col("split") == split)
    sns.boxplot(
        data=subset,
        x="label",
        y="seq_length",
        order=["normal", "abnormal"],
        ax=ax,
        showfliers=False,
    )
    ax.set_yscale("log")
    ax.set_title(f"Sequence length — {split} (log scale, no outliers)")
    ax.set_ylabel("# syscalls")
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

# %% [3. Syscall vocabulary — plot (top-30 train)]
TOP_N = 30
train_freq = stats._vocab_freq.get("train", {})
top_syscalls = sorted(train_freq, key=lambda k: train_freq[k], reverse=True)[:TOP_N]

rows_vocab = []
for split, counter in stats._vocab_freq.items():
    for name in top_syscalls:
        rows_vocab.append(
            {"split": split, "syscall": name, "count": counter.get(name, 0)}
        )
df_vocab = pl.DataFrame(rows_vocab)
df_vocab.write_csv(OUT_DIR / "03_vocab_freq_top30.csv")

fig, ax = plt.subplots(figsize=(14, 5))
sns.barplot(
    data=df_vocab.filter(pl.col("split") == "train"),
    x="syscall",
    y="count",
    ax=ax,
)
ax.set_title(f"Top-{TOP_N} syscall frequencies — train split")
ax.set_ylabel("count")
ax.tick_params(axis="x", rotation=60)
plt.tight_layout()
fig.savefig(OUT_DIR / "03_vocab_top30.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [4. Kernel version distribution — data]
rows_kv = [
    {"kernel_version": kv, "count": cnt}
    for kv, cnt in sorted(stats._kernel_version_counts.items())
]
df_kv = pl.DataFrame(rows_kv)
print("\nAbnormal recordings per kernel version:")
print(df_kv)
df_kv.write_csv(OUT_DIR / "04_kernel_version_counts.csv")

# %% [4. Kernel version distribution — plot]
fig, ax = plt.subplots(figsize=(14, 4))
sns.barplot(data=df_kv, x="kernel_version", y="count", ax=ax)
ax.set_title("Abnormal recordings per kernel version")
ax.set_xlabel("kernel version")
ax.set_ylabel("# recordings")
ax.tick_params(axis="x", rotation=60)
plt.tight_layout()
fig.savefig(OUT_DIR / "04_kernel_version_dist.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [5. Source category distribution — data]
rows_src = [
    {"source": src, "count": cnt} for src, cnt in stats._source_counts.most_common()
]
df_src = pl.DataFrame(rows_src)
print("\nRecordings per source category:")
print(df_src)
df_src.write_csv(OUT_DIR / "05_source_counts.csv")

# %% [5. Source category distribution — plot]
fig, ax = plt.subplots(figsize=(14, 5))
sns.barplot(data=df_src, x="source", y="count", ax=ax)
ax.set_title("Recordings per source directory")
ax.set_ylabel("# recordings")
ax.tick_params(axis="x", rotation=60)
plt.tight_layout()
fig.savefig(OUT_DIR / "05_source_dist.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [Save summary]
sys.stdout = _tee._stream
(OUT_DIR / "summary.txt").write_text(_tee.getvalue())
print(f"Results saved to {OUT_DIR}")
