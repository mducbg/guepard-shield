# %%

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from guepard.config import WindowConfig
from guepard.data_loader.corpus import DongTingCorpus
from guepard.data_loader.windowing import num_sliding_windows
from rich import box
from rich.console import Console
from rich.table import Table
from scipy import stats
from tqdm import tqdm

console = Console()

DATA_DIR = "../data/processed/DongTing"
OUTPUT_DIR = Path("results/diagnostic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_CONFIG = WindowConfig(window_size=64, stride=12)
SPLITS = ["DTDS-train", "DTDS-validation", "DTDS-test"]
CLASS_NAMES = {0: "normal", 1: "attack"}
COLORS = {0: "steelblue", 1: "tomato"}

# ═══════════════════════════════════════════════════════════════════════════════
# Load corpus + read all tokens (single pass over files)
# ═══════════════════════════════════════════════════════════════════════════════
console.rule("[bold blue]Loading Corpus")
corpus = DongTingCorpus(DATA_DIR)
console.print(f"Total sequences in index: [bold]{len(corpus.metadata)}[/bold]")

# Single-pass: read every file once, collect all data needed for all analyses
# seq_data stores only lightweight metadata — no tokens in memory
seq_data = []  # list of dicts: {seq_id, split, label, bug_name, metadata_len, actual_len, file_path}
file_issues = []

console.print("Scanning files (metadata only, no tokens loaded)...")
for m in tqdm(corpus.metadata, desc="Scanning", unit="seq"):
    entry = {
        "seq_id": m.seq_id,
        "split": m.seq_class,
        "label": m.label,
        "bug_name": m.bug_name,
        "metadata_len": m.seq_length,
        "actual_len": None,
        "file_path": m.file_path,
    }
    if not m.file_path.exists():
        file_issues.append((m.seq_id, "missing file"))
        seq_data.append(entry)
        continue
    try:
        # Read only to get length and check integrity — discard content immediately
        content = m.file_path.read_text(encoding="utf-8").strip()
        if not content:
            file_issues.append((m.seq_id, "empty file"))
            seq_data.append(entry)
            continue
        actual_len = content.count("|") + 1
        if actual_len != m.seq_length:
            file_issues.append(
                (
                    m.seq_id,
                    f"length mismatch: metadata={m.seq_length} actual={actual_len}",
                )
            )
        entry["actual_len"] = actual_len
    except Exception as e:
        file_issues.append((m.seq_id, f"read error: {e}"))
    seq_data.append(entry)

valid = [s for s in seq_data if s["actual_len"] is not None]
console.print(f"Valid sequences: [bold]{len(valid)}[/bold] / {len(seq_data)} total")


def stream_tokens(entries, desc=""):
    """Lazy token stream — reads one file at a time, yields (entry, tokens), never holds all in memory."""
    for s in tqdm(entries, desc=desc, unit="seq"):
        try:
            tokens = s["file_path"].read_text(encoding="utf-8").strip().split("|")
            yield s, tokens
        except Exception:
            continue


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: File Integrity
# ═══════════════════════════════════════════════════════════════════════════════
console.rule("[bold blue]Part 1: File Integrity")

if file_issues:
    console.print(f"[red]{len(file_issues)} issues found:[/red]")
    for seq_id, issue in file_issues[:20]:
        console.print(f"  {seq_id}: {issue}")
    if len(file_issues) > 20:
        console.print(f"  ... and {len(file_issues) - 20} more")
else:
    console.print("[green]OK — no missing/empty/mismatched files[/green]")

# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: Sequence-Level Distribution
# ═══════════════════════════════════════════════════════════════════════════════
console.rule("[bold blue]Part 2: Sequence-Length Distribution")

# Group by split × label
split_label_lens = defaultdict(lambda: defaultdict(list))  # split → label → [len]
for s in valid:
    split_label_lens[s["split"]][s["label"]].append(s["actual_len"])

for split in SPLITS:
    by = split_label_lens[split]
    table = Table(title=f"[bold]{split}[/bold]", box=box.SIMPLE_HEAD)
    table.add_column("Class", style="cyan")
    for col in ["N", "Min", "P25", "Median", "Mean", "P75", "P95", "Max", "Std"]:
        table.add_column(col, justify="right")
    for label, name in CLASS_NAMES.items():
        lens = by[label]
        if not lens:
            table.add_row(name, *(["-"] * 9))
            continue
        a = np.array(lens)
        table.add_row(
            name,
            str(len(a)),
            f"{a.min():,}",
            f"{int(np.percentile(a, 25)):,}",
            f"{int(np.median(a)):,}",
            f"{a.mean():,.0f}",
            f"{int(np.percentile(a, 75)):,}",
            f"{int(np.percentile(a, 95)):,}",
            f"{a.max():,}",
            f"{a.std():,.0f}",
        )
    console.print(table)

# Top-5 outliers per split
for split in SPLITS:
    metas = corpus.get_split(split)
    top5 = sorted(metas, key=lambda m: m.seq_length, reverse=True)[:5]
    console.print(f"  [yellow]Top-5 longest ({split}):[/yellow]")
    for m in top5:
        console.print(
            f"    {m.seq_id:35s} {CLASS_NAMES[m.label]:8s} {m.seq_length:>12,} syscalls  ({m.bug_name})"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# Part 3: Statistical Tests — seq_length normal vs attack
# ═══════════════════════════════════════════════════════════════════════════════
console.rule("[bold blue]Part 3: Statistical Tests")

table = Table(
    title="Mann-Whitney U — seq_length: normal vs attack", box=box.SIMPLE_HEAD
)
table.add_column("Split")
table.add_column("U statistic", justify="right")
table.add_column("p-value", justify="right")
table.add_column("Median normal", justify="right")
table.add_column("Median attack", justify="right")
table.add_column("Significant?", justify="center")

for split in SPLITS:
    by = split_label_lens[split]
    n_lens = np.array(by[0])
    a_lens = np.array(by[1])
    if len(n_lens) < 2 or len(a_lens) < 2:
        table.add_row(split, "-", "-", "-", "-", "-")
        continue
    u_stat, p_val = stats.mannwhitneyu(n_lens, a_lens, alternative="two-sided")
    sig = "[bold green]YES[/bold green]" if p_val < 0.05 else "[red]NO[/red]"
    table.add_row(
        split,
        f"{u_stat:,.0f}",
        f"{p_val:.2e}",
        f"{int(np.median(n_lens)):,}",
        f"{int(np.median(a_lens)):,}",
        sig,
    )
console.print(table)

# KS test
table2 = Table(
    title="Kolmogorov-Smirnov — seq_length distribution shape", box=box.SIMPLE_HEAD
)
table2.add_column("Split")
table2.add_column("KS statistic", justify="right")
table2.add_column("p-value", justify="right")
table2.add_column("Distribution differ?", justify="center")

for split in SPLITS:
    by = split_label_lens[split]
    n_lens = np.array(by[0])
    a_lens = np.array(by[1])
    if len(n_lens) < 2 or len(a_lens) < 2:
        table2.add_row(split, "-", "-", "-")
        continue
    ks_stat, p_val = stats.ks_2samp(n_lens, a_lens)
    sig = "[bold green]YES[/bold green]" if p_val < 0.05 else "[red]NO[/red]"
    table2.add_row(split, f"{ks_stat:.4f}", f"{p_val:.2e}", sig)
console.print(table2)

# Point-biserial correlation: seq_length ~ label (all sequences combined)
all_lens = np.array([s["actual_len"] for s in valid])
all_labels = np.array([s["label"] for s in valid])
r, p_val = stats.pointbiserialr(all_labels, all_lens)
console.print(
    f"\nPoint-biserial correlation (seq_length ~ label): r={r:.4f}, p={p_val:.2e}"
)
console.print(
    f"  → {'Strong' if abs(r) > 0.5 else 'Moderate' if abs(r) > 0.3 else 'Weak'} {'positive' if r > 0 else 'negative'} association"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Part 4: Window-Level Stats + Imbalance
# ═══════════════════════════════════════════════════════════════════════════════
console.rule("[bold blue]Part 4: Window-Level Stats")

win_stats = defaultdict(lambda: defaultdict(list))
for s in tqdm(valid, desc="Computing windows", unit="seq"):
    n = num_sliding_windows(s["actual_len"], WINDOW_CONFIG)
    win_stats[s["split"]][s["label"]].append(n)

for split in SPLITS:
    by = win_stats[split]
    table = Table(
        title=f"[bold]{split}[/bold] — Windows per Sequence", box=box.SIMPLE_HEAD
    )
    table.add_column("Class", style="cyan")
    for col in ["Total", "Min", "Median", "P90", "Max"]:
        table.add_column(col, justify="right")
    totals = {}
    for label, name in CLASS_NAMES.items():
        wpc = by[label]
        totals[label] = sum(wpc)
        a = np.array(wpc) if wpc else np.array([0])
        table.add_row(
            name,
            f"{totals[label]:,}",
            f"{a.min():,}",
            f"{int(np.median(a)):,}",
            f"{int(np.percentile(a, 90)):,}",
            f"{a.max():,}",
        )
    ratio = totals.get(1, 0) / max(totals.get(0, 1), 1)
    console.print(table)
    console.print(
        f"  [bold red]Window imbalance (attack/normal): {ratio:.1f}:1[/bold red]"
    )

# Recommended cap
all_normal_wpc = []
all_attack_wpc = []
for split in SPLITS:
    all_normal_wpc.extend(win_stats[split][0])
    all_attack_wpc.extend(win_stats[split][1])

p50_n = int(np.percentile(all_normal_wpc, 50)) if all_normal_wpc else 0
p90_n = int(np.percentile(all_normal_wpc, 90)) if all_normal_wpc else 0
pct_capped = np.mean(np.array(all_attack_wpc) > p90_n) * 100 if all_attack_wpc else 0

console.print(f"\n[bold green]Recommended max_windows_per_seq = {p90_n}[/bold green]")
console.print(f"  P50 normal: {p50_n:,} | P90 normal: {p90_n:,}")
console.print(f"  {pct_capped:.1f}% of attack sequences would be capped")

# ═══════════════════════════════════════════════════════════════════════════════
# Part 5: Syscall / Vocab Analysis
# ═══════════════════════════════════════════════════════════════════════════════
console.rule("[bold blue]Part 5: Syscall & Vocab Analysis")

syscall_counts = {0: Counter(), 1: Counter()}
syscall_seq_count = {0: Counter(), 1: Counter()}  # syscall → #sequences containing it

for s, tokens in stream_tokens(valid, desc="Counting syscalls"):
    syscall_counts[s["label"]].update(tokens)
    syscall_seq_count[s["label"]].update(set(tokens))

only_normal = set(syscall_counts[0]) - set(syscall_counts[1])
only_attack = set(syscall_counts[1]) - set(syscall_counts[0])
shared = set(syscall_counts[0]) & set(syscall_counts[1])
total_counter = syscall_counts[0] + syscall_counts[1]
top20 = total_counter.most_common(20)

console.print(
    f"Total unique syscalls:      {len(set(syscall_counts[0]) | set(syscall_counts[1])):>6,}"
)
console.print(f"  Shared:                   {len(shared):>6,}")
console.print(f"  Only in normal:           {len(only_normal):>6,}")
console.print(f"  Only in attack:           {len(only_attack):>6,}")


# Entropy of syscall distribution per class
def entropy(counter):
    total = sum(counter.values())
    return -sum((c / total) * np.log2(c / total) for c in counter.values() if c > 0)


h_normal = entropy(syscall_counts[0])
h_attack = entropy(syscall_counts[1])
console.print("\nSyscall distribution entropy:")
console.print(f"  Normal: {h_normal:.3f} bits")
console.print(f"  Attack: {h_attack:.3f} bits")
console.print(
    f"  → {'Attack more diverse' if h_attack > h_normal else 'Normal more diverse'}"
)

# Top-20 table
table = Table(title="Top-20 Syscalls", box=box.SIMPLE_HEAD)
table.add_column("Syscall", style="cyan")
table.add_column("Total count", justify="right")
table.add_column("Normal count", justify="right")
table.add_column("Attack count", justify="right")
table.add_column("Attack %", justify="right")
for syscall, total in top20:
    a_cnt = syscall_counts[1][syscall]
    n_cnt = syscall_counts[0][syscall]
    pct = 100 * a_cnt / total if total else 0
    table.add_row(syscall, f"{total:,}", f"{n_cnt:,}", f"{a_cnt:,}", f"{pct:.1f}%")
console.print(table)

# PMI: Pointwise Mutual Information — which syscalls are most predictive of attack?
n_normal_seqs = sum(1 for s in valid if s["label"] == 0)
n_attack_seqs = sum(1 for s in valid if s["label"] == 1)
n_total_seqs = len(valid)
p_attack = n_attack_seqs / n_total_seqs
p_normal = n_normal_seqs / n_total_seqs

pmi_scores = {}
for syscall in shared:
    p_s = (syscall_seq_count[0][syscall] + syscall_seq_count[1][syscall]) / n_total_seqs
    if p_s == 0:
        continue
    p_s_given_attack = syscall_seq_count[1][syscall] / max(n_attack_seqs, 1)
    if p_s_given_attack > 0:
        pmi_scores[syscall] = np.log2(p_s_given_attack / p_s)

top10_pmi_attack = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:10]
top10_pmi_normal = sorted(pmi_scores.items(), key=lambda x: x[1])[:10]

console.print("\n[bold]Top-10 syscalls most associated with ATTACK (PMI):[/bold]")
for s, pmi in top10_pmi_attack:
    console.print(f"  {s:30s}  PMI={pmi:+.3f}")
console.print("[bold]Top-10 syscalls most associated with NORMAL (PMI):[/bold]")
for s, pmi in top10_pmi_normal:
    console.print(f"  {s:30s}  PMI={pmi:+.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Part 6: Split Consistency — attack type overlap across splits
# ═══════════════════════════════════════════════════════════════════════════════
console.rule("[bold blue]Part 6: Split Consistency")


# Extract attack type from bug_name (use prefix before first digit or underscore)
def attack_type(bug_name: str) -> str:
    import re

    m = re.match(r"([A-Za-z]+)", bug_name)
    return m.group(1) if m else bug_name


split_attack_types = {}
for split in SPLITS:
    metas = corpus.get_split(split)
    types = Counter(attack_type(m.bug_name) for m in metas if m.label == 1)
    split_attack_types[split] = types

# Cross-split overlap
all_types = sorted(set().union(*[set(t.keys()) for t in split_attack_types.values()]))
table = Table(title="Attack Type Distribution Across Splits", box=box.SIMPLE_HEAD)
table.add_column("Attack type", style="cyan")
for split in SPLITS:
    table.add_column(split, justify="right")
for atype in all_types:
    row = [atype] + [str(split_attack_types[s].get(atype, 0)) for s in SPLITS]
    table.add_row(*row)
console.print(table)

# Vocab overlap across splits
split_vocabs = {}
for split in SPLITS:
    vocab = set()
    subset = [s for s in valid if split in s["split"]]
    for s, tokens in stream_tokens(subset, desc=f"Vocab {split}"):
        vocab.update(tokens)
    split_vocabs[split] = vocab

if len(SPLITS) >= 2:
    train_vocab = split_vocabs.get("DTDS-train", set())
    val_vocab = split_vocabs.get("DTDS-validation", set())
    test_vocab = split_vocabs.get("DTDS-test", set())
    console.print("\nVocab coverage:")
    if train_vocab and val_vocab:
        val_oov = val_vocab - train_vocab
        console.print(
            f"  Val OOV (not in train vocab):  {len(val_oov):>4} syscalls ({100 * len(val_oov) / max(len(val_vocab), 1):.1f}%)"
        )
    if train_vocab and test_vocab:
        test_oov = test_vocab - train_vocab
        console.print(
            f"  Test OOV (not in train vocab): {len(test_oov):>4} syscalls ({100 * len(test_oov) / max(len(test_vocab), 1):.1f}%)"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════
console.rule("[bold blue]Generating Plots")

# ── Figure 1: Distributions ──────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle(
    "DongTing — Sequence & Window Distributions", fontsize=13, fontweight="bold"
)

# 1a: seq_length histogram (log x + log y) — DTDS-train
ax = axes[0, 0]
for label, name in CLASS_NAMES.items():
    lens = split_label_lens["DTDS-train"][label]
    if lens:
        ax.hist(
            np.log10(np.array(lens) + 1),
            bins=40,
            alpha=0.6,
            label=name,
            color=COLORS[label],
        )
ax.set_title("seq_length histogram (DTDS-train, log10 x)")
ax.set_xlabel("log10(seq_length)")
ax.set_ylabel("Count")
ax.legend()

# 1b: CDF of seq_length — all splits combined
ax = axes[0, 1]
for label, name in CLASS_NAMES.items():
    all_lens = []
    for split in SPLITS:
        all_lens.extend(split_label_lens[split][label])
    if all_lens:
        sorted_lens = np.sort(all_lens)
        cdf = np.arange(1, len(sorted_lens) + 1) / len(sorted_lens)
        ax.plot(sorted_lens, cdf, label=name, color=COLORS[label])
ax.set_xscale("log")
ax.set_title("CDF: seq_length (all splits, log x)")
ax.set_xlabel("seq_length (log scale)")
ax.set_ylabel("CDF")
ax.axvline(
    WINDOW_CONFIG.window_size,
    color="gray",
    linestyle="--",
    alpha=0.5,
    label=f"window_size={WINDOW_CONFIG.window_size}",
)
ax.legend()

# 1c: seq_length vs windows per sequence (scatter, sampled)
ax = axes[0, 2]
for label, name in CLASS_NAMES.items():
    pts = [
        (s["actual_len"], num_sliding_windows(s["actual_len"], WINDOW_CONFIG))
        for s in valid
        if s["label"] == label and s["actual_len"] is not None
    ]
    if pts:
        xs, ys = zip(*pts[:500])  # sample up to 500 to avoid clutter
        ax.scatter(xs, ys, alpha=0.3, s=10, color=COLORS[label], label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("seq_length vs window count (log-log, sampled)")
ax.set_xlabel("seq_length")
ax.set_ylabel("windows per sequence")
ax.legend()

# 1d: Windows per sequence boxplot per split
ax = axes[1, 0]
positions = []
labels_bp = []
colors_bp = []
offset = 0
for split in SPLITS:
    for label, name in CLASS_NAMES.items():
        wpc = win_stats[split][label]
        if wpc:
            positions.append(offset)
            labels_bp.append(f"{split.split('-')[1][:3]}\n{name[:3]}")
            colors_bp.append(COLORS[label])
            ax.boxplot(
                [wpc],
                positions=[offset],
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=COLORS[label], alpha=0.6),
                medianprops=dict(color="black"),
            )
        offset += 1
    offset += 0.5
ax.set_xticks(range(len(labels_bp)))
ax.set_xticklabels(labels_bp, fontsize=7)
ax.set_title("Windows/seq by split × class (no outliers)")
ax.set_ylabel("Window count")

# 1e: Total windows stacked bar per split
ax = axes[1, 1]
x = np.arange(len(SPLITS))
w = 0.35
ax.bar(
    x - w / 2,
    [sum(win_stats[s][0]) for s in SPLITS],
    w,
    label="Normal",
    color=COLORS[0],
)
ax.bar(
    x + w / 2,
    [sum(win_stats[s][1]) for s in SPLITS],
    w,
    label="Attack",
    color=COLORS[1],
)
ax.set_xticks(x)
ax.set_xticklabels([s.split("-")[1] for s in SPLITS])
ax.set_title("Total windows per split")
ax.set_ylabel("Window count")
ax.legend()
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K")
)

# 1f: Sequence count per split (grouped bar)
ax = axes[1, 2]
n_seqs = [[len(split_label_lens[s][l]) for s in SPLITS] for l in [0, 1]]
ax.bar(x - w / 2, n_seqs[0], w, label="Normal", color=COLORS[0])
ax.bar(x + w / 2, n_seqs[1], w, label="Attack", color=COLORS[1])
ax.set_xticks(x)
ax.set_xticklabels([s.split("-")[1] for s in SPLITS])
ax.set_title("Sequence count per split")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig1_distributions.png", dpi=130)
plt.show()
console.print("[green]Saved: fig1_distributions.png[/green]")

# ── Figure 2: Syscall Analysis ────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(18, 7))
fig2.suptitle("DongTing — Syscall Analysis", fontsize=13, fontweight="bold")

# 2a: Top-30 syscalls by PMI (attack-associated positive, normal-associated negative)
ax = axes[0]
top15_attack = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:15]
top15_normal = sorted(pmi_scores.items(), key=lambda x: x[1])[:15]
combined = top15_attack + top15_normal
combined_sorted = sorted(combined, key=lambda x: x[1])
syscall_names, pmi_vals = zip(*combined_sorted)
bar_colors = [COLORS[1] if v > 0 else COLORS[0] for v in pmi_vals]
ax.barh(range(len(pmi_vals)), pmi_vals, color=bar_colors, alpha=0.8)
ax.set_yticks(range(len(syscall_names)))
ax.set_yticklabels(syscall_names, fontsize=8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Syscall PMI (positive=attack-assoc, negative=normal-assoc)")
ax.set_xlabel("PMI score")

# 2b: Syscall co-occurrence heatmap (top-15 by total frequency, sequence-level)
ax = axes[1]
top15_syscalls = [s for s, _ in total_counter.most_common(15)]
n = len(top15_syscalls)
cooc = np.zeros((n, n))
for s, tokens in stream_tokens(valid, desc="Co-occurrence matrix"):
    token_set = set(tokens)
    for i, si in enumerate(top15_syscalls):
        if si not in token_set:
            continue
        for j, sj in enumerate(top15_syscalls):
            if sj in token_set:
                cooc[i, j] += 1

# Normalize by diagonal (Jaccard-like: cooc(i,j) / (cooc(i,i) + cooc(j,j) - cooc(i,j)))
cooc_norm = np.zeros_like(cooc)
for i in range(n):
    for j in range(n):
        denom = cooc[i, i] + cooc[j, j] - cooc[i, j]
        cooc_norm[i, j] = cooc[i, j] / denom if denom > 0 else 0

im = ax.imshow(cooc_norm, cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(n))
ax.set_xticklabels(top15_syscalls, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n))
ax.set_yticklabels(top15_syscalls, fontsize=8)
ax.set_title("Syscall Co-occurrence (Jaccard similarity, top-15 by freq)")
plt.colorbar(im, ax=ax, label="Jaccard similarity")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_syscall_analysis.png", dpi=130)
plt.show()
console.print("[green]Saved: fig2_syscall_analysis.png[/green]")

# ── Figure 3: Statistical Analysis ───────────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("DongTing — Statistical Analysis", fontsize=13, fontweight="bold")

# 3a: Q-Q plot seq_length normal vs attack (DTDS-train)
ax = axes[0]
n_lens = np.sort(np.array(split_label_lens["DTDS-train"][0]))
a_lens = np.sort(np.array(split_label_lens["DTDS-train"][1]))
# Interpolate to same size for Q-Q
quantiles = np.linspace(0, 1, 100)
n_q = np.quantile(n_lens, quantiles)
a_q = np.quantile(a_lens, quantiles)
ax.scatter(n_q, a_q, s=15, alpha=0.7, color="purple")
mn = min(n_q.min(), a_q.min())
mx = max(n_q.max(), a_q.max())
ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="y=x (identical distributions)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Normal quantiles (log)")
ax.set_ylabel("Attack quantiles (log)")
ax.set_title("Q-Q plot: seq_length normal vs attack (DTDS-train)")
ax.legend(fontsize=8)

# 3b: Label correlation — seq_length binned by class proportion
ax = axes[1]
all_valid_lens = np.array([s["actual_len"] for s in valid])
all_valid_labels = np.array([s["label"] for s in valid])
bins = np.percentile(all_valid_lens, np.linspace(0, 100, 21))
bins = np.unique(bins)
bin_centers = (bins[:-1] + bins[1:]) / 2
attack_fracs = []
counts = []
for i in range(len(bins) - 1):
    mask = (all_valid_lens >= bins[i]) & (all_valid_lens < bins[i + 1])
    if mask.sum() > 0:
        attack_fracs.append(all_valid_labels[mask].mean())
        counts.append(mask.sum())
    else:
        attack_fracs.append(np.nan)
        counts.append(0)

sc = ax.scatter(bin_centers, attack_fracs, c=counts, cmap="viridis", s=60, zorder=3)
ax.set_xscale("log")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("seq_length bin center (log scale)")
ax.set_ylabel("Fraction of attack sequences")
ax.set_title(f"seq_length vs attack fraction (r={r:.3f}, p={p_val:.1e})")
plt.colorbar(sc, ax=ax, label="Sequences in bin")
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig3_statistical.png", dpi=130)
plt.show()
console.print("[green]Saved: fig3_statistical.png[/green]")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
console.rule("[bold blue]Summary")
console.print("[bold]Key findings:[/bold]")
for split in SPLITS:
    n0 = sum(win_stats[split][0])
    n1 = sum(win_stats[split][1])
    console.print(
        f"  {split}: window imbalance {n1 / max(n0, 1):.0f}:1 (attack/normal)"
    )

console.print(
    f"\n  Point-biserial r(seq_length, label) = {r:.4f} — attack sequences are {'longer' if r > 0 else 'shorter'}"
)
console.print(
    f"  Syscall entropy: normal={h_normal:.2f} bits, attack={h_attack:.2f} bits"
)
console.print(
    f"\n[bold yellow]Recommended max_windows_per_seq = {p90_n}[/bold yellow] (P90 of normal, {pct_capped:.0f}% attack seqs capped)"
)
console.print(f"Saved 3 figures to: {OUTPUT_DIR}/")
