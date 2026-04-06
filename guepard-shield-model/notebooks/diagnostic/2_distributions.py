# %% [markdown]
# # D02 — Sequence & Window Distributions + Statistical Tests
# Requires: d01_integrity.py has been run (seq_data / valid in scope, or re-run here).

# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from guepard.data_loader.windowing import num_sliding_windows
from guepard.evaluation.corpus_stats import (
    compute_window_stats,
    recommend_max_windows,
    scan_corpus_integrity,
)
from guepard.evaluation.statistical_tests import length_distribution_tests, point_biserial

from config import CLASS_NAMES, COLORS, CORPUS as corpus, COUNT_TOKENS, DATASET_NAME, OUTPUT_DIR, SPLITS, TRAIN_SPLIT, WINDOW_CONFIG

console = Console()

# %%  Load data (re-scan if running standalone)
seq_data, _ = scan_corpus_integrity(corpus.metadata, count_tokens=COUNT_TOKENS)
valid = [s for s in seq_data if s["actual_len"] is not None]

# %%
# ── Sequence-length stats ──────────────────────────────────────────────────────
console.rule("[bold blue]Sequence-Length Distribution")

split_label_lens: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
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
            name, str(len(a)), f"{a.min():,}", f"{int(np.percentile(a, 25)):,}",
            f"{int(np.median(a)):,}", f"{a.mean():,.0f}", f"{int(np.percentile(a, 75)):,}",
            f"{int(np.percentile(a, 95)):,}", f"{a.max():,}", f"{a.std():,.0f}",
        )
    console.print(table)

for split in SPLITS:
    metas = corpus.get_split(split)
    top5 = sorted(metas, key=lambda m: m.seq_length, reverse=True)[:5]
    console.print(f"  [yellow]Top-5 longest ({split}):[/yellow]")
    for m in top5:
        console.print(
            f"    {m.seq_id:35s} {CLASS_NAMES[m.label]:8s} {m.seq_length:>12,} syscalls  ({m.bug_name})"
        )

# %%
# ── Window-level stats ─────────────────────────────────────────────────────────
console.rule("[bold blue]Window-Level Stats")

win_stats = compute_window_stats(valid, WINDOW_CONFIG)

for split in SPLITS:
    by = win_stats.get(split, {})
    table = Table(title=f"[bold]{split}[/bold] — Windows per Sequence", box=box.SIMPLE_HEAD)
    table.add_column("Class", style="cyan")
    for col in ["Total", "Min", "Median", "P90", "Max"]:
        table.add_column(col, justify="right")
    totals = {}
    for label, name in CLASS_NAMES.items():
        wpc = by.get(label, [])
        totals[label] = sum(wpc)
        a = np.array(wpc) if wpc else np.array([0])
        table.add_row(
            name, f"{totals[label]:,}", f"{a.min():,}",
            f"{int(np.median(a)):,}", f"{int(np.percentile(a, 90)):,}", f"{a.max():,}",
        )
    ratio = totals.get(1, 0) / max(totals.get(0, 1), 1)
    console.print(table)
    console.print(f"  [bold red]Window imbalance (attack/normal): {ratio:.1f}:1[/bold red]")

rec = recommend_max_windows(win_stats)
console.print(f"\n[bold green]Recommended max_windows_per_seq = {rec['p90_n']}[/bold green]")
console.print(f"  P50 normal: {rec['p50_n']:,} | P90 normal: {rec['p90_n']:,}")
console.print(f"  {rec['pct_capped']:.1f}% of attack sequences would be capped")

# %%
# ── Statistical tests ──────────────────────────────────────────────────────────
console.rule("[bold blue]Statistical Tests")

test_results = length_distribution_tests(split_label_lens, SPLITS)

mw_table = Table(title="Mann-Whitney U — seq_length: normal vs attack", box=box.SIMPLE_HEAD)
mw_table.add_column("Split")
for col in ["U statistic", "p-value", "Median normal", "Median attack", "Significant?"]:
    mw_table.add_column(col, justify="right")
for res in test_results:
    mw = res["mannwhitney"]
    if mw:
        sig = "[bold green]YES[/bold green]" if mw["significant"] else "[red]NO[/red]"
        mw_table.add_row(
            res["split"], f"{mw['stat']:,.0f}", f"{mw['p_val']:.2e}",
            f"{mw['median_normal']:,}", f"{mw['median_attack']:,}", sig,
        )
    else:
        mw_table.add_row(res["split"], "-", "-", "-", "-", "-")
console.print(mw_table)

ks_table = Table(title="Kolmogorov-Smirnov — seq_length distribution shape", box=box.SIMPLE_HEAD)
ks_table.add_column("Split")
for col in ["KS statistic", "p-value", "Distribution differ?"]:
    ks_table.add_column(col, justify="right")
for res in test_results:
    ks = res["ks"]
    if ks:
        sig = "[bold green]YES[/bold green]" if ks["significant"] else "[red]NO[/red]"
        ks_table.add_row(res["split"], f"{ks['stat']:.4f}", f"{ks['p_val']:.2e}", sig)
    else:
        ks_table.add_row(res["split"], "-", "-", "-")
console.print(ks_table)

r, p_val = point_biserial(valid)
console.print(f"\nPoint-biserial correlation (seq_length ~ label): r={r:.4f}, p={p_val:.2e}")
console.print(
    f"  → {'Strong' if abs(r) > 0.5 else 'Moderate' if abs(r) > 0.3 else 'Weak'} "
    f"{'positive' if r > 0 else 'negative'} association"
)

# %%
# ── Figure 1: Distributions ───────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle(f"{DATASET_NAME} — Sequence & Window Distributions", fontsize=13, fontweight="bold")

ax = axes[0, 0]
for label, name in CLASS_NAMES.items():
    lens = split_label_lens[TRAIN_SPLIT][label]
    if lens:
        ax.hist(np.log10(np.array(lens) + 1), bins=40, alpha=0.6, label=name, color=COLORS[label])
ax.set_title(f"seq_length histogram ({TRAIN_SPLIT}, log10 x)")
ax.set_xlabel("log10(seq_length)")
ax.set_ylabel("Count")
ax.legend()

ax = axes[0, 1]
for label, name in CLASS_NAMES.items():
    all_lens = [l for split in SPLITS for l in split_label_lens[split][label]]
    if all_lens:
        sorted_lens = np.sort(all_lens)
        cdf = np.arange(1, len(sorted_lens) + 1) / len(sorted_lens)
        ax.plot(sorted_lens, cdf, label=name, color=COLORS[label])
ax.set_xscale("log")
ax.set_title("CDF: seq_length (all splits, log x)")
ax.set_xlabel("seq_length (log scale)")
ax.set_ylabel("CDF")
ax.axvline(WINDOW_CONFIG.window_size, color="gray", linestyle="--", alpha=0.5,
           label=f"window_size={WINDOW_CONFIG.window_size}")
ax.legend()

ax = axes[0, 2]
for label, name in CLASS_NAMES.items():
    pts = [(s["actual_len"], num_sliding_windows(s["actual_len"], WINDOW_CONFIG))
           for s in valid if s["label"] == label and s["actual_len"] is not None]
    if pts:
        xs, ys = zip(*pts[:500])
        ax.scatter(xs, ys, alpha=0.3, s=10, color=COLORS[label], label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("seq_length vs window count (log-log, sampled)")
ax.set_xlabel("seq_length")
ax.set_ylabel("windows per sequence")
ax.legend()

ax = axes[1, 0]
offset = 0
labels_bp = []
for split in SPLITS:
    for label, name in CLASS_NAMES.items():
        wpc = win_stats.get(split, {}).get(label, [])
        if wpc:
            labels_bp.append(f"{split.split('-')[1][:3]}\n{name[:3]}")
            ax.boxplot([wpc], positions=[offset], patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=COLORS[label], alpha=0.6),
                       medianprops=dict(color="black"))
        offset += 1
    offset += 0.5
ax.set_xticks(range(len(labels_bp)))
ax.set_xticklabels(labels_bp, fontsize=7)
ax.set_title("Windows/seq by split × class (no outliers)")
ax.set_ylabel("Window count")

ax = axes[1, 1]
x = np.arange(len(SPLITS))
w = 0.35
ax.bar(x - w/2, [sum(win_stats.get(s, {}).get(0, [])) for s in SPLITS], w, label="Normal", color=COLORS[0])
ax.bar(x + w/2, [sum(win_stats.get(s, {}).get(1, [])) for s in SPLITS], w, label="Attack", color=COLORS[1])
ax.set_xticks(x)
ax.set_xticklabels([s.split("-")[1] for s in SPLITS])
ax.set_title("Total windows per split")
ax.set_ylabel("Window count")
ax.legend()
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K")
)

ax = axes[1, 2]
n_seqs = [[len(split_label_lens[s][l]) for s in SPLITS] for l in [0, 1]]
ax.bar(x - w/2, n_seqs[0], w, label="Normal", color=COLORS[0])
ax.bar(x + w/2, n_seqs[1], w, label="Attack", color=COLORS[1])
ax.set_xticks(x)
ax.set_xticklabels([s.split("-")[1] for s in SPLITS])
ax.set_title("Sequence count per split")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig1_distributions.png", dpi=130)
plt.show()
console.print("[green]Saved: fig1_distributions.png[/green]")

# %%
# ── Figure 3: Statistical Analysis ────────────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle(f"{DATASET_NAME} — Statistical Analysis", fontsize=13, fontweight="bold")

ax = axes[0]
n_lens = np.sort(np.array(split_label_lens[TRAIN_SPLIT][0]))
a_lens = np.sort(np.array(split_label_lens[TRAIN_SPLIT][1]))
quantiles = np.linspace(0, 1, 100)
n_q = np.quantile(n_lens, quantiles) if len(n_lens) else np.zeros_like(quantiles)
a_q = np.quantile(a_lens, quantiles) if len(a_lens) else np.zeros_like(quantiles)
ax.scatter(n_q, a_q, s=15, alpha=0.7, color="purple")
mn, mx = min(n_q.min(), a_q.min()), max(n_q.max(), a_q.max())
ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="y=x (identical distributions)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Normal quantiles (log)")
ax.set_ylabel("Attack quantiles (log)")
ax.set_title(f"Q-Q plot: seq_length normal vs attack ({TRAIN_SPLIT})")
ax.legend(fontsize=8)

ax = axes[1]
all_valid_lens = np.array([s["actual_len"] for s in valid])
all_valid_labels = np.array([s["label"] for s in valid])
bins = np.unique(np.percentile(all_valid_lens, np.linspace(0, 100, 21)))
bin_centers = (bins[:-1] + bins[1:]) / 2
attack_fracs, counts = [], []
for i in range(len(bins) - 1):
    mask = (all_valid_lens >= bins[i]) & (all_valid_lens < bins[i + 1])
    attack_fracs.append(all_valid_labels[mask].mean() if mask.sum() > 0 else np.nan)
    counts.append(int(mask.sum()))
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

# %%
console.rule("[bold blue]Summary")
for split in SPLITS:
    n0 = sum(win_stats.get(split, {}).get(0, []))
    n1 = sum(win_stats.get(split, {}).get(1, []))
    console.print(f"  {split}: window imbalance {n1 / max(n0, 1):.0f}:1 (attack/normal)")
console.print(
    f"\n  Point-biserial r(seq_length, label) = {r:.4f} — attack sequences are "
    f"{'longer' if r > 0 else 'shorter'}"
)
console.print(f"[bold yellow]Recommended max_windows_per_seq = {rec['p90_n']}[/bold yellow] "
              f"(P90 of normal, {rec['pct_capped']:.0f}% attack seqs capped)")
