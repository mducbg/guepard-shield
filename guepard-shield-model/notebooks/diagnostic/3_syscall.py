# %% [markdown]
# # D03 — Syscall & Vocab Analysis + Split Consistency
# Requires: d01_integrity.py has been run (or re-run scan here).

# %%
import re
from collections import Counter

import matplotlib.pyplot as plt
from config import COLORS, CORPUS as corpus, COUNT_TOKENS, DATASET_NAME, OUTPUT_DIR, SPLITS, TOKEN_READER, TRAIN_SPLIT
from guepard.evaluation.corpus_stats import scan_corpus_integrity
from guepard.evaluation.syscall_analysis import (
    compute_pmi,
    cooccurrence_matrix,
    count_syscalls,
    stream_tokens,
    syscall_entropy,
)
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()

# %%
seq_data, _ = scan_corpus_integrity(corpus.metadata, count_tokens=COUNT_TOKENS)
valid = [s for s in seq_data if s["actual_len"] is not None]

# %%
# ── Syscall counts ─────────────────────────────────────────────────────────────
console.rule("[bold blue]Syscall & Vocab Analysis")

syscall_counts, syscall_seq_count = count_syscalls(valid, token_reader=TOKEN_READER)

only_normal = set(syscall_counts[0]) - set(syscall_counts[1])
only_attack = set(syscall_counts[1]) - set(syscall_counts[0])
shared = set(syscall_counts[0]) & set(syscall_counts[1])
total_counter: Counter = syscall_counts[0] + syscall_counts[1]

console.print(
    f"Total unique syscalls: {len(set(syscall_counts[0]) | set(syscall_counts[1])):>6,}"
)
console.print(f"  Shared:              {len(shared):>6,}")
console.print(f"  Only in normal:      {len(only_normal):>6,}")
console.print(f"  Only in attack:      {len(only_attack):>6,}")

h_normal = syscall_entropy(syscall_counts[0])
h_attack = syscall_entropy(syscall_counts[1])
console.print("\nSyscall distribution entropy:")
console.print(f"  Normal: {h_normal:.3f} bits")
console.print(f"  Attack: {h_attack:.3f} bits")
console.print(
    f"  → {'Attack more diverse' if h_attack > h_normal else 'Normal more diverse'}"
)

top20 = total_counter.most_common(20)
table = Table(title="Top-20 Syscalls", box=box.SIMPLE_HEAD)
table.add_column("Syscall", style="cyan")
for col in ["Total count", "Normal count", "Attack count", "Attack %"]:
    table.add_column(col, justify="right")
for syscall, total in top20:
    a_cnt = syscall_counts[1][syscall]
    n_cnt = syscall_counts[0][syscall]
    pct = 100 * a_cnt / total if total else 0
    table.add_row(syscall, f"{total:,}", f"{n_cnt:,}", f"{a_cnt:,}", f"{pct:.1f}%")
console.print(table)

# %%
# ── PMI ────────────────────────────────────────────────────────────────────────
n_attack_seqs = sum(1 for s in valid if s["label"] == 1)
n_total_seqs = len(valid)

pmi_scores = compute_pmi(syscall_seq_count, n_attack_seqs, n_total_seqs, shared)

top10_attack = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:10]
top10_normal = sorted(pmi_scores.items(), key=lambda x: x[1])[:10]

console.print("\n[bold]Top-10 syscalls most associated with ATTACK (PMI):[/bold]")
for s, pmi in top10_attack:
    console.print(f"  {s:30s}  PMI={pmi:+.3f}")
console.print("[bold]Top-10 syscalls most associated with NORMAL (PMI):[/bold]")
for s, pmi in top10_normal:
    console.print(f"  {s:30s}  PMI={pmi:+.3f}")

# %%
# ── Split Consistency ──────────────────────────────────────────────────────────
console.rule("[bold blue]Split Consistency")


def attack_type(bug_name: str) -> str:
    m = re.match(r"([A-Za-z]+)", bug_name)
    return m.group(1) if m else bug_name


split_attack_types = {}
for split in SPLITS:
    metas = corpus.get_split(split)
    split_attack_types[split] = Counter(
        attack_type(m.bug_name) for m in metas if m.label == 1
    )

all_types = sorted(set().union(*[set(t.keys()) for t in split_attack_types.values()]))
table = Table(title="Attack Type Distribution Across Splits", box=box.SIMPLE_HEAD)
table.add_column("Attack type", style="cyan")
for split in SPLITS:
    table.add_column(split, justify="right")
for atype in all_types:
    table.add_row(atype, *[str(split_attack_types[s].get(atype, 0)) for s in SPLITS])
console.print(table)

split_vocabs: dict[str, set] = {}
for split in SPLITS:
    vocab: set[str] = set()
    subset = [s for s in valid if split in s["split"]]
    for _, tokens in stream_tokens(subset, desc=f"Vocab {split}", token_reader=TOKEN_READER):
        vocab.update(tokens)
    split_vocabs[split] = vocab

train_vocab = split_vocabs.get(TRAIN_SPLIT, set())
for key in SPLITS[1:]:
    other = split_vocabs.get(key, set())
    if train_vocab and other:
        oov = other - train_vocab
        console.print(
            f"  {key} OOV (not in train vocab): {len(oov):>4} syscalls "
            f"({100 * len(oov) / max(len(other), 1):.1f}%)"
        )

# %%
# ── Figure 2: Syscall Analysis ─────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(18, 7))
fig2.suptitle(f"{DATASET_NAME} — Syscall Analysis", fontsize=13, fontweight="bold")

ax = axes[0]
top15_attack = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:15]
top15_normal = sorted(pmi_scores.items(), key=lambda x: x[1])[:15]
combined_sorted = sorted(top15_attack + top15_normal, key=lambda x: x[1])
syscall_names, pmi_vals = zip(*combined_sorted)
bar_colors = [COLORS[1] if v > 0 else COLORS[0] for v in pmi_vals]
ax.barh(range(len(pmi_vals)), pmi_vals, color=bar_colors, alpha=0.8)
ax.set_yticks(range(len(syscall_names)))
ax.set_yticklabels(syscall_names, fontsize=8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Syscall PMI (positive=attack-assoc, negative=normal-assoc)")
ax.set_xlabel("PMI score")

ax = axes[1]
top15_syscalls = [s for s, _ in total_counter.most_common(15)]
cooc_norm = cooccurrence_matrix(valid, top15_syscalls, token_reader=TOKEN_READER)
im = ax.imshow(cooc_norm, cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(len(top15_syscalls)))
ax.set_xticklabels(top15_syscalls, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(top15_syscalls)))
ax.set_yticklabels(top15_syscalls, fontsize=8)
ax.set_title("Syscall Co-occurrence (Jaccard similarity, top-15 by freq)")
plt.colorbar(im, ax=ax, label="Jaccard similarity")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_syscall_analysis.png", dpi=130)
plt.show()
console.print("[green]Saved: fig2_syscall_analysis.png[/green]")
