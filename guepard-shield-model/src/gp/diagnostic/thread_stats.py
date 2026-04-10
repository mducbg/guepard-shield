"""Thread-level diagnostic statistics for assessing graph-based model viability.

Each Recording stores syscalls, timestamps, and tids in parallel lists,
allowing per-thread sequence reconstruction without re-reading raw files.

Key questions answered:
  1. thread_count_dist   — how many graph nodes per recording? (feasibility)
  2. per_thread_seq_len  — are per-thread sequences long enough for node embeddings?
  3. syscall_bigram_graph — are normal vs attack transition graphs structurally different?
  4. attack_thread_isolation — is the attack TID distinguishable by syscall distribution?
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gp.data_loader.recording import Recording
from rich.console import Console
from rich.table import Table

console = Console()


def _array_stats(xs: list[int]) -> dict:
    if not xs:
        return {}
    a = np.array(xs)
    return {
        "n": len(xs),
        "min": int(a.min()),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": int(a.max()),
        "mean": float(a.mean()),
    }


def _split_by_tid(recording: Recording) -> dict[int, list[str]]:
    if recording.tid is None:
        return {}
    tid_seqs: dict[int, list[str]] = defaultdict(list)
    for syscall, tid in zip(recording.syscalls, recording.tid):
        tid_seqs[tid].append(syscall)
    return tid_seqs


def _kl_div(p: Counter, q: Counter) -> float:
    vocab = set(p) | set(q)
    total_p = sum(p.values()) + len(vocab)
    total_q = sum(q.values()) + len(vocab)
    kl = 0.0
    for w in vocab:
        prob_p = (p.get(w, 0) + 1) / total_p
        prob_q = (q.get(w, 0) + 1) / total_q
        kl += prob_p * np.log(prob_p / prob_q)
    return float(kl)


class ThreadStat:
    """Accumulate all thread-level statistics in a single pass.

    Usage::

        stat = ThreadStat()
        for recording in track(dataset, total=len(dataset)):
            stat.analyze(recording)

        stat.report_thread_count_dist(save_dir=OUT_DIR)
        stat.report_per_thread_seq_len(save_dir=OUT_DIR)
        ...
    """

    def __init__(self, top_n: int = 20) -> None:
        self._top_n = top_n

        # thread_count_dist
        self._normal_counts: list[int] = []
        self._attack_counts: list[int] = []
        self._scenario_counts: dict[str, list[int]] = defaultdict(list)

        # per_thread_seq_len
        self._all_lens: list[int] = []
        self._normal_lens: list[int] = []
        self._attack_lens: list[int] = []
        self._short_thread_fracs: list[float] = []

        # syscall_bigram_graph
        self._normal_bigrams: Counter = Counter()
        self._attack_bigrams: Counter = Counter()

        # attack_thread_isolation
        self._kl_values: list[float] = []
        self._attack_tid_sizes: list[int] = []
        self._normal_tid_sizes: list[int] = []

    # ------------------------------------------------------------------
    # single-pass accumulation
    # ------------------------------------------------------------------

    def analyze(self, recording: Recording) -> None:
        self._thread_count_dist(recording)
        self._per_thread_seq_len(recording)
        self._syscall_bigram_graph(recording)
        self._attack_thread_isolation(recording)

    def _thread_count_dist(self, rec: Recording) -> None:
        if rec.tid is None:
            return
        n_threads = len(set(rec.tid))
        scenario = (rec.metadata or {}).get("scenario", "unknown")
        if rec.label == 0:
            self._normal_counts.append(n_threads)
        else:
            self._attack_counts.append(n_threads)
        self._scenario_counts[scenario].append(n_threads)

    def _per_thread_seq_len(self, rec: Recording) -> None:
        tid_seqs = _split_by_tid(rec)
        if not tid_seqs:
            return
        lens = [len(v) for v in tid_seqs.values()]
        self._short_thread_fracs.append(sum(1 for len in lens if len < 10) / len(lens))
        self._all_lens.extend(lens)
        if rec.label == 0:
            self._normal_lens.extend(lens)
        else:
            self._attack_lens.extend(lens)

    def _syscall_bigram_graph(self, rec: Recording) -> None:
        bigrams = Counter(zip(rec.syscalls[:-1], rec.syscalls[1:]))
        if rec.label == 0:
            self._normal_bigrams.update(bigrams)
        else:
            self._attack_bigrams.update(bigrams)

    def _attack_thread_isolation(self, rec: Recording) -> None:
        if rec.label != 1:
            return
        tid_seqs = _split_by_tid(rec)
        if len(tid_seqs) < 2:
            return

        attack_tid = max(tid_seqs.keys())
        attack_counter = Counter(tid_seqs[attack_tid])
        normal_counter: Counter = Counter()
        for tid, seqs in tid_seqs.items():
            if tid != attack_tid:
                normal_counter.update(seqs)

        kl = _kl_div(attack_counter, normal_counter)
        self._kl_values.append(kl)
        self._attack_tid_sizes.append(len(tid_seqs[attack_tid]))
        self._normal_tid_sizes.append(
            sum(len(v) for tid, v in tid_seqs.items() if tid != attack_tid)
        )

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def report_thread_count_dist(self, save_dir: Path | None = None) -> None:
        n_stats = _array_stats(self._normal_counts)
        a_stats = _array_stats(self._attack_counts)

        t = Table(title="Thread Count per Recording")
        t.add_column("Metric", style="cyan")
        t.add_column("Normal", style="green")
        t.add_column("Attack", style="red")
        for k in ["n", "min", "p50", "p95", "max", "mean"]:
            t.add_row(
                k,
                f"{n_stats.get(k, '-'):.0f}"
                if isinstance(n_stats.get(k), float)
                else str(n_stats.get(k, "-")),
                f"{a_stats.get(k, '-'):.0f}"
                if isinstance(a_stats.get(k), float)
                else str(a_stats.get(k, "-")),
            )
        console.print(t)

        t2 = Table(title="Thread Count per Scenario (median / max)")
        t2.add_column("Scenario", style="cyan")
        t2.add_column("Recordings", justify="right")
        t2.add_column("Median threads", justify="right")
        t2.add_column("Max threads", justify="right")
        t2.add_column("GNN feasible?", justify="center")
        rows = []
        for scenario in sorted(self._scenario_counts):
            xs = np.array(self._scenario_counts[scenario])
            med = float(np.median(xs))
            mx = int(xs.max())
            feasible = "yes" if mx <= 64 else ("partial" if mx <= 256 else "no")
            t2.add_row(scenario, str(len(xs)), f"{med:.0f}", str(mx), feasible)
            rows.append([scenario, len(xs), med, mx, feasible])
        console.print(t2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, counts, label, color in [
            (axes[0], self._normal_counts, "Normal", "steelblue"),
            (axes[1], self._attack_counts, "Attack", "crimson"),
        ]:
            if not counts:
                ax.set_visible(False)
                continue
            ax.hist(
                np.log1p(counts), bins=40, color=color, alpha=0.7, edgecolor="black"
            )
            ax.set_xlabel("log(1 + thread count)")
            ax.set_ylabel("Recordings")
            ax.set_title(f"{label} recordings (n={len(counts)})")
        plt.suptitle("Thread Count Distribution (log scale)")
        plt.tight_layout()

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_dir / "thread_count_dist.png", dpi=150, bbox_inches="tight"
            )
            with open(save_dir / "thread_count_dist.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "scenario",
                        "n_recordings",
                        "median_threads",
                        "max_threads",
                        "gnn_feasible",
                    ]
                )
                writer.writerows(rows)
        plt.show()

    def report_per_thread_seq_len(self, save_dir: Path | None = None) -> None:
        a_s = _array_stats(self._all_lens)
        n_s = _array_stats(self._normal_lens)
        at_s = _array_stats(self._attack_lens)

        t = Table(title="Per-Thread Syscall Sequence Length")
        t.add_column("Statistic", style="cyan")
        t.add_column("All", style="magenta")
        t.add_column("Normal threads", style="green")
        t.add_column("Attack threads", style="red")
        for k in ["min", "p10", "p50", "p95", "max", "mean"]:
            t.add_row(
                k,
                f"{a_s.get(k, 0):,.0f}",
                f"{n_s.get(k, 0):,.0f}",
                f"{at_s.get(k, 0):,.0f}",
            )
        console.print(t)

        if self._short_thread_fracs:
            median_short = float(np.median(self._short_thread_fracs))
            console.print(
                f"\nMedian fraction of threads with < 10 syscalls per recording: "
                f"[yellow]{median_short * 100:.1f}%[/yellow]"
            )
            if median_short > 0.3:
                console.print(
                    "[yellow]Warning: >30% of threads have < 10 syscalls — "
                    "thread-level node features will be noisy. "
                    "Consider merging short threads or using syscall-bigram graph.[/yellow]"
                )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(
            np.log1p(self._all_lens),
            bins=50,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("log(1 + per-thread sequence length)")
        ax.set_ylabel("Thread count")
        ax.set_title(
            f"Per-Thread Sequence Length Distribution (n={len(self._all_lens):,} threads)"
        )
        ax.axvline(np.log1p(10), color="orange", linestyle="--", label="length=10")
        ax.axvline(np.log1p(100), color="red", linestyle="--", label="length=100")
        ax.legend()
        plt.tight_layout()

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_dir / "per_thread_seq_len.png", dpi=150, bbox_inches="tight"
            )
            with open(save_dir / "per_thread_seq_len.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["statistic", "all", "normal", "attack"])
                for k in ["min", "p10", "p50", "p95", "max", "mean"]:
                    writer.writerow(
                        [k, a_s.get(k, ""), n_s.get(k, ""), at_s.get(k, "")]
                    )
        plt.show()

    def report_syscall_bigram_graph(self, save_dir: Path | None = None) -> None:
        normal_edges = set(self._normal_bigrams)
        attack_edges = set(self._attack_bigrams)
        shared = normal_edges & attack_edges
        attack_only = attack_edges - normal_edges
        normal_only = normal_edges - attack_edges

        jaccard = (
            len(shared) / len(normal_edges | attack_edges)
            if (normal_edges | attack_edges)
            else 0.0
        )

        t = Table(title="Syscall Transition Graph: Normal vs Attack")
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="magenta")
        t.add_row("Normal unique edges", f"{len(normal_edges):,}")
        t.add_row("Attack unique edges", f"{len(attack_edges):,}")
        t.add_row("Shared edges", f"{len(shared):,}")
        t.add_row("Attack-only edges", f"{len(attack_only):,}")
        t.add_row("Normal-only edges", f"{len(normal_only):,}")
        t.add_row("Jaccard similarity", f"{jaccard:.4f}")
        if jaccard < 0.5:
            t.add_row(
                "Verdict",
                "[green]Low Jaccard → graphs differ → GNN can discriminate[/green]",
            )
        else:
            t.add_row(
                "Verdict",
                "[yellow]High Jaccard → graphs similar → GNN may struggle[/yellow]",
            )
        console.print(t)

        if attack_only:
            t2 = Table(title=f"Top {self._top_n} Attack-Only Transitions (a→b)")
            t2.add_column("From", style="cyan")
            t2.add_column("To", style="cyan")
            t2.add_column("Count", justify="right")
            rows = []
            for a, b in sorted(
                attack_only, key=self._attack_bigrams.__getitem__, reverse=True
            )[: self._top_n]:
                cnt = self._attack_bigrams[(a, b)]
                t2.add_row(a, b, f"{cnt:,}")
                rows.append([a, b, cnt, "attack_only"])
            console.print(t2)

        total_n = max(sum(self._normal_bigrams.values()), 1)
        total_a = max(sum(self._attack_bigrams.values()), 1)
        enrichment_rows = []
        for edge in self._attack_bigrams:
            a_rate = self._attack_bigrams[edge] / total_a
            n_rate = self._normal_bigrams.get(edge, 0) / total_n
            ratio = a_rate / (n_rate + 1e-9)
            enrichment_rows.append(
                (
                    edge,
                    self._attack_bigrams[edge],
                    self._normal_bigrams.get(edge, 0),
                    ratio,
                )
            )
        enrichment_rows.sort(key=lambda x: -x[3])

        t3 = Table(
            title=f"Top {self._top_n} Most Enriched Transitions in Attack vs Normal"
        )
        t3.add_column("From", style="cyan")
        t3.add_column("To", style="cyan")
        t3.add_column("Attack count", justify="right", style="red")
        t3.add_column("Normal count", justify="right", style="green")
        t3.add_column("Enrichment (A/N ratio)", justify="right")
        csv_rows_enriched = []
        for (a, b), a_cnt, n_cnt, ratio in enrichment_rows[: self._top_n]:
            t3.add_row(a, b, f"{a_cnt:,}", f"{n_cnt:,}", f"{ratio:.1f}x")
            csv_rows_enriched.append([a, b, a_cnt, n_cnt, f"{ratio:.2f}"])
        console.print(t3)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / "bigram_attack_only.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["from", "to", "count", "type"])
                for a, b in sorted(
                    attack_only, key=self._attack_bigrams.__getitem__, reverse=True
                ):
                    writer.writerow([a, b, self._attack_bigrams[(a, b)], "attack_only"])
            with open(save_dir / "bigram_enriched.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["from", "to", "attack_count", "normal_count", "enrichment_ratio"]
                )
                writer.writerows(csv_rows_enriched)
            with open(save_dir / "bigram_summary.json", "w") as f:
                json.dump(
                    {
                        "normal_unique_edges": len(normal_edges),
                        "attack_unique_edges": len(attack_edges),
                        "shared_edges": len(shared),
                        "attack_only_edges": len(attack_only),
                        "jaccard_similarity": jaccard,
                    },
                    f,
                    indent=2,
                )

    def report_attack_thread_isolation(self, save_dir: Path | None = None) -> None:
        if not self._kl_values:
            console.print(
                "[yellow]No attack recordings with multi-thread data.[/yellow]"
            )
            return

        arr = np.array(self._kl_values)
        t = Table(
            title=f"Attack Thread Isolation (n={len(self._kl_values)} attack recordings)"
        )
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="magenta")
        t.add_row("KL div min", f"{arr.min():.3f}")
        t.add_row("KL div mean", f"{arr.mean():.3f}")
        t.add_row("KL div median", f"{float(np.median(arr)):.3f}")
        t.add_row("KL div max", f"{arr.max():.3f}")
        t.add_row(
            "Recordings with KL > 1.0",
            f"{(arr > 1.0).sum()} ({(arr > 1.0).mean() * 100:.1f}%)",
        )
        t.add_row(
            "Recordings with KL > 5.0",
            f"{(arr > 5.0).sum()} ({(arr > 5.0).mean() * 100:.1f}%)",
        )
        t.add_row("Mean attack TID syscalls", f"{np.mean(self._attack_tid_sizes):.0f}")
        t.add_row("Mean normal TIDs syscalls", f"{np.mean(self._normal_tid_sizes):.0f}")
        console.print(t)

        if arr.mean() > 2.0:
            console.print(
                "[green]High KL → attack thread is syscall-distribution-distinct from normal threads. "
                "Thread-level node embeddings (BiLSTM per-thread) will carry strong signal.[/green]"
            )
        elif arr.mean() > 0.5:
            console.print(
                "[yellow]Moderate KL → partial isolation. "
                "Graph message-passing across threads adds discriminative power.[/yellow]"
            )
        else:
            console.print(
                "[red]Low KL → attack thread mimics normal thread behavior. "
                "Thread-level features alone insufficient; temporal ordering or cross-thread context needed.[/red]"
            )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(self._kl_values, bins=30, color="crimson", alpha=0.7, edgecolor="black")
        ax.axvline(
            1.0, color="orange", linestyle="--", label="KL=1 (moderate isolation)"
        )
        ax.axvline(5.0, color="red", linestyle="--", label="KL=5 (strong isolation)")
        ax.set_xlabel("KL Divergence D(attack_tid || normal_tids)")
        ax.set_ylabel("Count")
        ax.set_title("Attack Thread Syscall Distribution Isolation")
        ax.legend()
        plt.tight_layout()

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                save_dir / "attack_thread_isolation.png", dpi=150, bbox_inches="tight"
            )
            with open(save_dir / "attack_thread_isolation.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["kl_divergence", "attack_tid_syscalls", "normal_tids_syscalls"]
                )
                for kl, a, n in zip(
                    self._kl_values, self._attack_tid_sizes, self._normal_tid_sizes
                ):
                    writer.writerow([f"{kl:.4f}", a, n])
        plt.show()


def graph_model_summary(save_dir: Path | None = None) -> None:
    console.rule("Graph Model Architecture Recommendation")

    rec = [
        (
            "Thread-graph Transformer",
            "Nodes=threads, edges=co-scheduling",
            "Feasible only if median threads ≤ 64 per scenario",
            "High KL → strong node features; low Jaccard across threads",
        ),
        (
            "Syscall-bigram GNN",
            "Nodes=syscall types (~335), edges=bigram transitions",
            "Always feasible (fixed ~335 nodes); normal/attack Jaccard drives power",
            "Low bigram Jaccard → GNN can discriminate via edge weights",
        ),
        (
            "Hierarchical: BiLSTM per-thread + GNN across threads",
            "Each thread encoded by BiLSTM, then GNN aggregates",
            "Best of both: captures intra-thread patterns AND inter-thread structure",
            "Recommended if KL > 1 AND median threads ≤ 64",
        ),
        (
            "Thread clustering + GNN",
            "Cluster threads by behavior, ~4-8 cluster nodes",
            "Handles variable/large thread counts (ZipSlip=888 threads)",
            "Use when max threads >> 64 in some scenarios",
        ),
    ]

    t = Table(title="Graph Architecture Options", show_lines=True)
    t.add_column("Architecture", style="cyan", width=30)
    t.add_column("Graph definition", width=35)
    t.add_column("Feasibility condition", width=40)
    t.add_column("Signal source", width=40)
    for row in rec:
        t.add_row(*row)
    console.print(t)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "graph_model_summary.json", "w") as f:
            json.dump(
                [
                    {
                        "architecture": r[0],
                        "graph_definition": r[1],
                        "feasibility": r[2],
                        "signal_source": r[3],
                    }
                    for r in rec
                ],
                f,
                indent=2,
            )
