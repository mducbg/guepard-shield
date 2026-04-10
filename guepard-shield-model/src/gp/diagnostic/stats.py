"""Diagnostic statistics for syscall recordings — single-pass design."""

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gp.data_loader.recording import Recording
from rich.console import Console
from rich.table import Table

console = Console()


class Stat:
    """Accumulate all dataset statistics in a single pass over the recordings.

    Usage::

        stat = Stat(syscall_table=syscall_table)
        for recording in track(dataset, total=len(dataset)):
            stat.analyze(recording)

        stat.report_integrity(save_dir=OUT_DIR)
        stat.report_seq_lengths(save_dir=OUT_DIR)
        ...
    """

    def __init__(
        self,
        syscall_table: dict[str, int] | None = None,
        top_n: int = 30,
    ) -> None:
        self._syscall_table = syscall_table
        self._top_n = top_n

        # integrity
        self._total: int = 0
        self._normal: int = 0
        self._attack: int = 0
        self._empty: int = 0

        # seq_lengths
        self._normal_lengths: list[int] = []
        self._attack_lengths: list[int] = []

        # syscall_vocab
        self._all_counter: Counter = Counter()
        self._normal_counter: Counter = Counter()
        self._attack_counter: Counter = Counter()

        # scenario_balance
        self._scenario_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"normal": 0, "attack": 0, "total": 0}
        )

        # attack_timing
        self._fractions: list[float] = []

        # oov_rate
        self._oov_total: int = 0
        self._oov: int = 0
        self._oov_counter: Counter = Counter()

    # ------------------------------------------------------------------
    # single-pass accumulation
    # ------------------------------------------------------------------

    def analyze(self, recording: Recording) -> None:
        self._integrity(recording)
        self._seq_lengths(recording)
        self._syscall_vocab(recording)
        self._scenario_balance(recording)
        self._attack_timing(recording)
        if self._syscall_table is not None:
            self._oov_rate(recording)

    def _integrity(self, rec: Recording) -> None:
        self._total += 1
        if rec.label == 0:
            self._normal += 1
        else:
            self._attack += 1
        if len(rec.syscalls) == 0:
            self._empty += 1

    def _seq_lengths(self, rec: Recording) -> None:
        length = len(rec.syscalls)
        if rec.label == 0:
            self._normal_lengths.append(length)
        else:
            self._attack_lengths.append(length)

    def _syscall_vocab(self, rec: Recording) -> None:
        c = Counter(rec.syscalls)
        self._all_counter.update(c)
        if rec.label == 0:
            self._normal_counter.update(c)
        else:
            self._attack_counter.update(c)

    def _scenario_balance(self, rec: Recording) -> None:
        scenario = (rec.metadata or {}).get("scenario", "unknown")
        self._scenario_counts[scenario]["total"] += 1
        if rec.label == 0:
            self._scenario_counts[scenario]["normal"] += 1
        else:
            self._scenario_counts[scenario]["attack"] += 1

    def _attack_timing(self, rec: Recording) -> None:
        if rec.label != 1:
            return
        meta = rec.metadata or {}
        frac = None

        if "time" in meta and isinstance(meta["time"], dict):
            time_info = meta["time"]
            exploits = time_info.get("exploit", [])
            container_ready = time_info.get("container_ready", {})
            recording_time = meta.get("recording_time")
            if exploits and container_ready and recording_time:
                elapsed = exploits[0]["absolute"] - container_ready["absolute"]
                frac = max(0.0, min(1.0, elapsed / recording_time))

        elif "exploit_start_time" in meta and meta["exploit_start_time"] is not None:
            recording_time = meta.get("recording_time")
            if recording_time and recording_time > 0:
                frac = max(0.0, min(1.0, meta["exploit_start_time"] / recording_time))

        if frac is not None:
            self._fractions.append(frac)

    def _oov_rate(self, rec: Recording) -> None:
        assert self._syscall_table is not None
        for syscall in rec.syscalls:
            self._oov_total += 1
            if syscall not in self._syscall_table:
                self._oov += 1
                self._oov_counter[syscall] += 1

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_stats(lengths: list[int]) -> dict:
        if not lengths:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
        arr = np.array(lengths)
        return {
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    def report_integrity(self, save_dir: Path | None = None) -> None:
        ratio = (
            self._attack / self._normal
            if self._normal > 0
            else (float("inf") if self._attack > 0 else 0)
        )

        rows = [
            ("Total Recordings", str(self._total)),
            ("Normal (label=0)", str(self._normal)),
            ("Attack (label=1)", str(self._attack)),
            ("Class Ratio (attack/normal)", f"{ratio:.3f}"),
            ("Empty Recordings", str(self._empty)),
        ]

        table = Table(title="Data Integrity Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        for metric, value in rows:
            table.add_row(metric, value)
        console.print(table)

        if self._empty > 0:
            console.print(
                f"[yellow]Warning: {self._empty} recordings have no syscalls[/yellow]"
            )

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / "integrity.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerows(rows)

    def report_seq_lengths(self, save_dir: Path | None = None) -> None:
        all_lengths = self._normal_lengths + self._attack_lengths
        if not all_lengths:
            console.print("[yellow]No recordings to analyze[/yellow]")
            return

        all_stats = self._calc_stats(all_lengths)
        normal_stats = self._calc_stats(self._normal_lengths)
        attack_stats = self._calc_stats(self._attack_lengths)

        table = Table(title="Sequence Length Statistics")
        table.add_column("Statistic", style="cyan")
        table.add_column("All", style="magenta")
        table.add_column("Normal", style="green")
        table.add_column("Attack", style="red")
        for stat in ["min", "max", "mean", "median", "p95"]:
            fmt = ",.1f" if stat == "mean" else ",.0f"
            table.add_row(
                stat.capitalize(),
                f"{all_stats[stat]:{fmt}}",
                f"{normal_stats[stat]:{fmt}}",
                f"{attack_stats[stat]:{fmt}}",
            )
        console.print(table)

        p95_all = all_stats["p95"]
        recommended = 2 ** math.ceil(math.log2(max(p95_all, 1)))
        console.print(
            f"\n[bold]Recommended max_len:[/bold] {recommended}  (next power-of-2 ≥ p95={p95_all:.0f})"
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, lengths, stats, title, color in [
            (axes[0], all_lengths, all_stats, f"All (n={len(all_lengths)})", "blue"),
            (
                axes[1],
                self._normal_lengths,
                normal_stats,
                f"Normal (n={len(self._normal_lengths)})",
                "green",
            ),
            (
                axes[2],
                self._attack_lengths,
                attack_stats,
                f"Attack (n={len(self._attack_lengths)})",
                "red",
            ),
        ]:
            if not lengths:
                ax.set_visible(False)
                continue
            ax.hist(lengths, bins=50, color=color, alpha=0.7, edgecolor="black")
            ax.set_title(title)
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Count")
            ax.axvline(
                stats["mean"],
                color="red",
                linestyle="--",
                label=f"Mean: {stats['mean']:.0f}",
            )
            ax.axvline(
                stats["median"],
                color="black",
                linestyle="--",
                label=f"Median: {stats['median']:.0f}",
            )
            ax.legend()

        plt.tight_layout()

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / "seq_lengths.png", dpi=150, bbox_inches="tight")
            with open(save_dir / "seq_lengths.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["statistic", "all", "normal", "attack"])
                for stat in ["min", "max", "mean", "median", "p95"]:
                    writer.writerow(
                        [stat, all_stats[stat], normal_stats[stat], attack_stats[stat]]
                    )

        plt.show()

    def report_syscall_vocab(self, save_dir: Path | None = None) -> None:
        vocab_all = set(self._all_counter)
        vocab_normal = set(self._normal_counter)
        vocab_attack = set(self._attack_counter)
        attack_only = vocab_attack - vocab_normal

        total_all = sum(self._all_counter.values())
        total_normal = sum(self._normal_counter.values())
        total_attack = sum(self._attack_counter.values())

        table = Table(title="Syscall Vocabulary Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("All", style="magenta")
        table.add_column("Normal", style="green")
        table.add_column("Attack", style="red")
        table.add_row(
            "Unique Syscalls",
            str(len(vocab_all)),
            str(len(vocab_normal)),
            str(len(vocab_attack)),
        )
        table.add_row(
            "Total Syscalls", f"{total_all:,}", f"{total_normal:,}", f"{total_attack:,}"
        )
        console.print(table)

        if attack_only:
            console.print(
                f"\n[yellow]Syscalls only in attack ({len(attack_only)}):[/yellow]"
            )
            console.print(", ".join(sorted(attack_only)))

        top_syscalls = self._all_counter.most_common(self._top_n)

        table = Table(title=f"Top {self._top_n} Most Frequent Syscalls")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Syscall", style="cyan")
        table.add_column("Count", style="magenta", justify="right")
        table.add_column("Percentage", style="green", justify="right")
        for i, (syscall, count) in enumerate(top_syscalls, 1):
            pct = count / total_all * 100 if total_all > 0 else 0
            table.add_row(str(i), syscall, f"{count:,}", f"{pct:.2f}%")
        console.print(table)

        if not top_syscalls:
            return

        syscall_names, counts = zip(*top_syscalls)
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(
            range(len(syscall_names)), counts, color="steelblue", edgecolor="black"
        )
        ax.set_xticks(range(len(syscall_names)))
        ax.set_xticklabels(syscall_names, rotation=45, ha="right")
        ax.set_xlabel("Syscall")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Top {self._top_n} Syscall Frequencies")
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h,
                f"{int(h):,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.tight_layout()

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / "syscall_vocab.png", dpi=150, bbox_inches="tight")
            with open(save_dir / "syscall_vocab.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["rank", "syscall", "count", "percentage"])
                for i, (syscall, count) in enumerate(top_syscalls, 1):
                    pct = count / total_all * 100 if total_all > 0 else 0
                    writer.writerow([i, syscall, count, f"{pct:.4f}"])
            with open(save_dir / "attack_only_syscalls.txt", "w") as f:
                f.write("\n".join(sorted(attack_only)))

        plt.show()

    def report_scenario_balance(self, save_dir: Path | None = None) -> None:
        table = Table(title="Per-Scenario Class Balance")
        table.add_column("Scenario", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Normal", style="green", justify="right")
        table.add_column("Attack", style="red", justify="right")
        table.add_column("Attack %", justify="right")

        rows = []
        for scenario in sorted(self._scenario_counts):
            c = self._scenario_counts[scenario]
            pct = c["attack"] / c["total"] * 100 if c["total"] > 0 else 0
            table.add_row(
                scenario,
                str(c["total"]),
                str(c["normal"]),
                str(c["attack"]),
                f"{pct:.1f}%",
            )
            rows.append([scenario, c["total"], c["normal"], c["attack"], f"{pct:.2f}"])

        console.print(table)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / "scenario_balance.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["scenario", "total", "normal", "attack", "attack_pct"])
                writer.writerows(rows)

    def report_attack_timing(self, save_dir: Path | None = None) -> None:
        if not self._fractions:
            console.print("[yellow]No attack timing data available.[/yellow]")
            return

        arr = np.array(self._fractions)
        t = Table(title=f"Attack Timing (n={len(self._fractions)} attack recordings)")
        t.add_column("Statistic", style="cyan")
        t.add_column("Value", style="magenta")
        t.add_row("Min", f"{arr.min():.3f}")
        t.add_row("Mean", f"{arr.mean():.3f}")
        t.add_row("Median", f"{float(np.median(arr)):.3f}")
        t.add_row("Max", f"{arr.max():.3f}")
        t.add_row(
            "Attacks in startup zone (< 20%)", f"{(arr < 0.20).mean() * 100:.1f}%"
        )
        t.add_row(
            "Attacks in shutdown zone (> 80%)", f"{(arr > 0.80).mean() * 100:.1f}%"
        )
        console.print(t)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(self._fractions, bins=20, color="crimson", alpha=0.7, edgecolor="black")
        ax.axvline(
            0.20, color="orange", linestyle="--", label="startup boundary (0.20)"
        )
        ax.axvline(
            0.80, color="orange", linestyle="--", label="shutdown boundary (0.80)"
        )
        ax.set_xlabel("Attack start (fraction of recording duration)")
        ax.set_ylabel("Count")
        ax.set_title("When do attacks occur within the recording?")
        ax.legend()
        plt.tight_layout()

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / "attack_timing.png", dpi=150, bbox_inches="tight")
            with open(save_dir / "attack_timing.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["attack_start_fraction"])
                for frac in self._fractions:
                    writer.writerow([f"{frac:.6f}"])

        plt.show()

    def report_oov_rate(self, save_dir: Path | None = None) -> None:
        if self._oov_total == 0:
            console.print("[yellow]No syscalls to analyze.[/yellow]")
            return

        oov_pct = self._oov / self._oov_total * 100
        t = Table(title="Out-of-Vocabulary (OOV) Rate vs syscall_64.tbl")
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="magenta")
        t.add_row("Total syscalls", f"{self._oov_total:,}")
        t.add_row("OOV syscalls", f"{self._oov:,}")
        t.add_row("OOV rate", f"{oov_pct:.3f}%")
        t.add_row("Unique OOV names", str(len(self._oov_counter)))
        console.print(t)

        if self._oov_counter:
            t2 = Table(title="Top 10 OOV Syscall Names")
            t2.add_column("Syscall", style="cyan")
            t2.add_column("Count", justify="right")
            for name, count in self._oov_counter.most_common(10):
                t2.add_row(name, f"{count:,}")
            console.print(t2)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / "oov_rate.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["syscall", "count"])
                for name, count in self._oov_counter.most_common():
                    writer.writerow([name, count])
