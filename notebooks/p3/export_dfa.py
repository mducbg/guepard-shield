"""Phase 3 — Step 5: Export the best DFA config to dfa_config.json.

Run from project root:
    uv run notebooks/p3/export_dfa.py

Selection criterion: lowest FPR with TPR ≥ 0.5.
Reads grid_search.csv produced by eval_dfa.py.

Output:
    results/p3/dfa_config.json

JSON schema (consumed by guepard-shield Rust loader):
    n_states          int
    n_tokens          int
    start_state       int
    transition_table  list[list[int]]  # shape [n_states][n_tokens], -1 = no transition
    syscall_to_token  dict[str, int]   # syscall_nr_str -> token_id (-1 = OOV/skip)
    state_class       list[int]        # 0 = NORMAL, 1 = SUSPECT (low-freq training state)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gp.dfa.export import DFAExporter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "guepard-shield-model"))

HIDDEN_DIR  = PROJECT_ROOT / "results" / "p3" / "hidden_states"
CLUSTER_DIR = PROJECT_ROOT / "results" / "p3" / "clusters"
DFA_DIR     = PROJECT_ROOT / "results" / "p3" / "dfa_s1"
METRICS_DIR = PROJECT_ROOT / "results" / "p3" / "metrics"
VOCAB_PATH  = PROJECT_ROOT / "data" / "processed" / "p2" / "vocab.json"
SYSCALL_TBL = PROJECT_ROOT / "docs" / "syscall.tbl"
OUT_PATH    = PROJECT_ROOT / "results" / "p3" / "dfa_config.json"

MIN_DR = 0.5  # minimum per-recording detection rate
SYSCALL_MAP_SIZE = 512  # BPF Array size for SYSCALL_TO_TOKEN


def read_grid(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def select_best(rows: list[dict]) -> dict | None:
    """Return the row with minimum FPR subject to dr_rec ≥ MIN_DR.

    S1 configs are excluded: after subset construction their state IDs are
    subset-construction IDs, not K-Means cluster IDs, so state_tiers computed
    from cluster frequencies would be wrong.
    """
    s1_excluded = [r for r in rows if r["strategy"] == "S1"]
    if s1_excluded:
        print(
            f"  [note] Excluding {len(s1_excluded)} S1 config(s) from export "
            f"candidates (subset-construction state IDs incompatible with "
            f"cluster-frequency state_class)."
        )
    candidates = [
        r for r in rows
        if float(r["dr_rec"]) >= MIN_DR and r["strategy"] != "S1"
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: float(r["fpr"]))


def load_transitions(config_name: str) -> dict[tuple[int, int], int]:
    dfa_dir = DFA_DIR / config_name
    data = np.load(dfa_dir / "transitions.npz")
    src, tok, dst = data["src"], data["tok"], data["dst"]
    return {(int(s), int(t)): int(d) for s, t, d in zip(src, tok, dst)}


def compute_state_freqs(config_name: str, M: int) -> np.ndarray:
    """Count how many training windows land in each cluster state."""
    parts = config_name.split("_")
    K = int(parts[0][1:])
    labels_path = CLUSTER_DIR / f"K{K}" / "labels.dat"
    labels = np.memmap(labels_path, dtype="int32", mode="r", shape=(M,))
    freqs = np.bincount(np.asarray(labels), minlength=K).astype(np.int64)
    return freqs


def parse_syscall_tbl(tbl_path: Path) -> dict[str, int]:
    """Return {syscall_name: syscall_nr} from Linux syscall.tbl format."""
    name_to_nr: dict[str, int] = {}
    for line in tbl_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                nr = int(parts[0])
                name = parts[2]
                name_to_nr[name] = nr
            except ValueError:
                continue
    return name_to_nr


def build_flat_config(
    transitions: dict[tuple[int, int], int],
    vocab: dict[str, int],
    state_freqs: np.ndarray,
    state_tiers: dict[str, str],
    start_state: int,
    metadata: dict,
) -> dict:
    """Convert DFAExporter Format A → flat Format B consumed by the Rust loader.

    state_class encodes per-state classification:
        0 = NORMAL  (state visited frequently during training)
        1 = SUSPECT (state below the edge_percentile frequency threshold)
    """
    n_states = len(state_freqs)
    n_tokens = metadata["vocab_size"]

    # Flat 2D transition table: table[state][token] = next_state, -1 = missing.
    table: list[list[int]] = [[-1] * n_tokens for _ in range(n_states)]
    for (src, tok), dst in transitions.items():
        if src < n_states and tok < n_tokens:
            table[src][tok] = dst

    # syscall_to_token keyed by syscall_nr (string) → token_id.
    # Uses syscall.tbl to map name → nr; unknown syscalls get -1.
    name_to_nr = parse_syscall_tbl(SYSCALL_TBL)
    # Invert vocab: token_name → token_id (vocab is already name→id).
    syscall_to_token: dict[str, int] = {str(nr): -1 for nr in range(SYSCALL_MAP_SIZE)}
    for name, token_id in vocab.items():
        if name.startswith("<"):
            continue  # skip special tokens (<PAD>, <UNK>, etc.)
        nr = name_to_nr.get(name)
        if nr is not None and 0 <= nr < SYSCALL_MAP_SIZE:
            syscall_to_token[str(nr)] = token_id

    # state_class: 0 = NORMAL, 1 = SUSPECT.
    state_class: list[int] = [0] * n_states
    for state_id_str, tier in state_tiers.items():
        idx = int(state_id_str)
        if tier == "EDGE" and idx < n_states:
            state_class[idx] = 1

    n_suspect = sum(state_class)
    print(f"  state_class: {n_states - n_suspect} NORMAL, {n_suspect} SUSPECT")

    return {
        "n_states": n_states,
        "n_tokens": n_tokens,
        "start_state": start_state,
        "transition_table": table,
        "syscall_to_token": syscall_to_token,
        "state_class": state_class,
        "metadata": metadata,
    }


def load_start_state(config_name: str) -> int:
    """Read the start state saved by build_dfa.py."""
    parts = config_name.split("_")
    K = int(parts[0][1:])
    cluster_dir = CLUSTER_DIR / f"K{K}"
    # Older phase-3 runs persisted this artifact with the S1 suffix even
    # when the selected export is S3. Prefer the canonical filename but do
    # not silently change the evaluated DFA start state to zero.
    for name in ("start_state.txt", "start_state_s1.txt"):
        start_path = cluster_dir / name
        if start_path.exists():
            return int(start_path.read_text().strip())
    # Fallback: use the most frequent state overall.
    return 0


def main() -> None:
    csv_path = METRICS_DIR / "grid_search.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run eval_dfa.py first ({csv_path} missing)")

    rows = read_grid(csv_path)
    best = select_best(rows)
    if best is None:
        print(f"No config found with dr_rec ≥ {MIN_DR}. Best configs by dr_rec:")
        for r in sorted(rows, key=lambda r: float(r["dr_rec"]), reverse=True)[:5]:
            print(f"  {r['config']}: dr_rec={r['dr_rec']}  FPR={r['fpr']}")
        return

    config_name = best["config"]
    print(f"Best config: {config_name}")
    print(f"  FPR={best['fpr']}  dr_rec={best['dr_rec']}  fidelity={best['fidelity']}")
    print(f"  n_states={best['n_states']}  n_trans={best['n_trans']}")

    with open(HIDDEN_DIR / "info.json") as f:
        info = json.load(f)
    M = info["M"]

    parts = config_name.split("_")
    K = int(parts[0][1:])
    strategy = parts[1]
    theta_str = parts[2][1:] if len(parts) > 2 else ""

    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    transitions = load_transitions(config_name)
    state_freqs = compute_state_freqs(config_name, M)
    start_state = load_start_state(config_name)

    metadata = {
        "K": K,
        "strategy": strategy,
        "theta": float(theta_str) if theta_str else None,
        "vocab_size": info["vocab_size"],
        "n_states": int(best["n_states"]),
        "n_transitions": int(best["n_trans"]),
        "fpr": float(best["fpr"]),
        "dr_rec": float(best["dr_rec"]),
        "fidelity": float(best["fidelity"]),
    }

    # Get state_tiers from DFAExporter (computes EDGE threshold via percentile).
    exporter = DFAExporter()
    fmt_a = exporter.to_json(
        transitions=transitions,
        vocab=vocab,
        state_freqs=state_freqs,
        metadata=metadata,
    )
    state_tiers: dict[str, str] = fmt_a.get("state_tiers", {})

    config = build_flat_config(
        transitions=transitions,
        vocab=vocab,
        state_freqs=state_freqs,
        state_tiers=state_tiers,
        start_state=start_state,
        metadata=metadata,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nExported → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
    try:
        plot_best_dfa_stats(config_name, transitions, state_freqs)
    except Exception as e:
        print(f"Error plotting best DFA stats: {e}")


def plot_best_dfa_stats(
    config_name: str,
    transitions: dict[tuple[int, int], int],
    state_freqs: np.ndarray,
) -> None:
    plots_dir = PROJECT_ROOT / "results" / "p3" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_degrees: dict[int, int] = {}
    for (src, _tok), _dst in transitions.items():
        out_degrees[src] = out_degrees.get(src, 0) + 1
    for s in range(len(state_freqs)):
        if s not in out_degrees:
            out_degrees[s] = 0

    degrees = list(out_degrees.values())

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(degrees, bins=range(min(degrees), max(degrees) + 2), kde=False, color="dodgerblue")
    plt.xlabel("Out-Degree (Number of outgoing transitions)")
    plt.ylabel("Number of States")
    plt.title(f"State Out-Degree Distribution ({config_name})")
    plt.tight_layout()
    plt.savefig(plots_dir / "best_dfa_out_degrees.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    sorted_freqs = np.sort(state_freqs)[::-1]
    total = sum(state_freqs)
    pct = (sorted_freqs / total) * 100 if total > 0 else sorted_freqs
    plt.plot(np.arange(1, len(pct) + 1), pct, marker="o", color="forestgreen", markersize=4)
    plt.xlabel("State Rank (sorted by frequency)")
    plt.ylabel("Percentage of training windows (%)")
    plt.title(f"State Frequency Distribution ({config_name})")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(plots_dir / "best_dfa_state_frequencies.png", dpi=300)
    plt.close()

    print(f"Saved best DFA stats plots to {plots_dir}")


if __name__ == "__main__":
    main()
