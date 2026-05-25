"""Phase 3 — Step 3b: Build NFA/DFA transitions at stride=1 without 175 GB disk.

Stride=4 problem: transition token = syscall at pos+67, not the next single syscall.
Fix: reuse stride=4 centroids, stream recordings at stride=1, encode once per recording,
update all K NFAs in one pass. Hidden states discarded immediately (175 GB → ~1 MB).

Run from project root:
    uv run notebooks/p3/build_transitions_stride1.py \\
        --ckpt results/p2/checkpoints/best.ckpt [--K 64 128] [--sanity-recs 5]

Outputs: results/p3/dfa_s1/K{K}_{S1,S3,S4_t*}/transitions.npz
Plots:   results/p3/plots/s1_*.png
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import pairwise_distances_argmin
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "guepard-shield-model"))

from gp.data_loader.lidds_2021_loader import LiddS2021Loader, UNK_TOKEN
from gp.dfa.transitions import TransitionBuilder
from gp.model import SyscallTransformer

DATA_DIR    = PROJECT_ROOT / "data" / "extracted" / "LID-DS-2021"
VOCAB_PATH  = PROJECT_ROOT / "data" / "processed" / "p2" / "vocab.json"
CLUSTER_DIR = PROJECT_ROOT / "results" / "p3" / "clusters"
DFA_S1_DIR  = PROJECT_ROOT / "results" / "p3" / "dfa_s1"
PLOTS_DIR   = PROJECT_ROOT / "results" / "p3" / "plots"

S4_THETAS = [0.80, 0.90, 0.95, 0.99]


@torch.inference_mode()
def encode(tokens: np.ndarray, W: int, model: SyscallTransformer, batch_size: int) -> np.ndarray:
    wins = np.lib.stride_tricks.sliding_window_view(tokens, W)
    if not len(wins):
        return np.empty((0, model.hparams.d_model))
    return np.concatenate([
        model.encode(torch.from_numpy(wins[b : b + batch_size].astype(np.int64)).to(model.device))
             .cpu().numpy().astype(np.float32)
        for b in range(0, len(wins), batch_size)
    ])


def sanity_check(loader, vocab, model, centroids, W, n_recs, batch_size) -> bool:
    unk, d4, d1 = vocab[UNK_TOKEN], [], []
    with tqdm(total=n_recs, desc="  sanity check", unit="rec", ncols=80) as pbar:
        for i, rec in enumerate(loader.stream_split("train")):
            if i >= n_recs:
                break
            tokens = np.array([vocab.get(s.syscall, unk) for s in rec.syscalls], np.int32)
            rec.syscalls.clear()
            pbar.update(1)
            if len(tokens) < W + 1:
                continue
            for stride, lst in [(4, d4), (1, d1)]:
                H = encode(tokens[::stride] if stride > 1 else tokens, W, model, batch_size)
                lbl = pairwise_distances_argmin(H, centroids)
                lst.append(float(np.linalg.norm(H - centroids[lbl], axis=1).mean()))
    if not d4 or not d1:
        return True
    ratio = np.mean(d1) / np.mean(d4)
    print(f"  d4={np.mean(d4):.4f}  d1={np.mean(d1):.4f}  ratio={ratio:.3f} (threshold<1.30)")
    ok = ratio < 1.30
    print("  PASSED" if ok else "  WARNING: ratio >= 1.30")
    return ok


def stream_all_k(K_list, centroids_map, loader, vocab, model, W, batch_size):
    unk = vocab[UNK_TOKEN]
    nfas = {K: defaultdict(Counter) for K in K_list}
    start_counts = {K: Counter() for K in K_list}
    n_recs = sum(1 for _ in DATA_DIR.rglob("*/training/*/*.sc"))
    total_wins = 0

    with tqdm(total=n_recs, desc="stride=1 streaming", unit="rec", ncols=100) as pbar:
        for rec in loader.stream_split("train"):
            tokens = np.array([vocab.get(s.syscall, unk) for s in rec.syscalls], np.int32)
            rec.syscalls.clear()
            if len(tokens) >= W + 1:
                H = encode(tokens, W, model, batch_size)
                trans_tok = tokens[W : W + len(H) - 1]
                total_wins += len(H)
                for K in K_list:
                    lbl = pairwise_distances_argmin(H, centroids_map[K]).astype(np.int32)
                    start_counts[K][int(lbl[0])] += 1
                    pairs = np.stack([lbl[:-1], trans_tok, lbl[1:]], axis=1)
                    unique, cnts = np.unique(pairs, axis=0, return_counts=True)
                    nfa = nfas[K]
                    for (s, t, d), c in zip(unique.tolist(), cnts.tolist()):
                        nfa[(s, t)][d] += c
            pbar.update(1)
            pbar.set_postfix(wins=f"{total_wins:,}", nfa=f"{len(nfas[K_list[0]]):,}")

    return nfas, start_counts


def save_transitions(path: Path, trans: dict, n_states: int) -> None:
    keys = list(trans.keys())
    np.savez(path,
             src=np.array([k[0] for k in keys], np.int32) if keys else np.empty(0, np.int32),
             tok=np.array([k[1] for k in keys], np.int32) if keys else np.empty(0, np.int32),
             dst=np.array(list(trans.values()), np.int32) if keys else np.empty(0, np.int32),
             n_states=np.int32(n_states), n_trans=np.int32(len(trans)))


def resolve_and_save(K, nfa, vocab_size, start_cluster) -> dict[str, int]:
    builder = TransitionBuilder(np.arange(K, dtype=np.int32), np.zeros((K, 3), np.int32), vocab_size, stride=1)
    builder._nfa = nfa  # type: ignore[attr-defined]
    nd = builder.nd_rate()
    print(f"  nd_rate={nd:.1%}  NFA pairs={len(nfa):,}")
    results: dict[str, int] = {}

    def _save(tag, trans, n_states):
        out = DFA_S1_DIR / f"K{K}_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "nd_rate.txt").write_text(f"{nd:.6f}\n")
        save_transitions(out / "transitions", trans, n_states)
        results[tag] = len(trans)

    s1 = builder.resolve_s1(initial_cluster=start_cluster)
    if s1 is None:
        out = DFA_S1_DIR / f"K{K}_S1"
        out.mkdir(parents=True, exist_ok=True)
        (out / "state_explosion.txt").write_text("1\n")
        results["S1"] = 0
        print("  S1: state explosion")
    else:
        n_s1 = max((max(s, d) for (s, _), d in s1.items()), default=-1) + 1
        _save("S1", s1, n_s1)
        print(f"  S1: {n_s1} states, {len(s1)} transitions")

    _save("S3", builder.resolve_s3(), K)
    print(f"  S3: {K} states, {results['S3']} transitions")
    for theta in S4_THETAS:
        tag = f"S4_t{theta:.2f}"
        _save(tag, builder.resolve_s4(theta), K)
        print(f"  S4(θ={theta:.2f}): {results[tag]} transitions")

    return results


def save_plots(K_list, nfas, trans_per_k) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    Ks = [str(K) for K in K_list]

    fig, ax = plt.subplots(figsize=(7, 4))
    nd_vals = [100 * sum(1 for c in nfas[K].values() if len(c) > 1) / max(len(nfas[K]), 1) for K in K_list]
    ax.plot(K_list, nd_vals, marker="o", linewidth=2, color="crimson")
    ax.set(xlabel="K", ylabel="Non-Determinism (%)", title="NFA Non-Determinism vs. K (stride=1)", xticks=K_list)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "s1_nd_rate_vs_k.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sizes = [len(nfas[K]) for K in K_list]
    b = ax.bar(Ks, sizes, color=sns.color_palette("Blues_d", len(K_list)))
    ax.bar_label(b, labels=[f"{v:,}" for v in sizes], padding=3, fontsize=9)
    ax.set(xlabel="K", ylabel="NFA pairs", title="NFA Size vs. K (stride=1)")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "s1_nfa_pairs_vs_k.png", dpi=300); plt.close(fig)

    if trans_per_k:
        strategies = list(next(iter(trans_per_k.values())).keys())
        x, w = np.arange(len(K_list)), 0.8 / len(strategies)
        palette = sns.color_palette("tab10", len(strategies))
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, strat in enumerate(strategies):
            ax.bar(x + (i - len(strategies)/2 + 0.5)*w, [trans_per_k[K].get(strat, 0) for K in K_list],
                   w*0.9, label=strat, color=palette[i])
        ax.set(xlabel="K", ylabel="Transitions", title="DFA Transitions per Strategy (stride=1)",
               xticks=x, xticklabels=Ks)
        ax.legend(title="Strategy", bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout(); fig.savefig(PLOTS_DIR / "s1_transitions_vs_k.png", dpi=300); plt.close(fig)

    print(f"  Plots → {PLOTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(PROJECT_ROOT / "results/p2/checkpoints/best.ckpt"))
    parser.add_argument("--K", nargs="+", type=int, default=[64])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sanity-recs", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: SyscallTransformer = SyscallTransformer.load_from_checkpoint(args.ckpt, map_location=device)
    model.freeze()
    W = model.hparams.window_size
    vocab = LiddS2021Loader.load_vocab(VOCAB_PATH)
    loader = LiddS2021Loader(DATA_DIR)
    print(f"Device={device}  W={W}  vocab={len(vocab)}")

    centroids_map = {}
    for K in args.K:
        p = CLUSTER_DIR / f"K{K}/centroids.npy"
        if p.exists():
            centroids_map[K] = np.load(p).astype(np.float32)
        else:
            print(f"[skip] K={K}: centroids.npy not found")
    K_list = list(centroids_map)
    if not K_list:
        return

    if args.sanity_recs > 0:
        ok = sanity_check(loader, vocab, model, centroids_map[K_list[0]],
                          W, args.sanity_recs, args.batch_size)
        if not ok and input("Continue? [y/N] ").strip().lower() != "y":
            return

    DFA_S1_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DFA_S1_DIR / f"nfas_cache_{'_'.join(map(str, K_list))}.pkl"
    if cache_path.exists():
        print(f"Loading cached NFAs and start counts from {cache_path}...")
        with open(cache_path, "rb") as f:
            nfas, start_counts = pickle.load(f)
    else:
        nfas, start_counts = stream_all_k(
            K_list, centroids_map, loader, vocab, model, W, args.batch_size)
        print(f"Saving NFAs and start counts to cache: {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump((nfas, start_counts), f)

    trans_per_k = {}
    for K in K_list:
        print(f"\n══ K={K} ══")
        sc_path = CLUSTER_DIR / f"K{K}/start_state_s1.txt"
        fallback = CLUSTER_DIR / f"K{K}/start_state.txt"
        start = start_counts[K].most_common(1)[0][0] if start_counts[K] else (
            int(sc_path.read_text()) if sc_path.exists() else
            int(fallback.read_text()) if fallback.exists() else 0)
        sc_path.write_text(str(start))
        print(f"  start_cluster={start}")
        trans_per_k[K] = resolve_and_save(K, nfas[K], len(vocab), start)

    try:
        save_plots(K_list, nfas, trans_per_k)
    except Exception as e:
        print(f"Plot error (non-fatal): {e}")

    print(f"\nDone → {DFA_S1_DIR}")


if __name__ == "__main__":
    main()
