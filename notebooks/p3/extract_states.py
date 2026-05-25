"""Phase 3 — Step 1: Extract per-syscall hidden states from training recordings.

Run from project root:
    uv run notebooks/p3/extract_states.py \\
        --ckpt results/p2/checkpoints/best.ckpt

Two-pass algorithm:
    Pass 1 — stream train recordings, count M = total stride-4 windows.
    Pass 2 — re-stream, encode windows with model.encode(), write to memmap.

Outputs (results/p3/hidden_states/):
    train_H.dat    [M × 128]  float16 memmap — last-token hidden states
    train_meta.dat [M × 3]    int32 memmap   — (rec_id, pos_in_rec, token_id)
    info.json                               — shape / config metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "guepard-shield-model"))

from gp.data_loader.lidds_2021_loader import LiddS2021Loader, UNK_TOKEN
from gp.model import SyscallTransformer

DATA_DIR  = PROJECT_ROOT / "data" / "extracted" / "LID-DS-2021"
VOCAB_PATH = PROJECT_ROOT / "data" / "processed" / "p2" / "vocab.json"
OUT_DIR   = PROJECT_ROOT / "results" / "p3" / "hidden_states"

BATCH_SIZE = 512
STRIDE = 4


def count_pass(loader: LiddS2021Loader, W: int, n_recs: int) -> list[int]:
    """Pass 1: count stride-STRIDE windows per recording without GPU."""
    counts: list[int] = []
    for rec in tqdm(loader.stream_split("train"), desc="pass-1 counting", unit="rec", total=n_recs):
        n_windows = max(0, (len(rec.syscalls) - W) // STRIDE + 1)
        counts.append(n_windows)
    return counts


def encode_pass(
    loader: LiddS2021Loader,
    vocab: dict[str, int],
    W: int,
    window_counts: list[int],
    H_mmap: np.memmap,
    meta_mmap: np.memmap,
    model: SyscallTransformer,
    device: torch.device,
) -> None:
    """Pass 2: encode stride-STRIDE windows and write hidden states to memmaps."""
    unk = vocab[UNK_TOKEN]
    offset = 0
    M = len(H_mmap)

    with tqdm(total=M, desc="pass-2 encoding", unit="win", unit_scale=True) as pbar:
        for rec_idx, rec in enumerate(loader.stream_split("train")):
            n_windows = window_counts[rec_idx]
            if n_windows == 0:
                rec.syscalls.clear()
                continue

            tokens = np.array(
                [vocab.get(sc.syscall, unk) for sc in rec.syscalls], dtype=np.int32
            )
            rec.syscalls.clear()

            # stride-STRIDE windows: take every STRIDE-th window from the dense view
            windows_view = np.lib.stride_tricks.sliding_window_view(tokens, W)[::STRIDE]
            assert len(windows_view) == n_windows

            # Encode in batches
            for b_start in range(0, n_windows, BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, n_windows)
                batch = torch.from_numpy(
                    windows_view[b_start:b_end].astype(np.int64)
                ).to(device)

                with torch.inference_mode():
                    h = model.encode(batch)  # [B, D_MODEL]

                B = b_end - b_start
                H_mmap[offset : offset + B] = h.cpu().numpy().astype(np.float16)

                # rec_positions: actual start index in recording (0, STRIDE, 2*STRIDE, ...)
                rec_positions = np.arange(b_start, b_end, dtype=np.int32) * STRIDE
                meta_mmap[offset : offset + B, 0] = rec_idx
                meta_mmap[offset : offset + B, 1] = rec_positions
                # last token of each window = syscall at rec_position + W - 1
                meta_mmap[offset : offset + B, 2] = tokens[rec_positions + W - 1]

                offset += B
                pbar.update(B)

    assert offset == M, f"offset mismatch: {offset} vs {M}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to best.ckpt")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if output already exists")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info_path = OUT_DIR / "info.json"
    H_path    = OUT_DIR / "train_H.dat"
    meta_path = OUT_DIR / "train_meta.dat"

    if not args.force and info_path.exists():
        print(f"Hidden states already extracted at {OUT_DIR}. Use --force to redo.")
        return

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {args.ckpt} on {device}...")
    model: SyscallTransformer = SyscallTransformer.load_from_checkpoint(
        args.ckpt, map_location=device
    )
    model.eval()
    W = model.hparams.window_size
    D_MODEL = model.hparams.d_model

    vocab = LiddS2021Loader.load_vocab(VOCAB_PATH)
    vocab_size = len(vocab)
    loader = LiddS2021Loader(DATA_DIR)

    # Pre-count recordings for pass-1 progress bar
    print("Scanning training recordings...")
    n_recs = sum(1 for _ in DATA_DIR.rglob("*/training/*/*.sc"))
    print(f"  Found {n_recs:,} training recordings")

    # Pass 1 — count
    print("Pass 1: counting stride-STRIDE windows per recording...")
    window_counts = count_pass(loader, W, n_recs)
    M = sum(window_counts)
    print(f"  Total windows M = {M:,}  ({M * D_MODEL * 2 / 1e9:.1f} GB for H.dat)")

    # Allocate memmaps
    print("Allocating memmap files...")
    H_mmap    = np.memmap(H_path,    dtype="float16", mode="w+", shape=(M, D_MODEL))
    meta_mmap = np.memmap(meta_path, dtype="int32",   mode="w+", shape=(M, 3))

    # Pass 2 — encode
    print("Pass 2: encoding hidden states...")
    encode_pass(loader, vocab, W, window_counts, H_mmap, meta_mmap, model, device)

    # Flush
    del H_mmap, meta_mmap

    # Save info
    info = {
        "M": M,
        "D_model": D_MODEL,
        "vocab_size": vocab_size,
        "window_size": W,
        "stride": STRIDE,
        "H_dtype": "float16",
        "meta_dtype": "int32",
        "meta_cols": ["rec_id", "pos_in_rec", "token_id"],
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone. Outputs written to {OUT_DIR}")
    print(f"  train_H.dat    : {H_path.stat().st_size / 1e9:.1f} GB")
    print(f"  train_meta.dat : {meta_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
