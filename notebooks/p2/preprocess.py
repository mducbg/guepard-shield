"""Phase 2 preprocessing: .sc recordings → .npy windows.

Run once from the project root:
    python notebooks/p2/preprocess.py

Outputs to data/processed/p2/:
  train_X.npy, train_rec_ids.npy
  val_X.npy,   val_rec_ids.npy
  test_X.npy,  test_y.npy, test_rec_ids.npy
  vocab.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gp.data_loader.lidds_2021_loader import LiddS2021Loader, stream_and_window

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_NON_SCENARIO = {"_lidds_seq_lengths.json", "README.md"}
_TEST_SUBDIRS = ["test/normal", "test/normal_and_attack"]


def _count_recordings(data_dir: Path, subdirs: list[str]) -> int:
    total = 0
    for sc_dir in data_dir.iterdir():
        if not sc_dir.is_dir() or sc_dir.name in _NON_SCENARIO:
            continue
        for sub in subdirs:
            p = sc_dir / sub
            if p.exists():
                total += sum(1 for x in p.iterdir() if x.is_dir())
    return total


# ── constants ────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "extracted" / "LID-DS-2021"
OUT_DIR  = PROJECT_ROOT / "data" / "processed" / "p2"

WINDOW_SIZE = 64
STRIDE      = 32
MAX_WIN_TRAIN = None  # no cap
MAX_WIN_EVAL  = None  # no cap for val/test


def build_vocab(loader: LiddS2021Loader, min_freq: int = 2) -> dict[str, int]:
    total = _count_recordings(DATA_DIR, ["training"])
    counts: Counter[str] = Counter()
    for rec in tqdm(loader.stream_split("train"), desc="building vocab", unit="rec", total=total):
        for sc in rec.syscalls:
            counts[sc.syscall] += 1
        rec.syscalls.clear()

    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    if "<unknown>" in counts:
        vocab["<unknown>"] = 2
    for name, freq in sorted(counts.items()):
        if name == "<unknown>":
            continue
        if freq >= min_freq:
            vocab[name] = len(vocab)

    print(f"  vocab size: {len(vocab)}")
    return vocab


def window_train_val(
    loader: LiddS2021Loader,
    split: str,
    vocab: dict[str, int],
    max_win: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"Windowing {split}…")
    _, X, _, rec_ids = stream_and_window(
        loader.stream_split(split),
        vocab=vocab,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        max_windows_per_recording=max_win,
    )
    return X, rec_ids


def window_test(
    loader: LiddS2021Loader,
    vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window test recordings with per-window attack labels.

    Label rule: ATTACK if last syscall timestamp >= earliest exploit event (ns).
    """
    unk_id = vocab.get("<UNK>", 1)
    W, S = WINDOW_SIZE, STRIDE

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []

    total = _count_recordings(DATA_DIR, _TEST_SUBDIRS)
    for rec_id, rec in enumerate(tqdm(loader.stream_split("test"), desc="windowing test", unit="rec", total=total)):
        syscalls = rec.syscalls
        n = len(syscalls)
        if n == 0:
            rec.syscalls.clear()
            continue

        # Earliest exploit in nanoseconds
        exploit_ns: float | None = None
        if rec.is_exploit and rec.exploit_times:
            exploit_ns = min(rec.exploit_times) * 1e9

        tokens = np.array(
            [vocab.get(sc.syscall, unk_id) for sc in syscalls], dtype=np.int32
        )
        starts = list(range(0, max(1, n - W + 1), S))
        n_win = len(starts)

        rec_X = np.zeros((n_win, W), dtype=np.int32)
        rec_y = np.zeros(n_win, dtype=np.int32)

        for i, start in enumerate(starts):
            chunk_syscalls = syscalls[start : start + W]
            chunk_tokens   = tokens[start : start + W]
            rec_X[i, : len(chunk_tokens)] = chunk_tokens

            last_ts = chunk_syscalls[-1].timestamp
            if exploit_ns is not None and last_ts >= exploit_ns:
                rec_y[i] = 1

        rec.syscalls.clear()
        X_parts.append(rec_X)
        y_parts.append(rec_y)
        id_parts.append(np.full(n_win, rec_id, dtype=np.int32))

    if not X_parts:
        empty = np.empty((0, W), dtype=np.int32)
        return empty, np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(id_parts, axis=0),
    )


def save(out_dir: Path, **arrays: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        path = out_dir / f"{name}.npy"
        np.save(path, arr)
        print(f"  saved {path.name}: shape={arr.shape} dtype={arr.dtype}")


def main() -> None:
    loader = LiddS2021Loader(DATA_DIR)

    vocab = build_vocab(loader)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = OUT_DIR / "vocab.json"
    LiddS2021Loader.save_vocab(vocab, vocab_path)
    print(f"  vocab saved → {vocab_path}")

    train_X, train_rec_ids = window_train_val(loader, "train", vocab, MAX_WIN_TRAIN)
    print(f"  train windows: {len(train_X):,}")
    save(OUT_DIR, train_X=train_X, train_rec_ids=train_rec_ids)

    val_X, val_rec_ids = window_train_val(loader, "val", vocab, MAX_WIN_EVAL)
    print(f"  val windows: {len(val_X):,}")
    save(OUT_DIR, val_X=val_X, val_rec_ids=val_rec_ids)

    test_X, test_y, test_rec_ids = window_test(loader, vocab)
    print(f"  test windows: {len(test_X):,}  attack: {test_y.sum():,}")
    save(OUT_DIR, test_X=test_X, test_y=test_y, test_rec_ids=test_rec_ids)

    print("\nDone. Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
