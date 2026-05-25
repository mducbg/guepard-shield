"""Phase 2 evaluation script.

Run from project root:
    python notebooks/p2/eval.py --ckpt results/p2/checkpoints/best.ckpt

The script runs inference directly in RAM and reports window-level and
recording-level metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from lightning import Trainer
from sklearn.metrics import f1_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "guepard-shield-model"))

from gp.datamodule import SyscallDataModule
from gp.metrics import evaluate, select_threshold
from gp.model import SyscallTransformer

DATA_DIR   = PROJECT_ROOT / "data" / "processed" / "p2"


def predict_scores(
    trainer: Trainer,
    model: SyscallTransformer,
    dataloader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and concatenate all prediction batches in memory."""
    predictions = trainer.predict(model, dataloaders=dataloader)
    if not predictions:
        raise RuntimeError("No predictions returned from Trainer.predict()")

    last, maxnll, labels = zip(*predictions)
    batch_bar = tqdm(
        zip(last, maxnll, labels),
        total=len(predictions),
        desc="stacking predictions",
        unit="batch",
    )
    last_parts: list[np.ndarray] = []
    max_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for batch_last, batch_max, batch_labels in batch_bar:
        last_parts.append(batch_last.numpy().astype(np.float32, copy=False))
        max_parts.append(batch_max.numpy().astype(np.float32, copy=False))
        label_parts.append(batch_labels.numpy().astype(np.int32, copy=False))

    return (
        np.concatenate(last_parts),
        np.concatenate(max_parts),
        np.concatenate(label_parts),
    )


def recording_level_eval_grouped(
    scores: np.ndarray,
    labels: np.ndarray,
    rec_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Max-pool contiguous window scores/labels to recording level."""
    if len(scores) != len(labels) or len(scores) != len(rec_ids):
        raise ValueError("scores, labels, and rec_ids must have the same length")
    if len(scores) == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    starts = np.r_[0, np.flatnonzero(np.diff(rec_ids)) + 1]
    return (
        np.maximum.reduceat(scores, starts).astype(np.float32, copy=False),
        np.maximum.reduceat(labels, starts).astype(np.int32, copy=False),
    )


def sweep_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (best_f1, best_tau) over 200 threshold candidates."""
    thresholds = np.percentile(scores, np.linspace(50, 99.9, 200))
    f1s = np.asarray(
        [
            f1_score(labels, scores >= tau, zero_division=0)
            for tau in tqdm(
                thresholds,
                total=len(thresholds),
                desc="sweeping thresholds",
                unit="τ",
            )
        ],
        dtype=np.float32,
    )
    best_idx = int(f1s.argmax())
    return float(f1s[best_idx]), float(thresholds[best_idx])


def print_metrics(m: dict) -> None:
    report = m.get("_report")
    for k, v in m.items():
        if k == "_report":
            continue
        print(f"  {k:8s}: {v:.4f}")
    if report:
        print(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers for evaluation.",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tau", type=float, default=None,
                        help="Fixed threshold — skips val scoring")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    model = SyscallTransformer.load_from_checkpoint(args.ckpt)
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        precision="16-mixed",
        callbacks=[],
    )

    def make_dm(stage: str) -> SyscallDataModule:
        dm = SyscallDataModule(
            DATA_DIR,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        dm.setup(stage)
        return dm

    # ── test scores: load cache or run inference ──────────────────────────────
    SCORE_DIR = PROJECT_ROOT / "results" / "p2" / "scores"
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    
    test_last_path = SCORE_DIR / "test_last.npy"
    test_max_path = SCORE_DIR / "test_max.npy"
    test_labels_path = SCORE_DIR / "test_labels.npy"

    if test_last_path.exists() and test_max_path.exists() and test_labels_path.exists():
        print(f"\n  loading cached test scores from {SCORE_DIR}...")
        test_last = np.load(test_last_path)
        test_max = np.load(test_max_path)
        test_labels = np.load(test_labels_path)
    else:
        dm_test = make_dm("test")
        test_last, test_max, test_labels = predict_scores(
            trainer,
            model,
            dm_test.test_dataloader(),
        )
        del dm_test
        
        print(f"  saving test scores to {SCORE_DIR}...")
        np.save(test_last_path, test_last)
        np.save(test_max_path, test_max)
        np.save(test_labels_path, test_labels)

    print(f"\n  test windows : {len(test_last):,}")
    print(f"  attack windows: {int(test_labels.sum()):,} ({100*test_labels.mean():.1f}%)")

    # ── threshold selection ───────────────────────────────────────────────────
    if args.tau is not None:
        tau_win = tau_rec = args.tau
    else:
        dm_val = make_dm("fit")
        val_last, _, _ = predict_scores(
            trainer,
            model,
            dm_val.val_dataloader(),
        )
        del dm_val

        tau_win = select_threshold(val_last, percentile=99.5)

        val_rec_ids = np.load(DATA_DIR / "val_rec_ids.npy", mmap_mode="r")
        val_rec_scores, _ = recording_level_eval_grouped(
            val_last, np.zeros(len(val_last), dtype=np.int32), val_rec_ids
        )
        tau_rec = select_threshold(val_rec_scores, percentile=99.5)

        print(f"\n  val stats: min={val_last.min():.4f}  mean={val_last.mean():.4f}  "
              f"p99={np.percentile(val_last, 99):.4f}  max={val_last.max():.4f}")
        print(f"  τ_win (99.5th pct, window)   : {tau_win:.4f}")
        print(f"  τ_rec (99.5th pct, recording): {tau_rec:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # WINDOW-LEVEL
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("WINDOW-LEVEL")
    print("═"*60)

    print("\n── last-token NLL @ val-percentile threshold ──")
    print_metrics(evaluate(test_last, test_labels, tau_win))

    print("\n── last-token NLL: oracle optimal F1 ──")
    _, best_tau = sweep_f1(test_last, test_labels)
    print(f"  τ : {best_tau:.4f}")
    print_metrics(evaluate(test_last, test_labels, best_tau))

    print("\n── max-window NLL: oracle optimal F1 ──")
    _, best_tau = sweep_f1(test_max, test_labels)
    print(f"  τ : {best_tau:.4f}")
    print_metrics(evaluate(test_max, test_labels, best_tau))

    # ════════════════════════════════════════════════════════════════════════
    # RECORDING-LEVEL
    # ════════════════════════════════════════════════════════════════════════
    rec_id_path = DATA_DIR / "test_rec_ids.npy"
    if not rec_id_path.exists():
        print(f"\n[skip] {rec_id_path} not found")
        return

    print("\n" + "═"*60)
    print("RECORDING-LEVEL  (max-score per recording)")
    print("═"*60)

    test_rec_ids = np.load(rec_id_path, mmap_mode="r")
    rec_last, rec_labels = recording_level_eval_grouped(test_last, test_labels, test_rec_ids)
    rec_max,  _          = recording_level_eval_grouped(test_max,  test_labels, test_rec_ids)

    print(f"\n  recordings: {len(rec_labels):,}  "
          f"attack: {int(rec_labels.sum()):,} ({100*rec_labels.mean():.1f}%)")

    print("\n── last-token NLL (max-pooled) @ val-percentile threshold ──")
    print_metrics(evaluate(rec_last, rec_labels, tau_rec))

    print("\n── last-token NLL (max-pooled): oracle optimal F1 ──")
    _, best_tau = sweep_f1(rec_last, rec_labels)
    print(f"  τ : {best_tau:.4f}")
    print_metrics(evaluate(rec_last, rec_labels, best_tau))

    print("\n── max-window NLL (max-pooled): oracle optimal F1 ──")
    _, best_tau = sweep_f1(rec_max, rec_labels)
    print(f"  τ : {best_tau:.4f}")
    print_metrics(evaluate(rec_max, rec_labels, best_tau))


if __name__ == "__main__":
    main()
