"""E2 - Transformer CPU single-sample inference latency benchmark.

Usage from the project root:
    uv run notebooks/p4/benchmark_transformer.py

Output: p50 / p99 / p999 latency per window (CPU, no batch, no GPU).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "guepard-shield-model"))

from gp.model import SyscallTransformer  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "results" / "p2" / "checkpoints" / "best.ckpt"
VAL_X = PROJECT_ROOT / "data" / "processed" / "p2" / "val_X.npy"
RESULTS_DIR = PROJECT_ROOT / "results" / "p4"
TEXT_OUT = RESULTS_DIR / "e2_transformer_latency.txt"
JSON_OUT = RESULTS_DIR / "e2_transformer_latency.json"
N_WARMUP = 100
N_BENCH = 10_000


def percentile(data: list[int], p: float) -> int:
    idx = max(0, int(len(data) * p / 100) - 1)
    return sorted(data)[idx]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {CHECKPOINT} ...")
    model = SyscallTransformer.load_from_checkpoint(str(CHECKPOINT))
    model.eval()
    model = model.cpu()

    print(f"Loading val windows from {VAL_X} ...")
    val_x = np.load(str(VAL_X), mmap_mode="r")
    n_total = N_WARMUP + N_BENCH
    assert val_x.shape[0] >= n_total, (
        f"need {n_total} windows, only {val_x.shape[0]} available"
    )

    windows = torch.from_numpy(val_x[:n_total].astype("int64"))

    print(f"Warmup ({N_WARMUP} windows) ...")
    with torch.no_grad():
        for i in range(N_WARMUP):
            model.encode(windows[i].unsqueeze(0))

    print(f"Benchmarking {N_BENCH} windows (single-sample, CPU, no grad) ...")
    latencies_ns: list[int] = []
    with torch.no_grad():
        for i in range(N_WARMUP, N_WARMUP + N_BENCH):
            t0 = time.perf_counter_ns()
            model.encode(windows[i].unsqueeze(0))
            latencies_ns.append(time.perf_counter_ns() - t0)

    p50 = percentile(latencies_ns, 50)
    p99 = percentile(latencies_ns, 99)
    p999 = percentile(latencies_ns, 99.9)
    mean = int(sum(latencies_ns) / len(latencies_ns))

    report = "\n".join(
        [
            f"[Transformer CPU latency - {N_BENCH} single-sample windows]",
            f"  p50  : {p50:>10,} ns  ({p50 / 1e6:.2f} ms)",
            f"  p99  : {p99:>10,} ns  ({p99 / 1e6:.2f} ms)",
            f"  p999 : {p999:>10,} ns  ({p999 / 1e6:.2f} ms)",
            f"  mean : {mean:>10,} ns",
            f"Saved text: {TEXT_OUT}",
            f"Saved json: {JSON_OUT}",
            "",
        ]
    )
    TEXT_OUT.write_text(report)
    JSON_OUT.write_text(
        json.dumps(
            {
                "checkpoint": str(CHECKPOINT),
                "val_x": str(VAL_X),
                "n_warmup": N_WARMUP,
                "n_bench": N_BENCH,
                "latency_ns": {
                    "p50": p50,
                    "p99": p99,
                    "p999": p999,
                    "mean": mean,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print()
    print(report, end="")


if __name__ == "__main__":
    main()
