# Phase 4 Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three benchmark experiments (E1: eBPF latency histogram, E2: Transformer CPU timing, E3: nginx overhead) to quantify eBPF DFA speed vs. Transformer inference.

**Architecture:** E1 adds a `PerCpuArray<u64>` histogram to the existing eBPF program and dumps it in userspace after Ctrl-C. E2 is a standalone Python script that times `model.encode()` per-window on CPU. E3 is a shell script that drives `wrk` + `perf stat` on nginx in two conditions (baseline, DFA attached).

**Tech Stack:** Rust/aya-ebpf (E1 kernel side), Rust/aya (E1 userspace), Python/PyTorch/Lightning (E2), bash/wrk/perf/nginx (E3).

---

### Task 1: Add LATENCY_HIST map to eBPF program

**Files:**
- Modify: `guepard-shield-ebpf/src/main.rs`

The PerCpuArray avoids atomic ops — each CPU core maintains its own counter; userspace sums them. 32 buckets of 100 ns each covers 0–3200 ns (bucket 31 is overflow ≥ 3100 ns).

**Step 1: Add the import and map declaration**

In `guepard-shield-ebpf/src/main.rs`, extend the existing `use aya_ebpf::{...}` block and add the map.

Change the imports from:
```rust
use aya_ebpf::{
    helpers::bpf_get_current_pid_tgid,
    macros::{map, tracepoint},
    maps::{Array, HashMap},
    programs::TracePointContext,
};
```
to:
```rust
use aya_ebpf::{
    helpers::{bpf_get_current_pid_tgid, bpf_ktime_get_ns},
    macros::{map, tracepoint},
    maps::{Array, HashMap, PerCpuArray},
    programs::TracePointContext,
};
```

Then add the new map after the existing three maps (after `static PROCESS_STATE`):
```rust
// 32 buckets × 100 ns; bucket 31 = overflow (≥ 3100 ns). PerCpu avoids atomic ops.
#[map]
static LATENCY_HIST: PerCpuArray<u64> = PerCpuArray::with_max_entries(32, 0);
```

**Step 2: Wrap try_guepard_shield with timing**

Replace the `guepard_shield` function:
```rust
#[tracepoint]
pub fn guepard_shield(ctx: TracePointContext) -> u32 {
    let t0 = bpf_ktime_get_ns();
    let ret = match try_guepard_shield(ctx) {
        Ok(ret) => ret,
        Err(ret) => ret,
    };
    let elapsed = bpf_ktime_get_ns().saturating_sub(t0);
    let bucket = ((elapsed / 100).min(31)) as u32;
    if let Some(count) = LATENCY_HIST.get_ptr_mut(bucket) {
        unsafe { *count += 1 };
    }
    ret
}
```

**Step 3: Verify it compiles**

```bash
cargo check
```
Expected: `Finished` with zero errors. Warnings about unused imports are fine.

**Step 4: Commit**

```bash
git add guepard-shield-ebpf/src/main.rs
git commit -m "feat(ebpf): add LATENCY_HIST PerCpuArray for per-syscall latency measurement"
```

---

### Task 2: Dump histogram and print percentiles in userspace

**Files:**
- Modify: `guepard-shield/src/main.rs`

**Step 1: Add PerCpuArray import**

In `guepard-shield/src/main.rs`, change the existing aya import line from:
```rust
use aya::maps::Array;
```
to:
```rust
use aya::maps::{Array, PerCpuArray};
```

**Step 2: Add histogram dump after Ctrl-C**

Replace the current shutdown block:
```rust
    let ctrl_c = signal::ctrl_c();
    ctrl_c.await?;
    println!("Exiting...");

    Ok(())
```

with:
```rust
    let ctrl_c = signal::ctrl_c();
    ctrl_c.await?;
    println!("Exiting...");

    // Dump latency histogram collected during run.
    if let Some(map_ref) = ebpf.map("LATENCY_HIST") {
        if let Ok(hist) = PerCpuArray::<_, u64>::try_from(map_ref) {
            let mut counts = [0u64; 32];
            for bucket in 0..32u32 {
                if let Ok(per_cpu) = hist.get(bucket, 0) {
                    counts[bucket as usize] = per_cpu.iter().sum();
                }
            }
            print_latency_histogram(&counts);
        }
    }

    Ok(())
```

**Step 3: Add the helper function**

Add this function after `main()` (before the end of the file):
```rust
fn print_latency_histogram(counts: &[u64; 32]) {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        println!("[latency] no samples collected");
        return;
    }
    println!("\n[latency histogram] {} samples", total);
    println!("{:>12}  {:>10}  {:>10}", "bucket_ns", "count", "cumul%");
    let mut cumul = 0u64;
    for (i, &c) in counts.iter().enumerate() {
        cumul += c;
        let bucket_label = if i < 31 {
            format!("{:>6}", i * 100)
        } else {
            "≥3100".to_string()
        };
        println!("{:>12}  {:>10}  {:>9.2}%", bucket_label, c, cumul as f64 / total as f64 * 100.0);
    }
    // Compute p50, p99, p999
    for (label, threshold) in [("p50", 0.50f64), ("p99", 0.99), ("p999", 0.999)] {
        let target = (total as f64 * threshold).ceil() as u64;
        let mut acc = 0u64;
        for (i, &c) in counts.iter().enumerate() {
            acc += c;
            if acc >= target {
                let lo = i * 100;
                let hi = if i < 31 { (i + 1) * 100 } else { usize::MAX };
                println!("{}: {}–{} ns", label, lo, if hi == usize::MAX { "∞".to_string() } else { hi.to_string() });
                break;
            }
        }
    }
}
```

**Step 4: Verify it compiles**

```bash
cargo check
```
Expected: `Finished` with zero errors.

**Step 5: Commit**

```bash
git add guepard-shield/src/main.rs
git commit -m "feat(userspace): dump LATENCY_HIST histogram and print p50/p99/p999 on exit"
```

---

### Task 3: Create E2 Python Transformer benchmark script

**Files:**
- Create: `notebooks/p4/benchmark_transformer.py`

The script loads the trained checkpoint, takes 10,100 windows from `data/processed/p2/val_X.npy`, and times each `model.encode()` call individually (no batching, CPU only).

**Step 1: Create notebooks/p4/ directory**

```bash
mkdir -p notebooks/p4
```

**Step 2: Write the benchmark script**

Create `notebooks/p4/benchmark_transformer.py`:
```python
"""E2 — Transformer CPU single-sample inference latency benchmark.

Usage (from project root):
    uv run notebooks/p4/benchmark_transformer.py

Output: p50 / p99 / p999 latency per window (CPU, no batch, no GPU).
"""
from __future__ import annotations

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
N_WARMUP = 100
N_BENCH = 10_000


def percentile(data: list[int], p: float) -> int:
    idx = max(0, int(len(data) * p / 100) - 1)
    return sorted(data)[idx]


def main() -> None:
    print(f"Loading model from {CHECKPOINT} ...")
    model = SyscallTransformer.load_from_checkpoint(str(CHECKPOINT))
    model.eval()
    model = model.cpu()

    print(f"Loading val windows from {VAL_X} ...")
    val_x = np.load(str(VAL_X), mmap_mode="r")  # (N, 64) int32
    n_total = N_WARMUP + N_BENCH
    assert val_x.shape[0] >= n_total, f"need {n_total} windows, only {val_x.shape[0]} available"

    windows = torch.from_numpy(val_x[:n_total].astype("int64"))  # (n_total, 64)

    print(f"Warmup ({N_WARMUP} windows) ...")
    with torch.no_grad():
        for i in range(N_WARMUP):
            model.encode(windows[i].unsqueeze(0))  # [1, 64]

    print(f"Benchmarking {N_BENCH} windows (single-sample, CPU, no grad) ...")
    latencies_ns: list[int] = []
    with torch.no_grad():
        for i in range(N_WARMUP, N_WARMUP + N_BENCH):
            t0 = time.perf_counter_ns()
            model.encode(windows[i].unsqueeze(0))
            latencies_ns.append(time.perf_counter_ns() - t0)

    p50  = percentile(latencies_ns, 50)
    p99  = percentile(latencies_ns, 99)
    p999 = percentile(latencies_ns, 99.9)

    print(f"\n[Transformer CPU latency — {N_BENCH} single-sample windows]")
    print(f"  p50  : {p50:>10,} ns  ({p50 / 1e6:.2f} ms)")
    print(f"  p99  : {p99:>10,} ns  ({p99 / 1e6:.2f} ms)")
    print(f"  p999 : {p999:>10,} ns  ({p999 / 1e6:.2f} ms)")
    print(f"  mean : {int(sum(latencies_ns)/len(latencies_ns)):>10,} ns")


if __name__ == "__main__":
    main()
```

**Step 3: Run the script to verify it works**

```bash
uv run notebooks/p4/benchmark_transformer.py
```
Expected output (values approximate):
```
Loading model from results/p2/checkpoints/best.ckpt ...
Loading val windows from data/processed/p2/val_X.npy ...
Warmup (100 windows) ...
Benchmarking 10000 windows (single-sample, CPU, no grad) ...

[Transformer CPU latency — 10000 single-sample windows]
  p50  :    X,XXX,XXX ns  (X.XX ms)
  p99  :    X,XXX,XXX ns  (X.XX ms)
  p999 :    X,XXX,XXX ns  (X.XX ms)
  mean :    X,XXX,XXX ns
```
If import fails, verify `guepard-shield-model/` is on `sys.path` and `pyproject.toml` includes the model package.

**Step 4: Commit**

```bash
git add notebooks/p4/benchmark_transformer.py
git commit -m "feat(bench): add E2 Transformer CPU single-sample latency benchmark"
```

---

### Task 4: Create E3 nginx config and benchmark shell script

**Files:**
- Create: `notebooks/p4/nginx_bench.conf`
- Create: `notebooks/p4/run_e3.sh`

**Step 1: Create nginx config**

Create `notebooks/p4/nginx_bench.conf`:
```nginx
# Minimal nginx config for E3 benchmark — serves a single static 4KB file.
# Run as: nginx -c /absolute/path/to/nginx_bench.conf
worker_processes 1;
error_log /tmp/nginx_bench_error.log;
pid /tmp/nginx_bench.pid;

events {
    worker_connections 1024;
}

http {
    access_log off;
    sendfile on;
    keepalive_timeout 65;

    server {
        listen 8080;
        root /tmp/nginx_bench_root;
        location / {
            try_files /bench.html =404;
        }
    }
}
```

**Step 2: Create benchmark script**

Create `notebooks/p4/run_e3.sh`:
```bash
#!/usr/bin/env bash
# E3 — nginx live overhead benchmark
# Usage: sudo bash notebooks/p4/run_e3.sh <path-to-dfa-config>
# Example: sudo bash notebooks/p4/run_e3.sh results/p3/dfa_config.json
#
# Requires: nginx, wrk, perf (linux-tools), sudo
# Duration: ~3 minutes total (2 × 60s measurement + setup)

set -euo pipefail

DFA_CONFIG="${1:-results/p3/dfa_config.json}"
WRK_DURATION=60
WRK_THREADS=4
WRK_CONNECTIONS=100
WRK_URL="http://localhost:8080/"
RESULTS_DIR="results/p4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Sanity checks ──────────────────────────────────────────────────────────────
for cmd in nginx wrk perf; do
    command -v "$cmd" &>/dev/null || { echo "ERROR: $cmd not found"; exit 1; }
done
[[ -f "$DFA_CONFIG" ]] || { echo "ERROR: DFA config not found: $DFA_CONFIG"; exit 1; }
[[ -f "$PROJECT_ROOT/target/release/guepard-shield" ]] || {
    echo "ERROR: guepard-shield binary not found. Run: cargo build --release"
    exit 1
}

mkdir -p "$RESULTS_DIR"

# ── Setup nginx ────────────────────────────────────────────────────────────────
echo "[setup] Creating nginx static file..."
mkdir -p /tmp/nginx_bench_root
dd if=/dev/urandom bs=4096 count=1 2>/dev/null | base64 > /tmp/nginx_bench_root/bench.html

echo "[setup] Starting nginx..."
nginx -c "$SCRIPT_DIR/nginx_bench.conf" 2>/dev/null || true
sleep 1
NGINX_PID=$(cat /tmp/nginx_bench.pid 2>/dev/null) || { echo "ERROR: nginx failed to start"; exit 1; }
echo "[setup] nginx pid=$NGINX_PID"

# Warm up nginx
curl -s "$WRK_URL" > /dev/null
sleep 1

# ── Condition 1: Baseline ─────────────────────────────────────────────────────
echo ""
echo "=== Condition 1: Baseline (no eBPF) ==="
perf stat -e cycles,instructions,raw_syscalls:sys_enter \
    -p "$NGINX_PID" -- sleep "$WRK_DURATION" \
    2> "$RESULTS_DIR/baseline_perf.txt" &
PERF_PID=$!

wrk -t"$WRK_THREADS" -c"$WRK_CONNECTIONS" -d"${WRK_DURATION}s" "$WRK_URL" \
    > "$RESULTS_DIR/baseline_wrk.txt"

wait "$PERF_PID" || true
echo "[baseline] wrk done. Results in $RESULTS_DIR/baseline_wrk.txt"
cat "$RESULTS_DIR/baseline_wrk.txt"
echo "--- perf ---"
cat "$RESULTS_DIR/baseline_perf.txt"

sleep 2

# ── Condition 2: DFA attached ─────────────────────────────────────────────────
echo ""
echo "=== Condition 2: DFA attached ==="
"$PROJECT_ROOT/target/release/guepard-shield" --dfa "$DFA_CONFIG" &
DFA_PID=$!
sleep 1  # let eBPF attach and maps populate

perf stat -e cycles,instructions,raw_syscalls:sys_enter \
    -p "$NGINX_PID" -- sleep "$WRK_DURATION" \
    2> "$RESULTS_DIR/dfa_perf.txt" &
PERF_PID=$!

wrk -t"$WRK_THREADS" -c"$WRK_CONNECTIONS" -d"${WRK_DURATION}s" "$WRK_URL" \
    > "$RESULTS_DIR/dfa_wrk.txt"

wait "$PERF_PID" || true
kill "$DFA_PID" 2>/dev/null || true
wait "$DFA_PID" 2>/dev/null || true
echo "[dfa] wrk done. Results in $RESULTS_DIR/dfa_wrk.txt"
cat "$RESULTS_DIR/dfa_wrk.txt"
echo "--- perf ---"
cat "$RESULTS_DIR/dfa_perf.txt"

# ── Cleanup ────────────────────────────────────────────────────────────────────
echo ""
echo "[cleanup] Stopping nginx..."
nginx -s stop -c "$SCRIPT_DIR/nginx_bench.conf" 2>/dev/null || kill "$NGINX_PID" 2>/dev/null || true

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== E3 Summary ==="
echo "Results saved to $RESULTS_DIR/"
echo ""
echo "Baseline throughput:"
grep "Requests/sec" "$RESULTS_DIR/baseline_wrk.txt" || grep -E "req/s|Req/Sec" "$RESULTS_DIR/baseline_wrk.txt" || true
echo "DFA throughput:"
grep "Requests/sec" "$RESULTS_DIR/dfa_wrk.txt" || grep -E "req/s|Req/Sec" "$RESULTS_DIR/dfa_wrk.txt" || true
echo ""
echo "Baseline cycles:"
grep "cycles" "$RESULTS_DIR/baseline_perf.txt" | head -1 || true
echo "DFA cycles:"
grep "cycles" "$RESULTS_DIR/dfa_perf.txt" | head -1 || true
echo ""
echo "Syscalls/sec (from baseline):"
grep "sys_enter" "$RESULTS_DIR/baseline_perf.txt" || true
```

**Step 3: Make the script executable**

```bash
chmod +x notebooks/p4/run_e3.sh
```

**Step 4: Smoke-test nginx config (no wrk yet)**

```bash
# Create the static file manually
mkdir -p /tmp/nginx_bench_root
echo "bench" > /tmp/nginx_bench_root/bench.html

# Start nginx with absolute path
nginx -c "$(pwd)/notebooks/p4/nginx_bench.conf"

# Verify it serves
curl -s http://localhost:8080/ | head -c 20
echo ""

# Stop it
nginx -s stop -c "$(pwd)/notebooks/p4/nginx_bench.conf"
```
Expected: curl returns the content of bench.html without error.

**Step 5: Commit**

```bash
git add notebooks/p4/nginx_bench.conf notebooks/p4/run_e3.sh
git commit -m "feat(bench): add E3 nginx overhead benchmark script and config"
```

---

### Task 5: Create results/p4/ directory and build release binary

**Files:**
- Create: `results/p4/.gitkeep`

**Step 1: Create results directory**

```bash
mkdir -p results/p4
touch results/p4/.gitkeep
```

**Step 2: Build release binary**

```bash
cargo build --release
```
Expected: `Finished release [optimized]` — the binary lands at `target/release/guepard-shield`.

**Step 3: Commit**

```bash
git add results/p4/.gitkeep
git commit -m "chore: add results/p4/ directory for benchmark outputs"
```

---

### Task 6: Verify full E1 end-to-end (manual smoke test)

No code changes. This task verifies that E1 works before running the real experiment.

**Step 1: Start nginx to generate syscall traffic**

```bash
mkdir -p /tmp/nginx_bench_root && echo "bench" > /tmp/nginx_bench_root/bench.html
nginx -c "$(pwd)/notebooks/p4/nginx_bench.conf"
```

**Step 2: Run guepard-shield for 15 seconds**

```bash
sudo target/release/guepard-shield --dfa results/p3/dfa_config.json &
DFA_PID=$!

# Generate some load
wrk -t2 -c10 -d10s http://localhost:8080/ > /dev/null 2>&1 || true
sleep 5

# Send Ctrl-C signal
kill -SIGINT "$DFA_PID"
wait "$DFA_PID" 2>/dev/null || true
```

Expected output ends with:
```
[latency histogram] XXXXXX samples
     bucket_ns       count    cumul%
           0        XXXXX      XX.XX%
         100        XXXXX      XX.XX%
...
p50: 0–100 ns          ← typical DFA lookup is sub-100ns
p99: 200–300 ns
p999: 500–1000 ns
```
If the histogram shows `0 samples`, verify `LATENCY_HIST` map name matches between eBPF and userspace.

**Step 3: Stop nginx**

```bash
nginx -s stop -c "$(pwd)/notebooks/p4/nginx_bench.conf"
```

---

### Task 7: Run all three experiments and save raw output

No code changes. Run experiments in order and capture output.

**Step 1: Run E2 (Transformer timing)**

```bash
uv run notebooks/p4/benchmark_transformer.py 2>&1 | tee results/p4/e2_transformer_latency.txt
```
Expected: prints p50/p99/p999 latency. Takes ~5–15 minutes depending on CPU speed.

**Step 2: Run E3 (nginx overhead, requires sudo)**

Build release binary first if not done:
```bash
cargo build --release
```

Then run E3 (takes ~2.5 minutes):
```bash
sudo bash notebooks/p4/run_e3.sh results/p3/dfa_config.json 2>&1 | tee results/p4/e3_summary.txt
```
Expected: two sections of wrk output showing req/s, then a summary comparing cycles and throughput.

**Step 3: Run E1 (eBPF histogram, 60s collection)**

```bash
mkdir -p /tmp/nginx_bench_root && echo "bench" > /tmp/nginx_bench_root/bench.html
nginx -c "$(pwd)/notebooks/p4/nginx_bench.conf"

sudo target/release/guepard-shield --dfa results/p3/dfa_config.json 2>&1 &
DFA_PID=$!
wrk -t4 -c100 -d60s http://localhost:8080/ > /dev/null 2>&1
kill -SIGINT "$DFA_PID"
wait "$DFA_PID" 2>/dev/null | tee results/p4/e1_ebpf_histogram.txt

nginx -s stop -c "$(pwd)/notebooks/p4/nginx_bench.conf"
```

**Step 4: Commit results**

```bash
git add results/p4/
git commit -m "results: add Phase 4 benchmark raw output (E1/E2/E3)"
```

---

## Quick Reference: File Map

| File | Purpose |
|------|---------|
| `guepard-shield-ebpf/src/main.rs` | +`LATENCY_HIST` map + timing in `guepard_shield()` |
| `guepard-shield/src/main.rs` | +histogram read + `print_latency_histogram()` |
| `notebooks/p4/benchmark_transformer.py` | E2: CPU single-sample Transformer timing |
| `notebooks/p4/nginx_bench.conf` | nginx config for E3 |
| `notebooks/p4/run_e3.sh` | E3 automation: wrk + perf stat, 2 conditions |
| `results/p4/` | Output directory for all experiment results |

## Dependency Notes

- **wrk**: `sudo pacman -S wrk` (CachyOS) or compile from source
- **perf**: `sudo pacman -S linux-tools` (CachyOS) or `perf_${kernel_version}`
- **nginx**: `sudo pacman -S nginx`
- All Python deps already in `pyproject.toml` via `uv`
- eBPF build needs `nightly` Rust + `bpf-linker` (already configured in project)
