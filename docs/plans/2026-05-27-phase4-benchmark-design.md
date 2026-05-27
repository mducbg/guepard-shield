# Phase 4: eBPF DFA vs. Transformer — Benchmark Experiment Design

**Date:** 2026-05-27
**Status:** Approved

## Mục tiêu

Chứng minh tốc độ vượt trội của eBPF DFA so với Transformer inference cho syscall HIDS thông qua hai loại bằng chứng:

1. **Microbenchmark latency** — per-syscall/per-window latency tuyệt đối của từng phương pháp
2. **Live workload overhead** — % overhead thực tế trên nginx production workload

## Thiết kế tổng thể

```
E1: bpf_ktime_get_ns() histogram              E2: Python Transformer timing
    trong try_guepard_shield()                     10,000 windows, CPU single-sample
    ↓                                              ↓
    DFA p50/p99/p999 latency (ns)                 Transformer p50/p99/p999 latency (ms)
                    \                             /
                     ↓                          ↓
                   E3: nginx live overhead
                   wrk + perf stat, 3 conditions
                   Baseline / DFA attached / Transformer-theoretical
```

## E1 — Microbenchmark eBPF DFA Latency

**File:** `guepard-shield-ebpf/src/main.rs` + `guepard-shield/src/main.rs`

**Cơ chế:**

Thêm BPF map thứ 4 vào eBPF program:

```rust
#[map]
static LATENCY_HIST: Array<u64> = Array::with_max_entries(32, 0);
// bucket i = [i*100ns, (i+1)*100ns), bucket 31 = overflow (≥3100ns)
```

Trong `try_guepard_shield()`:

```rust
let t0 = bpf_ktime_get_ns();
// ... DFA logic ...
let elapsed = bpf_ktime_get_ns() - t0;
let bucket = (elapsed / 100).min(31) as u32;
// atomic increment LATENCY_HIST[bucket]
```

Userspace (sau 30s): đọc 32 buckets, tính p50/p99/p999.

**Điều kiện chạy:**
- nginx phục vụ static file dưới load wrk (cùng lúc với E3)
- Collect tối thiểu 1 triệu samples

**Output:** Bảng `[bucket_ns, count, cumulative%]`, highlight p50/p99/p999.

## E2 — Microbenchmark Transformer CPU Latency

**File:** `notebooks/p4/benchmark_transformer.py`

**Cơ chế:**

```python
import time
import torch
from gp.model import SyscallTransformer

model = SyscallTransformer.load_from_checkpoint("results/p2/checkpoints/best.ckpt")
model.eval().cpu()

# Load 10,000 windows từ val set
windows = load_val_windows(n=10_100)  # extra 100 for warmup

latencies_ns = []
with torch.no_grad():
    # Warmup
    for w in windows[:100]:
        model.encode(w.unsqueeze(0))

    # Benchmark
    for w in windows[100:]:
        t0 = time.perf_counter_ns()
        model.encode(w.unsqueeze(0))  # shape [1, 64]
        latencies_ns.append(time.perf_counter_ns() - t0)

# Report p50, p99, p999
```

**Điều kiện:**
- `torch.device("cpu")`, no GPU
- `torch.no_grad()`, `model.eval()`
- Không batch: mỗi lần gọi xử lý 1 window `[1, 64]`
- Cùng máy local với E1/E3

**Output:** p50, p99, p999 latency (ns → ms). Expected: ~1–10ms/window.

## E3 — Live Workload Overhead (nginx)

**Điều kiện benchmark:**

| Điều kiện | Mô tả |
|---|---|
| **Baseline** | nginx + wrk, không có eBPF attach |
| **DFA attached** | nginx + wrk + `guepard-shield --dfa results/p3/dfa_config.json` |
| **Transformer (theoretical)** | Tính từ E2: `syscall_rate × transformer_p50_latency` |

**Quy trình đo:**

```bash
# Setup: nginx serving static 4KB file, 1 worker process
nginx -c nginx_bench.conf

# Điều kiện 1: Baseline
wrk -t4 -c100 -d60s http://localhost:8080 > baseline_wrk.txt
perf stat -e cycles,instructions,raw_syscalls:sys_enter \
  -p $(pgrep nginx) -- sleep 60 2> baseline_perf.txt

# Điều kiện 2: DFA attached
sudo guepard-shield --dfa results/p3/dfa_config.json &
wrk -t4 -c100 -d60s http://localhost:8080 > dfa_wrk.txt
perf stat -e cycles,instructions,raw_syscalls:sys_enter \
  -p $(pgrep nginx) -- sleep 60 2> dfa_perf.txt

# Đo syscall rate (từ baseline_perf.txt): syscalls_per_sec
# Tính Transformer theoretical CPU: syscalls_per_sec × p50_transformer_ns / 1e9
```

**Metrics báo cáo:**

| Metric | Baseline | DFA attached | Transformer (theoretical) |
|--------|----------|--------------|--------------------------|
| nginx throughput (req/s) | — | Δ% | N/A |
| CPU cycles (nginx, 60s) | — | Δ% | — |
| Syscall rate (/s) | — | — | (input cho tính toán) |
| CPU cores needed for inline eval | 0 | ~0 | `rate × latency` |
| Feasible for realtime? | — | Yes | No (nếu > 1 core) |

## File changes

| File | Thay đổi |
|------|---------|
| `guepard-shield-ebpf/src/main.rs` | Thêm `LATENCY_HIST` map + timestamp logic |
| `guepard-shield/src/main.rs` | Thêm histogram dump sau Ctrl-C |
| `notebooks/p4/benchmark_transformer.py` | Tạo mới |
| `notebooks/p4/nginx_bench.conf` | nginx config tối giản cho benchmark |
| `notebooks/p4/run_e3.sh` | Shell script chạy E3 tự động |

## Expected results (hypothesis)

| Metric | DFA (eBPF) | Transformer (CPU) | Ratio |
|--------|-----------|-------------------|-------|
| p50 latency/event | ~200–500 ns | ~3–8 ms | ~10,000× |
| p99 latency/event | ~1–2 µs | ~15–30 ms | ~15,000× |
| CPU overhead @ 50k syscalls/s | < 2% | > 100% (infeasible) | — |

## Ghi chú

- Transformer chạy E2 trên CPU để phản ánh đúng điều kiện "inline enforcement" — không có GPU trong kernel path.
- Nếu Transformer p50 < 1ms (bất ngờ tốt), cần thêm chú thích rằng syscall rate thực tế của nginx là Xk/s và vẫn cần X cores để theo kịp.
- Kết quả E3 DFA overhead nên < 5% để đáp ứng mục tiêu §5.4.3.
