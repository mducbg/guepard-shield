#!/usr/bin/env bash
# E3 - nginx live overhead benchmark.
# Usage: sudo bash notebooks/p4/run_e3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/p4"

if [[ "${E3_CAPTURE:-0}" != "1" ]]; then
    mkdir -p "$RESULTS_DIR"
    E3_CAPTURE=1 bash "$0" 2>&1 | tee "$RESULTS_DIR/e3_summary.txt"
    exit "${PIPESTATUS[0]}"
fi

cd "$PROJECT_ROOT"

DFA_CONFIG="results/p3/dfa_config.json"
WRK_DURATION=60
WRK_THREADS=4
WRK_CONNECTIONS=100
WRK_URL="http://localhost:8080/"

for cmd in nginx wrk perf curl; do
    command -v "$cmd" >/dev/null || {
        echo "ERROR: $cmd not found"
        exit 1
    }
done

[[ -f "$DFA_CONFIG" ]] || {
    echo "ERROR: DFA config not found: $DFA_CONFIG"
    exit 1
}

[[ -f "$PROJECT_ROOT/target/release/guepard-shield" ]] || {
    echo "ERROR: guepard-shield binary not found. Run: cargo build --release"
    exit 1
}

mkdir -p "$RESULTS_DIR"

cleanup() {
    if [[ -n "${DFA_PID:-}" ]]; then
        kill -SIGINT "$DFA_PID" 2>/dev/null || true
        wait "$DFA_PID" 2>/dev/null || true
    fi
    nginx -s stop -c "$SCRIPT_DIR/nginx_bench.conf" 2>/dev/null || true
}
trap cleanup EXIT

echo "[setup] Creating nginx static file..."
mkdir -p /tmp/nginx_bench_root
dd if=/dev/urandom bs=4096 count=1 2>/dev/null | base64 > /tmp/nginx_bench_root/bench.html

echo "[setup] Starting nginx..."
nginx -c "$SCRIPT_DIR/nginx_bench.conf" 2>/dev/null || true
sleep 1
NGINX_PID="$(cat /tmp/nginx_bench.pid 2>/dev/null)" || {
    echo "ERROR: nginx failed to start"
    exit 1
}
echo "[setup] nginx master pid=$NGINX_PID"
# Worker process does the actual HTTP work; master only manages children.
NGINX_WORKER_PID="$(pgrep -P "$NGINX_PID" | head -1)"
if [[ -z "$NGINX_WORKER_PID" ]]; then
    echo "ERROR: cannot find nginx worker process (child of $NGINX_PID)"
    exit 1
fi
echo "[setup] nginx worker pid=$NGINX_WORKER_PID"

curl -s "$WRK_URL" >/dev/null
sleep 1

echo ""
echo "=== Condition 1: Baseline (no eBPF) ==="
perf stat -e cycles,instructions,raw_syscalls:sys_enter \
    -p "$NGINX_WORKER_PID" -- sleep "$WRK_DURATION" \
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

echo ""
echo "=== Condition 2: DFA attached ==="
"$PROJECT_ROOT/target/release/guepard-shield" \
    --dfa "$DFA_CONFIG" \
    --latency-out "$RESULTS_DIR/e1_ebpf_histogram_from_e3.txt" \
    --target-tgid "$NGINX_WORKER_PID" &
DFA_PID=$!
sleep 1
kill -0 "$DFA_PID" 2>/dev/null || {
    echo "ERROR: guepard-shield failed to start or exited before benchmark"
    wait "$DFA_PID" 2>/dev/null || true
    unset DFA_PID
    exit 1
}

perf stat -e cycles,instructions,raw_syscalls:sys_enter \
    -p "$NGINX_WORKER_PID" -- sleep "$WRK_DURATION" \
    2> "$RESULTS_DIR/dfa_perf.txt" &
PERF_PID=$!

wrk -t"$WRK_THREADS" -c"$WRK_CONNECTIONS" -d"${WRK_DURATION}s" "$WRK_URL" \
    > "$RESULTS_DIR/dfa_wrk.txt"

wait "$PERF_PID" || true
kill -SIGINT "$DFA_PID" 2>/dev/null || true
wait "$DFA_PID" 2>/dev/null || true
unset DFA_PID

echo "[dfa] wrk done. Results in $RESULTS_DIR/dfa_wrk.txt"
cat "$RESULTS_DIR/dfa_wrk.txt"
echo "--- perf ---"
cat "$RESULTS_DIR/dfa_perf.txt"

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
echo "Syscalls/sec input:"
grep "sys_enter" "$RESULTS_DIR/baseline_perf.txt" || true
echo ""
echo "Full E3 log saved to $RESULTS_DIR/e3_summary.txt"
