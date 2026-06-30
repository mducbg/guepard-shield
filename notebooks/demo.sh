#!/usr/bin/env bash
# One-command Guepard Shield defense demo: P3 DFA export, then P4 eBPF runtime.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DFA_CONFIG="results/p3/dfa_config.json"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

for path in \
    results/p3/hidden_states/info.json \
    results/p3/clusters/K64/labels.dat \
    results/p3/dfa_s1/K64_S3/transitions.npz \
    results/p3/metrics/grid_search.csv; do
    [[ -f "$path" ]] || fail "P3 demo artifact not found: $path"
done

for cmd in uv cargo nginx wrk perf sudo; do
    command -v "$cmd" >/dev/null || fail "$cmd is required"
done

echo "=== Guepard Shield defense demo ==="
echo "[1/4] Exporting DFA from persisted P3 hidden-state artifacts..."
uv run notebooks/p3/export_dfa.py
[[ -f "$DFA_CONFIG" ]] || fail "DFA export did not create: $DFA_CONFIG"

echo "[2/4] Building the eBPF agent as the current user..."
cargo build --release

echo "[3/4] Starting 60s baseline and 60s eBPF runs..."
echo "      sudo is needed only to attach the eBPF tracepoint."
sudo bash notebooks/p4/run_e3.sh

echo "[4/4] Demo completed. Evidence files:"
printf '  %s\n' \
    results/p4/baseline_wrk.txt \
    results/p4/dfa_wrk.txt \
    results/p4/e1_ebpf_histogram_from_e3.txt \
    results/p4/e3_summary.txt
