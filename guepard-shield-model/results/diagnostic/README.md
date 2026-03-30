# Diagnostic — EDA & Dataset Statistics

**Date:** 2026-03-26 (approx)
**Phase:** P1 — Data pipeline, EDA
**Script:** `notebooks/diagnostic.py`

## Files

| File | Description |
|------|-------------|
| `fig1_distributions.png` | Label distribution, sequence length distribution by class |
| `fig2_syscall_analysis.png` | Syscall frequency analysis, top-N syscalls by class |
| `fig3_statistical.png` | Statistical summary — window counts, imbalance ratio |

## Key Findings

- Window imbalance before capping: attack sequences có nhiều windows hơn normal do seq_length dài hơn (~125:1 window ratio)
- Sau `max_windows_per_seq=5`: window ratio giảm xuống ~2:1 (chấp nhận được)
- Top discriminative syscalls đã visible ở EDA → input cho SHAP analysis (C2)

## Thesis Reference

- **Section 5 (Datasets)**: EDA figures, dataset statistics
- **Section 11.1** note về class imbalance: "window imbalance (1:1.3-1.5) sau max_windows_per_seq"
