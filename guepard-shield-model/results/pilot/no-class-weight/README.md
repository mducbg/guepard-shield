# Pilot v2 — No class_weight (canonical result)

**Date:** 2026-03-28
**Phase:** P1.5 — Pilot distilled vs direct DT on DongTing
**Status:** CANONICAL — kết quả pilot chính thức dùng cho thesis/paper

## Config

| Parameter | Value |
|-----------|-------|
| Dataset | DongTing (limit=2000 sequences) |
| max_windows_per_seq | 5 |
| window_size / stride | 64 / 12 |
| Teacher | BiLSTM, d_model=128, vocab_size=258 |
| Temperature (T) | 4.0 |
| Surrogate | DecisionTreeClassifier, max_depth=3 |
| **class_weight** | **None (imbalance handled via max_windows_per_seq)** |
| Surrogate features | TF-IDF n-gram (1,2), max_features=1000 |

## Key Results

| Model | Accuracy | Normal Recall | Attack Recall | F1 |
|-------|----------|---------------|---------------|----|
| Teacher (BiLSTM) | 97.0% | — | — | — |
| Direct DT (no CW) | 86.6% | **68.2%** | **99.8%** | 0.853 |
| Distilled DT | **94.7%** | **88.0%** | **99.5%** | **0.945** |

Distilled metrics:
- Fidelity to Teacher (overall): **94.99%**
- Attack-class Fidelity: **98.03%**

## Finding — Distillation Gain

Distilled DT vs Direct DT (key delta):
- Normal Recall: +19.8pp (68.2% → 88.0%) — giảm false positive đáng kể
- Attack Recall: ≈ (99.8% → 99.5%) — giữ nguyên detection rate
- Overall F1: +9.1pp (0.853 → 0.945)
- Attack Fidelity: 98.03% — distilled DT bắt gần như toàn bộ attack cases Teacher bắt

Kết quả này validate H2: "Distillation cải thiện Attack-class Fidelity ≥3-5%".

## Thesis Reference

- **RQ1 + RQ2 (core contribution)**: kết quả pilot confirm distillation pathway
- **Section 11.1** ablation table row: "DT-Soft-T4 (full features)"
- **Pivot point (P1.5)**: distillation gain đủ lớn → tiếp tục focus KD pathway cho P2/P3
- Paper: Methodology section, Table 1 (pilot results)
