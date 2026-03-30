# Pilot v1 — class_weight='balanced' (archived)

**Date:** 2026-03-27
**Phase:** P1.5 — Pilot distilled vs direct DT on DongTing
**Status:** ARCHIVED — baseline showing problem with balanced class weights

## Config

| Parameter | Value |
|-----------|-------|
| Dataset | DongTing (limit=2000 sequences) |
| max_windows_per_seq | 5 |
| window_size / stride | 64 / 12 |
| Teacher | BiLSTM, d_model=128, vocab_size=258 |
| Temperature (T) | 4.0 |
| Surrogate | DecisionTreeClassifier, max_depth=3 |
| **class_weight** | **'balanced' (DecisionTreeClassifier)** |
| Surrogate features | TF-IDF n-gram (1,2), max_features=1000 |

## Key Results

| Model | Accuracy | Normal Recall | Attack Recall | F1 |
|-------|----------|---------------|---------------|----|
| Teacher (BiLSTM) | 97.2% | — | — | — |
| Direct DT (balanced CW) | 84.0% | **96.3%** | **75.2%** | 0.840 |
| Distilled DT | 94.7% | 88.0% | 99.5% | 0.945 |

Distilled metrics:
- Fidelity to Teacher (overall): **95.35%**
- Attack-class Fidelity: **98.48%**

## Finding (Research Proposal Section 11.1)

`class_weight='balanced'` **flips the bias** trong direct DT:
- Normal recall: 68% → 96% (tăng)
- Attack recall: 99.8% → 75.2% (giảm mạnh)

Attack recall 75% là không chấp nhận được cho security use case. Balanced weights không cải thiện overall performance — chỉ dịch chuyển bias. **Kết luận: không dùng `class_weight='balanced'`**, xử lý imbalance ở data level qua `max_windows_per_seq`.

Distilled DT KHÔNG bị ảnh hưởng bởi class_weight vì lúc này `fit_distilled()` chưa nhận `hard_labels` → không có sample weighting → kết quả distilled giống v2.

## Thesis Reference

- Section 11.1 (Ablation Design): negative result về balanced class weights
- Supports design decision: "Không dùng class_weight='balanced'"
- Compare với v2 để show distillation benefit độc lập với class weight choice
