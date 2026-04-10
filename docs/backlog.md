# Backlog

Dựa trên [research-proposal.md](research-proposal.md). Chỉ bao gồm **Core** (đủ cho thesis).

---

## P1 — Data Pipeline & EDA (Tháng 1-2)

- [x] Viết data loaders cho LID-DS-2019, LID-DS-2021 (`src/gp/data_loader/`)
  - `lidds_2019_loader.py`: Parses .txt files with rich features (timestamp, tid, syscall)
  - `lidds_2021_loader.py`: Parses .sc + .json files with pre-defined train/val/test splits
- [x] Diagnostic notebooks: integrity, sequence length distribution, syscall vocabulary
  - `notebooks/diagnostic/lidds_2019.py`, `lidds_2021.py`
  - `src/gp/diagnostic/stats.py`: integrity(), seq_lengths(), syscall_vocab()
- [x] Implement phase segmenter (sliding window trên syscall rate → startup/active/idle/shutdown)
  - `src/gp/phase/segmenter.py`: segment() → Phase per syscall
- [x] MITRE ATT&CK mapping cho từng scenario trong LID-DS-2019 và LID-DS-2021
  - `src/gp/mitre.py`: SCENARIO_TO_TECHNIQUES, techniques_for(), scenarios_for()

## P1.5 — Pilot (Tháng 2)

- [ ] Feature engineering baseline (n-gram counts) trên LID-DS-2019
- [ ] Train DT trực tiếp (hard label) trên LID-DS-2019
- [ ] Train Teacher nhỏ (LSTM hoặc shallow Transformer) trên LID-DS-2019
- [ ] Train DT distilled (soft label từ Teacher) trên LID-DS-2019
- [ ] So sánh distilled vs direct → quyết định có tiếp tục KD focus không

## P2 — Teacher Training (Tháng 3-4)

- [ ] **C1:** Train Transformer Teacher trên LID-DS-2021, target F1 ≥ 90%
- [ ] Train LSTM baseline, so sánh với Teacher — chọn model tốt hơn làm Teacher
- [ ] Temperature calibration: Platt scaling → T_calib
- [ ] Sweep T ∈ [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] → train surrogate DT mỗi T → chọn T*
- [ ] Checkpoint P2: soft label entropy attack > normal, reliability diagram, DT fidelity tăng theo T

## P3 — Rule Extraction & Ablation (Tháng 5-6)

- [ ] **C2:** SHAP feature selection — rank top-K features từ Teacher
- [ ] **C3 (Exp B):** Tree family — DT, HSTree, FIGS; distilled vs direct; full vs SHAP-selected
- [ ] **C4 (Exp C):** Rule ensemble — RuleFit, BoostedRules; distilled vs direct; full vs SHAP-selected
- [ ] **C5:** Ablation distillation — tách gain: Hard vs Soft-T1 vs Soft-T*; full vs SHAP features
- [ ] **C6:** Phase-aware ablation — single-policy vs per-phase surrogate trên cùng Teacher

## P4 — Evaluation & Deployment (Tháng 6-7)

- [ ] **C7a:** MITRE ATT&CK coverage matrix — map scenarios → technique IDs, kiểm tra rule detect được không
- [ ] **C7b:** Head-to-head Falco comparison — detection rate, FPR, rule count, complexity
- [ ] **C7c:** Case study 2-3 CVE — walk through syscall sequence → rule fire → so sánh Falco
- [ ] **C7d:** Rule auditability — informal review từ 1-2 security practitioner
- [ ] **C8:** Cross-scenario generalization — train trên subset scenarios của LID-DS → test trên held-out scenarios (kiểm tra generalization qua các loại attack khác nhau)
- [ ] **C9:** eBPF Rule Compiler — convert DT rules → if-else C → compile eBPF → verify → đo latency
- [ ] **C10:** Real workload evaluation — đo FPR và latency trên nginx/redis/postgres

## P4.5 — Paper Draft (Tháng 7)

- [ ] Đóng gói C1 + C3/C4 + C6 + C7 + C9 + C10 thành paper draft
- [ ] Submit cho advisor review
