# Phase 1.5 — Pilot Experiments

**Phase:** P1.5 — Early validation of distillation pathway on DongTing
**Goal:** Validate H2 trước khi invest vào P2 Teacher training full scale.
**Pivot criterion:** Nếu distillation gain nhỏ → chuyển focus comparative study, giảm emphasis KD.

## Experiment Runs

| Run | Folder | Date | class_weight | Direct DT Attack Recall | Distilled DT Attack Fidelity | Status |
|-----|--------|------|--------------|------------------------|------------------------------|--------|
| v1 | `v1-balanced-cw/` | 2026-03-27 | balanced | 75.2% | 98.48% | Archived — shows problem |
| v2 | `v2-no-cw/` | 2026-03-28 | None | 99.8% | 98.03% | **Canonical** |

## Pivot Decision

**v2 kết quả → tiếp tục KD pathway:**
- Distillation gain trên normal recall: +19.8pp (rõ ràng)
- Attack Fidelity 98% → Teacher knowledge được transfer tốt
- FPR thấp hơn direct DT → validate phase-aware hypothesis

## Negative Result (v1 → v2)

`class_weight='balanced'` trong direct DT không cải thiện performance — chỉ flip bias.
Kết quả này được ghi lại trong research proposal (Section 11.1) và là design decision có giá trị cho thesis.
