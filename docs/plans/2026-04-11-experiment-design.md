# Experiment Design

**Date:** 2026-04-11  
**Goal:** Find the simplest experiment that produces both high detection performance and human-readable rules sufficient for the paper. Experiments run sequentially. **When one succeeds, all remaining experiments are dropped.**

---

## Philosophy

Each experiment is a self-contained attempt at the full pipeline:

```
Data preprocessing → Detection model (P2) → Rule extraction (P3) → Evaluate rules
```

Experiments are ordered by: **novelty of contribution** (try more impactful things first) balanced by **possibility** (don't start with something unlikely to work) and **difficulty** (avoid blocking the thesis on hard implementation).

**Success criteria (must pass all three to stop):**
- Detection: AUROC ≥ 0.95 AND F1 ≥ 0.90 on test set
- Rule fidelity: ≥ 95% vs. the detection model
- Rule quality: rules are human-readable, FPR < 1%, mappable to ≥ 1 MITRE technique

If any criterion fails → mark experiment as partial result, continue to next.

---

## Exp 0 — Baseline: BiLSTM + Patch-type Trie

**Novel contribution:** Low (STIDE at patch granularity)  
**Possibility:** Very high  
**Difficulty:** Low  

**Goal:** Validate the full pipeline end-to-end. Establishes that detection → rule → eBPF works before committing to complex architectures.

**P2 — Detection model:**
- BiLSTM Approach A (sliding window, 512 syscalls, next-syscall prediction)
- Anomaly score = mean CE loss over window positions
- See `2026-04-09-p2-teacher-model-design.md` for full spec

**P3 — Rule extraction:**
- Cluster BiLSTM hidden states → K patch types (K=20, K-Means)
- Each recording → sequence of patch type IDs
- Scan normal data → build Trie of allowed next-patch-type for each prefix
- Deploy Trie as `BPF_MAP_TYPE_LPM_TRIE` in eBPF

**Rule example:**
> IF patch_sequence [Net_recv → File_IO] THEN allowed_next ∈ {Net_send, File_IO}. Otherwise → ALERT.

**Stop condition:** Pass all 3 success criteria → paper with limited novelty (baseline contribution). Acceptable if time is critical.

**If fail:** Detection F1 < 0.90 → debug BiLSTM before proceeding. Rule FPR too high → Trie is too strict, proceed to Exp 1 for FSA which tolerates unseen transitions better.

---

## Exp 1 — Primary Target: Option 2 (Patch Transformer) + Patch-state FSA

**Novel contribution:** High  
**Possibility:** Medium-high  
**Difficulty:** Medium  

**Goal:** Replace BiLSTM with a two-level Patch Transformer. Use cross-patch attention to directly derive a Finite State Automaton of normal behavioral episode transitions.

**P2 — Detection model:**
- Hierarchical Patch Transformer (Option 2 in `2026-04-11-transformer-architecture-ideas.md`)
- 512 syscalls → 32 patches × 16 syscalls → patch embeddings (128-dim)
- Cross-patch Transformer with Masked Patch Modeling objective
- Anomaly score = reconstruction loss of masked patches

**P3 — Rule extraction:**
1. Collect patch embeddings from all normal recordings
2. K-Means → K behavioral states (K=8–15, choose by silhouette score)
3. Label each state semantically (e.g., "Web serving", "File I/O", "Process init")
4. Cross-patch attention weights → transition probability matrix between states
5. FSA: keep transitions with probability above threshold θ as allowed
6. At inference: any transition not in FSA → ALERT

**Rule example:**
> State "Web serving" → allowed: {"Web serving", "Closing connection"}. Transition to "Shell execution" → ALERT.

**Why this works:** Cross-patch attention IS the transition matrix — no post-hoc approximation needed. Attention matrix → automaton is a direct extraction.

**Stop condition:** Pass all 3 success criteria → paper-ready. Strong novelty claim: MAE objective applied to syscall sequences + attention-derived FSA.

**If fail:** Patch FSA has too many states to be readable (K must be large for good detection) → proceed to Exp 2 which uses phase segmentation to reduce per-phase FSA complexity. If detection itself is weak → fall back to BiLSTM + FSA variant before Exp 2.

---

## Exp 2 — Enhancement: Phase-aware Segmentation on top of Exp 1

**Novel contribution:** Medium (orthogonal to Exp 1)  
**Possibility:** High (if Exp 1 succeeded)  
**Difficulty:** Low (runs on top of Exp 1 without retraining)  

**Goal:** Split the single FSA into per-phase FSAs (Startup / Active / Idle / Shutdown). Reduces FPR by not conflating noisy startup syscalls with active-phase behavior.

**How (no model retraining):**
1. Use patch type labels from Exp 1 to identify phase boundaries in each recording
   - Startup: patches dominated by `open/mmap/execve` types
   - Active: stable patch type distribution (high `recv/send/read/write`)
   - Idle: low patch rate, few transitions
   - Shutdown: patch rate decreasing, `close/munmap` dominant
2. Build one FSA per phase using the same attention-derived method from Exp 1
3. At inference: classify current phase by syscall rate → apply corresponding FSA

**Ablation:** Single-policy FSA (Exp 1) vs. per-phase FSA (Exp 2) — compare FPR per phase. This ablation is a standalone table in the paper.

**Stop condition:** Per-phase FPR meaningfully lower than single-policy (target: ≥ 15% reduction in startup-phase FPR) → add as second contribution alongside Exp 1. Together these are sufficient for a paper.

**If fail (phase segmentation adds no FPR benefit):** Report as negative result in thesis. Keep Exp 1 result as the paper contribution. Do not proceed to Exp 3 just to get a third contribution — quality over quantity.

---

## Exp 3 — Stretch: Option 3 (Cross-Thread Transformer) + Thread-role Interaction Graph

**Novel contribution:** Very high  
**Possibility:** Medium  
**Difficulty:** High  

**Goal:** Model inter-thread behavior explicitly. Replace per-patch attention with per-thread attention across concurrent processes. Extract an interaction graph of allowed thread role co-occurrences.

**Only attempt if:** Exp 1 + Exp 2 are complete AND there is sufficient time before the paper deadline.

**P2 — Detection model:**
- Cross-Thread Transformer (Option 3 in `2026-04-11-transformer-architecture-ideas.md`)
- Per-thread encoder → thread embeddings (128-dim) via CLS token
- Cross-thread Transformer with full self-attention across threads
- High-thread fallback: k-means cluster threads → K=8 representatives

**P3 — Rule extraction:**
1. Collect thread embeddings from all normal recordings
2. K-Means → K thread role types (K=6–10: "Web_handler", "DB_connector", "Logger", "Shell_executor", ...)
3. Cross-thread attention matrix → expected co-occurrence between role pairs
4. Interaction graph: edge (A, B) exists if co-occurrence probability > threshold θ
5. At inference: any thread role pair not connected in graph → ALERT

**Rule example:**
> Web_handler ↔ DB_connector: ALLOWED.
> Web_handler ↔ Shell_executor: not in graph → ALERT.

**Stop condition:** Pass all 3 success criteria + thread interaction graph is human-readable (analyst can name each cluster) → strongest possible paper contribution. Cross-thread attention for syscall AD is unexplored in literature.

**If fail:** Thread clustering does not produce semantically meaningful roles (clusters are not nameable) → report in thesis as negative result. Do not proceed to Exp 4.

---

## Exp 4 — Supplementary: Thread-level Anchors for MITRE Mapping

**Novel contribution:** Medium  
**Possibility:** Medium  
**Difficulty:** Medium  

**Goal:** Generate compact If-Then rules at thread role level using the Anchors algorithm. Used for MITRE ATT&CK mapping and case studies, not as the primary detection method.

**Only attempt alongside Exp 3** (requires thread role types from Option 3 clustering).

**How:**
1. Use thread role types from Exp 3
2. Run Anchors: find minimal set of role presence/absence conditions that anchor the prediction
3. Map each anchor rule to a MITRE ATT&CK technique

**Rule example:**
> IF DB_connector ∈ recording AND Shell_executor ∉ recording → NORMAL (coverage=87%, precision=99%)
> Maps to: absence of Shell_executor → no T1059 (Command Execution)

**Role in paper:** MITRE mapping table, case study section, qualitative analysis. Not a standalone contribution — supports Exp 3 results.

---

## Experiment Order Summary

```
Exp 0 (Baseline: BiLSTM + Patch Trie)
  ↓ if detection works but rules are too strict
Exp 1 (Option 2 + Patch FSA)          ← PRIMARY TARGET
  ↓ if success: add enhancement
Exp 2 (Phase-aware on top of Exp 1)   ← ENHANCEMENT
  ↓ if time permits
Exp 3 (Option 3 + Thread Graph)       ← STRETCH
  ↓ alongside Exp 3
Exp 4 (Thread Anchors + MITRE)        ← SUPPLEMENTARY
```

**Stop at the first experiment that passes all 3 success criteria.** Exp 2 is the exception — always run it if Exp 1 succeeds, since it requires no retraining and adds a free ablation.

---

## Decision Table

| Experiment | Pass | Fail |
|---|---|---|
| Exp 0 | Pipeline validated. Continue to Exp 1. | Debug detection model first. |
| Exp 1 | Paper-ready. Run Exp 2 for free enhancement. | Analyze failure: detection or FSA? |
| Exp 2 | Stronger paper (two contributions). Stop. | Report negative result. Stop if deadline close. |
| Exp 3 | Flagship contribution. Run Exp 4 alongside. | Report negative result in thesis. |
| Exp 4 | Adds MITRE analysis section. | Drop, not critical. |
