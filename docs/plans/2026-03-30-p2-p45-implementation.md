# P2–P4.5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement full RuleDistill pipeline — LID-DS Teacher training → surrogate distillation → eBPF rule compiler — ready for ACSAC paper submission.

**Architecture:** BiLSTM/Transformer Teacher trained on LID-DS-2021, knowledge distilled into DT + RuleFit surrogates via soft labels, rules compiled to eBPF enforcement layer.

**Tech Stack:** JAX/Keras (Teacher), scikit-learn (DT, permutation importance), imodels (RuleFit), Aya/Rust (eBPF), polars, matplotlib.

**No commits between tasks** — commit only at phase boundaries (end of P2, P3, P4, P4.5).

---

## Memory Budget (6GB VRAM RTX 3060, 32GB RAM)

Set at top of every notebook script:
```python
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"  # 5.1GB usable
```

**Teacher configs that fit in 6GB:**
- BiLSTM: `d_model=128, dropout=0.1` → ~50MB weights, batch_size=64 safe
- Transformer: `d_model=128, n_heads=4, n_layers=4, d_ff=256` → ~15MB weights, batch_size=32 safe (attention is O(seq²))
- Window size = 64 tokens (existing `WindowConfig` default) — keep this

**CPU-side (sklearn/imodels):**
- Permutation importance: subsample to 10k windows for speed, full test set for final metrics
- RuleFit: `max_rules=2000, n_estimators=500` — stays under 4GB RAM
- Never load all LID-DS sequences into memory at once; stream with `iter_sequences()`

---

## Task 1: Dependencies

**Files:** `guepard-shield-model/pyproject.toml`

Add to `dependencies`:
```
"imodels>=1.3.10",
"shap>=0.46.0",
```

Run: `cd guepard-shield-model && uv sync`

`shap` is only used for `GradientExplainer` on Teacher embeddings (thesis discussion). Feature selection uses `sklearn.inspection.permutation_importance` — already available.

---

## Task 2: LID-DS Corpus Loader

**Files:**
- Create: `guepard-shield-model/src/guepard/data_loader/lidds_corpus.py`

**Context:** LID-DS-2021 directory layout:
```
<scenario>/
    training/    # .sc files (syscall traces) + .json (metadata)
    validation/
    test/
```

`.sc` format (space-separated):
```
timestamp_ns  thread_id  pid  process_name  pid  syscall_name  direction  [args]
```

**Design:**
- Extend `SequenceMeta` with `scenario: str`, `has_exploit: bool`, `exploit_time_ns: int | None` — parse from `.json` sidecar
- Only keep exit events (`direction == "<"`) to avoid duplicate tokens — consistent with LID-DS literature
- `LIDDSCorpus(data_dir, scenarios: list[str])` — scans `<data_dir>/<scenario>/` for each scenario
- `iter_sequences(split) → Iterator[(seq_id, label, tokens: list[str])]` — Tier 1, syscall names only
- `iter_sequences_rich(split) → Iterator[(seq_id, label, events: list[dict])]` — Tier 2, dicts with `timestamp_ns`, `syscall`, `thread_id`
- Split determined by subdirectory: `training/` → train, `validation/` → val, `test/` → test
- Label: 0 = normal, 1 = attack — from `.json` `is_exploit` field

**Memory:** Stream `.sc` files line-by-line; never load entire file. Each `.sc` file is one recording (one sequence).

---

## Task 3: Phase Segmenter

**Files:**
- Create: `guepard-shield-model/src/guepard/data_loader/phase_segmenter.py`

**Input:** `list[tuple[int, str]]` — `(timestamp_ns, syscall_name)` for one recording
**Output:** `list[str]` — per-event phase label ∈ `{startup, active, idle, shutdown}`

**Algorithm:**
1. Sliding window of 100ms → compute `syscall_rate` (events/window) per window
2. Per-recording percentiles: P25, P75 of rate distribution
3. Phase assignment per window:
   - `startup`: first consecutive windows until rate variance stabilizes (rolling std < 0.1 × mean)
   - `active`: rate > P75 (and not startup)
   - `idle`: rate < P25
   - `shutdown`: last monotonically decreasing run at end of recording
   - `transition`: anything else (can merge into nearest phase for simplicity)
4. Expand window labels → per-event labels (all events in window get same label)

**`PhaseSegmenter` class:**
```python
class PhaseSegmenter:
    def __init__(self, window_ms: int = 100): ...
    def segment(self, events: list[tuple[int, str]]) -> list[str]: ...
```

---

## Task 4: Permutation Importance Selector

**Files:**
- Create: `guepard-shield-model/src/guepard/features/shap_selector.py`

**Purpose:** Select top-K TF-IDF features by permutation importance, reducing feature space for surrogate models.

**Design:**
```python
class PermImportanceSelector:
    def __init__(self, k: int = 100, subsample: int = 10_000): ...

    def fit(self, X_val, y_val, dt_model) -> None:
        # sklearn.inspection.permutation_importance on dt_model
        # subsample from val set if len > self.subsample
        # store top-K feature indices in self.selected_indices_

    def transform(self, X) -> sparse matrix:
        # return X[:, self.selected_indices_]

    def save(self, path) -> None:
        # np.save(path / "perm_importance.npy", self.importances_)
        # np.save(path / "selected_indices.npy", self.selected_indices_)
```

No SHAP here — just sklearn permutation importance on DT-Hard-Full (fast, no GPU needed).

---

## Task 5: Evaluation Metrics

**Files:**
- Create: `guepard-shield-model/src/guepard/evaluation/metrics.py`

**Functions:**

```python
def attack_fidelity(teacher_preds, surrogate_preds) -> float:
    """Fraction of attack samples where surrogate agrees with teacher."""
    attack_mask = teacher_preds == 1
    return np.mean(surrogate_preds[attack_mask] == teacher_preds[attack_mask])

def overall_fidelity(teacher_preds, surrogate_preds) -> float:
    return np.mean(teacher_preds == surrogate_preds)

def per_phase_fpr(y_true, y_pred, phase_labels) -> dict[str, float]:
    """FPR broken down by phase label. y_true=ground truth, y_pred=surrogate."""
    result = {}
    for phase in ["startup", "active", "idle", "shutdown"]:
        mask = np.array(phase_labels) == phase
        if mask.sum() == 0:
            continue
        # FPR = FP / (FP + TN) on normal samples only
        normal_mask = mask & (y_true == 0)
        if normal_mask.sum() == 0:
            result[phase] = 0.0
            continue
        fp = np.sum((y_pred[normal_mask] == 1))
        tn = np.sum((y_pred[normal_mask] == 0))
        result[phase] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return result

def rule_complexity(tree_or_rules) -> int:
    """Total number of conditions (literals) across all rules."""
    ...  # DT: sum of (depth of each leaf). RuleFit: sum of rule lengths.
```

---

## Task 6: P2 — Teacher Training Script

**Files:**
- Create: `guepard-shield-model/notebooks/p2_teacher_lidds.py`
- Modify: `guepard-shield-model/src/guepard/config.py` (add `LIDDSConfig`)

**Outline:**

```
Step 1: Config
  - LIDDSConfig: scenarios (5 in-dist), data_dir, output_dir
  - TeacherConfig: d_model=128, n_heads=4, n_layers=4, d_ff=256, batch_size=32 (Transformer)
                   d_model=128, batch_size=64 (BiLSTM)
  - Set XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

Step 2: Load LID-DS (combined 5 in-dist scenarios)
  - LIDDSCorpus(data_dir, in_dist_scenarios)
  - Build SyscallVocab from training split only
  - Build TeacherDataset for train + val

Step 3: Architecture Comparison
  - Train SyscallLSTM (BiLSTM) — 20 epochs, early stopping on val F1, patience=5
  - Train SyscallTransformer — same setup
  - Save both: best_bilstm.weights.h5, best_transformer.weights.h5
  - Log: teacher_comparison.json → {bilstm: {f1, accuracy}, transformer: {f1, accuracy}}
  - Winner = higher val F1. Must be ≥ 0.90 to proceed.

Step 4: Platt Scaling (T_calib)
  - Get Teacher logits on val set
  - Grid search T ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0} minimizing NLL on val
  - T_calib = argmin NLL

Step 5: Temperature Sweep
  - For each T ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}:
    - Generate soft labels from winner Teacher at temperature T (val set)
    - Train DT (depth=5) on soft labels from train set
    - Evaluate Attack Fidelity on val set
  - T* = T at peak Attack Fidelity on val set
  - Plot: T vs Attack Fidelity → save as temperature_sweep.png

Step 6: Generate Final Soft Labels
  - Use winner Teacher at T* → generate soft labels for ALL train sequences
  - Save per-sequence: soft_labels/<seq_id>.npy (2-element array [p_normal, p_attack])
  - Memory: process in batches of 256 sequences, not all at once

Step 7: P2 Checkpoint Validation
  - Check 1: entropy(attack soft labels) > entropy(normal soft labels) → compare means
  - Check 2: T_calib > 1.0
  - Check 3: Attack Fidelity at T=3.0 > Attack Fidelity at T=1.0
  - Save: p2_checkpoint.json {pass: bool, checks: {...}}
  - If any fail: print warning, do NOT raise exception (let human investigate)
```

**Output artifacts:** `results/p2/` — see design doc §4.3.

**Memory notes:**
- Transformer with window=64, batch=32: activation memory ~150MB → safe
- Use `jax.clear_caches()` between BiLSTM and Transformer training to free XLA cache
- Soft label generation: `jax.device_get()` after each batch to avoid accumulation on GPU

---

## Task 7: P3a — Exp B Decision Tree Ablation

**Files:**
- Create: `guepard-shield-model/notebooks/p3a_exp_b_dt.py`

**Outline:**

```
Step 1: Load artifacts
  - Load winner Teacher weights + vocab from results/p2/
  - Load LID-DS test set (in-dist 5 scenarios)
  - Build SyscallVectorizer (TF-IDF, full n-gram features)
  - X_train_full, X_val_full, X_test_full (sparse matrices — fit only on train)

Step 2: Ablation matrix
  For each config in [DT-Hard-Full, DT-Soft-T1-Full, DT-Soft-T*-Full, DT-Soft-T*-PERM, RF-Direct]:
    For each seed in [42, 43, 44]:
      - Load/generate appropriate labels (hard or soft at T=1.0 or T=T*)
      - If PERM: fit PermImportanceSelector on val set, transform train/test
      - Train DT (depth=5, no class_weight) on train set
      - Evaluate on TEST set: attack_fidelity, overall_fidelity, FPR, #rules, #conditions
      - #rules = number of leaves. #conditions = sum of all split conditions in leaves.
  Report mean ± std across 3 seeds

Step 3: Pareto frontier
  - For depth ∈ {3, 5, 7, 10} × DT-Soft-T*-Full config:
    - Train + evaluate Attack Fidelity vs #Rules
  - Scatter plot → pareto_dt.png

Step 4: Save
  - results/p3/exp_b_results.json
```

**Memory:** Sparse TF-IDF matrices for LID-DS — use `scipy.sparse`, never densify. For full n-gram feature space, vocab_size=512 → bigrams give 512²=262k features max; cap with `max_features=10000` in TfidfVectorizer.

---

## Task 8: P3b — Exp C RuleFit Ablation

**Files:**
- Create: `guepard-shield-model/notebooks/p3b_exp_c_rulefit.py`

**Outline:** Same structure as Task 7 but using `imodels.RuleFitClassifier`.

```python
from imodels import RuleFitClassifier

clf = RuleFitClassifier(
    max_rules=2000,       # cap to avoid memory explosion
    n_estimators=500,
    random_state=seed,
)
# RuleFit expects dense input — densify only subsample if full set > 50k windows
# For 50k+ windows: subsample 50k for training, evaluate on full test set
```

**Extra metric:** `rule_complexity` = total literals across all rules (from `clf.rules_`).

**Configs:** Same 5 configs as Exp B. DT results from Task 7 loaded and compared side-by-side in same results JSON.

**Output:** `results/p3/exp_c_results.json`

**Memory:** RuleFit densifies internally — subsample train to 50k windows max. Test set evaluation stays sparse-then-densify-in-chunks.

---

## Task 9: P3c — Phase Ablation

**Files:**
- Create: `guepard-shield-model/notebooks/p3c_phase_ablation.py`

**Setup:** Fix config = DT-Soft-T*-Full (winner from Exp B Task 7).

```
Step 1: Load winner Teacher + LID-DS rich sequences
  - iter_sequences_rich() → events with timestamps
  - PhaseSegmenter.segment() per recording → phase labels per event

Step 2: Window phase assignment
  - Assign each window its majority-phase label (use windowing.py sliding window)
  - Builds: X_train_by_phase[phase], y_train_by_phase[phase], soft_labels_by_phase[phase]

Step 3: Three policies
  Single-policy:
    - One DT trained on all windows, soft labels at T*
    - Evaluate per-phase FPR using per_phase_fpr()

  Per-phase (4 surrogates):
    - Train 4 DTs, one per phase, on phase-filtered windows
    - At test time: route each window to its phase-DT by phase label
    - Evaluate per-phase FPR + overall + Attack Fidelity

  Per-phase + PERM:
    - Same but with PermImportanceSelector per phase (fit independently per phase)

Step 4: Save
  - results/p3/phase_ablation.json
```

---

## Task 10: Rule Extractor + MITRE Mapper

**Files:**
- Create: `guepard-shield-model/src/guepard/evaluation/rule_extractor.py`
- Create: `guepard-shield-model/src/guepard/evaluation/mitre_mapper.py`

**rule_extractor.py:**
```python
def dt_to_rules(dt: DecisionTreeClassifier, feature_names: list[str]) -> list[str]:
    """Traverse decision tree, extract IF-THEN rules for ATTACK leaves only.
    Format: 'IF feat_a > 2 AND feat_b <= 0 THEN ATTACK (conf=0.97, support=0.43)'"""

def rulefit_to_rules(clf: RuleFitClassifier, threshold: float = 0.01) -> list[str]:
    """Extract rules with |coef| > threshold from RuleFit clf.rules_."""
```

**mitre_mapper.py:**
```python
# Hardcoded mapping from design doc §3
SCENARIO_TO_MITRE = {
    "CVE-2014-0160": {"technique": "T1190", "name": "Exploit Public-Facing App"},
    "CVE-2017-7529": {"technique": "T1190+T1083", "name": "Exploit + File Discovery"},
    "CWE-89":        {"technique": "T1190", "name": "SQL Injection"},
    "Bruteforce_CWE-307": {"technique": "T1110", "name": "Brute Force"},
    "EPS_CWE-434":   {"technique": "T1190+T1105", "name": "Ingress Tool Transfer"},
    # OOD
    "CVE-2020-9484": {"technique": "T1059", "name": "Deserialization"},
    "CVE-2019-5418": {"technique": "T1083", "name": "Path Traversal"},
    "ZipSlip":       {"technique": "T1105", "name": "Archive Extraction"},
}

def build_coverage_matrix(rules: list[str], test_traces: dict[str, list]) -> pd.DataFrame:
    """Per-scenario, per-rule: does rule fire on attack trace? Returns binary matrix."""
```

---

## Task 11: P4a — Rule Analysis & MITRE + Falco Comparison

**Files:**
- Create: `guepard-shield-model/notebooks/p4a_rule_analysis.py`

**Outline:**
```
Step 1: Load DT-Soft-T*-PERM from results/p3/ → extract rules via rule_extractor
  - Save: results/p4/rules_human_readable.txt

Step 2: MITRE coverage matrix
  - For each in-dist scenario test set:
    - Run each rule against syscall traces (simulate: check if feature conditions met)
    - Binary matrix: rule × scenario → detection y/n
  - Save: mitre_coverage_matrix.csv
  - Heatmap figure: mitre_heatmap.png

Step 3: Falco comparison (offline simulation)
  - Load falcosecurity/rules YAML (download once, store in data/)
  - Parse Falco rules: extract syscall name conditions
  - For same test traces: check Falco rule matches
  - Compare: {guepard: detection_rate, fpr, rule_count, mean_rule_length}
             vs {falco: same metrics}
  - Save: results/p4/falco_comparison.json

Step 4: Case studies (2 CVEs)
  - CVE-2014-0160 + CWE-89:
    - Load attack recording → extract syscall sequence around exploit_time_ns
    - Show which rules fire → map to MITRE → compare with Falco equivalent
  - Print formatted case study text (for manual copy to thesis)
```

**Falco comparison is offline** — no Falco daemon. Just parse YAML rules and simulate matching against LID-DS traces using syscall name matching.

---

## Task 12: P4b — Cross-Domain Validation

**Files:**
- Create: `guepard-shield-model/notebooks/p4b_cross_domain.py`

**Outline:**
```
Step 1: Load DT-Soft-T*-Full trained on LID-DS (from results/p3/)
  - Load LID-DS vocab (SyscallVocab)
  - Load DongTing test set via DongTingCorpus

Step 2: Feature alignment
  - LID-DS vocab syscall names ∩ DongTing vocab syscall names → intersection vocabulary
  - Refit TfidfVectorizer restricted to unigrams in intersection vocab only
  - Transform DongTing test set with aligned vectorizer
  - Log: vocab_intersection_size

Step 3: Evaluate (no re-training)
  - Run DT on aligned DongTing features
  - Generate Teacher soft labels on DongTing (Teacher also evaluated zero-shot)
  - Metrics: OOD Fidelity, OOD Attack Fidelity
  - Expected gap vs in-dist: 5–15% drop in fidelity
  - If gap > 15%: note as limitation in output JSON (don't re-train)

Step 4: Save
  - results/p4/cross_domain_results.json
    {ood_fidelity, ood_attack_fidelity, vocab_intersection_size, gap_vs_indist}
```

---

## Task 13: P4c — eBPF Rule Compiler

**Files:**
- Create: `guepard-shield-model/notebooks/p4c_ebpf_compiler.py`
- Modify: `guepard-shield/src/` (Rust eBPF program — new file in existing Aya project)

**Python side (p4c_ebpf_compiler.py):**

```
Step 1: Load DT-Soft-T*-PERM (depth=5) from results/p3/
  - Extract rules via rule_extractor.dt_to_rules()
  - Identify feature types: all are n-gram frequency counts (integers post-discretization)

Step 2: Discretization
  - For each feature threshold in DT splits (float TF-IDF values):
    - Convert to integer frequency bins: bin = round(raw_count / bin_width)
    - bin_width tuned so DT splits map to integer boundaries
    - Start with bin_width=1 (direct count); if Attack Fidelity drops > 2%, halve bin_width
  - Re-evaluate DT on test set with discretized features
  - Save: results/p4/discretization_validation.json
    {before_fidelity, after_fidelity, drop_pct, bin_width}

Step 3: C code generation
  - rule_extractor generates C if-else chain from discretized DT:
    ```c
    static __always_inline int evaluate_rules(struct syscall_counts *counts) {
        if (counts->execve > 2 && counts->open > 5) return 1; // ATTACK
        ...
        return 0;
    }
    ```
  - All integer arithmetic, no floats, no loops → eBPF verifier safe
  - Stack usage: (depth=5) × (features per branch × 8 bytes) << 512 bytes limit

Step 4: Compile + latency measurement (shell commands, may need manual run)
  - eBPF program: hook tracepoint/raw_syscalls/sys_enter
  - Per-PID sliding window counter in BPF_HASH map
  - bpf_ktime_get_ns() before/after evaluate_rules() → latency histogram
  - Run on sample of normal traffic traces
  - Target: median latency < 2µs

Step 5: Save latency results
  - results/p4/ebpf_latency_histogram.json
```

**Rust/eBPF side** (Aya — modify existing project structure):
- New file: `guepard-shield/src/ebpf/rule_evaluator.bpf.c` (or `.rs` if Aya Rust eBPF)
- Template: generated by Python `rule_extractor` with hardcoded if-else chain
- Build: `cargo xtask build-ebpf` (existing Aya workflow)

---

## Task 14: P4d — Real Workload Evaluation

**Files:**
- Create: `guepard-shield-model/notebooks/p4d_real_workload.py`

**Setup:**
- nginx + wrk in Docker (docker-compose)
- 30 minutes normal HTTP traffic
- Collect syscall trace via eBPF (from Task 13 program) or via `strace -p` on nginx PID

**Outline:**
```
Step 1: Generate normal traffic trace
  - Docker: nginx + wrk → 30min, 10 req/s
  - Save raw syscall trace to data/real_workload/nginx_trace.jsonl
  - Each line: {timestamp_ns, pid, syscall, thread_id}

Step 2: Apply rule set
  - Load DT-Soft-T*-PERM (from results/p3/)
  - Slide window over nginx trace → build feature vectors → predict
  - Or: replay through eBPF program via BPF perf buffer (if C9 compiled)

Step 3: Compute FPR
  - All traffic is normal → any detection = false positive
  - FPR = FP_windows / total_windows
  - Target: < 5%
  - If FPR > 5%: analyze which syscall combos trigger rules (top-3 FP patterns)
    Propose threshold adjustment in results JSON (don't re-train)

Step 4: Save
  - results/p4/real_workload_fpr.json
    {fpr, total_windows, fp_windows, top_fp_patterns, threshold_adjustment_proposal}
```

---

## Task 15: P4.5 — Paper Table Generator

**Files:**
- Create: `guepard-shield-model/notebooks/p45_paper_tables.py`

**Outline:**
```python
# Load all artifacts
p2 = load_json("results/p2/")
p3 = load_json("results/p3/")
p4 = load_json("results/p4/")

# Generate LaTeX tables
# 1. Ablation table (Exp B + C combined):
#    Rows: 5 configs. Cols: Attack Fidelity ± std, Overall Fidelity, FPR, #Rules, #Conds.
generate_ablation_latex(p3["exp_b"], p3["exp_c"])  # → ablation_table.tex

# 2. Phase ablation table:
#    Rows: 3 policies. Cols: per-phase FPR (4 cols), Overall FPR, Attack Fidelity, #Rules.
generate_phase_latex(p3["phase_ablation"])  # → phase_table.tex

# 3. Teacher comparison table:
#    BiLSTM vs Transformer: F1, accuracy, training time.
generate_teacher_latex(p2["teacher_comparison"])  # → teacher_table.tex

# Generate figures
# 4. Temperature sweep curve: T vs Attack Fidelity → temperature_curve.pdf
# 5. Pareto frontier: depth vs Attack Fidelity vs #Rules → pareto.pdf
# 6. MITRE coverage heatmap: → mitre_heatmap.pdf
# 7. Latency CDF: → latency_cdf.pdf

# All outputs → results/paper/
```

Each `generate_*` function reads from artifacts, never hardcodes numbers.

---

## Execution Order & Critical Path

```
Task 1 (deps)
  → Task 2 (LID-DS loader) + Task 3 (phase segmenter) [parallel]
  → Task 4 (perm importance) + Task 5 (metrics) [parallel]
  → Task 6 (P2 Teacher training)  ← needs Tasks 2-5
  → Task 7 (Exp B DT)             ← needs Task 6 output
  → Task 8 (Exp C RuleFit)        ← parallel with Task 7
  → Task 9 (phase ablation)       ← needs Task 7 winner config
  → Task 10 (rule extractor + MITRE mapper)
  → Task 11 (rule analysis)       ← needs Task 10
  → Task 12 (cross-domain)        ← needs Task 7 winner model
  → Task 13 (eBPF compiler)       ← needs Task 7 winner DT
  → Task 14 (real workload)       ← needs Task 13 eBPF program
  → Task 15 (paper tables)        ← needs all results/
```

**Commit points:** After Task 6 (P2 complete), after Task 9 (P3 complete), after Task 14 (P4 complete), after Task 15 (P4.5 complete).
