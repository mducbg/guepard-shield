# P3 — Rule Extraction & Ablation Design

**Phase:** P3 (Tháng 5-6)
**Covers:** C3 (Exp B: Tree Family), C4 (Exp C: Rule Ensemble), C5 (Distillation Ablation), C6 (Phase-aware Ablation)
**Deliverable:** Rule sets + Ablation report (`ablation_table.csv`, plots, per-phase FPR breakdown)

---

## 1. Context & Inputs

P3 consumes P2 artifacts directly:

| Artifact | Path |
|---|---|
| Teacher checkpoint (winner) | `results/p2/best_teacher_lidds.ckpt` |
| Vocabulary | `results/p2/vocab.json` |
| TF-IDF vectorizer | `results/p2/vectorizer.joblib` |
| Temperature sweep results | `results/p2/temperature_sweep.json` |

T* is read from `temperature_sweep.json` at runtime — no hardcoding.

**No PyTorch Lightning in P3.** The teacher checkpoint is loaded once per notebook for a single inference pass (soft label extraction). All surrogate training is pure sklearn/imodels.

---

## 2. Notebook Structure

```
notebooks/p3/
├── config.py              # shared constants and hyperparams
├── utils.py               # 3 shared functions
├── dt.py                  # DT — Exp B + C5 ablation + C6 phase-aware
├── hstree.py              # HSTree — Exp B
├── figs.py                # FIGS — Exp B
├── rulefit.py             # RuleFit — Exp C
├── boosted_rules.py       # BoostedRules — Exp C
└── comparison.py          # loads all results_*.json → tables + plots

results/p3/
├── results_dt.json
├── results_hstree.json
├── results_figs.json
├── results_rulefit.json
├── results_boosted_rules.json
├── ablation_table.csv
├── ablation_plots.png
└── phase_fpr_plot.png
```

Each method notebook is fully self-contained and can be run independently.
`comparison.py` is the only notebook with a dependency — it requires all 5 result JSONs.

---

## 3. `config.py`

```python
from pathlib import Path

# --- Data & artifact paths ---
DATA_DIR        = Path("../../data/processed/LID-DS-2021")
OUTPUT_DIR      = Path("../../results/p3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_PATH      = Path("../../results/p2/vocab.json")
VECTORIZER_PATH = Path("../../results/p2/vectorizer.joblib")
WINNER_CKPT     = Path("../../results/p2/best_teacher_lidds.ckpt")
SWEEP_PATH      = Path("../../results/p2/temperature_sweep.json")
COMPARISON_PATH = Path("../../results/p2/teacher_comparison.json")

IN_DIST_SCENARIOS = [
    "CVE-2014-0160",
    "CVE-2017-7529",
    "CWE-89-SQL-injection",
    "Bruteforce_CWE-307",
    "EPS_CWE-434",
]

# --- Windowing (must match P2) ---
SEED            = 42
WINDOW_SIZE     = 64
STRIDE          = 12
MAX_WINDOWS     = 10
BATCH_SIZE      = 1024
NUM_WORKERS     = 4

# --- SHAP ---
SHAP_TOP_K      = 50          # top-K features selected per method

# --- Surrogate hyperparams (NEVER vary between distilled/direct — only one var at a time) ---
DT_MAX_DEPTH        = 5
HSTREE_MAX_DEPTH    = 5
FIGS_MAX_RULES      = 12
RULEFIT_MAX_RULES   = 50
BOOSTED_MAX_RULES   = 50
```

---

## 4. `utils.py` — Three Shared Functions

### 4.1 `load_teacher_and_extract`

Loads the teacher checkpoint, runs a single inference pass over the requested split,
and returns features + labels needed for surrogate training.

```python
def load_teacher_and_extract(
    ckpt_path: Path,
    vocab_path: Path,
    vectorizer_path: Path,
    comparison_path: Path,
    sweep_path: Path,
    corpus,
    window_config: WindowConfig,
    split_name: str,
    max_windows: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X          : (N, F) float32 TF-IDF feature matrix
        y_hard     : (N,)   int   ground-truth hard labels
        y_soft_T1  : (N, 2) float soft probs at T=1.0
        y_soft_Tstar: (N, 2) float soft probs at T* from sweep
    """
```

T* is read from `sweep_path` internally. Returns soft labels at both T=1.0 and T*
so callers get all training targets in one pass.

### 4.2 `compute_metrics`

```python
def compute_metrics(
    surrogate,
    X: np.ndarray,
    y_hard: np.ndarray,
    teacher_preds: np.ndarray,
    label: str = "run",
) -> dict:
    """
    Returns:
        overall_fidelity  : float  agreement with teacher on full set
        attack_fidelity   : float  agreement on teacher-predicted-attack windows
        fpr               : float  false positive rate vs ground truth
        n_rules           : int    leaf nodes (DT/HSTree/FIGS) or active rules (RuleFit/Boosted)
        n_conditions      : int    total split nodes — proxy for decision complexity
    """
```

`teacher_preds` is `argmax(y_soft_Tstar)` — passed in by the caller so the function
stays stateless and works for both distilled and direct variants uniformly.

### 4.3 `save_results`

```python
def save_results(results: dict, path: Path) -> None:
    """JSON dump with ISO timestamp added automatically."""
```

---

## 5. Method Notebooks — Structure and imodels API

Every method notebook follows this 5-step pattern:

```
1. Load artifacts (config, vocab, vectorizer, corpus, splits)
2. Extract (X, y_hard, y_soft_T1, y_soft_Tstar) via utils.load_teacher_and_extract()
3. SHAP: fit fast RF on X_train → shap.TreeExplainer → select top-K indices → X_shap
4. Train all variants → compute_metrics() for each
5. save_results() → results/p3/results_<method>.json
```

### 5.1 Variants trained per method

| Variant key | Training target | Features |
|---|---|---|
| `hard_full` | `y_hard` | full |
| `soft_T1_full` | `y_soft_T1[:, 1] > 0.5` (thresholded) | full |
| `soft_Tstar_full` | `y_soft_Tstar` (float or thresholded, see §5.3) | full |
| `soft_Tstar_shap` | `y_soft_Tstar` | SHAP top-K |

Hyperparams (`max_depth`, `max_rules`) are **identical** across all variants of the same method.
The only thing that changes is the training target and/or feature set.

### 5.2 imodels API Reference

**DecisionTreeClassifier** (`sklearn`)
```python
from sklearn.tree import DecisionTreeClassifier, export_text
dt = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=SEED)
dt.fit(X_train, y_soft_Tstar[:, 1] > 0.5)   # or float y for soft training
rules_text = export_text(dt, feature_names=feature_names)
n_rules = dt.get_n_leaves()
n_conditions = dt.tree_.node_count - dt.get_n_leaves()
```

**HSTreeClassifier** (`imodels`)
```python
from imodels import HSTreeClassifier
# Hierarchical shrinkage wraps a DT — same fit/predict API
hs = HSTreeClassifier(max_leaf_nodes=2**DT_MAX_DEPTH, random_state=SEED)
hs.fit(X_train, y)
# Access underlying DT:
inner_dt = hs.estimator_
rules_text = export_text(inner_dt, feature_names=feature_names)
```

**FIGSClassifier** (`imodels`)
```python
from imodels import FIGSClassifier
figs = FIGSClassifier(max_rules=FIGS_MAX_RULES)
figs.fit(X_train, y)
figs.print_rules(feature_names=feature_names)   # human-readable
# figs.trees_ → list of small DTs (one per additive component)
n_rules = sum(t.get_n_leaves() for t in figs.trees_)
n_conditions = sum(t.tree_.node_count - t.get_n_leaves() for t in figs.trees_)
```

**RuleFitClassifier** (`imodels`)
```python
from imodels import RuleFitClassifier
# RuleFit expects integer labels — use teacher argmax, not raw soft probs
rf = RuleFitClassifier(max_rules=RULEFIT_MAX_RULES, random_state=SEED)
rf.fit(X_train, y_int, feature_names=feature_names)
active_rules = rf.rules_[rf.rules_["coef"].abs() > 0]
n_rules = len(active_rules)
n_conditions = active_rules["rule"].str.count("&").add(1).sum()
```

**BoostedRulesClassifier** (`imodels`)
```python
from imodels import BoostedRulesClassifier
# Also expects integer labels
br = BoostedRulesClassifier(n_estimators=BOOSTED_MAX_RULES, random_state=SEED)
br.fit(X_train, y_int)
# br.rules_ → list of 1-level DT stumps
n_rules = len(br.rules_)
n_conditions = n_rules   # each stump has exactly 1 split condition
```

### 5.3 Soft Label Handling per Method

| Method | Accepts float y? | How to use soft labels |
|---|---|---|
| DT | Yes (`criterion='entropy'`) | pass `y_soft[:, 1]` directly as float targets |
| HSTree | Yes (wraps DT) | same as DT |
| FIGS | Yes | pass `y_soft[:, 1]` directly |
| RuleFit | No | threshold: `(y_soft[:, 1] > 0.5).astype(int)` |
| BoostedRules | No | threshold: `(y_soft[:, 1] > 0.5).astype(int)` |

For RuleFit/BoostedRules the "soft" information is still present indirectly —
temperature scaling shapes the training distribution by moving borderline windows
across the threshold boundary, giving the surrogate richer boundary supervision
than hard labels alone.

---

## 6. `dt.py` — Phase-aware Ablation (C6)

After single-policy variants are complete, `dt.py` continues with C6:

```python
# Phase label assignment
for meta in train_metas:
    ts = read_sc_timestamps(meta.file_path)
    phase_labels[meta.seq_id] = segment_phases(ts)

# Build per-phase index: map each window → its phase
# Then filter X_train / y_soft_Tstar by phase

per_phase_results = {}
for phase in ["startup", "active", "idle", "shutdown"]:
    X_phase = X_train[phase_mask[phase]]
    y_phase = y_soft_Tstar[phase_mask[phase]]
    y_hard_phase = y_hard[phase_mask[phase]]

    dt_phase = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=SEED)
    dt_phase.fit(X_phase, y_phase[:, 1] > 0.5)

    per_phase_results[phase] = compute_metrics(
        dt_phase, X_phase, y_hard_phase, teacher_preds[phase_mask[phase]]
    )

# combined_fpr: weighted average by phase window count
combined_fpr = sum(
    per_phase_results[p]["fpr"] * phase_counts[p] for p in ALL_PHASES
) / total_windows
```

Phase-window mapping: for each (seq_id, win_idx) pair in `flat_index`, the phase is
the majority phase label of the tokens in that window — computed by calling
`segment_phases(timestamps)` on the parent recording and taking `Counter(labels[start:end]).most_common(1)`.

---

## 7. Results JSON Schema

```json
{
  "timestamp": "<ISO-8601>",
  "method": "dt",
  "dataset": "LID-DS-2021",
  "hyperparams": { "max_depth": 5, "shap_top_k": 50 },

  "single_policy": {
    "hard_full":       { "overall_fidelity": 0.0, "attack_fidelity": 0.0, "fpr": 0.0, "n_rules": 0, "n_conditions": 0 },
    "soft_T1_full":    { "overall_fidelity": 0.0, "attack_fidelity": 0.0, "fpr": 0.0, "n_rules": 0, "n_conditions": 0 },
    "soft_Tstar_full": { "overall_fidelity": 0.0, "attack_fidelity": 0.0, "fpr": 0.0, "n_rules": 0, "n_conditions": 0 },
    "soft_Tstar_shap": { "overall_fidelity": 0.0, "attack_fidelity": 0.0, "fpr": 0.0, "n_rules": 0, "n_conditions": 0 }
  },

  "per_phase": {
    "startup":      { "overall_fidelity": 0.0, "attack_fidelity": 0.0, "fpr": 0.0, "n_rules": 0, "n_conditions": 0 },
    "active":       { "overall_fidelity": 0.0, "attack_fidelity": 0.0, "fpr": 0.0, "n_rules": 0, "n_conditions": 0 },
    "idle":         { "overall_fidelity": 0.0, "attack_fidelity": 0.0, "fpr": 0.0, "n_rules": 0, "n_conditions": 0 },
    "shutdown":     { "overall_fidelity": 0.0, "attack_fidelity": 0.0, "fpr": 0.0, "n_rules": 0, "n_conditions": 0 },
    "combined_fpr": 0.0
  }
}
```

`per_phase` is present only in `results_dt.json`. All other method JSONs omit it.

---

## 8. `comparison.py` — Outputs

Loads all 5 result JSONs and produces:

**`ablation_table.csv`** — C5 ablation (rows = method × variant, columns = metrics):

| method | variant | attack_fidelity | overall_fidelity | fpr | n_rules | n_conditions |
|---|---|---|---|---|---|---|
| DT | hard_full | ... | ... | ... | ... | ... |
| DT | soft_T1_full | ... | ... | ... | ... | ... |
| ... | | | | | | |

**`ablation_plots.png`** — two subplots:
1. Grouped bar chart: Attack Fidelity for all method × variant combinations
2. Scatter: Attack Fidelity vs n_conditions (Pareto frontier), colored by distilled/direct

**`phase_fpr_plot.png`** — from `results_dt.json` only:
- Bar chart: FPR per phase (startup/active/idle/shutdown) for single-policy vs per-phase DT-Soft-T*

---

## 9. Open Questions (deferred to implementation)

- **Phase-window majority vote:** confirm `Counter(labels[start:end]).most_common(1)` is the right aggregation vs "any attack syscall in window" convention used in `lidds_label_fn`.
- **SHAP backend:** `shap.TreeExplainer` on the fast RF — confirm compatible with sparse TF-IDF output (may need `.toarray()` first).
- **RuleFit/BoostedRules float y:** empirically test whether passing soft probs directly raises an error vs silently rounding — use thresholding if so.
- **Recovery script for P2:** `teacher_comparison.json` and `best_teacher_lidds.ckpt` are missing from `results/p2/` — must be created before P3 notebooks can run (see previous analysis).
