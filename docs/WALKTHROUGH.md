# Project Walkthrough & Progress Report

This document tracks the actual implementation progress of the RuleDistill project. It serves as a technical bridge between sessions.

---

## ✅ Phase 1: EDA & Data Preprocessing (Completed)
- **Datasets Analyzed:** LID-DS-2021 (Primary), LID-DS-2019, DongTing.
- **Key Findings:** 
    - Syscall sequences are highly repetitive (normal server traffic).
    - Window size of 1000 is optimal for capturing attack context.
    - Test set is significantly larger than Train/Val and includes temporal metadata.
- **Data Pipeline:**
    - Built specialized loaders for `.sc` and `.json` formats.
    - Implemented **Exact Deduplication** (using `np.unique`) to reduce noise in training data.
    - Implemented **Window-level Labeling** using exploit timestamps from JSON metadata.

## ✅ Phase 2: Teacher Model Training (Completed)
- **Architecture:** 
    - **Model:** `SyscallTransformer` (PyTorch Lightning).
    - **Specs:** 4 Layers, 8 Attention Heads, Causal Masking.
    - **Optimization:** Register-buffer mask caching, CosineAnnealingLR.
- **Training Strategy:**
    - **Dynamic Random Subsampling:** Each epoch picks 50 fresh random windows per recording.
    - **Performance:** Reduced epoch time from 1h50m to ~12m on RTX 3060 via subsampling and `16-mixed` precision.
- **Evaluation Results** (mean aggregation, stride=1000):
    - **Window-level AUROC:** 0.8652
    - **Recording-level AUROC:** 0.9832
    - **Best Window F1:** 0.9033 (threshold=0.0027, unconstrained)
    - **Window @ FPR=1%:** F1=0.8704 (thr=1.06)
    - **Window @ FPR=5%:** F1=0.8702 (thr=0.81)
    - **Recording @ FPR=2%:** Recall=93.7%, F1=0.9673 (thr=0.74)
    - **Recording @ FPR=6%:** Recall=98.0%, F1=0.9887 (thr=0.60)
- **Key Findings:**
    - **Model comparison:** A retrained model with `max_windows_train=None` (using all windows, val_loss=0.2748) was evaluated against the original model (`max_windows_train=50`, val_loss=0.3455). Surprisingly, the **original model with higher val_loss performed significantly better** on window-level detection (AUROC 0.8652 vs 0.7971). This indicates that subsampling acts as an implicit regularizer that improves anomaly generalization.
    - **Max/p95 aggregation failed:** Window AUROC dropped to 0.18 with max aggregation because normal windows contain legitimate rare syscalls with higher max-NLL than many attack windows.
- **Key Decision:** 
    - **Keep the original model** (`best-transformer-epoch=29-val_loss=0.3455.ckpt`) as the Teacher for Phase 3.
    - Tier 2 (One-Class Autoencoder) design doc is archived at `docs/plans/tier2-oneclass-design.md` for future reference if needed.
- **Artifacts:** Results saved to `/results/evaluation/transformer/`

## 🔄 Phase 3: Rule Distillation (Ready to Start)
- **Goal:** Extract interpretable logic from the Teacher's Anomaly Scores (NLL).
- **Current Setup:** 
    - Teacher model trained and checkpointed at `results/checkpoints/transformer/`.
    - Preprocessed data has detailed attack labels (0/1) for every window.
    - Evaluation results confirm **recording-level AUROC=0.98** — sufficient for clean pseudo-label generation.
- **Recommended P3 Protocol:**
    1. Use checkpoint: `best-transformer-epoch=29-val_loss=0.3455.ckpt` (original model)
    2. Score all test recordings with Teacher (`mean` aggregation).
    3. Generate pseudo-labels:
        - **Positive (Attack):** recording score >= 0.74 (~94% recall, ~2% FPR)
        - **Negative (Normal):** recording score <= 0.30
        - **Gray zone (0.30-0.74):** Discard or use as unlabeled
    4. Extract eBPF-friendly features (syscall histogram + discriminative n-grams) from windows.
    5. Distill rules from positive/negative recordings using Greedy Decision Set.
    6. Evaluate rule fidelity vs. Teacher (>95% target) and rule FPR (<1% target).
    7. Export rules to JSON config for Rust/Aya ingestion (existing `guepard-shield-ebpf` crate).

---

## 🛠 Working with the ML Pipeline

### 1. Data Preparation
- **Train/Val:** `uv run python notebooks/p2/preprocess_lidds2021.py` (De-duplicated)
- **Test:** `uv run python notebooks/p2/preprocess_test_lidds2021.py` (Detailed labels)

### 2. Model Execution
- **Train:** `uv run python notebooks/p2/train_transformer.py`
- **Evaluate:** `uv run python notebooks/p2/evaluate_transformer.py`

### 3. Rule Extraction (P3)
- **Generate Pseudo-Labels:** `uv run python notebooks/p3/01_generate_pseudo_labels.py`
- **Extract Features:** `uv run python notebooks/p3/02_extract_features.py`
- **Learn Rules:** `uv run python notebooks/p3/03_learn_rules.py`
- **Evaluate Rules:** `uv run python notebooks/p3/04_evaluate_rules.py`
- **Export Rust Config:** `uv run python notebooks/p3/05_export_rust_config.py`
- **Map MITRE:** `uv run python notebooks/p3/06_map_mitre.py`

### 4. Key Paths
- **Checkpoints:** `results/checkpoints/transformer/`
- **Processed Data:** `data/processed/lidds2021/`
- **P3 Rules:** `results/p3_rule_extraction/rules/`
- **P3 Rust Config:** `results/p3_rule_extraction/rust/rule_config.json`
- **Source Code:** `guepard-shield-model/gp/`
- **Global Config:** Hyperparameters are managed in `gp.config` as global variables.

---

## ⚠️ Important Technical Notes
- **Workspace:** Always run commands from the project root.
- **Memory:** If OOM occurs, check `batch_size` (current: 64) and `accumulate_grad_batches` (current: 2).
- **Type Safety:** The project uses `uv run ty check` for strict type validation across both `src` and `notebooks`.
