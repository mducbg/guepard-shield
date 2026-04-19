# guepard-shield-model — Project Walkthrough

Quick-start document for resuming work in a new session. Covers project goal,
dataset facts, codebase layout, what has been built, and known gotchas.

**Current State:** Only Phase 1 (EDA) is completed. No model training code or preprocessed data exists.

---

## 1. Project Goal

**guepard-shield** is a master's research HIDS (Host-based Intrusion Detection
System) for Linux. The ML pipeline has four phases:

| Phase                 | Goal                                                                                                                      | Status                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **P1 — EDA**          | Understand datasets; explore data structure                                                                               | **Done**                                    |
| **P2 — Teacher**      | Transformer (NTP) on LID-DS-2021; target AUROC ≥ 0.95                                                                       | **Not started**  |
| **P3 — Distillation** | Extract interpretable rules from Teacher soft labels; convert to eBPF enforcement policy | **Not started** |
| **P4 — Deployment** | Compile rules to eBPF programs; measure enforcement latency on real workloads | **Not started** |

End-to-end flow: eBPF captures syscalls → Teacher Transformer calculates anomaly scores → rules distilled from Teacher → rules deployed as eBPF policy.

---

## 2. Datasets

### LID-DS-2021 (primary training dataset)

| Split | Normal | Exploit | Total  |
| ----- | ------ | ------- | ------ |
| train | 3,149  | 0       | 3,149  |
| val   | 885    | 0       | 885    |
| test  | 11,341 | 1,815   | 13,156 |

- **Format:** `.sc` text files (tab-separated) + `.json` metadata per recording
- **Note:** Training data is 100% normal (only test has exploits)

### LID-DS-2019 (cross-domain validation)

- ~11,000 recordings across 10 attack scenarios
- Different system call naming convention than 2021 version

### DongTing (cross-domain validation)

- 18,966 sequences
- Only contains syscall names (no arguments)
- Chinese dataset for additional validation

---

## 3. Codebase Layout

```
guepard-shield-model/
├── pyproject.toml          # uv project dependencies
├── src/gp/
│   ├── config.py           # Configuration and paths
│   ├── data_loader/
│   │   ├── recording.py            # Recording dataclass + .sc parser
│   │   ├── lidds_2021_loader.py    # LID-DS-2021 streaming loader
│   │   ├── lidds_2019.py           # LID-DS-2019 loader
│   │   ├── lidds_2021.py           # LID-DS-2021 main loader
│   │   └── dongting.py             # DongTing loader
│   └── diagnostic/                 # EDA diagnostic scripts
│       ├── stats.py                # General statistics utilities
│       ├── dongtingstats.py        # DongTing dataset statistics
│       └── lidds2019stats.py       # LID-DS-2019 statistics
├── notebooks/
│   └── p1/                          # EDA scripts only
│       ├── eda_lidds2021.py        # LID-DS-2021 EDA
│       ├── eda_lidds2019.py        # LID-DS-2019 EDA
│       └── eda_dongtingds.py       # DongTing EDA
└── results/
    ├── eda_lidds2021/              # LID-DS-2021 EDA outputs
    ├── eda_lidds2019/              # LID-DS-2019 EDA outputs
    ├── eda_dongting/               # DongTing EDA outputs
    └── eda_cross_dataset/          # Cross-dataset comparison
```

**What's NOT in the codebase:**
- ❌ Model training code (P2)
- ❌ Rule extraction code (P3)
- ❌ Preprocessed data files (.npy, array_record)
- ❌ Model checkpoints
- ❌ eBPF deployment code (P4)

---

## 4. P1 — EDA (Completed)

### 4.1 What Was Done

**Data Understanding:**
- Analyzed all three datasets (LID-DS-2021, LID-DS-2019, DongTing)
- Documented sequence lengths, class distribution, thread structure
- Identified OOV syscalls across datasets
- Analyzed attack timing patterns

**Key Findings:**
- **Sequence lengths:** Varies widely (10s to 100,000s of syscalls)
- **Vocabulary overlap:** ~70% syscall names overlap between 2019 and 2021 versions
- **Attack distribution:** Different scenarios in each dataset
- **Thread structure:** Multi-threaded recordings need special handling

### 4.2 EDA Outputs

Located in `results/eda_*/`:
- Sequence length distributions (CSV + PNG)
- Vocabulary analysis (top-20 frequencies, OOV lists)
- Attack scenario breakdowns
- Cross-dataset compatibility matrices
- Thread structure statistics
- Attack timing patterns (offset from warmup)

---

## 5. Run Order

```bash
cd guepard-shield-model

# Run EDA for all datasets
uv run python notebooks/p1/eda_lidds2021.py
uv run python python notebooks/p1/eda_lidds2019.py
uv run python notebooks/p1/eda_dongtingds.py
```

Results are saved to `results/eda_*/`.

---

## 6. What's Next (Future Work)

When ready to start P2 (Model Training), you will need to implement:

1. **Data Preprocessing Pipeline:**
   - Sliding window with tokenization
   - Vocabulary building (min_freq filtering)
   - Converting to model-ready format (.npy or tf.data)
   - Streaming to minimize RAM usage

2. **Transformer Model (NTP):**
   - Architecture: Embedding → Causal Transformer → Linear head
   - Next Token Prediction task on normal sequences
   - Anomaly scoring via average NLL per window

3. **Training Infrastructure:**
   - Loss: Cross-entropy with PAD masking
   - Optimizer: AdamW with cosine decay
   - Mixed precision (bfloat16)
   - Checkpoint saving

---

## 7. Known Gotchas

| Issue | Fix |
| :--- | :--- |
| **Memory usage** | EDA scripts load full recordings into RAM. Close other applications when running. |
| **Vocabulary mismatch** | LID-DS-2019 and 2021 use different syscall naming conventions. |
| **Dataset paths** | Ensure `cfg.sc_dir` in `config.py` points to the correct location. |
| **Exit syscalls only** | The loader filters for exit events (`<` in column 6 of .sc files). |

---

## 8. Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `src/gp/config.py` | All hyperparameters and paths |
| `src/gp/data_loader/lidds_2021.py` | Main LID-DS-2021 loader |
| `src/gp/data_loader/recording.py` | Recording dataclass and .sc parser |
| `notebooks/p1/eda_lidds2021.py` | Primary dataset EDA |
| `results/eda_lidds2021/` | EDA outputs and statistics |
