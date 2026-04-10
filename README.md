# README.md

## Project Overview

**guepard-shield** is a master's research project: a Linux host-based intrusion detection system that monitors syscalls via eBPF, trains a Transformer Teacher to detect post-exploitation attacks, then distills the Teacher's knowledge into interpretable rules (Decision Tree, RuleFit, FIGS) deployable back as eBPF enforcement policies.

**Stack:** Rust + Aya (eBPF enforcement), Keras + JAX (Teacher training), imodels/scikit-learn/SHAP (rule extraction).

## Build & Run (Rust)

Build requires `bpf-linker` in PATH (for cross-compiling the eBPF bytecode):

```bash
# Build userspace + eBPF (eBPF is cross-compiled via build.rs automatically)
cargo build

# Run (requires root; .cargo/config.toml sets runner = "sudo -E")
cargo run

# Release build
cargo build --release
```

The `guepard-shield` build script (`build.rs`) calls `aya_build::build_ebpf` to cross-compile `guepard-shield-ebpf` for the BPF target. The resulting bytecode is embedded in the userspace binary via `include_bytes_aligned!`.

## Python (ML / Data)

Python workspace lives in `guepard-shield-model/`. Uses `uv` for package management. Python 3.13 required.

```bash
# Install dependencies
uv sync

# Run a notebook (jupytext format)
uv run notebooks/<path/to/notebook>.py

# Type-check
uv run ty check
```

Notebooks in `guepard-shield-model/notebooks/` use **jupytext** format — cells delimited by `# %%` comments, not `.ipynb` JSON.

## Architecture

### Rust Workspace

| Crate                   | Target           | Role                                                                                                            |
| ----------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
| `guepard-shield`        | host (userspace) | Loads eBPF bytecode, attaches tracepoint `raw_syscalls/sys_enter`, drives tokio async runtime                   |
| `guepard-shield-ebpf`   | `bpf` (kernel)   | `#![no_std]` eBPF program; hooks `sys_enter` tracepoint, logs/processes syscall events                          |
| `guepard-shield-common` | both             | Shared data structures; compiled with `#![no_std]`, feature-gated: `user` feature enables `std`-dependent impls |

**Data flow:** kernel eBPF program captures syscalls → passes events to userspace via eBPF maps → userspace processes/forwards for ML inference or rule matching.

### Python ML Pipeline

Located in `guepard-shield-model/`:

```
data/
├── raw/        # read-only raw datasets
├── processed/  # parsed datasets: DongTing, LID-DS-2019, LID-DS-2021
└── splits/     # train/val/test split indices

src/gp/
├── config.py               # paths and constants
├── data_loader/            # one loader per dataset → list[Recording]
└── diagnostic/             # integrity, seq_length, vocab stats utilities

notebooks/
├── diagnostic/             # EDA notebooks (one per dataset)
├── p1/                     # data pipeline, phase segmenter
└── p2/                     # Teacher training, temperature calibration
```

### Datasets

| Dataset     | Format                                                  | Scenarios |
| ----------- | ------------------------------------------------------- | --------- |
| LID-DS-2021 | `.sc` (rich: timestamp/pid/tid/args) + `.json` metadata | 15        |
| LID-DS-2019 | `.txt` (same fields as `.sc`), flat                     | 10        |
| DongTing    | `.log` (pipe-separated syscall names)                   | —         |

## Key Design Decisions

- `guepard-shield-common` must remain `no_std` compatible; use the `user` feature flag to gate any `std` usage
- eBPF programs use dual license `MIT/GPL` (required for kernel helper access)
- The workspace `default-members` excludes `guepard-shield-ebpf` to avoid attempting to build the BPF target with normal `cargo build`; the build script handles cross-compilation automatically
- Polars (not Pandas) for dataframes; Keras (with JAX backend) for Teacher training; imodels for interpretable surrogate models
- All shared Python logic lives in `src/gp/`; notebooks are thin wrappers that call into `src/gp/` and visualize
