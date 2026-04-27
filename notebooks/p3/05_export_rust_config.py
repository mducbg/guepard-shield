# %%
import json
from pathlib import Path

from gp.rules.rust_codegen import RustConfigExporter
from gp.rules.decision_set import GreedyDecisionSet

# %%
RULES_JSON = Path("results/p3_rule_extraction/rules/decision_set_rules.json")
VOCAB_PATH = Path("results/eda_cross_dataset/vocab_lidds2021_train.txt")
OUTPUT_CONFIG = Path("results/p3_rule_extraction/rust/rule_config.json")

# %%
with open(VOCAB_PATH) as f:
    vocab = [line.strip() for line in f if line.strip()]

with open(RULES_JSON) as f:
    rules_data = json.load(f)

ds = GreedyDecisionSet.from_dict(rules_data, feature_names=[r["feature_name"] for r in rules_data])

# %%
exporter = RustConfigExporter(
    vocab=vocab,
    dangerous_syscalls=[
        "execve", "connect", "socket", "openat", "open",
        "chmod", "chown", "kill", "ioctl", "mmap", "mprotect",
    ],
    window_size=1000,
)
exporter.export(ds, OUTPUT_CONFIG)
print(f"\nNext steps (P4):")
print("  eBPF tracepoint counts syscalls into per-CPU BPF maps")
print("  Userspace Rust reads maps and evaluates rules against rule_config.json")
