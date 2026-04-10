# %% [markdown]
# # Thread / Graph Structure — Diagnostic Analysis
#
# Assesses whether graph-based models (GNN, Graph Transformer) are viable
# for the Teacher model, given that each `.sc` file is a multi-threaded trace.
#
# **Questions answered:**
# 1. How many threads per recording? (graph size / node count)
# 2. Are per-thread syscall sequences long enough for node embeddings?
# 3. Are normal vs attack syscall-bigram graphs structurally different?
# 4. Is the attack thread distinguishable by its syscall distribution?
# 5. Which graph architecture is recommended?

# %% Setup
from gp.config import RESULTS_DIR
from gp.data_loader.lidds_2021_loader import LIDDS2021Dataset
from gp.diagnostic.thread_stats import ThreadStat, graph_model_summary
from rich.progress import track

OUT_DIR = RESULTS_DIR / "diagnostic" / "thread_graph"

dataset = LIDDS2021Dataset(shuffle=True, seed=42)
print(f"Dataset: {len(dataset)} recordings")

# %% Single-pass analysis
stat = ThreadStat()
for recording in track(dataset, total=len(dataset)):
    stat.analyze(recording)

# %% 1. Thread Count Distribution
# → How many graph nodes per recording? Median > 64 makes Graph Transformer expensive.
# → Per-scenario breakdown flags scenarios where full thread-graph is infeasible.
stat.report_thread_count_dist(save_dir=OUT_DIR)

# %% 2. Per-Thread Sequence Length
# → Are individual thread sequences long enough for BiLSTM / local Transformer node embeddings?
# → Short threads (< 10 syscalls) produce noisy node features.
stat.report_per_thread_seq_len(save_dir=OUT_DIR)

# %% 3. Syscall Bigram (Transition) Graph
# → Build syscall transition graph for normal vs attack recordings.
# → Low Jaccard similarity → graphs are structurally different → GNN discriminates well.
# → Attack-only edges and enriched transitions = the signal a GNN would learn.
stat.report_syscall_bigram_graph(save_dir=OUT_DIR)

# %% 4. Attack Thread Isolation
# → For attack recordings: how different is the attack-spawned thread (highest TID)
#   from the normal worker threads (KL divergence)?
# → High KL → thread-level node embeddings carry strong signal (BiLSTM per-thread suffices).
# → Low KL → cross-thread message-passing (GNN edges) needed to detect the anomaly.
stat.report_attack_thread_isolation(save_dir=OUT_DIR)

# %% 5. Architecture Recommendation
# → Synthesises findings into a table of viable graph architectures.
graph_model_summary(save_dir=OUT_DIR)
