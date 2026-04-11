# Transformer Architecture Ideas for P2 + P3 (Brainstorming)

**Date:** 2026-04-11  
**Status:** Brainstorming — not yet adopted into plan  
**Context:** Real-time model inference on syscall sequences is infeasible → rules are the deployment artifact. These options are candidate architectures for P2 (detection model) and directly shape how P3 (rule extraction) works.

---

## Core Insight

**The architecture determines the atom of analysis, which determines what rules look like:**

| Architecture            | Atom                                  | Rule reads as                                           |
| ----------------------- | ------------------------------------- | ------------------------------------------------------- |
| BiLSTM (current plan)   | Individual syscall hidden states      | "Syscall X at position N is anomalous"                  |
| Option 2 (Patch)        | Behavioral episode (16-syscall patch) | "Episode type A followed by episode type B → ALERT"     |
| Option 3 (Cross-Thread) | Thread role (concurrent process)      | "Thread type A co-occurring with thread type B → ALERT" |

Rules become more abstract and human-readable as the atom grows larger.

**Attention weights are first-class in Transformers, not side effects.** Unlike LSTM hidden states that require post-hoc SHAP, Transformer attention maps directly answer "which parts of the input matter." This makes the path from model → rule more principled.

---

## Option 2 — Hierarchical Patch Transformer (MAE-style)

### Intuition

Group syscalls into fixed-size "behavioral episodes" (patches), then learn how episodes transition. A normal web server:

```
[accept, recv, read, ...]  →  [send, write, ...]  →  [accept, recv, ...]
    patch: receive request      patch: send response    patch: repeat
```

When an attacker injects a shell, the cross-patch Transformer cannot predict the anomalous patch from its neighbors → high reconstruction loss → high anomaly score.

### Architecture

```
Level 1 — Patch Encoder (local, shared weights):
  512 syscalls → 32 patches x 16 syscalls each
  Small Transformer (2 layers) per patch → patch embedding (128-dim)

Level 2 — Cross-patch Transformer (global):
  32 patch embeddings as tokens
  Objective: Masked Patch Modeling — mask ~30% of patches, predict from context
  Anomaly score at inference = reconstruction loss over masked patches
```

### Trade-offs

| Pro                                                       | Con                                                  |
| --------------------------------------------------------- | ---------------------------------------------------- |
| Novel — MAE applied to syscall sequences is underexplored | More hyperparameters than next-token prediction      |
| Patch boundary → rule boundary (clean correspondence)     | Window size and patch size interact, requires tuning |
| Two-level attention: local + global anomaly signal        | Harder to implement than BiLSTM                      |

---

## Option 3 — Cross-Thread Transformer

### Intuition

Many attacks involve behavior chains **across threads**, not just anomalous syscalls within one thread:

```
Thread A (web handler):    recv → read → write → ...
Thread B (file worker):    open → write → ...
Thread C (injected shell): execve → dup2 → socket   ← attacker
```

BiLSTM Approach B detects "thread C has anomalous syscalls" but cannot model whether thread C's _existence_ is anomalous given thread A's behavior. Cross-Thread Transformer applies full self-attention across threads — it learns which thread interactions are normal.

### Architecture

```
Step 1 — Per-thread encoder (shared weights):
  Each thread: truncate/pad to 256 syscalls
  Causal Transformer → CLS token = thread embedding (128-dim)

Step 2 — Cross-thread Transformer:
  K thread embeddings as K tokens + CLS_recording
  Full self-attention across threads
  Anomaly score = CE loss from Step 1 + reconstruction loss from Step 2
```

High-thread recordings (ZipSlip=1,239 threads): k-means cluster threads by unigram frequency → reduce to K=8 cluster representatives before Step 2.

### Trade-offs

| Pro                                                                       | Con                                               |
| ------------------------------------------------------------------------- | ------------------------------------------------- |
| Most novel — cross-thread attention for AD is unexplored                  | Higher implementation complexity                  |
| Attention matrix directly encodes which thread interactions are anomalous | Smaller training set (one sample = one recording) |
| Fits multi-process attack scenarios in LID-DS well                        | Clustering fallback adds a preprocessing step     |

---

## P3 Rule Extraction: Combinations

The 4 rule extraction ideas from the brainstorming session each fit differently depending on which architecture is used.

---

### Idea 1 — Patch-type Trie (Option 2 only)

**What:** Build a Trie (prefix tree) over **patch type sequences** rather than individual syscalls.

**How:**

1. Cluster patch embeddings → K patch types (e.g., K=20: "Net_recv", "File_IO", "Process_spawn", ...)
2. Each recording becomes a short sequence of patch type IDs
3. Scan normal data → record allowed next-patch-type for each prefix → Trie of safe paths

**Rule example:**

> IF patch_sequence [Net_recv → File_IO] THEN allowed_next ∈ {Net_send, File_IO}. Otherwise → ALERT.

**Why better than syscall-level Trie (STIDE):** Branching factor drops from 336 syscalls to ~20 patch types. Trie is smaller, less brittle, and deployable as `BPF_MAP_TYPE_LPM_TRIE` in eBPF.

**Role in paper:** Baseline / eBPF deployment prototype. Not a primary contribution — it is essentially STIDE at patch granularity.

---

### Idea 2 — Behavioral State FSA (Option 2 + Option 3)

**What:** Cluster the architecture's embeddings into abstract states, then build a Finite State Automaton of allowed transitions from normal data.

**Option 2 — Patch-state FSA:**

1. Cluster patch embeddings → K behavioral states (e.g., "Web serving", "File I/O", "Process init")
2. Cross-patch attention weights directly give transition probabilities between states
3. FSA: nodes = behavioral states, edges = transitions observed in normal recordings
4. At inference: any transition not in FSA → ALERT

Rule example:

> State "Web serving" → allowed: {"Web serving", "Closing connection"}. Transition to "Shell execution" → ALERT.

**Option 3 — Thread-role Interaction Graph:**
Threads are concurrent, not sequential → FSA becomes a graph instead of a chain.

1. Cluster thread embeddings → K thread role types (e.g., "Web_handler", "DB_connector", "Shell_executor")
2. Cross-thread attention matrix gives expected co-occurrence patterns between roles
3. Graph: nodes = thread roles, edges = allowed co-occurrence (weighted by attention)
4. At inference: any thread role co-occurrence not in graph → ALERT

Rule example:

> Web_handler ↔ DB_connector: ALLOWED. Web_handler ↔ Shell_executor: ANOMALOUS → ALERT.

**Why this combination is the strongest:** Cross-patch/cross-thread attention IS the FSA — no post-hoc approximation needed. Attention matrix → automaton is a direct, principled extraction.

**Role in paper:** Primary P3 method. Best novelty-to-complexity ratio.

---

### Idea 3 — Thread-level Anchors (Option 3)

**What:** Apply the Anchors algorithm at thread role level to find minimal sufficient conditions for "Normal."

**Why thread level fixes the coverage problem:** At syscall level, one anchor rule covers a tiny fraction of inputs. At thread role level (K=8-15 types), the feature space is small enough that anchors have meaningful coverage.

**How:**

1. Classify each thread in a recording into a thread role type (from Option 3 clustering)
2. Run Anchors: find the minimal set of role presence/absence conditions that anchor the prediction
3. Output: compact If-Then rules over thread roles

Rule example:

> IF DB_connector ∈ recording AND Shell_executor ∉ recording → NORMAL (coverage=87%, precision=99%)

**Why useful:** Directly maps to MITRE ATT&CK — "Shell_executor" thread → T1059 Command Execution. Easy for analyst to review and approve.

**Role in paper:** Supplementary method for MITRE mapping and case studies. Not primary (coverage is still limited compared to FSA).

---

### Idea 4 — Phase-aware Rule Segmentation (Option 2 + Option 3)

**What:** Use different rule sets per process lifecycle phase instead of a single global rule set. Reduces FPR in noisy phases (startup) and tightens control in critical phases (active).

**Option 2 — Patch phase segmentation:**
Patch types naturally cluster into phases — startup patches are dominated by `open/mmap/execve`, active patches by `recv/send/read/write`. Phase boundary = when dominant patch type shifts. Build separate Trie or FSA per phase.

**Option 3 — Thread composition per phase:**
Phase is defined by thread composition rather than syscall rate:

| Phase    | Expected thread roles                          |
| -------- | ---------------------------------------------- |
| Startup  | Loader threads, initializer threads            |
| Active   | Web_handler, DB_connector, Logger (stable set) |
| Idle     | Monitor thread only                            |
| Shutdown | Cleanup threads, terminating workers           |

Shell_executor appearing in Active phase → ALERT even if it might legitimately appear during Startup (e.g., install scripts).

**Why this matters:** A single-policy model trained on all phases must tolerate startup noise (`execve, mmap` bursts), which loosens the rules for the active phase where attacks actually happen. Per-phase rules let you tighten active-phase policy independently.

**Role in paper:** Orthogonal contribution applicable on top of any primary method (FSA or Trie). Strong security motivation, easy to ablate (single-policy vs. per-phase comparison).

---

## Summary: Recommended Combinations

| Combination                           | Role                                                  | Novelty                              |
| ------------------------------------- | ----------------------------------------------------- | ------------------------------------ |
| Option 2 + Idea 2 (Patch FSA)         | **Primary P3 method**                                 | High — attention matrix as automaton |
| Option 3 + Idea 2 (Thread-role Graph) | **Alternative / ablation for multi-thread scenarios** | Very high — unexplored in literature |
| Option 3 + Idea 3 (Thread Anchors)    | Supplementary — MITRE mapping, case studies           | Medium                               |
| Option 2 or 3 + Idea 4 (Phase-aware)  | Orthogonal contribution, ablation study               | Medium — strong security motivation  |
| Option 2 + Idea 1 (Patch Trie)        | Baseline / eBPF deployment demo                       | Low — STIDE variant                  |

**Suggested paper structure:** Option 2 + Idea 2 as primary method. Option 3 + Idea 2 as the multi-thread variant. Idea 4 (phase-aware) as an orthogonal enhancement applied to both. Idea 1 as the simple baseline to compare against. Idea 3 in the analysis/case study section.

---

## Next Step

These remain candidate approaches until P2 begins. Key decision points:

1. **Hardware feasibility:** Patch Transformer and Cross-Thread Transformer are heavier than BiLSTM — profile training time on available GPU before committing.
2. **Pilot:** Compare BiLSTM vs. Causal Transformer (simplest GPT-style, same next-token objective) on AUROC first. If no gain over BiLSTM, Option 2/3 complexity may not be justified.
3. **Dataset dependency:** If a future dataset has labeled attack data in training (binary classification), Option 3 + Idea 3 (Thread Anchors) becomes significantly more powerful since anchor precision improves with direct attack supervision.

See `2026-04-09-p2-teacher-model-design.md` for the current BiLSTM plan that these options would replace.
