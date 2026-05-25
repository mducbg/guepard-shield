# P3 Design: DFA Distillation từ Transformer

**Ngày:** 2026-05-19  
**Trạng thái:** Approved — sẵn sàng implement  
**Phụ thuộc:** `results/p2/checkpoints/best.ckpt`, `data/processed/p2/train_X.npy`

---

## 1. Mục tiêu

Trích xuất một DFA từ Transformer đã huấn luyện (Phase 2 teacher) và xuất ra `results/p3/dfa_config.json` — artifact sẵn sàng nạp vào eBPF maps ở Phase 4.

**Artifact đầu ra:** `dfa_config.json` + bảng metrics (FPR, TPR, fidelity, kích thước DFA)

**Tiêu chí thành công:** tìm được config (K, strategy, θ) cho FPR thấp nhất với TPR ≥ 0.5

---

## 2. Pipeline & Data Flow

```
train recordings
    │
    ▼  notebooks/p3/extract_states.py
results/p3/hidden_states/
    train_H.npy          [M × 128]   ← h_t tại mỗi syscall (stride-1 dense window)
    train_meta.npy       [M × 2]     ← (rec_id, pos_in_rec) để ghép cặp liên tiếp
    │
    ▼  notebooks/p3/cluster.py       (cho mỗi K ∈ {64, 128, 256, 512})
results/p3/clusters/K{k}/
    labels.npy           [M]
    centroids.npy        [K × 128]
    │
    ▼  notebooks/p3/build_dfa.py     (cho mỗi K, mỗi strategy)
results/p3/dfa/K{k}_S{s}[_t{theta}]/
    transitions.npz                  ← sparse transition relation/function
    nd_rate.txt                      ← % (state, token) có > 1 destination (diagnostic)
    │
    ▼  notebooks/p3/eval_dfa.py
results/p3/metrics/
    grid_search.csv                  ← K, strategy, θ, FPR, TPR, fidelity, n_states, n_trans
    │
    ▼  notebooks/p3/export_dfa.py   (best config: min FPR với TPR ≥ 0.5)
results/p3/dfa_config.json
```

Mỗi bước cache độc lập — có thể chạy lại từ bất kỳ bước nào mà không phải làm lại extraction.

---

## 3. Cấu trúc code

### 3.1 Thư viện mới: `gp/dfa/`

```
guepard-shield-model/gp/dfa/
    __init__.py
    transitions.py     ← TransitionBuilder (NFA building + S1–S4 resolution)
    evaluate.py        ← DFAEvaluator (FPR, TPR, fidelity)
    export.py          ← DFAExporter (→ dfa_config.json)
```

Cùng pattern với `gp/data_loader/` và `gp/diagnostic/`.

### 3.2 Scripts (thin wrappers, mỗi bước 1 file)

```
notebooks/p3/
    extract_states.py
    cluster.py
    build_dfa.py
    eval_dfa.py
    export_dfa.py
```

---

## 4. Chi tiết từng component

### 4.1 `gp/dfa/transitions.py` — `TransitionBuilder`

```python
class TransitionBuilder:
    def __init__(self, labels: np.ndarray, meta: np.ndarray, vocab_size: int)
    # labels: [M] cluster id; meta: [M, 2] (rec_id, pos)

    def build_nfa(self) -> dict[tuple[int,int], Counter[int]]
    # Key: (src_state, token), Value: Counter{dest_state: count}
    # Chỉ ghép cặp (t, t+1) có cùng rec_id

    def nd_rate(self) -> float
    # % cặp (state, token) có > 1 unique destination — in ra trước khi commit K

    def resolve_s1(self) -> dict[tuple[frozenset,int], frozenset]
    # Subset construction NFA→DFA. Cảnh báo nếu |states| > 10×K (state explosion).

    def resolve_s3(self) -> dict[tuple[int,int], int]
    # Majority voting: giữ destination phổ biến nhất cho mỗi (state, token)

    def resolve_s4(self, theta: float) -> dict[tuple[int,int], int]
    # Statistical pruning: chỉ giữ nhánh có tần suất ≥ theta
    # Các cặp (state, token) còn lại sau pruning → không có entry → REJECT tại runtime
```

**S2 (tăng K):** được cover bởi grid search qua K — không cần strategy riêng.

### 4.2 `gp/dfa/evaluate.py` — `DFAEvaluator`

**Evaluation mode: per-window, stateless** — mỗi window bắt đầu từ state 0. Nhất quán với P2 và đủ cho luận văn (stateful evaluation theo thread sẽ làm ở P4).

```python
class DFAEvaluator:
    def fpr_tpr(
        self,
        transitions: dict,           # {(state, token): next_state}
        test_X: np.ndarray,          # [N_test, W]
        test_labels: np.ndarray,     # [N_test] 0/1
    ) -> tuple[float, float]
    # Simulate token-by-token trên mỗi window
    # REJECT nếu bất kỳ transition nào không có trong bảng
    # FPR = reject_count / normal_windows
    # TPR = reject_count / attack_windows

    def fidelity(
        self,
        transitions: dict,
        val_X: np.ndarray,           # [N_val, W]
        teacher_decisions: np.ndarray,   # bool [N_val]: teacher NLL > oracle_tau
    ) -> float
    # Agreement rate DFA vs teacher trên val set
```

### 4.3 `gp/dfa/export.py` — `DFAExporter`

```python
class DFAExporter:
    def to_json(
        self,
        transitions: dict,
        centroids: np.ndarray,    # [K, 128] — cho P4 nếu cần re-assign
        vocab: dict[str, int],
        state_freqs: np.ndarray,  # [K] tần suất mỗi state trong train
        edge_percentile: int = 5, # states dưới ngưỡng này → EDGE tier
        metadata: dict = {},
    ) -> dict
```

---

## 5. Grid Search

| K | S1 | S3 | S4(θ=0.80) | S4(θ=0.90) | S4(θ=0.95) | S4(θ=0.99) |
|---|:--:|:--:|:----------:|:----------:|:----------:|:----------:|
| 64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 128 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 256 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 512 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Tổng: 24 configs** (4 K × 6 strategies).

**S1 cảnh báo:** nếu `|states_S1| > 10 × K`, report state explosion — ghi metrics nhưng không export.

**Chọn best:** `min FPR` với ràng buộc `TPR ≥ 0.5`.

---

## 6. Format `dfa_config.json`

```json
{
  "metadata": {
    "K": 128,
    "strategy": "S4",
    "theta": 0.95,
    "vocab_size": 102,
    "n_states": 128,
    "n_transitions": 9843,
    "fpr": 0.012,
    "tpr": 0.71,
    "fidelity": 0.94
  },
  "vocab": {
    "read": 3, "write": 4, "mmap": 5
  },
  "state_tiers": {
    "1": "EDGE", "17": "EDGE"
  },
  "transitions": {
    "0": { "3": 1, "4": 2, "5": 0 },
    "1": { "3": 3, "4": 1 }
  }
}
```

- `transitions[state][token] = next_state` — nếu không có entry → **REJECT** tại runtime
- `state_tiers` chỉ list state EDGE (NORMAL là default, tiết kiệm space)
- Format sẵn sàng đọc vào BPF map ở P4

---

## 7. Quyết định thiết kế quan trọng

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Hidden state scope | Stride-1 dense windows, last-token h | Context đầy đủ W cho mọi h_t; per-syscall DFA |
| Clustering | K-Means, K ∈ {64,128,256,512} | Đơn giản, interpretable, phù hợp với transition lookup |
| Resolution | Tất cả S1/S3/S4, grid search | So sánh thực nghiệm, chọn best theo data |
| Evaluation | Per-window stateless | Nhất quán với P2; stateful theo thread dành cho P4 |
| Baseline (STIDE) | Không làm | Ngoài scope luận văn hiện tại |
| Hyperparameters | Cố định tại training time | Không cần chỉnh ngưỡng tại deployment (khác với τ của P2) |

---

## 8. Diagnostic bắt buộc trước khi grid search

Trước khi chạy full grid, `build_dfa.py` in ra `nd_rate` cho mỗi K:

```
K=64:  nd_rate=42.3%  ← cao, S4 sẽ prune nhiều
K=128: nd_rate=28.1%
K=256: nd_rate=15.4%
K=512: nd_rate= 8.2%  ← thấp, S1 khả thi hơn
```

Nếu nd_rate > 50% ở mọi K → K-Means không nắm được state, cần re-evaluate approach.
