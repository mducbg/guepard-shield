# Hướng dẫn Dự án & Báo cáo Tiến độ

Tài liệu này theo dõi tiến độ triển khai thực tế của dự án Guepard Shield. Nó đóng vai trò như một cầu nối kỹ thuật giữa các phiên làm việc.

---

## ✅ Giai đoạn 1: EDA & Tiền xử lý dữ liệu (Đã hoàn thành)

- **Các bộ dữ liệu đã phân tích:** LID-DS-2021 (Chính), LID-DS-2019, DongTing.
- **Các phát hiện chính:**
  - Các chuỗi syscall có tính lặp lại rất cao (lưu lượng server bình thường).
  - Kích thước cửa sổ (Window size) `W` là một siêu tham số (hyperparameter) quan trọng để nắm bắt ngữ cảnh tấn công.
  - Tập Test lớn hơn đáng kể so với tập Train/Val và bao gồm cả metadata về thời gian.
- **Pipeline Dữ liệu:**
  - Xây dựng các loader chuyên dụng cho các định dạng `.sc` và `.json`.
  - Triển khai **Khử trùng lặp chính xác (Exact Deduplication)** (sử dụng `np.unique`) để giảm nhiễu trong dữ liệu huấn luyện.
  - Triển khai **Gán nhãn cấp độ cửa sổ (Window-level Labeling)**: một window là **ATTACK** nếu timestamp của syscall cuối trong window ≥ thời điểm exploit sớm nhất trong bản ghi (từ metadata JSON). Normal ngược lại.

---

## ✅ Giai đoạn 2: Mô hình Teacher (Đã hoàn thành)

- **Kiến trúc:** Decoder-only Transformer (phong cách LLaMA), pre-norm, RoPE, RMSNorm, GELU FFN, weight tying.
  - d_model=128, 4 layers, 4 heads, d_ff=512, window_size=64, vocab_size=102
- **Huấn luyện:** PyTorch Lightning, AdamW + cosine LR decay + 5% warmup, AMP 16-mixed.
  - Dataset: LID-DS-2021, toàn bộ 3,149 bản ghi train, W=64, S=32, không giới hạn số window.
  - Dừng lại ở epoch 7 (val_loss tốt nhất=0.3686 tại epoch 1, early stopping patience=5).
- **Điểm bất thường (Anomaly score):** Last-token NLL — $\text{score} = -\log P(s_W \mid s_1, \ldots, s_{W-1})$
- **Inference (Dự đoán):** Lightning `Trainer.predict()` + `predict_step` (lấy index token cuối cùng có nhận thức về PAD). Điểm số được lưu cache vào `results/p2/scores/{test_last,test_max,test_labels}.npy` sau lần chạy đầu tiên để tránh việc phải chạy lại inference cho 99 triệu window.

### Kết quả trên tập test LID-DS-2021

**Cấp độ Cửa sổ (Window-level)** (99,855,329 windows — 17.2% attack):

| Điểm số (Score) | AUROC      | PR-AUC     | Oracle F1 | Oracle τ | FPR@oracle |
| --------------- | ---------- | ---------- | --------- | -------- | ---------- |
| Last-token NLL  | **0.8503** | **0.7093** | **0.80**  | 2.606    | 2.85%      |
| Max-window NLL  | 0.35\*     | 0.19       | 0.08      | 4.000    | 57.4%      |

\*Max-window NLL có sự tương quan nghịch (anti-correlated) ở cấp độ window (AUROC < 0.5): các chuỗi tấn công sử dụng các syscall phổ biến trong các ngữ cảnh bất thường thay vì các token hiếm gặp đơn lẻ, do đó đỉnh NLL trong một cửa sổ bị chi phối bởi các syscall bình thường nhưng hiếm gặp.

Ngưỡng vận hành (Operational threshold) (val 99.5th pct, τ=5.404): F1≈0.01 — hạn chế mang tính cấu trúc của việc hiệu chuẩn ngưỡng, không phải là thất bại của mô hình. Phân phối điểm số val có mean=0.30, p99=4.34; sự phân tách attack/normal yêu cầu τ≈2.6, mức này nằm dưới phân vị 99 của tập bình thường.

**Cấp độ Bản ghi (Recording-level)** (13,156 recordings — 13.8% attack, thông qua max-pool cho mỗi bản ghi):

| Điểm số (Score)             | AUROC      | PR-AUC     | Oracle F1 | Oracle τ |
| --------------------------- | ---------- | ---------- | --------- | -------- |
| Last-token NLL (max-pooled) | 0.6559     | 0.3747     | 0.39      | 13.789   |
| Max-window NLL (max-pooled) | **0.6997** | **0.5490** | **0.55**  | 19.156   |

Max-window NLL đảo ngược dấu ở cấp độ bản ghi (AUROC dương) bởi vì các bản ghi tấn công chứa ít nhất một sự kiện khai thác nơi một syscall token đạt mức NLL rất cao — cơ chế max-pool làm nổi bật điểm dị biệt (outlier) đó. Ngưỡng cấp độ bản ghi theo phân vị của tập val τ_rec=15.099, cho F1=0.27.

**Kết luận:** Window AUROC 0.85 là đủ tốt cho vai trò teacher để chưng cất DFA ở Giai đoạn 3. Vấn đề hiệu chuẩn ngưỡng là nút thắt chính — Giai đoạn 3 sẽ thay thế ngưỡng liên tục này bằng một quyết định chấp nhận/từ chối mang tính cấu trúc (structural accept/reject decision).

### Các tệp chính (Key files)

| Tệp (File)                              | Mục đích (Purpose)                                                           |
| --------------------------------------- | ---------------------------------------------------------------------------- |
| `guepard-shield-model/gp/model.py`      | SyscallTransformer + `predict_step`                                          |
| `guepard-shield-model/gp/dataset.py`    | SyscallDataset (hỗ trợ mmap)                                                 |
| `guepard-shield-model/gp/datamodule.py` | LightningDataModule                                                          |
| `guepard-shield-model/gp/metrics.py`    | `select_threshold`, `evaluate`                                               |
| `notebooks/p2/preprocess.py`            | Chuyển đổi sliding windows từ .sc → .npy                                     |
| `notebooks/p2/train.py`                 | Entry point để huấn luyện                                                    |
| `notebooks/p2/eval.py`                  | Đánh giá: Lightning predict, cache điểm số, đánh giá cấp độ window+recording |
| `results/p2/checkpoints/best.ckpt`      | Checkpoint tốt nhất (epoch 1)                                                |
| `results/p2/scores/`                    | Điểm số test đã cache (test_last.npy, test_max.npy, test_labels.npy)         |
| `data/processed/p2/`                    | Các windows + rec_ids đã qua tiền xử lý (train/val/test)                     |

---

## ✅ Giai đoạn 3: Chưng cất Luật (Rule Distillation) — Đã hoàn thành (K=64)

**Mục tiêu:** Trích xuất một DFA từ các trạng thái ẩn (hidden states) của mô hình Transformer đã huấn luyện thông qua việc gom cụm K-Means, sau đó sử dụng nó cho việc phát hiện bất thường mà không cần ngưỡng (threshold-free anomaly detection).

### Pipeline đã triển khai

| Bước | Script | Output |
|------|--------|--------|
| 1. Extract hidden states (stride=4) | `notebooks/p3/extract_states.py` | `results/p3/hidden_states/train_H.dat` [M×128 float16] |
| 2. K-Means clustering | `notebooks/p3/cluster.py` | `results/p3/clusters/K64/centroids.npy` |
| 3. Build NFA/DFA transitions (stride=1) | `notebooks/p3/build_transitions_stride1.py` | `results/p3/dfa_s1/K64_{S3,S4_t*}/transitions.npz` |
| 4. Grid search evaluation | `notebooks/p3/eval_dfa.py` | `results/p3/metrics/grid_search.csv` |

**Giải pháp kỹ thuật chính — Stride mismatch:** Extract hidden states dùng stride=4 (để tránh lưu 175 GB), nhưng DFA phải hoạt động ở stride=1 (eBPF intercepts mọi syscall). Giải pháp: reuse centroids stride=4, stream lại toàn bộ recordings ở stride=1, encode hidden states và gán cluster — NFA được tích lũy on-the-fly, không cần lưu hidden states ra đĩa (175 GB → ~1 MB NFA cache).

### Kết quả — K=64, Strategy S3 (Chiến lược tốt nhất)

| Chỉ số | Giá trị | Ý nghĩa |
|--------|---------|---------|
| **DR_rec** | **90.85%** | 9/10 attack recordings bị phát hiện — metric chính |
| **FPR** | **1.41%** | 1/71 normal windows bị báo nhầm (cold-start; streaming thấp hơn) |
| Fidelity vs. Teacher | 98.12% | DFA phản chiếu 98.12% quyết định của Transformer gốc |
| States | 64 | Số trạng thái DFA |
| Transitions | 1,973 | Kích thước bảng tra cứu |
| BPF map size | ~210 KB | `int32[64][816]` — phù hợp giới hạn eBPF |

### Kết quả các chiến lược khác (K=64)

| Chiến lược | FPR | DR_rec | Kết luận |
|------------|-----|--------|----------|
| S1 (subset construction) | — | — | **State explosion** — ND rate 85.9% làm powerset construction tạo >640 states |
| **S3 (majority voting)** | **1.41%** | **90.85%** | **Dùng được — chiến lược tốt nhất** |
| S4 θ=0.80–0.99 | 97–99% | ~100% | **Không dùng được** — ND rate cao làm S4 loại bỏ quá nhiều transitions, normal sequences liên tục bị reject nhầm |

**Phát hiện quan trọng — S4 thất bại:** Giả thuyết thiết kế (§3.4.4) dự đoán S4 là "ứng viên chính" vì phân bố transitions skewed. Thực nghiệm cho thấy ngược lại: ND rate = 85.9% tại K=64 (hầu hết cặp (src, tok) dẫn đến nhiều đích khác nhau), khiến S4 với bất kỳ θ nào cũng loại bỏ quá nhiều transitions hợp lệ. S3 xử lý ND một cách graceful bằng cách giữ lại tất cả transitions với đích là mode của phân phối.

### Các tệp chính

| Tệp | Mục đích |
|-----|---------|
| `guepard-shield-model/gp/dfa/transitions.py` | `TransitionBuilder` — build NFA, resolve S1/S3/S4 |
| `guepard-shield-model/gp/dfa/evaluate.py` | `DFAEvaluator` — FPR, DR_rec, fidelity |
| `notebooks/p3/build_transitions_stride1.py` | Pipeline chính — streaming NFA building |
| `notebooks/p3/eval_dfa.py` | Grid search đánh giá tất cả configs |
| `results/p3/dfa_s1/K64_S3/transitions.npz` | **DFA cuối cùng** — 64 states, 1,973 transitions |
| `results/p3/metrics/grid_search.csv` | Toàn bộ kết quả grid search |

### Việc còn lại (cho grid search đầy đủ)

- Re-cluster K=128 với `n_init=10, batch_size=50_000` (K=128 hiện tại có 59% empty clusters do underfitting)
- Chạy `build_transitions_stride1.py --K 64 128` và `eval_dfa.py` để so sánh K sensitivity
- Export DFA sang C struct cho eBPF demo (`gp/dfa/export.py`)

---

## ✅ Giai đoạn 4: Triển khai (Deployment) — Hoàn thành

eBPF DFA enforcement được triển khai đầy đủ bằng Rust/Aya:

- **Agent userspace** (`guepard-shield/`): nạp DFA từ `dfa_config.json`, điền vào eBPF maps (`TRANSITION_TABLE`, `SYSCALL_TO_TOKEN`, `STATE_CLASS`), bắt sự kiện SUSPECT/ATTACK qua ring buffer.
- **Chương trình eBPF kernel** (`guepard-shield-ebpf/`): tracepoint `raw_syscalls/sys_enter`, tra cứu DFA O(1) mỗi syscall, phát hiện ATTACK (không có transition) và SUSPECT (trạng thái tần suất thấp).
- **Đo độ trễ**: histogram 32 bucket × 100 ns, p50/p99/p999 được ghi ra file.
- **Benchmark Transformer** (`notebooks/p4/benchmark_transformer.py`): đo throughput và latency của mô hình Teacher làm baseline so sánh.
