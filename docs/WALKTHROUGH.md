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
  - Triển khai **Gán nhãn cấp độ cửa sổ (Window-level Labeling)** sử dụng các dấu thời gian khai thác (exploit timestamps) từ metadata JSON.

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

## ⏳ Giai đoạn 3: Chưng cất Luật (Rule Distillation) — Chưa được triển khai

Thiết kế đã được lập tài liệu trong `docs/chapters/3_Methodology.md` (các phần 3.4.\*).

**Mục tiêu:** Trích xuất một DFA từ các trạng thái ẩn (hidden states) của mô hình Transformer đã huấn luyện thông qua việc gom cụm K-Means, sau đó sử dụng nó cho việc phát hiện bất thường mà không cần ngưỡng (threshold-free anomaly detection).

**Các đầu vào có sẵn:**

- `results/p2/checkpoints/best.ckpt` — Transformer đã huấn luyện (Phase 2 teacher)
- `data/processed/p2/train_X.npy`, `train_rec_ids.npy` — các window huấn luyện + ID của bản ghi
- `model.encode(x)` — trả về trạng thái ẩn của token cuối cùng với kích thước [B, d_model] để phục vụ gom cụm.

**Các bước cần triển khai:**

1. Trích xuất trạng thái ẩn trên tập train: chạy `encode()` trên tất cả các cửa sổ huấn luyện → ma trận [N, 128].
2. Gom cụm K-Means (K ∈ {100, 200, 500, 1000}) → gán nhãn trạng thái cho mỗi cửa sổ.
3. Xây dựng bảng chuyển đổi DFA: đối với mỗi cặp cửa sổ liên tiếp (cùng bản ghi, kề nhau theo stride), ghi lại `(state_i, syscall_last, state_j)`.
4. Giải quyết tính không xác định (non-determinism) (các chiến lược S1–S4, xem phần Phương pháp nghiên cứu 3.4.3).
5. Đánh giá: độ trung thực (fidelity) so với mô hình teacher, FPR trên các cửa sổ test bình thường, TPR trên các bản ghi tấn công.
6. Tìm kiếm lưới (Grid search) K × θ (ngưỡng cắt tỉa cho S4).

**Quyết định thiết kế chính:** DFA chấp nhận/từ chối dựa trên việc một bước chuyển đổi (transition) đã được nhìn thấy trong quá trình huấn luyện hay chưa — không cần tham số ngưỡng.

---

## ⏳ Giai đoạn 4: Triển khai (Deployment) — Theo kế hoạch

Mã nguồn Rust / Aya trong kho lưu trữ này hiện vẫn chỉ là khung (scaffold). Nó chưa được kết nối với mô hình Giai đoạn 2 đã huấn luyện hay một tạo tác (artifact) đã được chưng cất từ Giai đoạn 3.
