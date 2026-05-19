# Chương 5 — Kết quả thực nghiệm

Chương này hiện là một **mẫu kế hoạch đánh giá (planned evaluation template)** cho kiến trúc Giai đoạn 2 / Giai đoạn 3 mới. Hiện tại chưa có kết quả thực nghiệm duy trì cho Giai đoạn 2 hoặc Giai đoạn 3 trong codebase này.

## 5.1 Các tham số đánh giá

| Tham số | Mô tả | Các giá trị ứng viên |
|---------|-------|-----------------|
| `W` | Kích thước sliding window (syscalls) | 64, 128 |
| `K` | Số lượng trạng thái DFA (cụm K-Means) | 100, 200, 500, 1000 |
| `θ` | Ngưỡng cắt tỉa thống kê (S4) | 0.95, 0.99, 0.999 |
| `\|Σ\|` | Kích thước bộ từ vựng (tên syscall) | 102 (baseline hiện tại); dự kiến composite tokenization |
| `d` | Transformer hidden dimension | Sẽ xác định (TBD) |
| `L` | Số lớp Transformer | Sẽ xác định (TBD) |

## 5.2 Các bộ dữ liệu (Datasets)

| Bộ dữ liệu | Phân chia (Split) | Mục đích sử dụng |
|---------|-------|-----|
| LID-DS-2021 | train / val / test (đã chia sẵn) | Đánh giá chính |
| LID-DS-2019 | — | Cross-domain generalization |
| DongTing | — | Cross-domain generalization |

Nhãn của window được trích xuất từ các dấu thời gian khai thác (exploit timestamps) trong metadata JSON. Không sử dụng nhãn cấp độ bản ghi (recording-level).

## 5.3 Các chỉ số đánh giá

| Chỉ số | Cấp độ | Mục tiêu |
|--------|-------|--------|
| Detection AUROC | Window | > 0.95 |
| Detection F1 | Window | > 0.90 |
| DFA Fidelity vs. Teacher | Window | > 95% |
| DFA FPR | Window | < 1% |
| eBPF latency overhead | Per-syscall | < 2 µs |
| Real workload FPR | Window | < 5% |
| MITRE ATT&CK coverage | — | Được báo cáo |

## 5.4 Phương pháp mô phỏng

### 5.4.1 Phase 2 Baselines

So sánh Teacher Transformer với:
- Mô hình tần suất N-gram (n=5)
- LSTM next-token predictor (theo phong cách DeepLog)

Chỉ số: AUROC và F1 cấp độ window trên tập test LID-DS-2021.

### 5.4.2 Giai đoạn 3 — So sánh chiến lược giải quyết tính không xác định

Đối với mỗi chiến lược S1–S4, đo lường:
- Tỷ lệ xung đột (conflict rate - tỷ lệ các bước chuyển mơ hồ)
- Số lượng trạng thái DFA (sau khi giải quyết)
- Fidelity so với Teacher trên LID-DS-2021
- FPR cấp độ window trên các window test bình thường

Tìm kiếm lưới (Grid search) trên K ∈ {100, 200, 500, 1000} và θ ∈ {0.95, 0.99, 0.999} cho S4.

### 5.4.3 Giai đoạn 4 — Đánh giá tại Runtime

Khối lượng công việc (Workloads): nginx, redis, postgres chạy trên một máy ảo Linux.
- Đo lường eBPF overhead trên mỗi syscall so với baseline (không có eBPF).
- Đo lường FPR (các window công việc bình thường bị gắn nhãn Edge hoặc Reject).
- Tiêm (Inject) các kịch bản tấn công LID-DS; đo lường TPR.

## 5.5 Kết quả Giai đoạn 2

### 5.5.1 Thiết lập thực nghiệm

| Siêu tham số (Hyperparameter) | Giá trị |
|----------------|-------|
| Window size W | 64 |
| Stride S | 32 |
| d_model | 128 |
| Layers / Heads | 4 / 4 |
| d_ff | 512 |
| Dropout | 0.1 |
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| LR schedule | Cosine decay + 5% linear warmup |
| Batch size | 256 |
| Max epochs | 20 (early stopping patience=5) |
| Precision | 16-mixed |
| Vocab size | 102 |

Quá trình huấn luyện đã sử dụng toàn bộ 3,149 bản ghi huấn luyện từ LID-DS-2021 (không giới hạn số window trên mỗi bản ghi).
Mô hình dừng ở epoch thứ 7 (val_loss tốt nhất ở epoch 1).

### 5.5.2 Hội tụ huấn luyện (Training Convergence)

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 0 | 0.5292 | 0.3811 |
| 1 | 0.3830 | **0.3686** ← tốt nhất |
| 2 | 0.3762 | 0.3706 |
| 3 | 0.3732 | 0.3709 |
| 4 | 0.3713 | 0.3695 |
| 5 | 0.3699 | 0.3688 |
| 6 | 0.3688 | 0.3691 |

Val loss sớm đi vào trạng thái bão hòa (plateau) trong khi train loss tiếp tục giảm, cho thấy mô hình đã đạt đến điểm tổng quát hóa ổn định mà không bị overfitting (khoảng cách train/val < 0.001).

### 5.5.3 Đánh giá trên tập kiểm tra (Test Set)

Tập kiểm tra: 99,855,329 windows trên tất cả các kịch bản test của LID-DS-2021 (17.2% là attack windows).

Anomaly score: NLL của token cuối cùng — $\text{score} = -\log P(s_W \mid s_1, \ldots, s_{W-1})$.

Thống kê điểm số trên tập val (normal windows):

| Thống kê | Giá trị |
|---------|-------|
| min | 0.0000 |
| mean | 0.3024 |
| p99 | 4.3359 |
| max | 18.1250 |

| Chỉ số | Giá trị |
|--------|-------|
| AUROC | **0.8503** |
| PR-AUC | **0.7093** |

#### Các biến thể ngưỡng (Threshold variants)

| Ngưỡng | τ | Precision (attack) | Recall (attack) | F1 (attack) | FPR |
|-----------|---|--------------------|-----------------|-------------|-----|
| 99.5th pct của val | 5.4040 | 0.10 | 0.00 | 0.01 | 0.51% |
| μ + 3σ của val | — | 0.07 | 0.01 | 0.02 | 2.59% |
| Oracle optimal | **2.606** | **0.85** | **0.75** | **0.80** | 2.85% |

Tại ngưỡng oracle (τ = 2.606):

```
              precision    recall  f1-score   support

      normal       0.95      0.97      0.96  82,664,861
      attack       0.85      0.75      0.80  17,190,468

    accuracy                           0.93  99,855,329
   macro avg       0.90      0.86      0.88  99,855,329
weighted avg       0.93      0.93      0.93  99,855,329
```

#### Max-window NLL (tín hiệu phụ)

Max-window NLL — $\text{score} = \max_{t} \left[ -\log P(s_t \mid s_1, \ldots, s_{t-1}) \right]$ — đạt **AUROC = 0.35 ở cấp độ window**, tức là *tương quan âm* với attack windows. Điều này phản ánh thực tế rằng token NLL cao nhất trong một window thường bị chiếm bởi các syscall hiếm của normal code (socket setup, stat với flag bất thường,...) thay vì các syscall trong chuỗi tấn công vốn dùng các syscall phổ biến (read/write/mmap) theo pattern lạ. Max-window NLL không được dùng làm tín hiệu bất thường ở cấp độ window.

### 5.5.4 Thảo luận

**Các chỉ số chính (độc lập với ngưỡng).** AUROC = 0.85 và PR-AUC = 0.71 là các thước đo có thẩm quyền về khả năng phân biệt của mô hình, vì chúng tóm tắt hiệu suất trên tất cả các ngưỡng có thể có. Điều này xác nhận rằng last-token NLL là một tín hiệu bất thường không giám sát hiệu quả: Transformer được huấn luyện duy nhất trên các chuỗi syscall bình thường đã xếp hạng các attack windows cao hơn các normal windows trong 85% các cặp.

**Ngưỡng Oracle đóng vai trò là một giới hạn trên.** Kết quả F1 = 0.80 tại τ = 2.606 thu được bằng cách quét các ngưỡng so với nhãn test và chọn giá trị tối đa hóa. Điều này chỉ có giá trị như một giới hạn trên về mặt lý thuyết — nó định lượng năng lực tiềm ẩn của mô hình giả sử có kiến thức hoàn hảo về ngưỡng. Nó không thể được coi là một kết quả vận hành (operational result).

**Hạn chế của ngưỡng vận hành (Operational threshold).** Trong triển khai không giám sát, không có nhãn tấn công nào có sẵn tại thời điểm lựa chọn ngưỡng. Việc hiệu chuẩn τ từ phân vị thứ 99.5 của điểm số val bình thường mang lại F1 ≈ 0.01, vì phân phối điểm val tập trung sâu bên dưới vùng phân tách giữa attack windows và normal windows. Đây là một hạn chế về mặt cấu trúc của việc hiệu chuẩn dựa trên phân vị (percentile-based), không phải là lỗi của mô hình: AUROC xác nhận rằng việc xếp hạng là chính xác; ngưỡng đơn giản là được đặt sai điểm vận hành.

**Ý nghĩa đối với Giai đoạn 3.** Nguyên nhân gốc rễ của sự thất bại của ngưỡng vận hành là việc hiệu chuẩn dựa trên phân vị yêu cầu biết điểm số tấn công nằm ở đâu so với điểm số bình thường — thông tin vốn không có sẵn nếu không có dữ liệu bất thường được dán nhãn. Giai đoạn 3 giải quyết vấn đề này bằng cách thay đổi cơ chế ra quyết định thay vì cải thiện việc lựa chọn ngưỡng.

Thay vì tạo ra một điểm số liên tục để so sánh với τ, DFA của Giai đoạn 3 đưa ra một quyết định có tính cấu trúc: một bước chuyển syscall được chấp nhận nếu nó đã được quan sát trong dữ liệu huấn luyện bình thường, và bị từ chối trong trường hợp ngược lại. Điều này loại bỏ hoàn toàn nhu cầu xác định ngưỡng trên một trục điểm số. Một chuỗi là bất thường không phải vì điểm số của nó vượt quá một giá trị nào đó, mà vì nó cố gắng thực hiện một bước chuyển trạng thái mà automaton chưa từng thấy trong hành vi bình thường.

Giai đoạn 3 không phải là không có tham số — số lượng trạng thái DFA K và chiến lược giải quyết tính không xác định vẫn yêu cầu sự lựa chọn. Tuy nhiên, các tham số này chi phối độ chi tiết của không gian trạng thái đã học được thay vì vị trí của ranh giới quyết định trên một phân phối liên tục, khiến chúng ít nhạy cảm hơn với việc thiếu dữ liệu bất thường được dán nhãn.

## 5.5.5 Đánh giá ở cấp độ bản ghi (Recording-Level)

Tập kiểm tra: **13,156 bản ghi** (11,341 normal / 1,815 attack — 13.8% attack rate).

Phương pháp: max-pool điểm số window theo `rec_id`, nghĩa là điểm số của một bản ghi = max NLL trên tất cả các window thuộc bản ghi đó.
Ngưỡng riêng biệt `τ_rec` được hiệu chuẩn từ phân vị thứ 99.5 của điểm số max-pooled trên tập val (`τ_rec = 15.0989`).

#### Last-token NLL (max-pooled)

| Chỉ số | Val-percentile (τ=15.10) | Oracle (τ=13.79) |
|--------|--------------------------|------------------|
| AUROC | 0.6559 | 0.6559 |
| PR-AUC | 0.3747 | 0.3747 |
| F1 (attack) | 0.27 | 0.39 |
| FPR | 1.82% | 4.70% |

```
── oracle optimal (τ = 13.789) ──
              precision    recall  f1-score   support

      normal       0.90      0.95      0.92     11341
      attack       0.52      0.32      0.39      1815

    accuracy                           0.87     13156
   macro avg       0.71      0.63      0.66     13156
weighted avg       0.84      0.87      0.85     13156
```

#### Max-window NLL (max-pooled)

Đáng chú ý: dù max-window NLL tương quan *âm* với attack ở cấp độ window (AUROC=0.35), khi max-pool theo bản ghi, tín hiệu này trở nên *dương* và tốt hơn last-token NLL ở cấp độ bản ghi:

| Chỉ số | Oracle (τ=19.16) |
|--------|-----------------|
| AUROC | **0.6997** |
| PR-AUC | **0.5490** |
| F1 (attack) | **0.55** |
| FPR | 1.99% |

```
── oracle optimal (τ = 19.156) ──
              precision    recall  f1-score   support

      normal       0.91      0.98      0.95     11341
      attack       0.78      0.43      0.55      1815

    accuracy                           0.90     13156
   macro avg       0.84      0.70      0.75     13156
weighted avg       0.90      0.90      0.89     13156
```

#### Thảo luận

**Drop từ window sang recording.** AUROC giảm từ 0.85 (window) xuống 0.66 (recording, last-token) phản ánh hai hiệu ứng: (1) max-pool trên hàng trăm windows khiến normal recordings cũng tích lũy xác suất cao có một window có điểm số cao do ngẫu nhiên; (2) ngưỡng `τ_rec=15.10` quá cao so với phần lớn điểm tấn công, bỏ sót nhiều bản ghi attack thực sự.

**Nghịch lý max-window NLL.** Hiệu ứng nghịch đảo (anti-correlation ở window, positive ở recording) xảy ra vì: các bản ghi tấn công *chắc chắn* chứa ít nhất một sự kiện exploit — tại đó một syscall đơn lẻ có thể đạt NLL rất cao dù các syscall xung quanh bình thường. Khi max-pool, outlier đó nổi lên. Ngược lại, ở cấp độ window, tấn công thường dùng các syscall phổ biến theo context lạ, không phải các syscall có NLL đơn lẻ cao.

**Ý nghĩa cho Giai đoạn 3.** Cả hai tín hiệu đều cho thấy khả năng phân biệt ở recording-level còn hạn chế với cách tiếp cận ngưỡng liên tục. Giai đoạn 3 (DFA) thay thế cơ chế này hoàn toàn: thay vì so sánh điểm số với τ, DFA từ chối ngay khi gặp bước chuyển trạng thái chưa từng xuất hiện trong training bình thường.

---

## 5.6 Kết quả Giai đoạn 3 — So sánh chiến lược giải quyết tính không xác định

*(Chưa có kết quả hiện tại. Sẽ được điền sau khi có triển khai Giai đoạn 3 mới.)*

## 5.7 Kết quả Giai đoạn 3 — Độ nhạy của K (K Sensitivity)

*(Chưa có kết quả hiện tại. Sẽ được điền sau khi có triển khai Giai đoạn 3 mới.)*

## 5.8 Kết quả Giai đoạn 4 — Độ trễ Runtime

*(Sẽ được điền.)*

## 5.9 Độ bao phủ MITRE ATT&CK

*(Sẽ được điền. Ánh xạ các mẫu trạng thái từ chối của DFA với các kỹ thuật MITRE: T1059 Command and Scripting Interpreter, T1055 Process Injection, T1003 Credential Dumping, v.v.)*
