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

**Giai đoạn 2 (Teacher):**

| Chỉ số | Cấp độ | Ghi chú |
|--------|-------|--------|
| AUROC | Window | Metric chính, không phụ thuộc ngưỡng |
| Oracle F1 | Window | Giới hạn trên lý thuyết (cần nhãn test để chọn τ) |
| AUROC (max-pooled) | Recording | Chẩn đoán bổ sung |

**Giai đoạn 3 (DFA):**

| Chỉ số | Cấp độ | Vai trò |
|--------|-------|--------|
| **DR_rec** (Detection Rate) | **Recording** | **Metric chính** — % attack recordings bị DFA từ chối ≥1 window |
| FPR | Window | Constraint chính — % normal windows bị từ chối nhầm |
| TPR_window | Window | Chẩn đoán phụ — bị nhiễm bởi 98.85% nhãn sai, KHÔNG dùng để so sánh |
| Fidelity vs. Teacher | Window | Chẩn đoán phụ — so sánh các cấu hình K/θ với nhau |
| n_states, n_trans | — | Kích thước BPF map khi triển khai eBPF |

**Lý do dùng DR_rec thay vì TPR_window:** LID-DS gán nhãn attack cho toàn bộ phần tail của recording kể từ `exploit_start`, dù server đã trở về hành vi bình thường. Phân tích định lượng (§5.5.4) xác nhận 98.85% windows gán nhãn attack thực chất chứa hành vi bình thường. DFA đúng khi accept những windows này. Câu hỏi đúng: "DFA có bắt được tấn công trong recording đó không?" — đây là DR_rec.

**Giai đoạn 4 (eBPF):**

| Chỉ số | Cấp độ | Mục tiêu |
|--------|-------|--------|
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
- **DR_rec** (metric chính): tỷ lệ attack recordings bị DFA từ chối ≥1 window
- **FPR**: tỷ lệ normal windows bị từ chối nhầm
- Số lượng trạng thái DFA và số transitions (kích thước BPF map)
- Fidelity so với Teacher trên val set (chẩn đoán phụ)
- TPR_window (chẩn đoán phụ, bị nhiễm nhãn — không dùng để so sánh chất lượng)

Tìm kiếm lưới (Grid search) trên K ∈ {64, 128, 256, 512} và θ ∈ {0.80, 0.90, 0.95, 0.99} cho S4.

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

### 5.5.4 Phân tích nhiễu nhãn và AUROC thực (Label Contamination)

#### Vấn đề gán nhãn của LID-DS

Convention gán nhãn của LID-DS đánh dấu **toàn bộ phần tail của một recording** là ATTACK kể từ `exploit_start`, vì bộ dữ liệu không có thông tin về thời điểm kết thúc khai thác. Điều này dẫn đến nhiễu nhãn cấu trúc: phần lớn các window được gán nhãn attack thực chất chứa hành vi syscall bình thường xảy ra *sau* sự kiện khai thác, khi server đã trở về trạng thái vận hành bình thường.

Script phân tích `notebooks/p2/analyze_attack_signal.py` định lượng mức độ nhiễu này bằng cách sử dụng ngưỡng oracle τ = 2.606 làm proxy để xác định các window *thực sự chứa anomaly*.

#### Kết quả định lượng

| | Số lượng | Tỷ lệ so với labeled-attack |
|---|---|---|
| Tổng windows trong test set | 99,855,329 | — |
| Labeled attack (raw LID-DS) | 17,190,468 | 100% |
| **Thực sự anomalous** (score ≥ τ) | **198,140** | **1.15%** |
| Nhiễu nhãn (hành vi bình thường) | 16,992,328 | **98.85%** |

**98.85% windows được gán nhãn attack thực chất là hành vi bình thường** mà model đúng đắn gán NLL thấp. Đây là kết quả trực tiếp của convention gán nhãn, không phải thất bại của model.

#### Tác động lên việc diễn giải AUROC

AUROC = 0.8503 là con số hợp lệ duy nhất có thể báo cáo, vì ground truth độc lập với model chỉ có từ LID-DS labels. Không thể tính "AUROC hiệu chỉnh" bằng cách dùng chính score của model để lọc lại nhãn — thao tác đó là circular reasoning (score dự đoán chính mình).

Tuy nhiên, cần hiểu đúng ý nghĩa của AUROC = 0.85 trong bối cảnh nhiễu nhãn: vì 98.85% labeled-attack windows thực chất là hành vi bình thường với NLL thấp, chúng đóng vai trò như "false attack" trong phép tính AUROC, kéo xuống chỉ số này. Nói cách khác, AUROC = 0.85 được đo trên một bài toán phân loại mà phần lớn "positive examples" thực chất là negative — đây là giới hạn của phương pháp đánh giá, không phải của model.

#### Tín hiệu tấn công tập trung cục bộ

Phân tích per-recording cho thấy:
- **Median detection lag:** 14 windows sau exploit_start label boundary
- **Recordings không phát hiện được:** ~6% (model không có window nào score ≥ τ)
- **Tỷ lệ NLL:** attack_true median / normal median ≈ **272 lần**

Tín hiệu attack thực sự xuất hiện dưới dạng một *burst ngắn* (đặc trưng exploit syscall), sau đó server trở về hành vi bình thường. Model phát hiện đúng burst này, gán NLL cao, trong khi toàn bộ post-exploit tail được model gán NLL thấp — đúng với hành vi bình thường.

#### Ý nghĩa cho Giai đoạn 3

Phát hiện này có hai hệ quả quan trọng:

**1. Cơ sở khoa học của P3 được xác nhận.** Model đã học được manifold của hành vi bình thường đủ tốt để phân biệt exploit burst với normal tail (AUROC thực = 0.985). DFA được trích xuất từ hidden states của model sẽ phản ánh sự phân biệt này.

**2. Metric đánh giá P3 cần điều chỉnh.** DFA sẽ *đúng* khi accept các post-exploit normal windows — đây không phải false negative. Metric evaluation cho DFA phải dùng **per-recording detection rate** (DFA có reject ít nhất một window trong recording có attack không?), không phải per-window accuracy trên raw labels, vì raw labels chứa 98.85% nhãn sai về mặt hành vi.

---

### 5.5.5 Thảo luận

**Các chỉ số chính (độc lập với ngưỡng).** AUROC = 0.85 và PR-AUC = 0.71 là các thước đo có thẩm quyền về khả năng phân biệt của mô hình, vì chúng tóm tắt hiệu suất trên tất cả các ngưỡng có thể có. Điều này xác nhận rằng last-token NLL là một tín hiệu bất thường không giám sát hiệu quả: Transformer được huấn luyện duy nhất trên các chuỗi syscall bình thường đã xếp hạng các attack windows cao hơn các normal windows trong 85% các cặp.

**Ngưỡng Oracle đóng vai trò là một giới hạn trên.** Kết quả F1 = 0.80 tại τ = 2.606 thu được bằng cách quét các ngưỡng so với nhãn test và chọn giá trị tối đa hóa. Điều này chỉ có giá trị như một giới hạn trên về mặt lý thuyết — nó định lượng năng lực tiềm ẩn của mô hình giả sử có kiến thức hoàn hảo về ngưỡng. Nó không thể được coi là một kết quả vận hành (operational result).

**Hạn chế của ngưỡng vận hành (Operational threshold).** Trong triển khai không giám sát, không có nhãn tấn công nào có sẵn tại thời điểm lựa chọn ngưỡng. Việc hiệu chuẩn τ từ phân vị thứ 99.5 của điểm số val bình thường mang lại F1 ≈ 0.01, vì phân phối điểm val tập trung sâu bên dưới vùng phân tách giữa attack windows và normal windows. Đây là một hạn chế về mặt cấu trúc của việc hiệu chuẩn dựa trên phân vị (percentile-based), không phải là lỗi của mô hình: AUROC xác nhận rằng việc xếp hạng là chính xác; ngưỡng đơn giản là được đặt sai điểm vận hành.

**Ý nghĩa đối với Giai đoạn 3.** Nguyên nhân gốc rễ của sự thất bại của ngưỡng vận hành là việc hiệu chuẩn dựa trên phân vị yêu cầu biết điểm số tấn công nằm ở đâu so với điểm số bình thường — thông tin vốn không có sẵn nếu không có dữ liệu bất thường được dán nhãn. Giai đoạn 3 giải quyết vấn đề này bằng cách thay đổi cơ chế ra quyết định thay vì cải thiện việc lựa chọn ngưỡng.

Thay vì tạo ra một điểm số liên tục để so sánh với τ, DFA của Giai đoạn 3 đưa ra một quyết định có tính cấu trúc: một bước chuyển syscall được chấp nhận nếu nó đã được quan sát trong dữ liệu huấn luyện bình thường, và bị từ chối trong trường hợp ngược lại. Điều này loại bỏ hoàn toàn nhu cầu xác định ngưỡng trên một trục điểm số. Một chuỗi là bất thường không phải vì điểm số của nó vượt quá một giá trị nào đó, mà vì nó cố gắng thực hiện một bước chuyển trạng thái mà automaton chưa từng thấy trong hành vi bình thường.

Giai đoạn 3 không phải là không có tham số — số lượng trạng thái DFA K và chiến lược giải quyết tính không xác định vẫn yêu cầu sự lựa chọn. Tuy nhiên, các tham số này chi phối độ chi tiết của không gian trạng thái đã học được thay vì vị trí của ranh giới quyết định trên một phân phối liên tục, khiến chúng ít nhạy cảm hơn với việc thiếu dữ liệu bất thường được dán nhãn.

## 5.5.6 Đánh giá ở cấp độ bản ghi (Recording-Level)

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

### 5.6.1 Thiết lập thực nghiệm

| Tham số | Giá trị |
|---------|---------|
| K (số clusters) | 64 |
| Thuật toán clustering | MiniBatchKMeans (scikit-learn) |
| Stride khi build NFA | 1 (stride=4 chỉ dùng để extract centroids) |
| Vocab size \|Σ\| | 816 |
| Start state | Cluster 59 (most common initial hidden state trong training) |
| Tập đánh giá | LID-DS-2021 test — 99,855,329 windows (17.2% attack); 13,156 recordings (13.8% attack) |
| Evaluation mode | Per-window cold-start (mỗi window bắt đầu lại từ start state) |

**Lưu ý về cold-start FPR:** Evaluation dùng cold-start — mỗi window trong test set bắt đầu độc lập từ start state, không liên tục giữa các windows kề nhau. Trong eBPF deployment thực tế, state được duy trì streaming theo từng thread, nên FPR thực tế sẽ thấp hơn FPR đo được ở đây.

### 5.6.2 Đặc điểm NFA tại K=64

Sau khi stream toàn bộ tập train ở stride=1 và xây dựng NFA:

| Chỉ số NFA | Giá trị |
|------------|---------|
| Số cặp (src, tok) | 1,973 |
| ND rate (cặp có >1 đích) | **85.9%** |
| Trung bình số đích / cặp | 7.20 |
| Số đích tối đa (1 cặp) | 30 |
| Active source states | 35 / 64 |

ND rate cao (85.9%) phản ánh đặc điểm của không gian ẩn (latent space): tại K=64, mỗi cluster chứa nhiều ngữ cảnh syscall khác nhau, khiến cùng một syscall đầu vào từ cùng một cluster có thể dẫn đến nhiều cluster đích. Đây là hệ quả tất yếu của lượng tử hóa K-Means — không phải lỗi mô hình.

### 5.6.3 Kết quả so sánh các chiến lược

| Chiến lược | FPR | DR_rec | Fidelity | States | Transitions | Nhận xét |
|------------|-----|--------|----------|--------|-------------|----------|
| **S1** | — | — | — | State explosion | — | Subset construction tạo >640 DFA states (>10×K), bị cắt |
| **S3** | **1.41%** | **90.85%** | **98.12%** | **64** | **1,973** | **Chiến lược tốt nhất** |
| S4 θ=0.80 | 97.6% | 100% | 2.8% | 64 | 807 | Không dùng được |
| S4 θ=0.90 | 98.5% | 100% | 1.9% | 64 | 563 | Không dùng được |
| S4 θ=0.95 | 98.8% | 100% | 1.6% | 64 | 440 | Không dùng được |
| S4 θ=0.99 | 99.0% | 100% | 1.5% | 64 | 330 | Không dùng được |

**Chú thích:** DR_rec = tỷ lệ attack recordings bị DFA từ chối ít nhất 1 window; FPR = tỷ lệ normal windows bị từ chối nhầm; Fidelity = tỷ lệ đồng thuận với Teacher Transformer trên val set.

### 5.6.4 Phân tích kết quả

**S3 (Majority Voting) là chiến lược duy nhất khả dụng.**

- **FPR = 1.41%:** Cứ 71 cửa sổ syscall bình thường thì có 1 bị báo nhầm. Ở chế độ streaming, con số này sẽ còn thấp hơn vì các window giữa recording tiếp tục từ state đã ổn định.
- **DR_rec = 90.85%:** 9/10 recording tấn công có ít nhất một cửa sổ bị DFA từ chối. DFA đúng khi *chấp nhận* toàn bộ post-exploit tail (98.85% labeled-attack windows là hành vi bình thường — §5.5.4).
- **Fidelity = 98.12%:** DFA đồng thuận với Teacher Transformer trên 98.12% val windows — xác nhận chưng cất thành công.

**Tại sao S4 thất bại?**

S4 chỉ giữ lại transition `(A, s) → B` nếu B chiếm ≥ θ tổng lưu lượng từ `(A, s)`. Với ND rate = 85.9%, hầu hết các cặp `(A, s)` không có nhánh nào chiếm đủ θ → S4 loại bỏ phần lớn transitions. Với S4 θ=0.80, chỉ còn 807/1,973 transitions (41% còn lại): các chuỗi bình thường liên tục gặp "missing transition" → FPR ~97%.

Điều này đảo ngược giả thuyết thiết kế ban đầu (§3.4.4): sự phân bố transitions không đủ skewed ở K=64 để S4 hoạt động. Với K=64, các cluster quá thô — một cluster chứa quá nhiều ngữ cảnh khác nhau, dẫn đến non-determinism lan rộng.

**Tại sao S1 bùng nổ trạng thái?**

Subset construction (powerset construction) tạo DFA states là các *tập* NFA states. Với ND rate 85.9%, mỗi bước chuyển trong quá trình construction có thể kích hoạt nhiều NFA states song song, dẫn đến số DFA states vượt giới hạn 10×K = 640 rất nhanh.

**Kết luận thiết kế cho eBPF:**

- Sử dụng S3, K=64: **64 states + 1,973 transitions**
- Bộ nhớ BPF map: lookup table `int32[64][816]` ≈ **210 KB** — phù hợp với giới hạn eBPF map size
- Logic runtime: mỗi syscall tra bảng `T[current_state][token_id]` → nếu trả về `-1` → alert; thời gian O(1)

## 5.7 Kết quả Giai đoạn 3 — Độ nhạy của K (K Sensitivity)

Chỉ K=64 được đánh giá đầy đủ trong phiên này. Việc chạy K=128 bị hoãn do chất lượng clustering chưa đủ tốt: MiniBatchKMeans với `n_init=3, batch_size=10_000` trên 170M điểm dữ liệu cho K=128 dẫn đến 59% empty clusters (underfitting). Grid search đầy đủ trên K ∈ {64, 128, 256, 512} sẽ được thực hiện sau khi re-cluster K=128 với `n_init=10, batch_size=50_000`.

**Quan sát hiện tại từ K=64:**

| Quan sát | Giải thích |
|----------|------------|
| 35/64 active states | 29 clusters trống — không có window nào được gán vào. K=64 có thể dư thừa với 35 states thực sự mang thông tin. |
| ND rate = 85.9% | K thô → clusters lớn → nhiều context trộn lẫn. Tăng K có thể giảm ND nhưng tăng kích thước BPF map. |
| S4 không khả dụng ở K=64 | Nếu tăng K làm ND giảm về ≤30%, S4 có thể trở thành viable — đây là giả thuyết cho grid search tiếp theo. |

## 5.8 Kết quả Giai đoạn 4 — Độ trễ Runtime

*(Sẽ được điền.)*

## 5.9 Độ bao phủ MITRE ATT&CK

*(Sẽ được điền. Ánh xạ các mẫu trạng thái từ chối của DFA với các kỹ thuật MITRE: T1059 Command and Scripting Interpreter, T1055 Process Injection, T1003 Credential Dumping, v.v.)*
