# Chương 3 — Phương pháp nghiên cứu

Giai đoạn 1 và 2 đã được triển khai. Thiết kế của Giai đoạn 3 đã được hoàn thiện nhưng chưa được lập trình. Giai đoạn 4 hiện đang ở dạng khung (scaffold).

## 3.1 Tổng quan

Guepard Shield là một pipeline gồm bốn giai đoạn, chuyển đổi các bản ghi syscall thô thành một DFA được thực thi trong kernel:

```
[Phase 1] EDA + Tiền xử lý dữ liệu
          - Tokenization tên syscall, gán nhãn window
          ↓
[Phase 2] Teacher: Decoder-only Transformer
          - Next-Token Prediction trên các chuỗi syscall bình thường
          - Đánh giá: điểm NLL cấp độ window → AUROC / F1
          ↓
[Phase 3] Student: Trích xuất DFA
          - Thu thập các trạng thái ẩn lớp cuối h_t
          - Gom cụm K-Means → K trạng thái rời rạc
          - Xây dựng bảng chuyển đổi δ(q, token) → q'
          - Giải quyết tính không xác định (chiến lược S1–S4)
          - Xuất tệp dfa_config.json
          ↓
[Phase 4] Thực thi tại Runtime (eBPF)
          - Duyệt DFA theo từng thread thông qua tra cứu BPF map
          - Phản hồi ba cấp độ: Normal / Edge / Rejecting
          - Vòng lặp phản hồi suspect window tới Transformer ngoại tuyến
```

Tại thời điểm thực thi (runtime), không có khái niệm "ghi lại" (recording). Hệ thống xử lý một luồng syscall liên tục theo từng thread. Chỉ tồn tại các sliding windows có kích thước `W`.

Sự tương tác giữa hai sub-agents vận hành:

```
                 LUỒNG SYSCALL LIÊN TỤC (theo từng thread)
                                  │
┌─────────────────────────────────▼──────────────────────────────────────┐
│                    eBPF SUB-AGENT  (Kernel Space)                      │
│                                                                        │
│  ┌──────────────────────────────────────┐                              │
│  │  Syscall Name Tokenizer              │                              │
│  │  Syscall_Name → token_id             │                              │
│  └─────────────────┬────────────────────┘                              │
│                    │                                                   │
│  ┌─────────────────▼────────────────────┐                              │
│  │  DFA State Machine                   │                              │
│  │  transition_table[(state, token)]    │ ← BPF_MAP_TYPE_HASH          │
│  │  thread_state[TID]                   │ ← BPF_MAP_TYPE_HASH          │
│  └──────────┬──────────────┬────────────┘                              │
│             │              │              │                            │
│          NORMAL           EDGE         REJECT                          │
│          tiếp tục    thu thập window   BLOCK / KILL / ALERT            │
│                            │                                           │
└────────────────────────────│───────────────────────────────────────────┘
                             │  perf_event ring buffer
                             │  (suspect window + TID + state trace)
┌────────────────────────────▼───────────────────────────────────────────┐
│                    Rust Agent  (User Space)                            │
│            nhận window → chuyển tiếp tới Transformer                   │
└────────────────────────────┬───────────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────────────┐
│              Transformer Sub-Agent  (Phân tích ngoại tuyến)            │
│                                                                        │
│   ┌─────────────────────┐                                              │
│   │  Tính điểm NLL      │──── True Positive ──▶  ALERT / LOG           │
│   │  trên suspect window│                                              │
│   └─────────────────────┘──── False Positive ──▶ ┌──────────────────┐  │
│                                (hành vi mới      │  Trích xuất lại  │  │
│                                 hợp lệ)          │  DFA & thêm bước │  │
│                                                  └────────┬─────────┘  │
└───────────────────────────────────────────────────────────│────────────┘
                                                            │
                                                   cập nhật map BPF nguyên tử
                                                            │
                                                            ▼
                                                 ┌────────────────────────┐
                                                 │  DFA cập nhật tại chỗ  │
                                                 │  (không nạp lại kernel)│
                                                 └────────────────────────┘
```

---

## 3.2 Giai đoạn 1 — Tiền xử lý dữ liệu

### 3.2.1 Tokenization

Mỗi sự kiện syscall được ánh xạ tới tên syscall của nó, sau đó được tra cứu trong một bộ từ vựng (vocabulary) cố định:
```math
\text{token\_id} = \text{vocab}[\text{Syscall\_Name}]
```
Bộ từ vựng được xây dựng từ dữ liệu huấn luyện với một ngưỡng tần suất tối thiểu. Các tên syscall không xác định tại thời điểm inference được ánh xạ tới `<UNK>`. Kích thước bộ từ vựng hiện tại |Σ| ≈ 102.

**Lý do:** Tên syscall tạo thành một bảng chữ cái hữu hạn và ổn định. Bộ từ vựng bao quát hầu hết các syscall quan sát được trong các khối lượng công việc server bình thường, thỏa mãn yêu cầu của DFA về một bảng chữ cái đầu vào hữu hạn mà không cần kỹ thuật phức tạp hơn.

**Hướng phát triển — composite tokenization:** Một phần mở rộng dự kiến sẽ ánh xạ mỗi syscall tới một token hỗn hợp `(Syscall_Name, ReturnCode_Bucket, Syscall_Params)`, trong đó các mã trả về được phân nhóm (ví dụ: thành công, EPERM, EAGAIN, lỗi-khác). Điều này sẽ tăng tính phong phú của bảng chữ cái — phân biệt một lệnh `open` thành công với một lệnh thất bại — với chi phí là bộ từ vựng và BPF map lớn hơn. Việc này được tạm hoãn cho đến khi baseline của Giai đoạn 2 triển khai thành công.

### 3.2.2 Sliding Windows và Gán nhãn

Chuỗi syscall của 1 chương trình được phân đoạn thành các sliding windows gối đầu lên nhau có kích thước `W`. `W` là một siêu tham số (hyperparameter) (phạm vi ứng viên: 64–128).

Các window được gán nhãn bằng cách sử dụng dấu thời gian khai thác (exploit timestamps) từ metadata của bộ dữ liệu (tệp JSON của LID-DS): một window là **attack** nếu nó trùng lặp với khoảng thời gian khai thác, ngược lại là **normal**. Nhãn này chỉ được sử dụng cho đánh giá ngoại tuyến — không dùng để huấn luyện.

**Lý do chọn W nhỏ:**

- Hành vi tấn công (điểm khai thác) có tính cục bộ tạm thời (temporal locality) cao — thường hoàn thành trong vòng vài chục syscall.
- W nhỏ hơn buộc Transformer phải tổng quát hóa, tạo ra các trạng thái ẩn gom cụm sạch hơn.
- W nhỏ hơn → ít trạng thái DFA hơn → eBPF map nhỏ hơn → chiếm ít bộ nhớ kernel hơn.

---

## 3.3 Giai đoạn 2 — Teacher: Huấn luyện Transformer

### 3.3.1 Kiến trúc

Decoder-only Transformer với causal (lower-triangular) self-attention mask. Đầu vào: chuỗi các syscall-name token ID. Đầu ra: phân phối xác suất token tiếp theo trên Σ.

Causal mask đảm bảo $h_t$ được tính toán chỉ bằng $(s_1, \ldots, s_{t-1})$, khớp với tính chất một chiều, thời gian thực của DFA khi vận hành.

### 3.3.2 Mục tiêu huấn luyện

Next-Token Prediction với Cross-Entropy Loss:

$$\mathcal{L} = -\frac{1}{N} \sum_{t=1}^{N} \log P_\theta(s_{t+1} \mid s_1, \ldots, s_t)$$

Được huấn luyện **chỉ trên các chuỗi bình thường** (phát hiện bất thường không giám sát). Mô hình học $P(\text{hành vi bình thường})$; các bất thường biểu hiện dưới dạng các window có xác suất thấp (NLL cao).

Kỹ thuật Teacher Forcing cho phép huấn luyện song song toàn bộ chuỗi trên GPU.

### 3.3.3 Điểm bất thường (Anomaly Score)

Đối với một window $[s_1, \ldots, s_W]$, điểm bất thường là **NLL của token cuối cùng**:

$$\text{score}(w) = -\log P_\theta(s_W \mid s_1, \ldots, s_{W-1})$$

Chỉ token cuối cùng được tính điểm. $W-1$ token đứng trước đóng vai trò làm ngữ cảnh. Điều này phản chiếu quy tắc gán nhãn window: một window được dán nhãn attack nếu dấu thời gian syscall cuối cùng của nó nằm trong khoảng thời gian khai thác, vì vậy điểm bất thường nhắm vào cùng một vị trí.

Đánh giá chính: AUROC và F1 cấp độ window so với nhãn exploit-timestamp. AUROC và F1 cấp độ bản ghi (recording-level) (gom nhóm điểm số tối đa cho mỗi bản ghi) được báo cáo như một chẩn đoán bổ sung để phù hợp với giao thức đánh giá LID-DS tiêu chuẩn.

---

## 3.4 Giai đoạn 3 — Student: Trích xuất DFA

### 3.4.1 Định nghĩa Trạng thái ẩn (Hidden State)

**Trạng thái ẩn** là vector nhúng đầu ra (output embedding) lớp cuối cùng của **token cuối cùng** trong mỗi lượt forward pass:

$$h_t = \text{TransformerFinalLayer}(s_1, \ldots, s_t)[\text{vị trí } t] \in \mathbb{R}^d$$

Nhờ cơ chế causal self-attention, $h_t$ mã hóa toàn bộ ngữ cảnh nhân quả $(s_1, \ldots, s_{t-1})$. Đây là thành phần tương đương nhất với trạng thái ẩn của RNN và là ứng viên tự nhiên nhất để biểu diễn trạng thái DFA.

### 3.4.2 Rời rạc hóa trạng thái qua K-Means

Chạy Transformer đã huấn luyện trên tất cả các window huấn luyện bình thường. Thu thập tất cả các vector $h_t$ — mỗi token một vector cho mỗi window. Áp dụng gom cụm K-Means với $K$ cụm.

Mỗi centroid của cụm $c_k$ định nghĩa một trạng thái DFA $q_k \in Q$. Một trạng thái ẩn $h$ được ánh xạ tới trạng thái $q_k$ trong đó $k = \arg\min_j \|h - c_j\|_2$.

$K$ là một siêu tham số chi phối sự **đánh đổi giữa độ chính xác (fidelity) và tính gọn nhẹ (compactness)**:

- K thấp → DFA thô, eBPF map nhỏ, tỷ lệ xung đột không xác định cao hơn, khả năng xảy ra false positives.
- K cao → DFA tinh, map lớn hơn, tỷ lệ xung đột thấp hơn, rủi ro overfitting vào các chuỗi huấn luyện.

### 3.4.3 Xây dựng bước chuyển (Transition)

Đối với mỗi cặp token liên tiếp $(s_t, s_{t+1})$ trong dữ liệu huấn luyện:

1. Tính $h_t$ và $h_{t+1}$ qua forward pass.
2. Ánh xạ tới các trạng thái: $A = \text{cluster}(h_t)$, $B = \text{cluster}(h_{t+1})$.
3. Ghi lại bước chuyển ứng viên: $\delta(A,\ s_{t+1}) \to B$.

Kết quả thô là một **quan hệ** chuyển đổi (NFA), không phải là một hàm, vì cùng một cặp $(A, s)$ có thể ánh xạ tới các trạng thái đích khác nhau trên các chuỗi khác nhau.

### 3.4.4 Giải quyết tính không xác định (Non-determinism Resolution)

Tính không xác định phát sinh khi phép chiếu K-Means làm mất thông tin: hai chuỗi có lịch sử khác nhau có thể rơi vào cùng một cụm A, sau đó phân tách tại đầu vào $s$. Bốn chiến lược giải quyết được đánh giá bằng thực nghiệm:

| ID     | Chiến lược              | Cơ chế                                                                                                     | Sự đánh đổi                                                                            |
| ------ | ----------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **S1** | NFA→DFA Determinization | Xây dựng tập hợp con (Subset construction): các trạng thái DFA là các _tập hợp_ trạng thái NFA.            | Chính xác; có thể gây bùng nổ trạng thái (state explosion).                            |
| **S2** | Tăng K                  | Gom cụm tinh hơn làm giảm mất mát thông tin khi chiếu.                                                     | BPF map lớn hơn; cần hiệu chuẩn K.                                                     |
| **S3** | Majority Voting         | Với mỗi $(A, s)$, giữ lại trạng thái đích thường xuyên nhất.                                               | Đơn giản; các nhánh thiểu số bị loại bỏ âm thầm.                                       |
| **S4** | Statistical Pruning (θ) | Đếm tần suất các nhánh. Chỉ giữ lại các nhánh có tần suất $\ge \theta$ (ví dụ: 99%). Loại bỏ phần còn lại. | Đồ thị gọn nhẹ; khai thác sự phân bố syscall cực kỳ lệch (skewed). **Ứng viên chính.** |

**Lý do chọn S4 làm ứng viên chính:** Các khối lượng công việc server bình thường có tính lặp lại cao. Sự phân bố các bước chuyển bị lệch mạnh: một nhánh chiếm ưu thế (happy path), trong khi các nhánh thiểu số ($< \theta$) là các tạo vật của quá trình lượng tử hóa K-Means hơn là sự biến đổi hành vi thực sự.

### 3.4.5 Xuất DFA

DFA hoàn thiện được xuất ra `dfa_config.json` và được nạp vào các eBPF maps:

| BPF Map            | Key (Khóa)             | Value (Giá trị)          |
| ------------------ | ---------------------- | ------------------------ |
| `transition_table` | `(state_id, token_id)` | `next_state_id`          |
| `state_tier`       | `state_id`             | `{NORMAL, EDGE, REJECT}` |
| `thread_state`     | `TID`                  | `current_state_id`       |

---

## 3.5 Giai đoạn 4 — Thực thi tại Runtime (eBPF)

### 3.5.1 Bước DFA trên mỗi syscall

Trên mỗi sự kiện syscall, chương trình eBPF:

1. Đọc `current_state` từ `thread_state[TID]`.
2. Tra cứu `token_id = vocab[Syscall_Name]`.
3. Tra cứu `next_state = transition_table[(current_state, token)]`.
4. Nếu không tồn tại mục nhập → **Trạng thái từ chối (Rejecting State)** (bước chuyển không xác định).
5. Ghi `next_state` vào `thread_state[TID]`.
6. Đọc `state_tier[next_state]` và hành động tương ứng.

Chi phí: hai lần tra cứu BPF map O(1) trên mỗi syscall. Không có context switch sang userspace cho các bước chuyển bình thường.

### 3.5.2 Phản hồi phân tầng (Tiered Response)

| Tầng                     | Tiêu chí                                                           | Hành động                                                                                   |
| ------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| **Normal**               | Tầng của `next_state` = NORMAL                                     | Tiếp tục. Không tốn thêm chi phí ngoài việc cập nhật map.                                   |
| **Edge (Vùng xám)**      | Tầng của `next_state` = EDGE, hoặc token là OOV (không có trong Σ) | Thu thập window, gửi tới Rust Agent → Transformer ngoại tuyến để phân tích sâu. Không chặn. |
| **Rejecting (Tấn công)** | Không có mục nhập trong `transition_table` cho `(state, token)`    | BLOCK / KILL / ALERT.                                                                       |

**Định nghĩa trạng thái Edge:** Các trạng thái có tần suất xuất hiện trong thời gian huấn luyện thấp hơn một ngưỡng phân vị (percentile threshold). Những trạng thái này đại diện cho các hành vi hiếm gặp nhưng đã từng thấy trong quá trình huấn luyện — đủ nghi ngờ để cần phân tích sâu, nhưng không chặn ngay lập tức.

### 3.5.3 Khả năng kháng Mimicry Attack

Kẻ tấn công chèn các syscall trông có vẻ lành tính vào chuỗi khai thác để kéo giãn nó qua các ranh giới window sẽ khiến con trỏ DFA đi theo một lộ trình chuyển đổi không điển hình. Vì DFA là một whitelist nghiêm ngặt, ngay cả các token lành tính trong một ngữ cảnh trạng thái bất thường cũng sẽ thiếu các bước chuyển hợp lệ — DFA sẽ từ chối bất kể độ dài của phần đệm (padding). Thuộc tính này là hệ quả từ cấu trúc DFA, không phải là một quy tắc riêng biệt.

### 3.5.4 Vòng lặp cập nhật liên tục

Nếu Transformer xác định một window Edge thu thập được là false positive (ví dụ: do cập nhật phần mềm giới thiệu hành vi mới):

1. Mẫu chuyển đổi mới được thêm vào tập huấn luyện.
2. DFA được trích xuất lại (hoặc cập nhật tăng dần).
3. Các mục nhập `transition_table` mới được ghi nguyên tử vào BPF map.

Không cần nạp lại kernel. Con trỏ trạng thái theo từng thread tiếp tục từ vị trí hiện tại của nó.
