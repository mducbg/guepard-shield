# Chương 2 — Tổng quan tài liệu

Chương này đánh giá nền tảng nghiên cứu tạo động lực cho kiến trúc **tiếp theo** của Guepard Shield. Chương này nên được xem như tài liệu hỗ trợ và đặt vấn đề, chứ không phải là bằng chứng cho thấy Phase 2 hoặc Phase 3 đã được triển khai trong trạng thái hiện tại của kho lưu trữ.

## 2.1 Phạm vi nghiên cứu

Luận văn này tập trung vào **syscall-level anomaly detection** cho các máy chủ Linux chạy các khối lượng công việc (workloads) của server. Phạm vi được giới hạn bởi:

- **Monitor layer:** Giao diện System call — lớp phần mềm thấp nhất trước kernel.
- **Attacker model:** Post-exploitation (giả định mã độc đã được thực thi). Kẻ tấn công thực hiện reverse shell, privilege escalation, exfiltration, hoặc persistence. Black-box: kẻ tấn công không biết DFA hiện tại.
- **Ngoài phạm vi:** Adversarial evasion chống lại thuật toán học máy, kernel-level rootkits, phát hiện xâm nhập ở network-layer.

**Các bộ dữ liệu (Datasets) trong phạm vi:**

| Dataset     | Size                             | Role                     |
| ----------- | -------------------------------- | ------------------------ |
| LID-DS-2021 | 17,190 recordings, 15 scenarios  | Primary — train/val/test |
| LID-DS-2019 | ~11,000 recordings, 10 scenarios | Cross-domain validation  |
| DongTing    | 18,966 sequences                 | Cross-domain validation  |

## 2.2 Các nghiên cứu liên quan

### 2.2.1 Syscall Anomaly Detection

- **STIDE / N-gram models:** Đếm tần suất các n-gram của syscall; gắn cờ (flag) các sai lệch so với phân phối huấn luyện. Nhanh nhưng bỏ lỡ các phụ thuộc dài hạn (long-range dependencies). Không có ngữ cảnh trình tự vượt quá n.
- **Các phương pháp dựa trên HMM:** Mô hình hóa các chuỗi syscall dưới dạng hidden Markov chains. Nắm bắt được một phần cấu trúc trình tự nhưng khả năng biểu diễn hạn chế; chi phí tốn kém khi mở rộng từ vựng (vocabulary).
- **LSTM / GRU anomaly detection (DeepLog, v.v.):** Next-token prediction trên các chuỗi syscall sử dụng mạng nơ-ron hồi quy (recurrent networks). Đạt AUROC cao trên các benchmark theo phong cách LID-DS. Không thể triển khai trong kernel. Không có lộ trình cho enforcement.
- **Transformer-based detection:** Cơ chế Attention nắm bắt ngữ cảnh dài hơn so với LSTM. Là phương pháp tiên tiến nhất (State of the art) trên bộ dữ liệu LID-DS-2021. Có cùng hạn chế triển khai như LSTM.

### 2.2.2 Trích xuất Automata từ Mạng nơ-ron

- **Weiss et al. (2018) — Trích xuất Automata từ RNNs:** Gom cụm (Cluster) các RNN hidden states bằng k-means; xây dựng DFA từ các chuyển đổi (transitions) quan sát được. Hoạt động tốt cho RNNs (trạng thái hồi quy tự nhiên). Chưa được áp dụng cho Transformers hoặc syscall HIDS.
- **Các nghiên cứu sau đó:** Mở rộng cho LSTM, GRU, và các mô hình attention. Phát hiện chính: chất lượng gom cụm quyết định độ chính xác (fidelity) của DFA. Giải quyết tính không xác định (non-determinism resolution) là thách thức mở (open challenge) chính.
- **Khoảng trống (Gap):** Chưa có nghiên cứu trước đây nào trích xuất DFA từ các mô hình Transformer syscall và triển khai chúng vào một lớp kernel enforcement.

### 2.2.3 Security Enforcement dựa trên eBPF

- **Falco:** Giám sát Syscall thông qua eBPF với các quy tắc YAML được viết thủ công. Sẵn sàng cho môi trường production nhưng các quy tắc do con người viết; không học từ dữ liệu.
- **Tetragon:** Enforcement dựa trên eBPF với chính sách dạng mã (policy-as-code). Khoảng trống tương tự — các chính sách do con người tạo ra.
- **Khoảng trống (Gap):** Không có công cụ hiện tại nào tự động trích xuất các chính sách thực thi eBPF từ một mô hình nơ-ron đã được huấn luyện.

## 2.3 Cơ sở lý thuyết: Decoder-only Transformer

Một Decoder-only Transformer với causal self-attention xử lý các chuỗi token một cách autoregressively. Causal mask đảm bảo token $s_t$ chỉ chú ý (attends) tới $s_1, \ldots, s_{t-1}$. Được huấn luyện với next-token prediction:

$$\mathcal{L} = -\sum_t \log P(s_{t+1} \mid s_1, \ldots, s_t)$$

Đầu ra của lớp cuối cùng (final-layer output) $h_t \in \mathbb{R}^d$ tại vị trí $t$ mã hóa toàn bộ causal context — nó là thành phần tương tự nhất với RNN hidden state và là cơ sở để trích xuất trạng thái DFA.

## 2.4 Cơ sở lý thuyết: Deterministic Finite Automaton

Một DFA là một bộ 5 thành phần (5-tuple) $(Q, \Sigma, \delta, q_0, F)$ trong đó:

- $Q$: tập hợp hữu hạn các trạng thái
- $\Sigma$: bảng chữ cái đầu vào hữu hạn (finite input alphabet)
- $\delta: Q \times \Sigma \to Q$: hàm chuyển đổi (transition function)
- $q_0 \in Q$: trạng thái bắt đầu (initial state)
- $F \subseteq Q$: tập hợp các trạng thái chấp nhận (accepting states) (hoặc, trong nghiên cứu này là trạng thái từ chối - rejecting states)

Một DFA xử lý đầu vào từng ký hiệu (symbol) một với độ phức tạp O(1) cho mỗi bước. Thuộc tính này biến DFA thành mục tiêu triển khai tự nhiên cho eBPF.

## 2.5 Cơ sở lý thuyết: eBPF

eBPF (Extended Berkeley Packet Filter) cho phép các chương trình đã được xác minh (verified programs) chạy trong Linux kernel mà không cần một kernel module. Các thuộc tính chính:

- Các chương trình được gắn vào các tracepoints hoặc LSM hooks — kích hoạt trên mọi syscall khớp (matching).
- BPF maps (các kho lưu trữ key-value được chia sẻ giữa kernel và userspace) cung cấp khả năng tra cứu (lookup) với độ phức tạp O(1).
- Giới hạn stack (Stack limit): 512 bytes. Toàn bộ trạng thái (state) phải nằm trong BPF maps.
- Cập nhật map nguyên tử (Atomic map updates) cho phép tải lại nóng (hot-reload) các DFA transitions mà không cần khởi động lại kernel.

