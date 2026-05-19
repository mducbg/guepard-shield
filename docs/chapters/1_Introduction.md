# Chương 1 — Giới thiệu

## 1.1 Đặt vấn đề

Khối lượng công việc trên Linux server (web servers, databases, microservices) là mục tiêu thường xuyên của các cuộc post-exploitation attacks: reverse shells, privilege escalation, data exfiltration, và persistence. Những cuộc tấn công này để lại dấu vết hệ thống tại lớp syscall — giao diện phần mềm thấp nhất trước khi tương tác với kernel.

Host-based Intrusion Detection Systems (HIDS) hoạt động ở cấp độ syscall cung cấp khả năng quan sát chi tiết nhất về hành vi của tiến trình. Tuy nhiên, các phương pháp hiện tại đối mặt với một sự đánh đổi cơ bản:

- **Các mô hình deep learning** (Transformers, LSTMs) đạt được độ chính xác phát hiện cao nhưng không thể chạy bên trong kernel — inference latency cao hơn nhiều bậc so với mức cho phép của việc chặn bắt syscall trong thời gian thực.
- **Các hệ thống dựa trên quy tắc** (ví dụ: các quy tắc mặc định của Falco) chạy ở tốc độ của kernel nhưng được xây dựng thủ công, cứng nhắc và không bắt được các mẫu trình tự phức tạp đặc trưng của các cuộc tấn công.

Chưa có nghiên cứu hiện tại nào giải quyết được khoảng cách này một cách toàn diện (end-to-end): từ một mô hình nơ-ron đã được huấn luyện đến một cơ chế phát hiện được thực thi trong kernel một cách tự động.

## 1.2 Bối cảnh và các vấn đề nghiên cứu

Các nghiên cứu HIDS dựa trên syscall hiện nay thường dừng lại ở các bảng báo cáo độ chính xác. Các hạn chế chính bao gồm:

| Phương pháp                               | Ưu điểm                             | Hạn chế                                                                   |
| ----------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------- |
| Transformer / LSTM anomaly detection      | AUROC cao trên các bộ dữ liệu chuẩn | Không thể chạy trong kernel; không có lộ trình triển khai                 |
| Quy tắc dựa trên N-gram / tần suất        | Nhanh, có thể triển khai            | Thiếu ngữ cảnh trình tự; các ngưỡng (thresholds) được điều chỉnh thủ công |
| Các quy tắc mặc định của Falco / Tetragon | Sẵn sàng cho môi trường production  | Được viết bởi con người, không có khả năng học từ dữ liệu                 |
| Trích xuất Automata từ RNNs               | Cầu nối giữa neural ↔ discrete      | Chưa được áp dụng cho syscall HIDS hoặc triển khai trên eBPF              |

Khoảng cách thực thi (enforcement gap) — giữa những gì một mô hình học được và những gì có thể thực thi ở tốc độ kernel — là vấn đề cốt lõi chưa được giải quyết.

## 1.3 Mục tiêu nghiên cứu và Khung khái niệm

**Mục tiêu chính:** Xây dựng một pipeline đầu-cuối (end-to-end) tự động chuyển đổi kiến thức học được của một Decoder-only Transformer thành một Deterministic Finite Automaton (DFA) có thể triển khai dưới dạng chương trình eBPF kernel.

**Mô hình vận hành (Runtime model):** Hệ thống hoạt động trên một luồng syscall liên tục của mỗi thread. Không có khái niệm "ghi lại" (recording) tại thời điểm thực thi — chỉ có các sliding windows gồm `W` syscalls. Mỗi window được đánh giá dựa trên trạng thái của DFA. Một cảnh báo sẽ được đưa ra khi DFA rơi vào trạng thái từ chối (rejecting state).

**Pipeline:**

```
Syscall stream
        ↓
[Phase 1] EDA + Data Preprocessing
        ↓
[Phase 2] Decoder-only Transformer — Next-Token Prediction trên các chuỗi syscall
        ↓  (trạng thái ẩn - hidden states)
[Phase 3] DFA Extraction — K-Means clustering → transition table → non-determinism resolution
        ↓  (dfa_config.json)
[Phase 4] eBPF Enforcement — Tra cứu DFA với độ phức tạp O(1) trên mỗi syscall, trạng thái theo từng thread
```

**Tại sao dùng DFA thay vì model inference:** Một bước chuyển trạng thái (transition) của DFA là một thao tác tra cứu BPF map với độ phức tạp O(1). Việc thực thi Transformer inference trên một window có độ phức tạp O(W·d²) — không khả thi ở tốc độ kernel. DFA mã hóa ranh giới quyết định (decision boundary) mà mô hình đã học được sang một định dạng mà kernel có thể thực thi.

## 1.4 Các đóng góp dự kiến

1. **Pipeline đầu-cuối:** Syscall HIDS từ phát hiện bất thường bằng mạng nơ-ron → trích xuất DFA → thực thi tại eBPF kernel, với phép đo độ trễ (latency) trên các khối lượng công việc thực tế.
2. **Trích xuất DFA từ các trạng thái ẩn của Transformer:** Hình thức hóa việc ánh xạ từ liên tục sang rời rạc (K-Means trên final-layer embeddings), bao gồm nghiên cứu về các chiến lược giải quyết tính không xác định (non-determinism resolution strategies) (S1–S4).
3. **Khả năng chống lại mimicry attack thông qua cấu trúc DFA:** Lập luận chính thức rằng các cuộc tấn công mimicry dựa trên kỹ thuật chèn dữ liệu đệm (padding) sẽ khiến con trỏ DFA trôi vào các trạng thái biên (edge states), từ đó không thể phục hồi bất kể độ dài của phần đệm.
4. **Đánh giá trên LID-DS và DongTing:** AUROC/F1 ở cấp độ window cho Teacher; fidelity và FPR cho DFA Student; độ trễ thực thi trên các khối lượng công việc nginx, redis, postgres.
5. **Phân tích độ bao phủ MITRE ATT&CK:** Ánh xạ các mẫu trạng thái từ chối (rejecting-state patterns) của DFA với các kỹ thuật tấn công đã biết.

Đây là các mục tiêu của luận văn, không phải là các kết quả đã hoàn thành trong trạng thái hiện tại của kho lưu trữ.

## 1.5 Cấu trúc luận văn

_(Sẽ được viết sau khi hoàn thành dự thảo tất cả các chương.)_
