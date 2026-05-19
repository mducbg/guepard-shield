# Chương 6 — Kết luận

## 6.1 Tóm tắt

*(Chương này hiện là phần giữ chỗ. Chưa thể đưa ra kết luận nào cho Giai đoạn 2 / Giai đoạn 3 vì các giai đoạn đó chưa được triển khai trong trạng thái hiện tại của kho lưu trữ.)*

Hiện tại, bản tóm tắt có thể xác nhận được của kho lưu trữ này hẹp hơn:

1. Giai đoạn 1 EDA và tiền xử lý (preprocessing) đã hoàn thành.
2. Các giai đoạn Transformer, trích xuất DFA và thực thi trong kernel (kernel-enforcement) sau này vẫn là các đề xuất công việc.
3. Bất kỳ kết luận nào trong tương lai về độ trung thực (fidelity), chi phí vận hành (runtime overhead), hoặc khả năng kháng mimicry (mimicry resilience) phải chờ một chu kỳ triển khai và đánh giá mới.

## 6.2 Các hạn chế

- **Static DFA:** DFA nắm bắt các hành vi có thể quan sát được trong tập huấn luyện. Các hành vi phần mềm hợp lệ mới (cập nhật, các đường dẫn mã mới) có thể gây ra false positives cho đến khi DFA được cập nhật.
- **Sự hội tụ của K-Means:** Chất lượng của DFA phụ thuộc vào chất lượng của việc gom cụm trạng thái ẩn (hidden state clustering). Các trạng thái ẩn của Transformer có thể không được gom cụm sạch cho tất cả các kịch bản của LID-DS.
- **Syscall-name vocabulary:** Bộ từ vựng hiện tại (~102 tokens) chỉ mã hóa tên syscall, bỏ qua mã trả về (return code) và ngữ cảnh đối số (argument context). Điều này hạn chế khả năng phân biệt đối với các syscall mà hành vi của chúng thay đổi theo kết quả (ví dụ: một lệnh `open` thành công so với một lệnh thất bại).
- **Quản lý vòng đời thread (Thread lifecycle management):** Các thread tồn tại trong thời gian ngắn hoặc các thread có rất ít syscall có thể không tích lũy đủ ngữ cảnh để theo dõi trạng thái DFA có ý nghĩa.

## 6.3 Hướng phát triển trong tương lai

- **Online DFA update:** Sử dụng K-Means tăng dần (Incremental K-Means) hoặc gom cụm trực tuyến (online clustering) để cập nhật các trạng thái DFA mà không cần huấn luyện lại toàn bộ khi quan sát thấy các mẫu hành vi mới.
- **Đánh giá đối kháng (Adversarial evaluation):** Các cuộc tấn công White-box mimicry attacks chống lại một DFA đã biết; nghiên cứu xem liệu kẻ tấn công có kiến thức về cấu trúc DFA có thể tạo ra các lộ trình chuyển đổi hợp lệ thông qua automaton hay không.
- **Multi-host DFA sharing:** Huấn luyện một DFA dùng chung từ các dấu vết syscall trên nhiều máy chủ; đo lường khả năng tổng quát hóa.
- **Các phương pháp biểu diễn trạng thái thay thế:** Thay thế K-Means bằng lượng tử hóa đã học (learned quantization - VQ-VAE) để có các ranh giới cụm tốt hơn; so sánh fidelity với baseline K-Means.
- **Mở rộng sang các container workloads:** Áp dụng pipeline cho các dịch vụ vi mô được container hóa (containerized microservices); đánh giá các chính sách DFA cho mỗi container.
- **Composite tokenization:** Mở rộng bộ từ vựng sang các cặp `(Syscall_Name, ReturnCode_Bucket)`, phân nhóm các mã trả về (thành công, EPERM, EAGAIN, lỗi-khác). Điều này làm tăng tính phong phú của bảng chữ cái và được kỳ vọng sẽ cải thiện cả khả năng phân biệt của Transformer và tính đặc hiệu của các bước chuyển DFA.
