# thầy C:

- Hệ thống phát hiện của bạn có sự kế thừa và sử dụng lại một số thành phần từ các nghiên cứu khác. Bạn hãy làm rõ và làm nổi bật trong luận văn đâu là đóng góp mới, đóng góp chính mang tính sáng tạo của bản thân bạn trong nghiên cứu này?
- Hãy phân tích và nêu rõ ưu điểm của giải pháp phát hiện tấn công thực hiện ở cấp độ hệ điều hành (Kernel space) so với các phương pháp phát hiện khác hiện nay (ví dụ: các phương pháp phân tích log ở User space).
- xem xét các phương pháp khác tự tìm K tối ưu thay vì fix cứng (bổ sung vào future work)
- bổ sung tính gắn kết giữa tên đề tài và thực nghiệm: bổ sung định nghĩa cụ thể về hành vi truy cập dữ liệu trái phép và các tiêu chí rõ ràng

# thầy M:

- Hiệu quả của việc chuyển đổi từ không gian các trạng thái ẩn (hidden states) sang các trạng thái của mô hình tự động hữu hạn đơn định (DFA) phụ thuộc rất lớn vào phương pháp phân cụm. Tại sao trong luận văn bạn lại chỉ lựa chọn và sử dụng duy nhất thuật toán K-means? Cần bổ sung phần thảo luận sâu hơn về lý do khoa học chọn thuật toán này (thay vì chỉ giải thích do thuật toán phổ biến).
- Tại sao bạn lại cố định kích thước của cửa sổ trượt là 64 lời gọi hệ thống (system calls)? Cần đưa vào luận văn sự phân tích, thực nghiệm hoặc lập luận khoa học để chứng minh tại sao con số 64 này là tối ưu đối với mô hình của bạn.
- Về mặt lý thuyết, khi tăng số lượng cụm $K$ (số lượng trạng thái DFA) thì sẽ giảm bớt sự mất mát thông tin khi ánh xạ từ một tập dữ liệu rất lớn sang tập nhỏ. Tuy nhiên, kết quả thực nghiệm trong luận văn lại cho thấy khi số cụm tăng lên (ví dụ: trạng thái $K=64$) thì kết quả lại kém đi. Bạn hãy giải thích rõ nguyên nhân của hiện tượng này và làm rõ mối quan hệ giữa số lượng trạng thái DFA được chọn với hiệu quả phát hiện tấn công.

# thầy Toàn:

- Khi mô hình $DFA$ phát hiện ra một lời gọi hệ thống (system call) bị nghi ngờ và chuyển tiếp luồng dữ liệu sang cho mô hình Transformer phân tích sâu, làm thế nào hệ thống của bạn có thể tách biệt và theo dõi (track) chính xác các chuỗi syscall đó thuộc về đúng tiến trình hành vi tấn công (attack) mà không bị lẫn với các tiến trình bình thường khác? (Gợi ý chỉnh sửa: Làm rõ việc sử dụng định danh tiến trình - Process ID để phân tách luồng dữ liệu).
- Trong tập dữ liệu thực nghiệm của bạn, tỷ lệ các lời gọi hệ thống bị gắn nhãn "nghi ngờ" và cần đẩy lên cho mô hình Transformer xử lý chiếm khoảng bao nhiêu phần trăm? Cần bổ sung số liệu này để đánh giá hiệu năng thực tế của hệ thống.

# thầy Thọ:

- Tại Slide số 15, mô hình cấu hình với 64 trạng thái (states) và 102 lời gọi hệ thống (syscalls). Về mặt toán học, phép tính của bạn để ra dung lượng 204 KB là bị nhầm lẫn (do nhầm lẫn giữa vocabulary của Transformer sang token ID). Hãy tính toán lại chính xác thông số này trong luận văn (kết quả chuẩn xác phải là khoảng 25 - 26 KB).
