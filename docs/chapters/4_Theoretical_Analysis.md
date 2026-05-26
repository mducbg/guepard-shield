# Chương 4 — Phân tích lý thuyết

Chương này phân tích các thuộc tính dự kiến của kiến trúc Giai đoạn 2 / Giai đoạn 3 / Giai đoạn 4 đã đề xuất. Các lập luận này là khung lý thuyết cho các công việc trong tương lai, không phải là sự xác thực của một pipeline đã được triển khai trong codebase hiện tại.

## 4.1 Độ phức tạp tính toán

### 4.1.1 Transformer Inference (Ngoại tuyến)

Lượt forward pass trên một window gồm W token với model dimension d và L lớp:

$$O(L \cdot W^2 \cdot d)$$

Không khả thi tại runtime đối với các giá trị điển hình (W=128, d=256, L=6 → hàng triệu FLOPs mỗi window, độ trễ ở mức mili giây).

### 4.1.2 Duyệt DFA (Runtime)

Mỗi syscall yêu cầu:
- 1 lần tra cứu BPF hash map cho bước chuyển (transition): O(1)
- 1 lần tra cứu BPF hash map cho phân loại (category/tier): O(1)
- 1 lần ghi vào BPF hash map cho trạng thái mới: O(1)

Tổng cộng: **O(1) cho mỗi syscall**, bất kể độ dài chuỗi hay kích thước mô hình.

### 4.1.3 Trích xuất K-Means (Ngoại tuyến)

Thu thập các trạng thái ẩn: một lượt forward pass trên dữ liệu huấn luyện — O(N · L · W² · d) trong đó N là số lượng window huấn luyện. Sự hội tụ của K-Means: O(N · K · d · I) trong đó I là số lần lặp (iterations). Cả hai đều là chi phí ngoại tuyến thực hiện một lần.

---

## 4.2 Kích thước DFA và Bộ nhớ eBPF

Bảng chuyển đổi (transition table) có tối đa $K \times |\Sigma|$ mục nhập. Với K=500, |Σ|=1000: có 500,000 mục nhập. Với 8 bytes cho mỗi key và 4 bytes cho mỗi value, dung lượng khoảng ~6 MB — nằm tốt trong giới hạn của `BPF_MAP_TYPE_HASH` (mặc định max_entries có thể cấu hình lên đến hàng triệu).

Trong thực tế, bảng này có tính thưa thớt (sparse): chỉ các bước chuyển quan sát được trong dữ liệu huấn luyện mới được điền vào. Các cặp (A, token) không quan sát được sẽ mặc định là trạng thái từ chối (rejecting).

---

## 4.3 Tỷ lệ không xác định và Lựa chọn chiến lược

Định nghĩa **tỷ lệ xung đột (conflict rate)** của quá trình trích xuất DFA là:

```math
\text{conflict\_rate} = \frac{|\{(A, s) : |\{\delta(A,s)\}| > 1\}|}{|\{(A, s) : \delta(A,s) \neq \emptyset\}|}
```
**Khẳng định:** Đối với dữ liệu syscall có phân phối bước chuyển bị lệch mạnh (strongly skewed), tỷ lệ xung đột giảm đơn điệu theo K và được giảm thiểu hơn nữa bởi chiến lược cắt tỉa S4.

**Lập luận:** Nếu hai chuỗi đạt đến cụm A thông qua các lịch sử khác nhau nhưng hành xử giống hệt nhau trong tương lai (cùng gán trạng thái tiếp theo), thì không có xung đột phát sinh. Xung đột chỉ xảy ra khi K-Means gộp hai lịch sử mà Transformer có thể phân biệt được. Tăng K làm giảm việc gộp như vậy. Cắt tỉa S4 loại bỏ các xung đột gây ra bởi nhiễu lượng tử hóa hiếm gặp (tần suất < θ) mà không loại bỏ các bước chuyển thực sự.

**Kế hoạch xác thực thực nghiệm:** Đo tỷ lệ xung đột trên K ∈ {100, 200, 500, 1000} và θ ∈ {0.95, 0.99, 0.999} trên LID-DS-2021. Báo cáo fidelity (DFA AUROC so với Teacher AUROC) và FPR dưới dạng các hàm kết hợp của (K, θ).

---

## 4.4 Khả năng kháng Mimicry Attack

**Mối đe dọa:** Kẻ tấn công biết phương pháp chung (DFA whitelist) nhưng không biết DFA cụ thể. Kẻ tấn công chèn k syscall đệm (padding) giữa mỗi syscall khai thác để kéo giãn chuỗi khai thác và tránh việc rơi vào một window phát hiện.

**Khẳng định:** Việc đệm làm cho con trỏ trạng thái DFA trôi vào các trạng thái edge hoặc rejecting độc lập với k.

**Lập luận:** Giả sử $p_1, p_2, \ldots, p_k$ là các syscall đệm. Mỗi $p_i$ là một bước chuyển hợp lệ trong phân phối huấn luyện — nhưng là từ trạng thái hiện tại của DFA, không phải từ trạng thái trên lộ trình khai thác (exploit path). Sau syscall đệm đầu tiên, DFA chuyển sang một trạng thái $q'$ được đạt tới bằng cách quan sát $p_1$ từ một trạng thái thuộc exploit-path. $q'$ có khả năng chưa bao giờ được quan sát trong quá trình huấn luyện (vì các trạng thái exploit-path không có trong dữ liệu huấn luyện — mô hình chỉ được huấn luyện trên dữ liệu bình thường). Do đó $q'$ hoặc là một trạng thái edge hoặc không có các bước chuyển đi cho các token tiếp theo.

**Hạn chế:** Lập luận này giữ vững nghiêm ngặt khi các trạng thái exploit-path tách biệt với các trạng thái normal-path trong DFA. Nếu K quá nhỏ và các trạng thái bình thường cũng như khai thác bị gộp vào cùng một cụm, khả năng kháng cự sẽ giảm sút. Đây là một động lực để chọn K đủ lớn.

---

## 4.5 Giới hạn độ trung thực của DFA (DFA Fidelity Bound)

**Định nghĩa:** Fidelity = tỷ lệ các window kiểm tra mà DFA và Transformer đồng ý về phán quyết bất thường (trên/dưới ngưỡng).

**Giới hạn trên:** Fidelity bị giới hạn bởi lỗi lượng tử hóa của K-Means. Về mặt hình thức, nếu hai window $w_1, w_2$ tạo ra các quỹ đạo DFA giống hệt nhau (cùng chuỗi state ID) nhưng có điểm NLL của Transformer khác nhau, chúng sẽ nhận được cùng một phán quyết từ DFA. Mất mát fidelity tỷ lệ thuận với tỷ lệ xảy ra các va chạm (collisions) như vậy.

Tăng K làm giảm va chạm nhưng tăng kích thước map. Giá trị K tối ưu sẽ tối thiểu hóa FPR + (1 - Fidelity) tùy thuộc vào các giới hạn về kích thước map.
