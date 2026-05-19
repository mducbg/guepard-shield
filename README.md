# Guepard Shield: Cơ chế phát hiện xâm nhập hệ thống sử dụng thông tin mức Kernel

**Guepard Shield** là một khung nghiên cứu cho Hệ thống phát hiện xâm nhập trên máy chủ (Host-based Intrusion Detection - HIDS), nhằm thu hẹp khoảng cách giữa độ chính xác của Deep Learning và hiệu suất xử lý ở mức Kernel.

## 🎯 Mục tiêu dự án

Mục tiêu nghiên cứu dài hạn là nghiên cứu cách thức chuyển đổi việc phát hiện bất thường dựa trên syscall từ các thực nghiệm học máy ngoại tuyến (offline) sang việc thực thi nhẹ nhàng ở phía kernel bằng eBPF.

Dự án được chia thành 4 giai đoạn cốt lõi:

1. **Giai đoạn 1:** Phân tích dữ liệu (EDA) và Tiền xử lý.
2. **Giai đoạn 2 (Teacher):** Huấn luyện mô hình Transformer (Decoder-only) để học hành vi hệ thống bình thường.
3. **Giai đoạn 3 (Student):** Chưng cất tri thức từ Transformer sang Automata hữu hạn đơn định (DFA).
4. **Giai đoạn 4 (Deployment):** Thực thi DFA trong Linux Kernel bằng eBPF với độ phức tạp O(1).

## 🏗 Cấu trúc Monorepo

Dự án được tổ chức dưới dạng một workspace thống nhất cho cả Python (ML) và Rust (eBPF).

- **`guepard-shield-model/`**: Logic cốt lõi bằng Python.
  - `gp/`: Package chính (có thể import qua `import gp`). Chứa các bộ nạp dữ liệu, chẩn đoán và đường dẫn dùng chung.
- **`guepard-shield-ebpf/`**: Mã nguồn eBPF (Rust) chạy trong kernel.
- **`guepard-shield/`**: Agent chạy ở phía người dùng (Rust) sử dụng thư viện Aya.
- **`data/`**: Các bộ dữ liệu syscall (LID-DS, DongTing).
- **`results/`**: Các kết quả thực nghiệm, checkpoint mô hình và biểu đồ.
- **`notebooks/`**: Các script nghiên cứu theo từng giai đoạn.
- **`docs/`**: Tài liệu dự án và nội dung luận văn (tiếng Việt).

## 🛠 Cài đặt & Phát triển

Dự án sử dụng **UV** cho Python và **Cargo** cho Rust. Mọi câu lệnh đều được thiết kế để chạy từ **Thư mục gốc của dự án**.

### 1. Môi trường Python

```bash
# Đồng bộ môi trường và cài đặt dependencies
uv sync
```

### 2. Chạy Pipeline ML (Phase 1 & 2)

```bash
# P1: EDA
uv run notebooks/p1/eda_lidds2021.py

# P2: Tiền xử lý cửa sổ syscall
uv run notebooks/p2/preprocess.py

# P2: Huấn luyện mô hình Transformer
uv run notebooks/p2/train.py

# P2: Đánh giá mô hình (Inference & Metrics)
uv run notebooks/p2/eval.py --ckpt results/p2/checkpoints/best.ckpt
```

### 3. Biên dịch Rust/eBPF (Giai đoạn 4)

```bash
cargo build --release
```

## 📜 Tiến độ dự án

- **Giai đoạn 1 (EDA):** ✅ **Hoàn thành**. Đã phân tích các bộ dữ liệu LID-DS và DongTing.
- **Giai đoạn 2 (Teacher):** ✅ **Hoàn thành**. Mô hình Transformer đạt AUROC 0.85 trên tập test LID-DS-2021.
- **Giai đoạn 3 (Student):** ⏳ **Đang triển khai**. Thiết kế thuật toán trích xuất DFA đã hoàn tất.
- **Giai đoạn 4 (Deployment):** ⏳ **Kế hoạch**. Đã xây dựng khung (scaffold) eBPF.

Để biết chi tiết kỹ thuật và kết quả thực nghiệm cụ thể, vui lòng xem:
👉 **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)** (Báo cáo tiến độ chi tiết)
