Báo cáo Tổng hợp Nghiên cứu: Trích xuất Luật Bảo mật Khả diễn từ Mô hình Transformer dựa trên Syscall và Triển khai eBPF

1. Tổng quan Kiến trúc Hệ thống: Từ Ingestion đến Enforcement

Dưới góc độ kiến trúc hệ thống, quy trình phát hiện xâm nhập hiện đại dựa trên syscall được cấu trúc thành một pipeline khép kín gồm: Data Ingestion (eBPF) -> Tokenization (BPE) -> Teacher Inference (Transformer) -> Uncertainty Routing -> Student Execution (Rules in Kernel). Trong sơ đồ này, mô hình Transformer đóng vai trò là "Teacher" nắm bắt ngữ nghĩa phức tạp, trong khi các tập luật được trích xuất (Student) thực thi trực tiếp tại mức kernel để đảm bảo hiệu suất runtime.

2. Hệ thống IDS dựa trên Syscall: Từ Mô hình Truyền thống đến Transformer

Sự chuyển dịch chiến lược từ phân tích chữ ký tĩnh sang phân tích hành vi động qua syscall là tất yếu trong môi trường container. Syscall đại diện cho ranh giới bảo mật cuối cùng giữa ứng dụng và kernel. Cơ chế Self-attention của Transformer cho phép xử lý các chuỗi sự kiện dài và nắm bắt các phụ thuộc xa (long-range dependencies), vượt qua hạn chế của RNN/LSTM về gradient biến mất. Tuy nhiên, các kiến trúc này đòi hỏi phương pháp tiền xử lý khắt khe. Việc sử dụng Byte-Pair Encoding (BPE) là then chốt để nén các chuỗi syscall thô thành các token có nghĩa, giúp mô hình tractability tốt hơn trên các tập dữ liệu như MH-1M.

Các nghiên cứu trọng tâm:

- [Senoussi & Salmi, 2026 ] "Next-Generation IDS: Leveraging Transformer and Hybrid Deep Architectures for Robust Cyber Threat Detection" — Preprint. Sử dụng Transformer để nhận diện chuỗi tấn công dài, vượt trội hơn LSTM/CNN truyền thống trên tập NGIDS-DS.
- [Al Siam et al., 2026 ] "TransCall: A Transformer-Driven Framework for Zero-Day Malware Detection Using System Call Sequences" — IEEE ICAIC. Xử lý syscall như ngôn ngữ tự nhiên để phát hiện zero-day.
- [Fournier et al., 2023] "Language Models for Novelty Detection in System Call Traces" — arXiv. Đánh giá Transformer đạt F1/AUROC >95%. Lưu ý kỹ thuật: Nghiên cứu chỉ ra LSTM đôi khi vẫn vượt trội hơn Transformer trong một số bài kiểm tra độ mới (novelty), cho thấy sự cần thiết của việc tối ưu hóa kiến trúc thay vì mặc định sử dụng Transformer.
- [Duan et al., 2023] "DongTing: A large-scale dataset for anomaly detection of the Linux kernel" — J. of Systems & Software. Cung cấp tập dữ liệu syscall quy mô lớn, thu thập qua eBPF từ 200+ phiên bản kernel.
- [Mvula et al., 2023] "Evaluating Word Embedding Feature Extraction Techniques for HIDS" — Discover Data. Cảnh báo rủi ro: Word2Vec gây ra hiện tượng rò rỉ dữ liệu (data leakage) do các mẫu trùng lặp, khuyến nghị sử dụng các phương pháp encoding thay thế như BPE.

3. Giải thích và Trích xuất Luật từ Mạng Neural (XAI & Rule Extraction)

XAI là yêu cầu bắt buộc trong điều tra số (forensic accountability). Thay vì sử dụng các phương pháp hậu kiểm (post-hoc) tốn kém, kiến trúc này ưu tiên mô hình đại diện (surrogate modeling) thông qua chưng cất tri thức (Knowledge Distillation). Mục tiêu là chuyển đổi tri thức từ Transformer sang các tập luật IF-THEN tường minh như RuleFit, Anchors hoặc Decision Trees để thực thi tại kernel.

Các nghiên cứu trọng tâm:

- [Ables et al., 2024] "Eclectic Rule Extraction for Explainability of Deep Neural Network based IDS" — arXiv. Đề xuất phương pháp trích xuất luật hỗn hợp đạt độ trung thành (fidelity) 99.9% so với mô hình gốc.
- [Abou El Houda et al., 2022] "A Novel IoT-Based Explainable Deep Learning Framework for Intrusion Detection" — IEEE Network. Ứng dụng RuleFit để trích xuất luật quyết định từ mô hình deep learning.
- [Herbinger et al., 2023] "Leveraging Model-Based Trees as Interpretable Surrogate Models for Model Distillation" — ECAI Workshops. Đánh giá sự đánh đổi giữa tính khả diễn (interpretability) và độ trung thành (fidelity).
- [Friedman et al., 2023] "Learning Transformer Programs" — NeurIPS. Chuyển đổi Transformer thành các chương trình logic rời rạc (RASP), giúp con người có thể kiểm chứng trực tiếp.

4. Bảo mật Runtime và Thực thi Chính sách qua eBPF

eBPF đại diện cho bước tiến kiến trúc cho phép chạy các chương trình sandboxed trong kernel với độ trễ cực thấp. Việc đưa logic kiểm tra (luật đã trích xuất) vào kernel giúp loại bỏ chi phí chuyển ngữ cảnh (context switch). Tuy nhiên, trình xác thực (verifier) của eBPF áp đặt các giới hạn nghiêm ngặt: stack size 512 bytes và không cho phép vòng lặp vô hạn, buộc các luật Student phải cực kỳ tối ưu.

Các nghiên cứu trọng tâm:

- [Her et al., 2025 ] "An In-Depth Analysis of eBPF-Based System Security Tools in Cloud-Native Environments" — IEEE Access. So sánh hiệu năng: Tetragon có mức chiếm dụng CPU khoảng 73% và thời gian phát hiện trung bình (MTTD) cực nhanh là 1.178s cho các ca thoát khỏi container, tối ưu hơn so với KubeArmor (overhead lên tới 88%).
- [Zhang et al., 2024] "Real-Time Intrusion Detection and Prevention with Neural Network in Kernel Using eBPF" — IEEE/IFIP DSN. Triển khai suy luận trong kernel bằng số học số nguyên, đạt thời gian thực thi chỉ 5 microseconds.
- [Bachl et al., 2021] "A Flow-Based IDS Using Machine Learning in eBPF" — arXiv. Chứng minh việc thực thi cây quyết định trong eBPF tăng 20% hiệu suất so với userspace.
- [Ryu et al., 2026 ] "Hybrid Runtime Detection of Malicious Containers Using eBPF" — CMC. Kết hợp metadata luồng mạng và syscall để đạt độ chính xác 98.39%.

5. Khai thác Lô-gic Thời gian (Temporal Logic Mining)

Trong tấn công đa giai đoạn, thứ tự sự kiện quan trọng hơn tần suất. Khai thác lô-gic thời gian (TLM) cho phép biểu diễn hành vi dưới dạng các công thức LTL/STL. Các công thức này có thể được biên dịch trực tiếp thành các monitors trạng thái hiệu quả trong eBPF.

Các nghiên cứu trọng tâm:

- [Raha et al., 2022] "SCARLET: Scalable Anytime Algorithms for Learning Fragments of Linear Temporal Logic" — TACAS. Thuật toán học công thức LTL từ các trace syscall thực tế.
- [Bartocci et al., 2022] "Survey on Mining Signal Temporal Logic Specifications" — Information and Computation. Tổng quan về học STL, cung cấp nền tảng xây dựng đặc tả bảo mật tự động.
- [Aliabadi et al., 2021] "ARTINALI#: Efficient Intrusion Detection for Cyber-Physical Systems" — IJ CIP. Giảm 69% chi phí runtime bằng cách tối ưu hóa phủ đặc tả bảo mật qua mạng Bayesian.

6. Học Luật Bayesian và Cơ chế Uncertainty Routing

Hệ thống bảo mật cần định lượng sự không chắc chắn (uncertainty) để đưa ra quyết định an toàn. Cơ chế Uncertainty Routing đóng vai trò là bộ cân bằng tải: Các luật Student (khả diễn, tốc độ cao) xử lý các hành vi "biết rõ là bình thường" trong kernel; các mẫu có độ bất định cao (OOD) sẽ được chuyển lên mô hình Teacher (Transformer) ở userspace để phân tích sâu.

Các nghiên cứu trọng tâm:

- [Yang et al., 2017] "Scalable Bayesian Rule Lists" — ICML. Tìm kiếm danh sách luật tối ưu cân bằng giữa độ chính xác và tính thưa (sparsity) – yếu tố sống còn để vượt qua eBPF verifier.
- [Wang & Lin, 2021] "Hybrid Predictive Models: When an Interpretable Model Collaborates with a Black-box Model" — JMLR. Cơ sở lý thuyết cho việc phối hợp giữa mô hình khả diễn và hộp đen dựa trên phân phối xác định.
- [Perini et al., 2023] "Expected Anomaly Posterior (EAP)" — NeurIPS. Sử dụng uncertainty để phân loại các dị thường thực sự so với nhiễu dữ liệu.

7. Dịch chuyển Phân phối (Concept Drift) và Phát hiện OOD

Môi trường Kubernetes thường xuyên thay đổi (di trú container, cập nhật version) dẫn đến "Normality Shift". Nếu không có khả năng phát hiện drift, các luật eBPF sẽ nhanh chóng lỗi thời và gây ra dương tính giả.

Các nghiên cứu trọng tâm:

- [CAShift, 2025 ] "Benchmarking Log-Based Cloud Attack Detection under Normality Shift" — FSE. Khẳng định các IDS hiện tại thường thất bại khi ứng dụng cập nhật phiên bản hoặc thay đổi cấu trúc hạ tầng.
- [Yang et al., 2021] "CADE: Detecting and Explaining Concept Drift Samples for Security Applications" — USENIX Security. Phát hiện và giải thích mẫu drift bằng contrastive learning.
- [Han et al., 2023] "OWAD: Anomaly Detection in the Open World" — NDSS. Xử lý sự dịch chuyển dữ liệu lành tính mà không gây ra hiện tượng quên thảm họa.

8. Khảo sát và Tài liệu Tổng quan (Surveys)

Tài liệu Phạm vi Đóng góp lý thuyết quan trọng
Capuano et al. (2022) XAI trong Cybersecurity Cảnh báo XAI có thể bị khai thác cho tấn công đối kháng (Adversarial XAI).
Neupane et al. (2022) X-IDS Survey Chỉ ra sự thiếu hụt các chỉ số đánh giá độ trung thành (fidelity) đặc thù cho bảo mật.
Alam et al. (2022) NLP trong HIDS Xác nhận việc ứng dụng Transformer/BPE vào syscall vẫn còn ở giai đoạn sơ khởi.
Rjoub et al. (2023) XAI cho An ninh mạng Phân tích sâu các kỹ thuật giải thích tiền kiểm (ante-hoc) và hậu kiểm (post-hoc).

9. Tổng kết: Chuẩn so sánh và Thách thức kỹ thuật

9.1. Các chuẩn so sánh (Baselines) chính

1. DeepLog (LSTM): Chuẩn mực modeling chuỗi log/syscall theo NLP.
2. STIDE: Phương pháp n-gram kinh điển để đánh giá hiệu quả phát hiện so với Deep Learning.
3. Isolation Forest: Baseline không giám sát hàng đầu nhờ tính hiệu quả thực thi cao.
4. ADFA-LD & DongTing: Hai bộ dữ liệu tiêu chuẩn để benchmark khả năng tổng quát hóa.

9.2. Rủi ro kỹ thuật và Thách thức chưa được giải quyết (Identified Gaps)

- Tối ưu hóa ràng buộc eBPF: Thiếu nghiên cứu về việc tích hợp trực tiếp các ràng buộc của verifier (branching factor, stack depth) vào quá trình huấn luyện Student.
- Uncertainty Routing trong Kernel: Chưa có công trình nào triển khai tính toán xác suất Bayesian (BRL) trực tiếp tại kernel hook để kích hoạt chuyển tiếp mẫu OOD.
- Fidelity-Interpretability Trade-off: Thiếu khung đánh giá đa chiều so sánh giữa nhiều loại Student (Decision Tree vs. BRL vs. LTL) trên cùng một khối lượng syscall.
- Tính trung thực của luật (Faithfulness Gap): Nguy cơ "hallucination" khi luật trích xuất không phản ánh đúng logic thực tế của Transformer Teacher.

  9.3. Định hướng nghiên cứu bổ sung

- Verified eBPF Programs: Tìm kiếm các phương pháp kiểm chứng hình thức (formal verification) cho mã C được sinh tự động từ luật.
- Lượng tử hóa (Quantization): Áp dụng số học số nguyên (integer-only arithmetic) cho các tập luật Student để tương thích hoàn toàn với giới hạn tập lệnh của eBPF.
- Cedar Policy Mapping: Chuyển đổi luật khả diễn sang các ngôn ngữ quản lý chính sách hiện đại để tích hợp vào hệ sinh thái Cloud-native.
