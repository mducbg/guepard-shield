# Master Research: Trích xuất Security Rule từ Mô hình Transformer Phân loại Syscall

**Công cụ:** PyTorch Lightning, Rust, Aya (eBPF), Cedar/CEL, Falco.

---

## 1. ABSTRACT

- **Bối cảnh:** Môi trường container ngày càng trở thành mục tiêu tấn công ở tầng kernel. Giám sát syscall thời gian thực là lớp phòng thủ quan trọng trong Cloud-native security.
- **Vấn đề:** Mô hình Deep Learning (Transformer) phát hiện syscall bất thường với độ chính xác cao nhưng là "hộp đen" — không giải thích được quyết định cho security analyst và không tái sử dụng được trong quy trình bảo mật thực tế.
- **Giải pháp:** Pipeline **RuleDistill** — dùng Transformer làm Teacher Oracle để distill tri thức phát hiện tấn công thành **security rule dạng human-readable**. Sau khi diễn giải thành công, các rule có thể triển khai trên bất kỳ enforcement engine nào phù hợp (Falco cho hiệu năng, Cedar/CEL cho khả năng biểu diễn cao hơn, v.v.).
- **Kết quả mong đợi:** Rule set đạt Fidelity >95% so với Teacher, FPR <1%, ngắn gọn đủ để analyst đọc và hiểu được.

---

## 2. INTRODUCTION

### 2.1. Động lực và Vấn đề

Syscall là giao diện duy nhất giữa userspace và kernel — mọi hành vi tấn công (container escape, privilege escalation, lateral movement) đều để lại dấu vết tại tầng này.

**Hướng truyền thống** (n-gram, STIDE, rules viết tay) — human-readable, overhead thấp — nhưng FPR cao, không scale khi attack landscape thay đổi, và không capture pattern đa bước hay cross-thread.

**Hướng Deep Learning** (Transformer, LSTM) — phát hiện chính xác — nhưng output là nhãn nhị phân: analyst không biết _tại sao_ một chuỗi syscall bị đánh dấu bất thường, không thể kiểm tra hay tin tưởng quyết định của model trong bối cảnh security.

**Khoảng trống:** Chưa có cơ chế nào đồng thời đạt được độ chính xác của DL và tính diễn giải cho con người — rule có thể đọc, hiểu, và audit. Luận văn này lấp khoảng trống đó qua knowledge distillation từ Transformer sang interpretable rule set. Câu hỏi về _triển khai_ rule (engine nào, latency bao nhiêu) là vấn đề kỹ thuật thứ cấp sau khi diễn giải thành công.

### 2.2. Câu hỏi nghiên cứu

- **RQ1:** Phương pháp trích xuất rule nào (trong 7 phương pháp đề xuất) đạt cân bằng tốt nhất giữa Fidelity, FPR, Rule Complexity, và tính diễn giải cho analyst?
- **RQ2:** Rich features (thread interleaving, timing, syscall arguments) cải thiện bao nhiêu so với syscall sequence thuần túy trong việc tạo ra rule chính xác và vẫn ngắn gọn đủ để đọc hiểu?
- **RQ3:** Rule trích xuất generalize cross-scenario và cross-domain đến mức nào, và khi nào cần retrain?

### 2.3. Giả thuyết

- **H1:** Decision Tree Surrogate (Exp B) và RuleFit/Skope-Rules (Exp C) sẽ đạt Fidelity >95% với rule set đủ compact và tường minh nhờ tính chất inherently rule-based.
- **H2:** Rich features (Tier 2) cải thiện Fidelity ít nhất 3–5% so với sequence-only (Tier 1) trên LID-DS, đặc biệt cho attack khai thác cross-thread patterns.
- **H3:** Cross-scenario fidelity thấp hơn in-distribution 10–15%, nhưng Feedback Loop có thể thu hẹp khoảng cách sau 2–3 chu kỳ retrain.

### 2.4. Đóng góp

1. **RuleDistill Pipeline** — quy trình end-to-end từ Transformer Teacher đến deployable security rule, với Rule Compiler tự động và Shadow Mode validation.
2. **So sánh có hệ thống 7 phương pháp trích xuất rule** (Exp A–G) theo thứ tự ưu tiên khả thi, trên 2 feature tiers và 3 datasets, bao gồm Temporal Logic Mining và Neuro-Symbolic — hai hướng chưa từng áp dụng cho syscall-based HIDS.
3. **Segment Embeddings** ($E_{pid}$) cho đa luồng — cho phép Teacher học pattern cross-thread và distill thành rule conditions tường minh.
4. **Interpretability Taxonomy (L1–L5)** — phân loại rule theo mức độ diễn giải được: từ atomic boolean condition đến symbolic predicate, kèm trade-off fidelity tương ứng.

### 2.5. Phạm vi và giới hạn

**Trong phạm vi:** Trích xuất và đánh giá rule từ Transformer đã train trên datasets công khai (LID-DS, DongTing); POC deployment minh họa khả năng áp dụng thực tế.

**Ngoài phạm vi:** Tối ưu hóa latency và lựa chọn enforcement engine cụ thể; xây dựng Transformer SOTA; đánh giá trên production cluster; adversarial robustness — tất cả là future work.

---

## 3. BACKGROUND & RELATED WORKS

### 3.1. Syscall-based Intrusion Detection

**Truyền thống:** N-gram (Forrest et al., 1996), STIDE, HMM (Warrender et al., 1999) — đơn giản nhưng FPR cao, không nắm bắt được long-range dependencies hay cross-thread patterns.

**Deep Learning:** DeepLog/LSTM (Du et al., 2017), CNN 1D, WaveNet (Ring et al., 2021) — accuracy cao nhưng black-box.

**Transformer:** Fournier et al. (2023) đạt F1/AUROC >95% nhưng LSTM đôi khi vượt Transformer trên novelty tests — lý do luận văn này tập trung vào pipeline distillation thay vì chứng minh Transformer tốt nhất. Chen et al. (2021): BERT đạt FPR thấp nhất. Grimmer et al. (2022): tách thread ID trên LID-DS cải thiện detection cho 6/7 algorithms.

**Tokenization:** BPE được khuyến nghị thay Word2Vec (Mvula et al., 2023 — Word2Vec gây data leakage).

### 3.2. eBPF trong Security Monitoring

eBPF cho phép thu thập syscall events từ kernel với overhead thấp — dùng làm data collection layer trong luận văn này. Bachl et al. (2021) và Zhang et al. (2024, DSN) chứng minh rule đơn giản và network nhỏ có thể chạy in-kernel, gợi ý hướng tối ưu hóa khi rule L1–L2 đã được validate. BeaCon (Kang et al., 2025) tự động sinh Seccomp profiles qua eBPF — không dùng ML, không capture behavioral patterns.

### 3.3. Explainable AI & Knowledge Distillation

**KD cho IDS:** DistillGuard, DistillMal, KD-XVAE — đều distill cho network traffic. **Chưa có công trình nào distill Transformer cho syscall-based HIDS** — đây là gap chính.

**Rule Extraction:** Ables et al. (2024) đạt fidelity 99.9% từ DNN-based IDS. Abou El Houda et al. (2022): RuleFit cho IoT IDS. Friedman et al. (2023, NeurIPS): Transformer → RASP programs. CORELS (Angelino et al., 2018): certifiably optimal rule lists. Wang & Lin (2021, JMLR): hybrid interpretable + black-box cho uncertain cases.

**Interpretability frameworks:** Doshi-Velez & Kim (2017), Lipton (2018) — nền tảng lý thuyết cho Taxonomy L1–L5.

### 3.4. Temporal Logic & Neuro-Symbolic

**LTL/STL Mining:** SCARLET (Raha et al., 2022), Roy et al. (2023, AAAI) — học LTL từ positive-only logs, phù hợp IDS. Output là công thức ordering tường minh: `G(open → F[0,5] execve)`.

**Neuro-Symbolic:** LTN (Badreddine et al., 2022), DeepProbLog (Manhaeve et al., 2021). Onchis & Istin (2022) áp dụng LTN cho NIDS — chưa áp dụng cho syscall HIDS.

### 3.5. Concept Drift

CADE (Yang et al., 2021, USENIX Security), CAShift (FSE, 2025), OWAD (Han et al., 2023, NDSS) — motivate Feedback Loop trong Section 6.

### 3.6. Research Gaps

1. Chưa có KD từ Transformer Teacher cho syscall-based HIDS.
2. Chưa có comparative study đa chiều (fidelity–complexity–interpretability) giữa nhiều loại surrogate trên cùng syscall task.
3. Chưa có pipeline tự động bridge ML-learned patterns với deployable security policy.
4. Temporal Logic và Neuro-Symbolic chưa được áp dụng cho syscall HIDS với mục tiêu sinh interpretable rule.

---

## 4. PROPOSED METHOD

### 4.1. Kiến trúc RuleDistill

Nguyên tắc cốt lõi: **Transformer learns offline — Rules explain and enforce online.**

```
[Offline]
  Syscall Traces → Feature Engineering → Transformer Teacher
                                               ↓
                                        Rule Extraction (Exp A–G)
                                               ↓
                                        Rule Set (human-readable)
                                               ↓
                                        Rule Compiler
                                               ↓
                              Enforcement engine phù hợp
                         (Falco, Cedar/CEL, OPA, hoặc tương đương)

[Online — không inference model]
  eBPF collector → syscall sequence → Rule evaluation → Alert/Block
```

Sau khi training, Teacher không tham gia inference path. Rule evaluation được thực hiện bởi enforcement engine — lựa chọn engine là quyết định kỹ thuật tùy ngữ cảnh triển khai, không phải mục tiêu cố định của nghiên cứu.

### 4.2. Data Pipeline & Preprocessing

| Tầng         | Dataset           | Quy mô                           | Features                        |
| ------------ | ----------------- | -------------------------------- | ------------------------------- |
| **Tier 1**   | DongTing (DT2022) | 18,966 sequences                 | Syscall name sequence only      |
| **Tier 1**   | LID-DS-2019       | ~11,000 recordings, 10 scenarios | + timestamp, thread_id, args    |
| **Tier 1+2** | LID-DS-2021       | 17,190 recordings, 15 scenarios  | + exploit timestamps, pre-split |

Tier 1 chạy trên cả 3 datasets; Tier 2 chỉ trên LID-DS. DongTing đóng vai trò cross-domain validation (kernel bugs vs. container attacks). Preprocessing: so sánh simple ID mapping vs. BPE tokenization, semantic filtering, sliding window. Class imbalance được phân tích trong EDA (Phase 1) để chọn strategy phù hợp.

### 4.3. Xử lý đa luồng — Tier 2

**Segment Embeddings:** $V_{input} = E_{syscall} + E_{pos} + E_{pid}$, trong đó $E_{pid}$ dùng Relative Thread ID (theo thứ tự xuất hiện trong window, max $K \leq 16$) thay vì raw PID. **Hierarchical Aggregation:** xử lý process con trước, nhúng vector vào chuỗi cha tại điểm `fork`/`clone`.

### 4.4. Teacher Model

Encoder-only Transformer (cân nhắc Longformer/BigBird cho chuỗi dài). Target: F1 ≥ 95%, so sánh với LSTM baseline. Nếu LSTM vượt Transformer trên một dataset cụ thể, dùng mô hình tốt hơn làm Teacher. Output: soft labels + attention weights làm supervision signal cho rule extraction.

### 4.5. Domain-Informed Rule Extraction (Structural Priors)

Thay vì để surrogate tự do học từ data, luận văn tích hợp domain knowledge có sẵn làm **structural prior** cho quá trình rule extraction:

- **Nguồn prior:** MITRE ATT&CK techniques (T1055 Process Injection, T1611 Escape to Host...), Falco default rules, Sigma rules cho Linux — cung cấp danh sách syscall đã biết là nguy hiểm theo ngữ cảnh (vd: `ptrace`, `process_vm_readv`, `execve` từ container, `mount` với flags bất thường).
- **Kỹ thuật tích hợp:**
  - _Feature weighting:_ Tăng trọng số cho các features tương ứng syscall/condition đã có trong knowledge base khi train surrogate (RuleFit, BRL).
  - _Cost-sensitive splitting:_ Ép Decision Tree (Exp B) ưu tiên split trên các syscall mà Falco/MITRE đã flag — giảm tree depth trong khi giữ coverage cho known attack patterns.
  - _Prior-informed anchoring:_ Khởi tạo Anchors (Exp E) từ Falco rule conditions thay vì random — tăng tốc convergence và sinh rule gần với ngôn ngữ mà analyst đã quen thuộc.
- **Lợi ích kép:** (1) Rule sinh ra align với mental model của security analyst, dễ review và trust hơn; (2) surrogate không cần "tái phát minh" các pattern đã được cộng đồng security xác nhận — tập trung capacity vào phát hiện pattern mới mà Teacher đã học được.
- **Đánh giá:** So sánh surrogate có prior vs. không prior trên cùng metrics (fidelity, complexity, interpretability) để đo lường giá trị thực sự của domain knowledge injection.

### 4.6. Interpretability Taxonomy (L1–L5)

Phân loại rule theo mức độ diễn giải được cho analyst — từ dễ nhất đến phức tạp nhất:

| Mức    | Loại               | Ví dụ                                                           | Đặc điểm                            |
| ------ | ------------------ | --------------------------------------------------------------- | ----------------------------------- |
| **L1** | Atomic boolean     | `"execve" in window`                                            | Một điều kiện, đọc tức thì          |
| **L2** | Conjunctive        | `"ptrace" in window AND count("open") > 8`                      | Vài điều kiện, scan nhanh           |
| **L3** | Ordered sequence   | `"open" PRECEDES "execve" WITHIN 5 events`                      | Intent bảo mật rõ ràng              |
| **L4** | Probabilistic      | `anomaly_score(window) > 0.85` + feature breakdown              | Cần giải thích thêm về score        |
| **L5** | Symbolic predicate | `suspicious(window) :- high_entropy(args), cross_thread(t1,t2)` | Expressive nhất, cần tooling verify |

Mục tiêu lý tưởng: fidelity cao với mức diễn giải thấp. Mỗi phương pháp (Exp A–G) được đánh giá trên cả fidelity lẫn mức diễn giải trung bình của rule set sinh ra.

---

## 5. EXPERIMENTAL DESIGN

### 5.1. Bảy phương pháp — theo thứ tự ưu tiên khả thi

| Ưu tiên | Exp | Phương pháp                                                                                                                                                              | Mức diễn giải | Độ phức tạp     |
| ------- | --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------- | --------------- |
| ★★★★★   | A   | **Attention Heuristics** — Top-K syscall attention cao → boolean presence rule                                                                                           | L1–L2         | Thấp            |
| ★★★★☆   | B   | **Decision Tree Surrogate** — KD từ soft labels → IF-THEN path conditions                                                                                                | L1–L2         | Thấp–Trung bình |
| ★★★★☆   | C   | **RuleFit / Skope-Rules** — rule set ngắn gọn từ feature importance                                                                                                      | L2            | Trung bình      |
| ★★★☆☆   | D   | **Bayesian Rule Lists (BRL/CORELS)** — IF-THEN-ELSE với uncertainty routing (Wang & Lin 2021): rule xử lý clear cases, uncertain cases escalate lên Teacher hoặc analyst | L2–L3         | Trung bình–Cao  |
| ★★★☆☆   | E   | **Anchors** — điều kiện tối thiểu per attack type, precision cao                                                                                                         | L2            | Cao             |
| ★★☆☆☆   | F   | **Temporal Logic Mining (LTL/STL)** — SCARLET/Roy et al. cho positive-only logs; capture ordering constraints                                                            | L3            | Cao             |
| ★★☆☆☆   | G   | **Neuro-Symbolic (LTN/DeepProbLog)** — logic bottleneck trong Teacher, learned predicates đọc trực tiếp                                                                  | L4–L5         | Rất cao         |

Mỗi experiment chạy **2 lần** (Tier 1 × 3 datasets + Tier 2 × LID-DS) → **14 runs**. Exp F và G có thể giới hạn trên LID-DS-2021 nếu thời gian hạn chế.

**Baselines:** STIDE (lower bound), DeepLog/LSTM (so sánh Teacher), Isolation Forest (unsupervised), Random Forest (~94% accuracy trên tabular features — performance ceiling cho surrogate methods), hand-written Falco rules (benchmark định tính về readability).

### 5.2. Metrics

**Fidelity** $= |\{x: R(x)=\arg\max T(x)\}| / |D|$

| #   | Metric                                               | Target             |
| --- | ---------------------------------------------------- | ------------------ |
| 1   | **Fidelity** (công thức trên)                        | >95%               |
| 2   | **Accuracy** (vs. ground truth)                      | >93%               |
| 3   | **FPR**                                              | <1%                |
| 4   | **Rule Complexity** (số rule × điều kiện trung bình) | Càng thấp càng tốt |
| 5   | **Interpretability Level** (L1–L5 trung bình)        | Càng thấp càng tốt |
| 6   | **OOD Fidelity** (cross-scenario, cross-domain)      | >85%               |

Ngoài ra: Weighted Fidelity (trọng số theo Teacher confidence) và Attack-class Fidelity (riêng trên attack samples).

### 5.3. OOD Evaluation

Cross-scenario: train web scenarios (Nginx, Rails) → test DB/CMS (MySQL, CouchDB). Cross-domain: LID-DS ↔ DongTing. Temporal split để mô phỏng production drift.

---

## 6. DEPLOYMENT & FEEDBACK LOOP

### 6.1. Rule Compiler & Enforcement

Rule Compiler chuyển rule set sang định dạng phù hợp với enforcement engine được chọn. Lựa chọn engine phụ thuộc vào yêu cầu triển khai cụ thể:

- **Falco** — enforcement in-kernel qua eBPF, latency microsecond, phù hợp rule L1–L2 đơn giản.
- **Cedar/CEL** — evaluation ở userspace, expressive hơn, phù hợp rule L3–L5 hoặc khi cần audit trail đầy đủ.
- **Kết hợp:** Falco làm fast-path cho rule đơn giản; Cedar làm audit trail song song.

Tiêu chí lựa chọn là data-driven: dựa trên interpretability level và feature composition của rule set thực nghiệm, không được định sẵn từ đầu.

### 6.2. Shadow Mode

Teacher và Rule chạy song song trên cùng syscall stream. Disagreement giữa hai được monitor liên tục — khi fidelity drop dưới ngưỡng, trigger Feedback Loop.

### 6.3. Feedback Loop

Disagreement cases → Human review → Bổ sung training data → Retrain surrogate → Compile rule mới → Canary deploy. Safeguards: rate limiting, anomaly detection trên feedback, rollback khi accuracy drop.

---

## 7. TIMELINE

| Giai đoạn | Thời gian   | Nội dung                                             | Deliverable           |
| --------- | ----------- | ---------------------------------------------------- | --------------------- |
| **P1**    | Tháng 1–2   | Data pipeline, EDA, eBPF collector POC               | Pipeline + EDA report |
| **P2**    | Tháng 3–4   | Train Teacher, so sánh LSTM baseline                 | Teacher checkpoint    |
| **P3**    | Tháng 5–7   | Exp A–E, Interpretability Taxonomy, Rule Compiler v1 | Rule sets + Compiler  |
| **P4**    | Tháng 8     | OOD evaluation, Exp F (LTL/STL Mining)               | OOD report + Exp F    |
| **P5**    | Tháng 9     | Shadow Mode POC, Exp G (Neuro-Symbolic, exploratory) | POC demo + Exp G      |
| **P6**    | Tháng 10    | Feedback Loop POC, tổng hợp so sánh A–G              | Full comparison table |
| **P7**    | Tháng 11–12 | Viết luận văn, bảo vệ                                | Thesis draft          |

**Contingency:** Exp F và G chạy trên LID-DS-2021 subset. Ưu tiên tuyệt đối: Exp A–E + Rule Compiler + Shadow Mode POC trước tháng 10.

---

## 8. CONCLUSION

Luận văn đề xuất pipeline **RuleDistill** với luận điểm trung tâm: _một security rule mà analyst có thể đọc và hiểu được có giá trị cao hơn một mô hình chính xác mà không ai giải thích được_.

Đóng góp cốt lõi là Interpretability Taxonomy (L1–L5) và comparative study 7 phương pháp distillation — cung cấp guidance thực tiễn về trade-off giữa fidelity, complexity, và interpretability. Khi rule đã được diễn giải thành công, việc lựa chọn enforcement engine (Falco, Cedar, hay bất kỳ công cụ nào) là vấn đề kỹ thuật thứ cấp tùy ngữ cảnh triển khai.

**Future work:** Tối ưu hóa enforcement pipeline cho các mức rule phức tạp hơn; adversarial robustness; đánh giá trên production cluster.
