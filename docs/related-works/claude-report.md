# Literature map for interpretable security rules from Transformer-based syscall IDS

**This survey identifies 55+ papers spanning Transformer-based syscall intrusion detection, explainable rule extraction, eBPF enforcement, temporal logic mining, Bayesian rule learning, and distribution drift—directly supporting a thesis on knowledge-distilling a Transformer teacher into deployable interpretable surrogates.** The most critical finding is that no existing work combines all these elements: the intersection of Transformer-based syscall classification, multi-surrogate knowledge distillation, and eBPF/Falco deployment represents a genuine and well-supported research gap. Below, each area is organized with verified papers, followed by key baselines, surveys, and a gap analysis.

---

## Area 1: Transformers and deep learning for syscall-based intrusion detection

This area is nascent but growing. Most syscall-based HIDS still rely on LSTM/CNN architectures, with Transformer adoption just beginning—making a Transformer teacher model a timely contribution.

**1. Fournier, Aloise & Costa — "Language Models for Novelty Detection in System Call Traces" (2023, arXiv:2309.02206).** The most directly relevant paper: compares **LSTM, Transformer, and Longformer** on syscall traces for novelty detection, treating syscalls as a language modeling problem. Introduces a large-scale open-source dataset of 2M+ web requests with seven behaviors. Achieves F-score and AuROC >95% on most novelties, explicitly demonstrating Transformer superiority for long-range syscall dependencies with positional encodings.

**2. Fournier, Aloise, Azhari & Tetreault — "On Improving Deep Learning Trace Analysis with System Call Arguments" (2021, IEEE/ACM MSR 2021, pp. 120–130).** Addresses the limitation that most DL approaches only use syscall event names, ignoring arguments. Proposes embedding and encoding syscall names alongside arguments (PID, timestamp, return value) using attention-based architectures. Directly relevant to encoding scheme design for syscall sequences.

**3. Chen, Tran, Thumati, Bhuyan & Ding — "Data Curation and Quality Assurance for ML-based Cyber Intrusion Detection" (2021, arXiv:2105.10041).** One of the first papers applying **BERT and GPT-2** to raw syscall integer sequences for HIDS. Evaluates 10 ML/DL models across **11 datasets including ADFA-LD, UNM, and MIT**. BERT achieves the lowest false positive rate. Syscall sequences are tokenized into length-6 subsequences, with multi-process traces grouped by PID.

**4. Ring, Van Oort, Durst, White, Near & Skalka — "Methods for Host-based Intrusion Detection with Deep Learning" (2021, ACM Digital Threats, Vol. 2, No. 4).** The largest comparison of DL models for syscall HIDS (**540 training trials**). Evaluates WaveNet, LSTM, and CNN/GRU on ADFA-LD and the new PLAID dataset. Proposes ALAD (Application Level Anomaly Detection). The key modern baseline that any Transformer approach must compare against.

**5. Grimmer, Kaelble & Rahm — "Improving Host-Based Intrusion Detection Using Thread Information" (2022, CRITIS/Springer LNCS).** Critically addresses **multi-thread interleaving**: shows detection improves for 6 of 7 algorithms when syscall streams are separated by thread ID. Evaluated on **LID-DS**, which provides thread IDs, process names, timestamps, and arguments. Essential for handling real container workloads.

**6. Grimmer, Kaelble, Nirsberger, Schulze, Rucks, Hoffmann & Rahm — "Dataset Report: LID-DS 2021" (2022, CRITIS/Springer LNCS, pp. 63–73).** Presents the **LID-DS 2021 dataset**: ~1,137 trace files per scenario from Docker-based environments with thread IDs, data buffers, and rich syscall parameters. Contains realistic CWE-based attacks. The richest modern syscall dataset for container-specific HIDS evaluation.

**7. El Khairi, Caselli, Knierim, Peter & Continella — "Contextualizing System Calls in Containers for Anomaly-Based Intrusion Detection" (2022, ACM CCSW'22, pp. 9–21).** Container-specific HIDS using **graph-based syscall sequence encoding** (syscall sequence graphs fed to an autoencoder). Uses 100ms time-window segmentation to handle multi-process interleaving from web servers. Evaluates 20 attack scenarios. Demonstrates that contextual syscall analysis is essential for container environments.

**8. Duan, Fu, Cai, Chen & Sun — "DongTing: A Large-Scale Dataset for Anomaly Detection of the Linux Kernel" (2023, Journal of Systems and Software, Vol. 203).** Introduces DongTing: **83× larger than ADFA-LD** with 12,116 abnormal and 6,850 normal sequences across 26 kernel releases. Benchmarks CNN, LSTM, and WaveNet. Cross-dataset evaluation shows superior generalization compared to ADFA-LD-trained models. The newest generation benchmark.

**9. Mvula, Branco, Jourdan & Viktor — "Evaluating Word Embedding Feature Extraction Techniques for HIDS" (2023, Discover Data, Vol. 1, No. 1).** Systematically evaluates **Word2Vec and GloVe** embeddings for syscall feature extraction across four datasets (ADFA-LD, NGIDS-DS, WWW2019, LID-DS 2021). Reveals that Word2Vec introduces data leakage via duplicated samples. Directly relevant to tokenization/encoding scheme selection.

**10. CAFTrans — "Channel Features and API Frequency-Based Transformer Model for Malware Identification" (2024, Electronics/MDPI, Vol. 13, No. 3).** Applies a **Transformer encoder with 1D channel attention** to API call sequences (Windows mal-api-2019 dataset). The tokenization and frequency-based feature extraction approach for call sequence classification transfers directly to syscall-based HIDS.

**11. Alam, Mvula, Branco, Jourdan & Viktor — "NLP Methods in Host-based Intrusion Detection Systems: A Systematic Review" (2022, arXiv:2201.08066).** Comprehensive survey of NLP methods for syscall-based HIDS covering the full pipeline: preprocessing, feature extraction (n-grams, Word2Vec, GloVe, TF-IDF), and DL models. Identifies 24+ studies and explicitly notes that **Transformer application to syscall HIDS remains underexplored**.

---

## Area 2: Explainability and rule extraction from neural network classifiers

This area is rich with methods but almost entirely disconnected from syscall-based security—creating a clear opportunity for the thesis.

**1. Friedman, Wettig & Chen — "Learning Transformer Programs" (2023, NeurIPS 2023, Oral).** Trains modified Transformers that can be automatically converted into **discrete, human-readable programs** based on RASP. Demonstrates that trained Transformers can be faithfully represented as interpretable symbolic programs. Directly addresses rule/program extraction from Transformer architectures.

**2. Badreddine, d'Avila Garcez, Serafini & Spranger — "Logic Tensor Networks" (2022, Artificial Intelligence, Vol. 303).** Foundational neuro-symbolic framework introducing "Real Logic"—a fully differentiable first-order logic where logical symbols are grounded onto data via neural computational graphs. Enables end-to-end integration of symbolic rules with neural learning, supporting multi-label classification and relational reasoning.

**3. Manhaeve, Dumančić, Kimmig, Demeester & De Raedt — "Neural Probabilistic Logic Programming in DeepProbLog" (2021, Artificial Intelligence, Vol. 303; originally NeurIPS 2018).** Integrates deep learning with probabilistic logic programming via "neural predicates." Supports symbolic and subsymbolic representations with end-to-end training. Key neuro-symbolic framework for combining neural pattern recognition with logical rule-based reasoning.

**4. Onchis & Istin — "A Neuro-Symbolic Classifier with Optimized Satisfiability for Monitoring Security Alerts" (2022, Applied Sciences, 12(22):11502).** Directly applies **Logic Tensor Networks to network intrusion detection** (KDD-Cup'99). The classifier incorporates first-order fuzzy logic reasoning within a neural framework for interpretable multi-label classification of security alerts. Concrete proof-of-concept for neuro-symbolic IDS, though limited to network traffic.

**5. Sharma, Sharma, Lal & Roy — "Explainable AI for Intrusion Detection in IoT Networks: A Deep Learning Based Approach" (2024, Expert Systems with Applications, Vol. 238).** Applies **SHAP and LIME** as post-hoc explanation methods to DNN/CNN-based IoT IDS on NSL-KDD and UNSW-NB15. Demonstrates both local and global explanations for attack classification. Representative of XAI applied to DL-based security.

**6. Gaspar, Ferreira & Oliveira — "Explainable AI for IDS: LIME and SHAP Applicability on Multi-Layer Perceptron" (2024, IEEE Access).** Applies LIME and SHAP to MLP-based IDS with **perturbation analysis to validate explanation fidelity**. Includes user survey measuring interpretability improvements. Directly addresses fidelity of post-hoc explanations in the IDS context.

**7. Abou El Houda, Senhaji Hafid & Khoukhi — "A Novel IoT-Based Explainable Deep Learning Framework for Intrusion Detection" (2022, IEEE Network Magazine).** Two-stage XAI framework: DNN-based IDS plus **RuleFit and SHAP explanations**. RuleFit extracts interpretable decision rules from the DNN's behavior. The clearest example of **RuleFit applied to security/intrusion detection** in the literature.

**8. Herbinger, Dandl, Ewald, Loibl & Casalicchio — "Leveraging Model-Based Trees as Interpretable Surrogate Models for Model Distillation" (2023, ECAI 2023 Workshops, Springer CCIS Vol. 1947).** Systematically investigates model-based trees as surrogates for distilling black-box models. Compares four tree algorithms on **fidelity, interpretability, stability, and interaction capture**. Provides a rigorous framework for evaluating the fidelity-interpretability trade-off.

**9. Ribeiro, Singh & Guestrin — "Anchors: High-Precision Model-Agnostic Explanations" (2018, AAAI 2018, pp. 1527–1535).** Foundational paper introducing **Anchors**—IF-THEN rules with explicit coverage and precision guarantees, computed via reinforcement learning. Model-agnostic and widely cited in security XAI literature. Essential reference for the Anchors surrogate in the thesis pipeline.

**10. Mane & Rao — "Explaining Network Intrusion Detection System Using Explainable AI Framework" (2021, arXiv:2103.07110).** Applies five explanation methods to DNN-based NIDS: SHAP, LIME, CEM, ProtoDash, and **Boolean Decision Rules via Column Generation (BRCG)**. Demonstrates explicit rule extraction from a trained DNN for intrusion detection on NSL-KDD.

**11. "A Synergistic Approach in Network Intrusion Detection by Neurosymbolic AI" (2024, arXiv:2406.00938).** Comprehensive evaluation of neuro-symbolic architectures (LTN, DeepProbLog, DeepStochLog, NeurASP, Neural Logic Machines) specifically for NIDS. Reviews how these frameworks embed symbolic rules into neural frameworks for anomaly detection. Addresses the neural-symbolic gap for detecting both known and unknown attacks.

---

## Area 3: eBPF-based runtime security and policy enforcement

This area has strong industry traction but limited academic coverage, especially regarding ML integration—a key thesis opportunity.

**1. He, Guo, Xing, Che, Sun, Liu, Xu & Li — "Cross Container Attacks: The Bewildered eBPF on Clouds" (2023, USENIX Security '23, pp. 5971–5988).** Demonstrates offensive eBPF capabilities including cross-container escapes on major cloud providers (Google, AWS, Azure). Shows that existing eBPF security tools (Falco, Datadog) can be blinded by malicious eBPF programs. Proposes "CapBits" for fine-grained eBPF access control.

**2. Her, Kim, Kim & Lee — "An In-Depth Analysis of eBPF-Based System Security Tools in Cloud-Native Environments" (2025, IEEE Access, Vol. 13).** Comprehensive comparative evaluation of **KubeArmor, Falco, Tetragon, and Tracee**: internal architectures, eBPF hook points, policy enforcement pipelines, and performance overhead (KubeArmor up to ~88%, Tetragon ~73%). Includes real-world CVE case studies. The most thorough academic comparison of eBPF security tools.

**3. Zhang, Chen, He, Chen & Li — "Real-Time Intrusion Detection and Prevention with Neural Network in Kernel Using eBPF" (2024, IEEE/IFIP DSN 2024, pp. 416–427).** Redesigns **neural network inference to run entirely within eBPF** constraints (integer-only arithmetic, 512-byte stack). Achieves F1 of 0.992 with 3,000–5,000ns inference time and 5KB memory. Demonstrates feasibility of in-kernel ML-based IDS via eBPF—closest existing work to the thesis's deployment vision.

**4. Xing, Wang, Torabi, Zhang, Lei & Sun — "A Hybrid System Call Profiling Approach for Container Protection" (2023, IEEE TDSC).** Hybrid static+dynamic analysis for **automated syscall whitelist generation** for containers. Identifies three execution phases in container lifecycle and enforces phase-specific whitelists via seccomp-BPF. Directly relevant to automated policy generation for containerized workloads.

**5. Fournier, Afchain & Baubeau — "Runtime Security Monitoring with eBPF" (2021, SSTIC 2021).** Describes the **Datadog Runtime Security Agent** architecture: eBPF-based HIDS with container-aware metadata enrichment. Hooks deeper than syscall level to retrieve kernel-internal information (e.g., overlayfs layer data). Details the agent's rule language and policy engine.

**6. Lopes, Martins, Correia, Serrano & Nunes — "Container Hardening Through Automated Seccomp Profiling" (2020, ACM WOC'20, pp. 31–36).** Automatically generates **custom seccomp profiles** using eBPF-based syscall tracing during unit test execution. Integrates into CI/CD pipelines. Demonstrates that custom seccomp profiles mitigate zero-day vulnerabilities and reduce container attack surface.

**7. Wang & Chang — "Design and Implementation of an IDS by Using Extended BPF in the Linux Kernel" (2022, JNCA, Vol. 198).** Combines kernel-space eBPF pattern matching with userspace rule matching (modified Snort ruleset). Achieves **3× throughput improvement** over traditional Snort. Foundational work on eBPF-accelerated signature-based IDS.

**8. Ryu, Kim, Lee, Kim, Choi & Kim — "Hybrid Runtime Detection of Malicious Containers Using eBPF" (2026, CMC, Vol. 86, No. 3).** Hybrid eBPF framework collecting both network flow metadata and syscall traces for multi-class malicious container classification. Host-based syscall detection achieves **98.39% accuracy**. Demonstrates that combining telemetry modalities resolves single-source classification ambiguities.

---

## Area 4: Temporal logic mining from execution traces

This area provides the formal foundations for extracting human-readable temporal specifications from labeled traces—a key surrogate model type in the thesis.

**1. Raha, Roy, Fijalkow & Neider — "Scalable Anytime Algorithms for Learning Fragments of Linear Temporal Logic" (2022, TACAS 2022, LNCS Vol. 13243, pp. 263–280).** Introduces **SCARLET**, a scalable anytime algorithm for learning LTL formulas from positive and negative trace examples. Constructs formulas an order of magnitude larger than prior SAT-based methods. Directly applicable to inferring temporal behavioral rules from labeled syscall traces.

**2. Bartocci, Mateis, Nesterini & Ničković — "Survey on Mining Signal Temporal Logic Specifications" (2022, Information and Computation, Vol. 289).** Comprehensive survey of STL specification mining covering template-based vs. template-free, supervised vs. unsupervised, and multiple algorithmic paradigms (decision trees, genetic algorithms, constraint solving). Foundational reference for the temporal logic mining component.

**3. Roy, Gaglione, Baharisangari, Neider, Xu & Topcu — "Learning Interpretable Temporal Properties from Positive Examples Only" (2023, AAAI 2023, Vol. 37, pp. 6507–6515).** Learns LTL_f formulas and DFAs from **positive examples only**, using conciseness and language minimality as regularizers. Particularly relevant for security where only logs of normal behavior are available—learned specifications flag deviations as potential intrusions.

**4. Gaglione, Neider, Roy, Topcu & Xu — "MaxSAT-Based Temporal Logic Inference from Noisy Data" (2021, ATVA 2021; extended in ISSE 2022).** First work using **MaxSAT solvers for temporal logic inference** from noisy traces. Combines MaxSAT-based learning with decision tree construction. Applicable to mining security rules from imperfect syscall data where labeling errors exist.

**5. Bartocci, Mateis, Nesterini & Ničković — "Mining Hyperproperties using Temporal Logics" (2023, ACM TECS, Vol. 22, No. 5s).** First approach for mining **HyperSTL specifications** relating multiple execution traces. Essential for security policies like non-interference that cannot be expressed as single-trace temporal logic. Uses syntax-guided synthesis.

**6. Bombara & Belta — "Offline and Online Learning of Signal Temporal Logic Formulae Using Decision Trees" (2021, ACM TCPS, Vol. 5, No. 3).** Decision-tree-based STL formula inference with both offline and online algorithms. Demonstrated on **anomaly detection in maritime security** (distinguishing normal vessel paths from threat patterns). Directly applicable as a temporal logic surrogate for security anomaly detection.

**7. Peng, Liang, Han, Luo, Du, Wan, Ye & Zheng — "PURLTL: Mining LTL Specification from Imperfect Traces in Testing" (2023, ASE 2023 NIER Track).** Neural-based method for mining LTL from **imperfect traces** without templates or negative examples. Uses differentiable LTL path checking. Relevant because real-world syscall logs are often incomplete.

**8. Saveri & Bortolussi — "Retrieval-Augmented Mining of Temporal Logic Specifications from Data" (2024, ECML PKDD 2024, LNCS Vol. 14947).** Retrieval-augmented STL learning for binary classification of regular vs. anomalous behavior. Directly targets anomaly detection through temporal logic classification.

---

## Area 5: Bayesian Rule Lists and uncertainty-aware rule learning

These papers establish the theoretical and practical foundation for BRL as a surrogate model with built-in uncertainty quantification and selective prediction.

**1. Letham, Rudin, McCormick & Madigan — "Interpretable Classifiers Using Rules and Bayesian Analysis" (2015, Annals of Applied Statistics, Vol. 9, No. 3, pp. 1350–1371).** **Foundational BRL paper.** Generative model yielding a posterior distribution over decision lists. Pre-mines frequent patterns, then learns a Bayesian decision list. Built-in uncertainty quantification: each rule's probability estimate reflects model confidence, enabling identification of uncertain cases for routing.

**2. Yang, Rudin & Seltzer — "Scalable Bayesian Rule Lists" (2017, ICML, PMLR 70:3921–3930).** SBRL achieves **two orders of magnitude speedup** over original BRL while fully optimizing over rule lists. Critical for practical deployment in security domains requiring real-time, scalable rule extraction with calibrated probability estimates.

**3. Wang, Rudin, Doshi-Velez, Liu, Klampfl & MacNeille — "A Bayesian Framework for Learning Rule Sets for Interpretable Classification" (2017, JMLR, Vol. 18, pp. 1–37).** Introduces **Bayesian Rule Sets (BRS)**—unordered rule sets in DNF with user-settable priors. Complementary to BRL: while BRL produces ordered lists, BRS produces unordered sets. The Bayesian framework enables uncertainty-aware classification with domain-expert-tunable interpretability.

**4. Angelino, Larus-Stone, Alabi, Seltzer & Rudin — "Learning Certifiably Optimal Rule Lists for Categorical Data" (2018, JMLR, Vol. 19, pp. 1–77).** **CORELS** uses branch-and-bound optimization to find **provably optimal rule lists** with a certificate of optimality. Critical for security domains where rule completeness guarantees matter—no simpler or more accurate rule list exists.

**5. Geifman & El-Yaniv — "SelectiveNet: A Deep Neural Network with an Integrated Reject Option" (2019, ICML, PMLR 97:2151–2159).** Three-headed architecture for **selective prediction** with end-to-end rejection training. Foundational for the uncertainty routing concept: interpretable rules handle high-confidence cases while uncertain cases are routed to the full Transformer teacher.

**6. Rudin — "Stop Explaining Black Box ML Models for High Stakes Decisions and Use Interpretable Models Instead" (2019, Nature Machine Intelligence, Vol. 1, pp. 206–215).** Highly influential argument that for **safety-critical systems**, inherently interpretable models should replace post-hoc explanations. Discusses BRL, SBRL, and CORELS as practical alternatives. Foundational framing for why rule-based surrogates are preferable in security contexts.

**7. Wang & Lin — "Hybrid Predictive Models: When an Interpretable Model Collaborates with a Black-box Model" (2021, JMLR, Vol. 22, pp. 1–38).** **Directly addresses uncertainty routing.** An interpretable model (BRS/SBRL) handles clear-cut cases while a black-box handles the rest. Optimizes a principled objective balancing accuracy, interpretability, and transparency fraction. Demonstrates Pareto-efficient trade-offs—the theoretical foundation for cascading interpretable rules with a Transformer fallback.

---

## Area 6: Distribution drift and OOD detection for deployed security systems

These papers address the critical operational challenge of maintaining detection accuracy as attack patterns and normal behavior evolve post-deployment.

**1. Yang, Guo, Hao, Ciptadi, Ahmadzadeh, Xing & Wang — "CADE: Detecting and Explaining Concept Drift Samples for Security Applications" (2021, USENIX Security '21, pp. 2327–2344).** Detects individual drifting samples using **contrastive representation learning** and provides semantic explanations for drift. Evaluated on Android malware (Drebin) and **network IDS (CIC-IDS2018)**. Detects unseen attack families. The leading work on per-sample drift detection with explanations for security.

**2. Barbero, Pendlebury, Pierazzi & Cavallaro — "Transcending TRANSCEND: Revisiting Malware Classification in the Presence of Concept Drift" (2022, IEEE S&P '22, pp. 1165–1183).** Formalizes **conformal prediction-based drift detection** with a reject option—quarantining likely-misclassified samples. TRANSCENDENT introduces refined conformal evaluators tested over 5 years of malware data. Principled OOD/drift detection applicable to any deployed security classifier.

**3. Andresini, Pendlebury, Pierazzi, Loglisci, Appice & Cavallaro — "INSOMNIA: Towards Concept-Drift Robustness in Network Intrusion Detection" (2021, ACM AISec '21, pp. 111–122).** Semi-supervised NIDS with **active learning for continuous model updates** under drift. Uses uncertainty sampling and XAI to interpret model reactions to distribution shifts. Extends the TESSERACT temporal evaluation framework to IDS.

**4. Wang — "ENIDrift: A Fast and Adaptive Ensemble System for Network Intrusion Detection under Real-World Drift" (2022, ACSAC '22, pp. 785–798).** Introduces iP2V (incremental feature extraction) and adaptive ensemble construction. Creates **RWDIDS**, the first real-world drift dataset for NIDS with intense concept drift. Achieves **up to 69.78% F1 improvement** and 100% F1 against adversarial attacks.

**5. Chen, Ding & Wagner — "Continuous Learning for Android Malware Detection" (2023, USENIX Security '23, pp. 1127–1144).** Documents severe concept drift: **F1 drops from 0.99 to 0.76 within 6 months**. Proposes contrastive learning with active learning, showing similarity-based uncertainty is more drift-robust than traditional confidence measures. Directly applicable to any security classifier facing evolving threats.

**6. Han, Wang, Chen, Wang, Yu, Wang, Zhang, Wang, Jin, Yang, Shi & Yin — "OWAD: Anomaly Detection in the Open World" (2023, NDSS '23).** First work tackling **normality shift** (distribution shift of benign data) for unsupervised security anomaly detection. Detects, explains, and adapts to shift while avoiding catastrophic forgetting. Evaluated on network IDS (Kyoto/AnoShift), log-based anomaly detection (BGL), and APT detection (LANL). Deployed in a real-world power grid SCADA system.

**7. Yang, Zheng, Li, Xu, Wang & Ngai — "ReCDA: Concept Drift Adaptation with Representation Enhancement for Network Intrusion Detection" (2024, KDD '24, pp. 3818–3828).** Two-stage approach: self-supervised drift-aware representation learning plus weakly-supervised classifier tuning with **minimal labeled drifting samples**. Demonstrates superior adaptability under varying drift degrees.

---

## XAI and cybersecurity survey papers

Four major surveys map the XAI-cybersecurity intersection and highlight the need for interpretable IDS:

**1. Capuano, Fenza, Loia & Stanzione — "Explainable Artificial Intelligence in CyberSecurity: A Survey" (2022, IEEE Access, Vol. 10, pp. 93575–93600).** Analyzes 300+ papers across IDS, malware, phishing, botnets, fraud, and forensics. Provides a comprehensive XAI taxonomy for cybersecurity and identifies real-time XAI deployment as an open challenge.

**2. Neupane, Ables, Anderson, Mittal, Rahimi, Banicescu & Seale — "Explainable Intrusion Detection Systems (X-IDS): A Survey" (2022, arXiv:2207.06236 / IEEE Access).** IDS-specific XAI survey proposing a human-in-the-loop X-IDS architecture. Identifies three critical gaps: defining explainability for IDS, tailoring explanations to stakeholders, and **designing evaluation metrics for explanations**.

**3. Rjoub et al. — "A Survey on Explainable AI for Cybersecurity" (2023, IEEE TNSM).** Reviews XAI for network-driven cybersecurity threats. Covers post-hoc and ante-hoc techniques. Calls for multi-modal explanation approaches.

**4. Takahashi & Zhang — "Explainable AI for Cybersecurity: A Literature Survey" (2022, Annals of Telecommunications).** Spans 2000–2022 across 8 digital libraries. Categorizes XAI by explanation type and cybersecurity domain.

---

## Essential baselines for syscall anomaly detection

Any thesis in this space must position against these established methods:

- **STIDE** (Hofmeyr, Forrest & Somayaji, 1998, Journal of Computer Security): The canonical n-gram baseline—builds a database of normal syscall n-grams and flags mismatches. All subsequent syscall IDS papers compare against it.
- **Creech & Hu (2014, IEEE Trans. Computers)**: Introduced semantic structure analysis of syscalls (contiguous and discontiguous patterns) and created the **ADFA-LD dataset**—the standard benchmark for a decade.
- **Ring et al. (2021, ACM DTRAP)**: Largest DL comparison (540 trials) on ADFA-LD and PLAID—WaveNet, LSTM, GRU baselines. The modern DL baseline.
- **LID-DS 2021** (Grimmer et al.): The most feature-rich container-based syscall dataset with thread information.
- **DongTing** (Duan et al., 2023): 83× larger than ADFA-LD, cross-dataset generalization benchmark.
- **Bridges et al. (2019, ACM Computing Surveys)**: Comprehensive survey organizing syscall HIDS by data source, documenting One-Class SVM, k-NN, HMM, and frequency-based baselines on ADFA-LD.

---

## Eight literature gaps this thesis uniquely fills

The gap analysis reveals that the proposed thesis sits at an intersection that **no existing work occupies**. Each gap is well-supported by the absence of relevant papers across all six research areas:

1. **No Transformer teacher for syscall classification exists as a distillation source.** While Fournier et al. (2023) compare Transformers on syscalls, no work uses a Transformer specifically as a knowledge distillation teacher. Existing KD-based IDS work (network domain only) distills to a single smaller neural network, not interpretable surrogates.

2. **Multi-surrogate knowledge distillation for IDS is entirely absent.** No work distills a security model into multiple diverse interpretable surrogates simultaneously. The thesis's six-surrogate approach (RuleFit, Decision Trees, BRL, Anchors, Temporal Logic, Neuro-Symbolic) would provide complementary explanation types that XAI surveys explicitly call for.

3. **Knowledge distillation has never been applied to syscall-based HIDS.** All identified KD+IDS papers operate on network traffic features (NSL-KDD, UNSW-NB15). Syscall sequences have fundamentally different characteristics—sequential, variable-length, categorical—that make distillation a novel challenge.

4. **No academic work bridges ML-learned rules with eBPF/Falco deployment.** Falco uses hand-written YAML rules exclusively. Zhang et al. (2024, DSN) run neural networks in eBPF but don't extract interpretable rules. Translating distilled decision trees or temporal logic into Falco-compatible rules for real-time enforcement is completely unaddressed.

5. **Container-specific syscall anomaly detection with explainability is unexplored.** El Khairi et al. (2022) do container syscall detection without XAI; XAI+IDS surveys focus on network IDS. Container workloads have unique syscall patterns (namespace isolation, ephemeral processes) requiring tailored explainability.

6. **Temporal logic surrogates for syscall pattern explanation are novel.** While temporal logic mining tools exist (SCARLET, Bombara & Belta) and syscall IDS exists, no work automatically extracts temporal logic formulas from a trained DL model to describe syscall anomaly patterns.

7. **Neuro-symbolic approaches have not been applied to syscall-based HIDS.** Existing neuro-symbolic IDS (Onchis & Istin, 2022) operates on network features only. A neuro-symbolic surrogate encoding OS semantics and container policies alongside learned syscall patterns is unexplored.

8. **No systematic fidelity-accuracy-interpretability evaluation framework exists for IDS surrogates.** While individual papers evaluate single surrogates, no work provides a head-to-head comparison of multiple surrogate types on the same HIDS task across fidelity (faithfulness to teacher), detection accuracy, and interpretability. The thesis would deliver the first such multi-dimensional evaluation, addressing gaps identified by Neupane et al. (2022).

---

## Conclusion

This literature map identifies **55+ verified papers** across all six research areas, revealing a well-defined and defensible research niche. The Transformer-for-syscalls space is nascent (Fournier et al. 2023 and Chen et al. 2021 are the primary references), XAI-for-IDS is active but network-focused, and eBPF security tools lack ML integration entirely. The thesis's unique contribution lies in the **end-to-end pipeline**: Transformer teacher → multi-surrogate distillation → eBPF/Falco deployment—a combination no existing work attempts. The strongest theoretical foundations come from BRL/SBRL (Letham et al. 2015; Yang et al. 2017) for uncertainty-aware rules, Wang & Lin (2021) for hybrid interpretable-black-box routing, and SCARLET (Raha et al. 2022) for temporal logic mining. The most pressing empirical gap is the absence of any fidelity comparison across surrogate types on syscall data, which the thesis is positioned to fill definitively.