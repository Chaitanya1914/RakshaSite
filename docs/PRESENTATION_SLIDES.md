# RakshaSite: Presentation Slide Deck Outline
## Multimodal Agentic Framework for Construction Safety Compliance Monitoring
### Faculty Review & Final Project Viva (10 Credits) — CSE (AI & ML), COER University

---

### Slide 1: Title & Project Identity
- **Header**: RakshaSite: Autonomous Multimodal AI for Construction Site Safety
- **Subheader**: Real-Time PPE Detection, LangGraph Multi-Agent Reasoning & BOCW Act Statutory Compliance
- **Presenter**:
  - Chaitanya (ML & Systems Architect Lead, Roll No: 2200101001)
  - Arshi (Data Engineering & Regulatory Research Lead, Roll No: 2200101002)
- **Academic Context**: B.Tech 5th Semester Final Project | Guided by Dr. R. K. Sharma
- **Visual Asset**: Hero split visual — Worksites with AI bounding boxes on left; automated BOCW legal audit report on right.
- **Speaker Notes**:
  > "Good morning respected faculty members and examiners. Today we present RakshaSite — an autonomous, multimodal AI compliance system designed specifically for the Indian construction sector. Unlike traditional computer vision tools that stop at drawing boxes, RakshaSite bridges the gap between raw pixel detection, machine-learned risk assessment, and legally enforceable statutory compliance under India's BOCW Act, 1996."

---

### Slide 2: The Crisis: Indian Construction Fatalities & Enforcement Deficit
- **Key Metrics (The Problem)**:
  - **38 Fatalities Per Day**: Construction is India's most hazardous sector (NCRB, 2023).
  - **11% Global Occupational Deaths**: India accounts for a disproportionate share of global worksite fatalities (ILO).
  - **80% Preventable**: The vast majority of deaths are due to missing basic PPE (helmets, high-visibility vests) and falls from height.
  - **The Enforcement Gap**: 1,800 labour inspectors nationwide tasked with monitoring over 50,000,000 workers (1 inspector per 27,000 workers).
- **Statutory Mandate**: Building and Other Construction Workers (BOCW) Act, 1996 mandates PPE enforcement with penalties under Section 47 (fines and imprisonment up to 3 months).
- **Visual Asset**: Infographic showing the enforcement pyramid deficit: 5 Crore Workers vs 1,800 Inspectors.
- **Speaker Notes**:
  > "In India, 38 construction workers lose their lives every single day. While we have robust safety laws like the BOCW Act of 1996, enforcement is completely broken. A single inspector is responsible for over 27,000 workers. Manual inspections occur only after tragedy strikes. RakshaSite transforms this reactive, complaint-driven process into 24/7 automated, proactive vigilance."

---

### Slide 3: Research Gap & The Architectural Limitations of Existing Tools
- **Comparative Analysis**:
  1. *Academic YOLO Systems (Wu et al., Nath et al.)*: Pure perception only. They output coordinates without understanding worksite danger context or legal obligations.
  2. *Enterprise Safety AI (Intenseye, viAct)*: Proprietary cloud solutions costing ₹25 Lakh to ₹42 Lakh per year. Grounded in US OSHA standards with zero alignment to Indian statutory frameworks.
  3. *Digital Checklists (SafetyCulture / iAuditor)*: Manual form-filling apps requiring human data entry — zero automated visual intelligence.
- **Identified Research Gap**: No existing system unifies real-time object detection with multi-agent reasoning, non-linear risk modeling, and automated Indian legal citation.
- **Visual Asset**: 2x2 Feature Matrix comparing Intenseye, Academic YOLO, Manual Audits, and RakshaSite across Affordability, Reasoning, and Indian Legal Grounding.
- **Speaker Notes**:
  > "When we reviewed existing solutions, we discovered three critical gaps: commercial tools like Intenseye cost upwards of 30,000 dollars a year and only understand US OSHA rules. Academic papers stop at basic bounding boxes. No system in India reasons about risk severity or maps violations to statutory penal codes. RakshaSite addresses all three limitations."

---

### Slide 4: System Architecture: The 4-Layer Multimodal Pipeline
- **Four Distinct Operational Layers**:
  1. **Perception Layer**: YOLOv8n (fine-tuned) + DeepSORT Multi-Object Tracking + Spatial Zone Inference.
  2. **Intelligence Layer (LangGraph 4-Agent Graph)**:
     - *Agent 1 (Detection)*: 5-frame temporal confirmation filter (eliminates false positives).
     - *Agent 2 (Risk)*: XGBoost model trained on 100k worksite incident records ($R^2 = 0.9998$).
     - *Agent 3 (Compliance)*: Hybrid ChromaDB RAG citing exact BOCW Act sections with 100% deterministic fallback.
     - *Agent 4 (Report)*: ReportLab engine generating signed PDF audit reports with anti-hallucination verification.
  3. **Data Layer**: PostgreSQL / SQLite structured logs + ChromaDB vector embeddings + MLflow tracking.
  4. **Presentation Layer**: FastAPI REST service + Executive Live Demo Viewer (`/demo`) + OpenAPI contracts.
- **Visual Asset**: Architectural Flowchart showing video feed flowing through Perception -> LangGraph -> Persistence -> Executive Viewer.
- **Speaker Notes**:
  > "Here is our 4-layer architecture. We start at the Perception Layer with fine-tuned YOLOv8 and DeepSORT tracking. Detections pass into our LangGraph Intelligence Layer, where four specialized agents collaborate: filtering noise, evaluating hazard severity, retrieving legal statutes via RAG, and compiling audit reports. Finally, results are served instantly via FastAPI."

---

### Slide 5: Perception Layer Engineering: Fine-Tuning & Hallucination Defense
- **Overcoming the Real-World Domain Gap**:
  - Baseline YOLOv8n trained on generic data achieved only **8.3% mAP50** on the critical `no_helmet` class on real construction photos.
  - Fine-tuned on Ultralytics Construction-PPE dataset with canonical remapping and AMP on RTX 3050 GPU.
  - Achieved **64.9% mAP50 on `no_helmet` (7.8x gain)** and **82.85% overall mAP50**.
- **3-Tier Hallucination & False Positive Defense**:
  - *Confidence Gate*: Detection confidence threshold $\ge 0.70$.
  - *Spatial Scale Filter*: Bounding box area $\ge 1\%$ of image dimensions.
  - *Spatial Containment Reasoning*: Verifies whether head and torso regions inside a tracked `person` bounding box contain required PPE.
  - *Temporal Confirmation Streak*: 5 consecutive frames required before raising an alert.
- **Visual Asset**: Side-by-side comparison images: baseline failing to detect bare heads vs fine-tuned model detecting `no_helmet` with confidence HUD.
- **Speaker Notes**:
  > "Pretrained models fail terribly on real construction sites because bare heads blend into chaotic backgrounds. Our baseline model scored just 8.3% on missing helmets. By fine-tuning on real worksite data and implementing spatial containment rules — checking the upper 25% of worker boxes for helmets — we achieved a 7.8x jump in violation detection accuracy, while 5-frame temporal tracking eliminates transient noise."

---

### Slide 6: Intelligence Layer: LangGraph Multi-Agent Reasoning & BOCW RAG
- **Autonomous Multi-Agent Collaboration**:
  - **Agent 1 (Detection Validation)**: Maintains per-worker temporal streaks; rejects transient noise.
  - **Agent 2 (Risk Assessment)**: XGBoost classifier evaluates 5 non-linear features: violation type, spatial hazard zone, proximity to heavy machinery, duration streak, and time-of-day ($R^2 = 0.9998$). Outputs 0–100 risk score and severity tier (LOW/MEDIUM/HIGH/CRITICAL).
  - **Agent 3 (Statutory Compliance)**: Hybrid semantic retrieval via ChromaDB (`all-MiniLM-L6-v2`) over BOCW Act 1996 and Central Rules 1998, with deterministic fallback for 100% citation reliability.
  - **Agent 4 (Report Generation)**: Synthesizes structured audit logs and generates professional ReportLab PDF reports with anti-hallucination verification.
- **Visual Asset**: LangGraph StateGraph execution diagram showing state transitions and data contracts between agents.
- **Speaker Notes**:
  > "Our Intelligence Layer uses LangGraph to coordinate four specialized agents. Agent 2 uses XGBoost to assess risk — recognizing that a missing helmet on a 10-meter scaffold is vastly more hazardous than one on ground level. Agent 3 uses RAG to pull the exact statutory clause from the BOCW Act. If the LLM is offline, a deterministic fallback ensures legal citations are never hallucinated and 100% compliant."

---

### Slide 7: Live System Demonstration & Artifact Walkthrough
- **One-Command Execution**: `python demo_runner.py --mode both`
- **Generated Core Artifacts**:
  1. *Annotated Video (`reports/demo_annotated_output.mp4`)*: H.264 MSMF stream with persistent worker IDs, risk badges, and violation HUD.
  2. *Violation Snapshot (`reports/demo_snapshot_violation.jpg`)*: Frame capture of highest-severity incident.
  3. *Statutory Audit Log (`reports/demo_audit_log.json`)*: Machine-readable audit trail citing Section 38(1)(a) and Rule 46.
  4. *Official PDF Compliance Report (`reports/safety_audit.pdf`)*: Formal legal compliance report ready for site management.
- **Executive Demo Viewer**: Live web interface served at `http://localhost:8000/demo` displaying real-time video, alerts, and PDF download.
- **Visual Asset**: Screenshot of the Executive Demo Viewer showing the live video player, violation HUD, compliance gauge (60/100), and citation banner.
- **Speaker Notes**:
  > "Everything you see runs with a single command. The system processes construction video at 98 frames per second on GPU or 18 FPS on CPU. Within seconds, it produces an annotated H.264 video, a high-resolution violation snapshot, a complete JSON audit log, and an official PDF compliance report ready for labor inspectors."

---

### Slide 8: Empirical Validation & 3-Tier Ablation Study
- **Held-Out Validation Benchmark (200 Real Worksite Photos)**:
  - Overall mAP50: **0.8285** | Overall Recall: **0.7595** | Overall Precision: **0.8802**.
  - Worker-Level Violation Detection: Combined Rule achieves **F1 = 0.7474** (80.04% accuracy across 481 workers).
- **3-Tier Ablation Study Results on 50 Real Validation Photos (`tests/test_cases.json`)**:
  - *Tier 1 (YOLO Only)*: Precision 100.0%, Recall 78.26%, F1 87.80%, False Positive Rate 0.0%, Citation Accuracy 0.0%.
  - *Tier 2 (YOLO + Static Rules)*: Precision 95.45%, Recall 91.30%, F1 93.33%, False Positive Rate 50.0%, Citation Accuracy 88.0%.
  - *Tier 3 (Full 4-Agent Pipeline)*: **Precision 100.0%, Recall 100.0%, F1 100.0%, False Positive Rate 0.0%, Hazard Accuracy 56.0% (88% on critical), Citation Accuracy 100.0%**.
- **Visual Asset**: Grouped bar chart illustrating the dramatic progression from Tier 1 to Tier 3 in Recall, False Positive Elimination, and Statutory Accuracy.
- **Speaker Notes**:
  > "To prove the scientific merit of our multi-agent architecture, we conducted an ablation study over our 50 real photographic test cases. Raw YOLO misses 22% of violations due to occlusions and has zero legal understanding. Adding heuristic rules improves recall to 91% but introduces false alarms. Only our full 4-agent LangGraph pipeline achieves 100% recall, zero false alarms, and flawless statutory citation accuracy."

---

### Slide 9: Statutory Compliance, Legal Defensibility & Competitive Edge
- **BOCW Act (1996) Alignment**:
  - *Section 38(1)(a) & Rule 46*: Industrial safety helmets conforming to IS 2925.
  - *Section 40(2)(r) & Rule 45*: High-visibility retro-reflective vests.
  - *Section 39 & Rules 175–185*: Fall protection and scaffolding safety (IS 3696 / IS 3521).
  - *Section 47*: Statutory penalties (₹2,000 fine, 3 months imprisonment, ₹100/day ongoing).
- **Engineering Excellence & Zero Git Bloat**:
  - Full clone reproducibility solved via `scripts/fetch_data.py` (0 blobs > 25MB).
  - 100% offline fallback capability — zero dependency on external cloud APIs during inspection.
  - Cost comparison: < ₹4,000 academic budget vs ₹30,00,000/year enterprise alternatives (**>99% cost reduction**).
- **Visual Asset**: Table highlighting BOCW legal provisions alongside corresponding detected PPE classes and statutory penalties.
- **Speaker Notes**:
  > "Every citation generated by RakshaSite is legally grounded in India's BOCW Act and BIS standards. Furthermore, we solved the 25 MB Git limit without external bloat: a fresh clone fetches verified assets via SHA-256 hashes in seconds. Our system is fully offline-resilient, meaning site managers can run safety audits deep inside basement excavation sites without internet connectivity."

---

### Slide 10: Conclusion, Future Roadmap & Viva Defense Q&A
- **Summary of Key Contributions**:
  1. Fine-tuned YOLOv8n achieving 0.8285 mAP50 with a 7.8x gain in missing helmet detection.
  2. First Indian construction safety framework integrating 4-agent LangGraph reasoning.
  3. Grounded BOCW Act RAG pipeline delivering 100% Top-1 statutory citation accuracy.
  4. 269/269 passing automated test suite with full fresh-clone reproducibility.
- **Future Roadmap**:
  - *Phase 2*: Full-body harness & tethering detection using pose estimation.
  - *Phase 3*: Multilingual audio alerts in Hindi, Bengali, and Tamil via site loudspeakers.
  - *Phase 4*: Drone-mounted aerial worksite inspection.
- **Call to Action**: System live and ready for examiner inspection.
- **Visual Asset**: RakshaSite logo, GitHub repository QR code, and "Thank You / Questions?" title card.
- **Speaker Notes**:
  > "In conclusion, RakshaSite bridges artificial intelligence, spatial safety engineering, and Indian labour law. We have delivered a fully verified, 269-test-passing system that runs in real time on commodity hardware. We thank our guide and the department, and we are now open for your questions."
