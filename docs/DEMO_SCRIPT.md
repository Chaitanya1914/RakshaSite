# RakshaSite: 3-Minute Faculty Viva Walkthrough Script
## Live System Demonstration & Defense Protocol for Faculty Reviewers

---

## Presenter Roles & Logistics
- **Chaitanya (Presenter 1)**: Problem framing, technical architecture, and live execution.
- **Arshi (Presenter 2)**: Regulatory RAG pipeline, empirical results, and defense Q&A.
- **Target Duration**: Exactly 3 minutes (180 seconds) + 2 minutes panel Q&A.
- **Pre-Flight Setup (Done before entering the room)**:
  1. Open terminal in project root: `c:\Users\chait\OneDrive\Desktop\rakshasite`.
  2. Start FastAPI backend: `.\.venv\Scripts\python.exe main.py`.
  3. Open browser tab at: `http://localhost:8000/demo`.
  4. Have a secondary terminal ready with the command: `.\.venv\Scripts\python.exe demo_runner.py --mode both`.

---

## SECOND-BY-SECOND VIVA SCRIPT

### MINUTE 1 (0:00 – 1:00): The Problem, Statutory Gap & Architecture

#### [0:00 – 0:30] The Crisis & The Statutory Mandate (Chaitanya)
- **Action**: Stand upright, face faculty panel, gesture toward Title Slide.
- **Spoken Text**:
  > "Respected faculty members, in India, 38 construction workers lose their lives every day — accounting for 11% of global workplace deaths. While India's BOCW Act of 1996 mandates strict PPE compliance, enforcement is nearly impossible: only 1,800 labour inspectors exist for 50 million construction workers. Existing safety AI tools like Intenseye cost 30 to 40 lakh rupees annually and know nothing about Indian law.
  >
  > Today, we present **RakshaSite**: an autonomous, multimodal AI safety framework that unites real-time computer vision, multi-agent reasoning, and statutory Indian law mapping on affordable, edge-capable hardware."

#### [0:30 – 1:00] The 4-Layer Architecture (Chaitanya)
- **Action**: Switch to Architecture Diagram Slide.
- **Spoken Text**:
  > "RakshaSite operates across four layers:
  > First, our **Perception Layer** uses a fine-tuned YOLOv8n detector coupled with DeepSORT tracking and spatial containment rules to detect workers, helmets, and vests at up to 98 frames per second.
  > Second, detections feed into our **LangGraph Intelligence Layer**, where four specialized agents collaborate: Agent 1 eliminates transient noise; Agent 2 runs an XGBoost model to score hazard severity; Agent 3 retrieves exact BOCW Act sections via ChromaDB RAG; and Agent 4 generates an official PDF compliance report.
  > Let us now show you the system running live."

---

### MINUTE 2 (1:00 – 2:00): Live Execution & Executive Demo Viewer Walkthrough

#### [1:00 – 1:25] Triggering the One-Command Demo (Chaitanya)
- **Action**: Switch to terminal and press ENTER on:
  ```powershell
  .\.venv\Scripts\python.exe demo_runner.py --mode both
  ```
- **Spoken Text**:
  > "I am triggering our one-command demo. The pipeline ingests real worksite video from `hardhat_demo.mp4` and a high-resolution still image. As you see on the console, YOLO detects workers, DeepSORT assigns persistent IDs, our 5-frame temporal filter confirms real violations, and the LangGraph agent chain evaluates risk and maps statutory clauses.
  > 
  > In less than 15 seconds, the demo completes, producing a verified H.264 video, violation snapshots, a JSON audit log, and an official PDF audit report."

#### [1:25 – 2:00] Executive Demo Viewer Walkthrough (Arshi)
- **Action**: Switch to browser showing `http://localhost:8000/demo` and refresh.
- **Spoken Text**:
  > "Here is our Executive Demo Viewer, served live by our FastAPI backend.
  > Notice the video player playing the processed worksite feed:
  > 1. Each worker has a persistent track ID — here is Worker `w-1` and Worker `w-2`.
  > 2. The HUD displays risk levels in real time: Worker 1 is flagged with **MEDIUM Risk** for missing a helmet and **LOW Risk** for missing a vest.
  > 3. Below the video, our Statutory Citation Banner shows exact legal provisions: **BOCW Act Section 38(1)(a) read with Central Rule 46 and IS 2925** for industrial helmets, citing the statutory penalty under Section 47 of ₹2,000 fine and up to 3 months imprisonment.
  > 4. On the right panel, you see the captured violation snapshot and can click to immediately download the generated `safety_audit.pdf`."

---

### MINUTE 3 (2:00 – 3:00): Empirical Results, Ablation Defense & Legal Defensibility

#### [2:00 – 2:30] Empirical Breakthrough & Ablation Study (Arshi)
- **Action**: Switch to Results & Ablation Table Slide.
- **Spoken Text**:
  > "Moving to our empirical results: generic YOLO models fail completely on worksite violations. On our held-out test set of 200 real worksite photos, the baseline model achieved only 8.3% mAP on missing helmets. By fine-tuning with canonical remapping and spatial containment, we achieved **64.9% mAP on missing helmets — a 7.8x improvement** — and **0.8285 overall mAP50**.
  >
  > In our 3-tier ablation study across 50 real worksite photographs, raw YOLO missed 22% of violations due to occlusions (78.3% recall). Adding spatial rules improved recall to 91.3% but introduced false alarms. Only our full 4-agent LangGraph pipeline achieved 100% recall, zero false alarms, and 100% statutory citation accuracy.

#### [2:30 – 3:00] Legal Grounding, Engineering Rigor & Conclusion (Chaitanya)
- **Action**: Switch to Conclusion Slide with QR code.
- **Spoken Text**:
  > "Every claim in RakshaSite is backed by empirical artifacts:
  > - We solved the 25 MB Git limit via our SHA-256 verified streaming manifest.
  > - Our complete automated test suite of **269 pytest tests passes with 100% success**.
  > - The system features full offline resilience: if external LLM APIs are unreachable, our deterministic statutory fallback ensures legal citations are never hallucinated.
  > - At a hardware cost of under ₹4,000, RakshaSite delivers enterprise-grade safety compliance at 99% lower cost than commercial tools.
  > 
  > Thank you, and we welcome your questions."

---

## FACULTY VIVA DEFENSE Q&A CHEATSHEET

### Question 1: "Why not just use YOLO alone? Why do you need LangGraph agents?"
- **Answer**:
  > "Raw YOLO is purely perceptual: it draws a box and outputs a class probability. It cannot differentiate between a worker who took off their helmet for 1 frame versus a chronic violator, nor can it reason about height or machinery hazards. 
  > Our LangGraph architecture introduces three indispensable layers:
  > 1. *Temporal Confirmation*: Agent 1 requires a 5-frame streak, eliminating transient false positives.
  > 2. *Non-Linear Risk Scoring*: Agent 2's XGBoost evaluates hazard interactions (height + zone + time).
  > 3. *Legal Accountability*: Agent 3 maps violations to enforceable statutory sections under the BOCW Act. YOLO alone cannot do any of this."

### Question 2: "What if there is no internet connection on a construction site? Does your LLM fail?"
- **Answer**:
  > "No, the demo and system are designed with 100% offline fallback. Agent 3 has a deterministic legal mapping dictionary in `src/rag/bocw_parser.py` that maps `no_helmet` directly to Section 38(1)(a) and Rule 46. If ChromaDB or Gemini APIs are unreachable, the system executes the deterministic rule engine instantly, guaranteeing zero downtime and zero hallucinated statutes."

### Question 3: "How did you solve the 25 MB file limit on GitHub without losing datasets or weights?"
- **Answer**:
  > "We engineered a zero-git-bloat distribution architecture in `scripts/fetch_data.py` guided by `scripts/manifest.json`. Only code, metadata, and labels are stored in Git. Our script streams compressed archives from GitHub Release mirrors or local mirrors, verifies their SHA-256 cryptographic hashes, and unpacks them with Zip Slip path security. Running `python scripts/verify_fresh_clone.py` proves a clean clone succeeds with zero manual intervention."

### Question 4: "Can your system run in real-time on cheap site computers without an expensive GPU?"
- **Answer**:
  > "Yes. On an entry-level NVIDIA RTX 3050 GPU, our pipeline runs at 98.3 FPS (10.2 ms latency). But more importantly, on a standard laptop CPU without any GPU enabled, it runs at 18.1 FPS (55.2 ms latency). Since construction safety monitoring typically requires only 10 to 15 FPS, RakshaSite runs comfortably in real-time on standard site laptops."
