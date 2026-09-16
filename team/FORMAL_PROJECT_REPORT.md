# PROJECT REQUIREMENT REPORT
## For Faculty Review — Semester Project (10 Credits)

---

## TITLE PAGE

| Field | Details |
|---|---|
| **Project Title** | RakshaSite: A Multimodal Agentic Framework for Real-Time Construction Site Safety Compliance Monitoring |
| **Department** | Computer Science & Engineering (AI & ML) |
| **University** | COER University, Roorkee, Uttarakhand |
| **Academic Year** | 2026–27 |
| **Semester** | 5th Semester |
| **Submitted By** | Chaitanya (Roll No: ___________) |
| **Guided By** | Prof. ___________ |
| **Date of Submission** | September 2026 |

---

## 1. ABSTRACT

Construction site fatalities in India account for 38 deaths per day (NCRB), with 80% of incidents being preventable through proper Personal Protective Equipment (PPE) compliance. Despite the existence of the Building and Other Construction Workers (BOCW) Act, 1996, enforcement remains manual and inadequate — approximately 1,800 labour inspectors are tasked with monitoring over 5 crore construction workers nationwide.

This project proposes **RakshaSite**, an autonomous, AI-driven safety compliance monitoring system that combines real-time Computer Vision (YOLOv8) for PPE violation detection, a Multi-Agent reasoning pipeline (LangGraph) for autonomous risk assessment and regulatory mapping, and Retrieval-Augmented Generation (RAG) for citing exact BOCW Act sections — all integrated into a live web dashboard. The system transitions safety monitoring from a manual, reactive process to an automated, proactive, and legally-grounded compliance framework.

**Keywords:** Computer Vision, Multi-Agent Systems, Agentic AI, RAG, Construction Safety, BOCW Act, YOLOv8, LangGraph, Deep Learning, Generative AI.

---

## 2. INTRODUCTION

### 2.1 Background

The Indian construction industry employs over 5 crore workers and contributes approximately 9% to the national GDP. However, it is also one of the most hazardous sectors globally, with India accounting for 11% of worldwide occupational fatalities (ILO, 2023). The primary causes of construction-related deaths include falls from height (34%), being struck by objects (22%), and electrocution (15%) — the majority of which are directly attributable to non-compliance with basic PPE norms such as wearing helmets, safety vests, and harnesses.

The Building and Other Construction Workers (Regulation of Employment and Conditions of Service) Act, 1996, mandates employers to provide and enforce PPE usage at all construction sites. Non-compliance attracts penalties of up to ₹2,000 per day and imprisonment of up to 3 months under Section 47 of the Act. Despite this legal framework, enforcement remains complaint-driven and severely understaffed.

### 2.2 Motivation

Current approaches to construction safety monitoring are:

1. **Manual Inspection:** Labour inspectors physically visit sites — infeasible at scale (1 inspector per 27,000 workers).
2. **CCTV Surveillance:** Cameras exist at many sites but are passively monitored by security guards who are not trained in safety compliance.
3. **Enterprise AI Solutions:** Companies like Intenseye (USA) and viAct (Hong Kong) offer AI-based safety monitoring, but at costs of $30,000–$50,000/year — unaffordable for 99% of Indian construction firms.

There exists a clear need for an **affordable, intelligent, India-specific** safety monitoring system that can:
- Detect PPE violations in real-time using Computer Vision.
- Reason about the severity and regulatory implications using AI agents.
- Generate compliance reports citing exact legal provisions.

### 2.3 Objective of the Project

The objective of this project is to design and implement an autonomous, multi-agent AI system that:

1. Detects construction site safety violations (helmet, vest) in real-time from video feeds using YOLOv8.
2. Assesses the risk severity of each violation using a trained ML model (XGBoost).
3. Maps each violation to the specific section of the BOCW Act, 1996 using a RAG pipeline.
4. Generates natural-language compliance reports using Generative AI (LLM).
5. Presents all outputs on a real-time web dashboard with analytics.

---

## 3. LITERATURE SURVEY

### 3.1 Existing Systems and Research

| S.No. | Title / System | Author / Company | Year | Methodology | Limitation |
|---|---|---|---|---|---|
| 1 | "Hard Hat Detection using YOLOv5" | Wu et al. | 2021 | YOLOv5 on custom dataset, 91.3% mAP | Detection only — no risk assessment, no regulatory mapping, no report generation |
| 2 | "PPE Detection for Construction Safety" | Nath et al. | 2020 | Faster R-CNN for helmet + vest detection | Slow inference (~5 FPS), not suitable for real-time |
| 3 | Intenseye (Commercial) | Intenseye Inc., USA | 2020–present | Proprietary CV models for workplace safety | $50,000/year, US OSHA-focused, no Indian law mapping |
| 4 | viAct (Commercial) | viAct Ltd., Hong Kong | 2019–present | CV-based construction monitoring | $30,000/year, traditional pipeline (detect → alert), no agentic reasoning |
| 5 | SafetyCulture / iAuditor | SafetyCulture Pty Ltd. | 2012–present | Digital checklists for safety audits | Fully manual — zero AI detection capability |
| 6 | "ReAct: Synergizing Reasoning and Acting in Language Models" | Yao et al. | 2023 | LLM agents that reason and take actions | Foundational framework for agentic AI; not applied to construction safety |
| 7 | "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" | Lewis et al. | 2020 | Combining retrieval with generation for factual outputs | Foundational RAG paper; not applied to legal/regulatory domains in India |

### 3.2 Research Gap Identified

| Gap | Explanation |
|---|---|
| **Gap 1: Detection without Intelligence** | All existing CV-based PPE detection systems stop at drawing a bounding box. None assess risk severity or provide regulatory context. |
| **Gap 2: No Indian Regulatory Mapping** | No existing system — academic or commercial — maps detected violations to specific sections of the BOCW Act, 1996. |
| **Gap 3: No Autonomous Reasoning** | Existing systems use traditional ML pipelines (input → model → output). None employ multi-agent autonomous reasoning where AI agents collaborate to assess, contextualize, and report violations. |
| **Gap 4: Affordability** | Enterprise solutions cost ₹25–42 lakh/year. No affordable solution exists for the Indian market. |

### 3.3 How This Project Addresses the Gaps

RakshaSite bridges all four gaps by combining:
- **Real-time CV detection** (addressing Gap 1 partially),
- **Multi-agent agentic reasoning** (addressing Gap 3),
- **RAG-based Indian regulatory mapping** (addressing Gap 2),
- **Open-source, low-cost architecture** (addressing Gap 4).

---

## 4. PROPOSED SYSTEM

### 4.1 System Overview

RakshaSite is a four-layer system:

| Layer | Function | Core Technology |
|---|---|---|
| **Perception Layer** | Detect workers and PPE from video | YOLOv8, OpenCV, DeepSORT |
| **Intelligence Layer** | Autonomous multi-agent reasoning | LangGraph, XGBoost, RAG |
| **Data Layer** | Persist violations, regulations, analytics | PostgreSQL, ChromaDB, MLflow |
| **Presentation Layer** | Real-time dashboard and reports | Streamlit, ReportLab (PDF) |

### 4.2 System Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                   VIDEO INPUT                             │
│          (IP Camera / Webcam / Uploaded Video)             │
└─────────────────────────┬────────────────────────────────┘
                          │ Frames (15-30 FPS)
                          ▼
┌──────────────────────────────────────────────────────────┐
│               PERCEPTION LAYER                            │
│                                                           │
│   YOLOv8n ──► Bounding Boxes ──► DeepSORT Tracking       │
│   (Helmet, Vest, Person)         (Worker ID Assignment)   │
│                                                           │
│   OpenCV: Zone mapping, spatial analysis                  │
└─────────────────────────┬────────────────────────────────┘
                          │ Structured Detection JSON
                          ▼
┌──────────────────────────────────────────────────────────┐
│             INTELLIGENCE LAYER (LangGraph)                │
│                                                           │
│   Agent 1          Agent 2          Agent 3        Agent 4│
│   ┌──────────┐    ┌──────────┐    ┌──────────┐   ┌──────┐│
│   │Detection │    │  Risk    │    │Compliance│   │Report││
│   │Validation│───▶│Assessment│───▶│  (RAG)   │──▶│(LLM) ││
│   │          │    │(XGBoost) │    │          │   │      ││
│   └──────────┘    └──────────┘    └──────────┘   └──────┘│
└─────────────────────────┬────────────────────────────────┘
                          │ Compliance Report JSON
                          ▼
┌──────────────────────────────────────────────────────────┐
│               DATA & PERSISTENCE LAYER                    │
│                                                           │
│   PostgreSQL          ChromaDB           MLflow           │
│   (Violation logs,    (BOCW Act          (Model versions, │
│    worker data,        embeddings,        experiment       │
│    analytics)          IS 3696)           tracking)        │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              PRESENTATION LAYER                           │
│                                                           │
│   Streamlit Dashboard:                                    │
│   • Live video with bounding boxes                        │
│   • Real-time violation alerts with BOCW citations        │
│   • Compliance score (0-100)                              │
│   • Analytics (charts, heatmaps, trends)                  │
│   • PDF report download                                   │
└──────────────────────────────────────────────────────────┘
```

### 4.3 Data Flow Diagram (DFD) — Level 0 (Context Diagram)

```
                    ┌───────────────┐
  Video Feed ──────▶│               │──────▶ Real-time Alerts
                    │  RakshaSite   │
  BOCW Act PDF ────▶│    System     │──────▶ Compliance Reports
                    │               │
  Site Config ─────▶│               │──────▶ Analytics Dashboard
                    └───────────────┘
```

### 4.4 Data Flow Diagram — Level 1

```
┌──────────┐     ┌───────────┐     ┌─────────────┐     ┌────────────┐
│  Video   │     │  Process  │     │   Agent     │     │  Generate  │
│  Input   │────▶│  Frames   │────▶│   Pipeline  │────▶│  Reports   │
│          │     │  (YOLO)   │     │  (LangGraph)│     │  (LLM)     │
└──────────┘     └─────┬─────┘     └──────┬──────┘     └─────┬──────┘
                       │                  │                   │
                       ▼                  ▼                   ▼
                 ┌───────────┐     ┌────────────┐      ┌───────────┐
                 │ Detection │     │ Violation  │      │ Report    │
                 │ Data Store│     │ Log (SQL)  │      │ Storage   │
                 └───────────┘     └────────────┘      └───────────┘
```

---

## 5. MODULE DESCRIPTION

### Module 1: Video Ingestion & Frame Extraction Module
- **Input:** Live video stream (webcam/IP camera) or uploaded video file.
- **Processing:** OpenCV `VideoCapture` extracts frames at 15-30 FPS. Frames are resized to 640×640 for YOLO input.
- **Output:** Individual frames passed to the detection pipeline.
- **Technology:** Python, OpenCV.

### Module 2: Object Detection & Tracking Module
- **Input:** Extracted video frames.
- **Processing:** YOLOv8n model performs inference to detect three classes — `person`, `helmet`, `vest`. DeepSORT algorithm assigns persistent tracking IDs to each worker across frames for temporal consistency.
- **Output:** Structured JSON with bounding boxes, class labels, confidence scores, and worker IDs.
- **Technology:** Ultralytics YOLOv8, DeepSORT, OpenCV.

### Module 3: Detection Validation Agent
- **Input:** Raw YOLO detections.
- **Processing:** Filters false positives (confidence < 0.7). Confirms violations only if detected consistently across 5+ consecutive frames for the same tracked worker. Prevents spurious single-frame alerts.
- **Output:** Validated violation records.
- **Technology:** Python, LangGraph.

### Module 4: Risk Assessment Agent
- **Input:** Validated violation records + zone metadata (e.g., elevated platform, ground level, near machinery).
- **Processing:** An XGBoost classifier scores each violation on a 0–100 severity scale based on features: `violation_type`, `zone_danger_level`, `worker_proximity_to_hazard`, `consecutive_violation_count`, `time_of_day`.
- **Output:** Risk score (0–100) and severity label (LOW / MEDIUM / HIGH / CRITICAL).
- **Technology:** XGBoost (scikit-learn), LangGraph.

### Module 5: Compliance Agent (RAG Pipeline)
- **Input:** Validated violation with risk score.
- **Processing:** Converts violation description into a vector embedding using Sentence Transformers (`all-MiniLM-L6-v2`). Performs semantic search over ChromaDB (containing BOCW Act 1996 sections, IS 3696 safety codes) to retrieve the most relevant regulatory clause. Returns exact section number, quoted legal text, and applicable penalty.
- **Output:** Regulatory citation JSON.
- **Technology:** Sentence Transformers, ChromaDB, LangChain, LangGraph.

### Module 6: Report Generation Agent (Generative AI)
- **Input:** Combined output from all upstream agents.
- **Processing:** An LLM (Gemini 1.5 Flash or Llama 3 8B via Ollama) generates:
  - **Instant alert** (one-line, displayed on dashboard).
  - **Incident report** (detailed PDF with screenshot, violation details, regulation citation, recommended action).
  - **Shift-end summary** (aggregated report for the past 8 hours).
- **Output:** Natural-language text + formatted PDF.
- **Technology:** Gemini API / Ollama, ReportLab (PDF generation), LangGraph.

### Module 7: Database & Analytics Module
- **Input:** All violation records, compliance scores, worker data.
- **Processing:** Stores structured violation data in PostgreSQL. Provides analytical queries for trend analysis, repeat offender identification, and compliance scoring.
- **Output:** Queryable database + analytics data for the dashboard.
- **Technology:** PostgreSQL, SQLAlchemy, Pandas.

### Module 8: Dashboard & Visualization Module
- **Input:** Real-time detection data + historical analytics.
- **Processing:** Renders live video feed with bounding boxes, real-time violation alerts, compliance score gauge (0–100), and analytical charts (bar charts, heatmaps, trend lines).
- **Output:** Interactive web dashboard accessible via browser.
- **Technology:** Streamlit, Plotly, Folium (for spatial heatmaps).

---

## 6. TECHNOLOGY STACK

### 6.1 Software Requirements

| Category | Technology | Version | Purpose |
|---|---|---|---|
| **Programming Language** | Python | 3.10+ | Core development language |
| **Deep Learning Framework** | Ultralytics YOLOv8 | 8.x | Object detection model |
| **Computer Vision** | OpenCV | 4.8+ | Video processing, frame extraction, image manipulation |
| **Object Tracking** | DeepSORT | — | Multi-object tracking with persistent IDs |
| **Classical ML** | scikit-learn, XGBoost | 1.3+, 2.0+ | Risk severity classification |
| **Agentic Framework** | LangGraph | 0.1+ | Multi-agent orchestration |
| **RAG Framework** | LangChain | 0.2+ | Retrieval-Augmented Generation pipeline |
| **Vector Database** | ChromaDB | 0.4+ | Embedding storage and semantic search |
| **Embedding Model** | Sentence Transformers | 2.x | Text embedding for RAG |
| **Generative AI** | Gemini 1.5 Flash API / Ollama (Llama 3) | — | Natural language report generation |
| **Relational Database** | PostgreSQL / SQLite | 15+ / 3 | Violation logs, analytics |
| **ORM** | SQLAlchemy | 2.0+ | Database abstraction |
| **MLOps** | MLflow | 2.x | Experiment tracking, model versioning |
| **Model Optimization** | ONNX Runtime | 1.16+ | INT8 quantization for edge deployment |
| **Web Framework** | Streamlit | 1.30+ | Dashboard and UI |
| **PDF Generation** | ReportLab | 4.x | Compliance report PDF output |
| **Visualization** | Plotly, Matplotlib | 5.x, 3.x | Charts, graphs, heatmaps |
| **API Framework** | FastAPI | 0.100+ | REST API endpoints |

### 6.2 Hardware Requirements

| Component | Minimum Specification | Recommended |
|---|---|---|
| **Processor** | Intel i5 / Ryzen 5 (8th Gen+) | Intel i7 / Ryzen 7 |
| **RAM** | 8 GB | 16 GB |
| **GPU (for training)** | Google Colab T4 (free) | Colab Pro A100 |
| **GPU (for inference)** | CPU-only (YOLOv8n runs at ~15 FPS on CPU) | NVIDIA GTX 1650+ for 30+ FPS |
| **Storage** | 20 GB free | 50 GB free |
| **Camera** | Laptop webcam (720p) | IP camera (1080p) |
| **Internet** | Required for Gemini API calls | Ollama eliminates this dependency |

---

## 7. DATASET DESCRIPTION

### 7.1 Training Datasets

| Dataset | Source | Images | Classes | Format | License |
|---|---|---|---|---|---|
| SHEL5K | Academic (Shi et al., 2023) | 5,000 | Helmet, No Helmet | YOLO | Free for research |
| Hard Hat Workers | Kaggle | 5,000+ | Helmet, Head, Person | YOLO-ready | Open |
| Pictor-v3 | Academic (Delhi et al., 2020) | 1,330 | Helmet, Vest, Worker | COCO/YOLO | Free for research |
| Construction Safety | Roboflow Universe | Multiple sets | PPE classes | YOLOv8 export | Free tier |
| **Total** | — | **12,000+** | **Multi-class** | **YOLO format** | **Free** |

### 7.2 Knowledge Base for RAG

| Document | Source | Sections | Purpose |
|---|---|---|---|
| BOCW Act, 1996 | legislative.gov.in | ~85 sections | Primary regulatory reference |
| BOCW Central Rules, 1998 | legislative.gov.in | ~250 rules | Detailed compliance rules |
| IS 3696 (Parts 1–3) | Bureau of Indian Standards | ~120 clauses | Scaffolding safety codes |
| IS 7069 | Bureau of Indian Standards | ~80 clauses | Material handling safety |

### 7.3 Data Preprocessing Strategy

1. **Merge** all datasets into a unified YOLOv8 format with consistent class IDs.
2. **Augmentation:** Random brightness (±30%), Gaussian blur, perspective warp, mosaic augmentation — to simulate real-world site conditions (dust, poor lighting, rain).
3. **Split:** 70% training, 15% validation, 15% testing.
4. **Regulatory text:** PDF → text extraction → chunking by section (200-400 tokens per chunk) → embedding via Sentence Transformers → storage in ChromaDB.

---

## 8. FEASIBILITY ANALYSIS

### 8.1 Technical Feasibility

| Component | Feasibility | Justification |
|---|---|---|
| YOLOv8 training | ✅ High | Pre-trained model, free GPU (Colab), well-documented library |
| LangGraph agents | ✅ High | Well-documented framework with tutorials; student has prior RAG experience (SolarSentinel project) |
| RAG pipeline | ✅ High | Student has built RAG with Ollama + ChromaDB previously |
| Streamlit dashboard | ✅ High | Student has built multiple Streamlit apps (ATFOS, SolarSentinel) |
| Edge deployment (ONNX) | ✅ Medium | New skill but well-documented; Ultralytics has one-line export command |

### 8.2 Economic Feasibility

| Item | Cost |
|---|---|
| Development (3 months) | ₹3,000–3,500 |
| All datasets | ₹0 (free/open-source) |
| Compute (Google Colab Pro) | ₹1,800 (2 months) |
| Demo props | ₹350 |
| **Total** | **< ₹4,000** |

### 8.3 Operational Feasibility

- The system runs entirely on a standard laptop for demo purposes.
- No special lab equipment, no external hardware, no institutional permissions required.
- The live demo uses only the laptop's built-in webcam.

---

## 9. PROJECT TIMELINE (3 Months / 12 Weeks)

| Week | Phase | Tasks | Deliverable |
|---|---|---|---|
| 1 | **Data Preparation** | Download datasets (SHEL5K, Hard Hat Workers, Pictor-v3), merge, clean, standardize class labels | Unified dataset in YOLOv8 format |
| 2 | **Model Training** | Train YOLOv8n on Google Colab, hyperparameter tuning, MLflow logging | Trained model (mAP ≥ 85%) |
| 3 | **Model Optimization** | ONNX export, INT8 quantization, webcam inference testing | Edge-ready model file |
| 4 | **RAG Pipeline** | Parse BOCW Act PDF, chunk by section, embed into ChromaDB, test retrieval accuracy | Working compliance lookup |
| 5 | **Agent 1 & 2** | Build Detection Validation Agent + Risk Assessment Agent (XGBoost) | Agents returning validated violations with risk scores |
| 6 | **Agent 3 & 4** | Build Compliance Agent (RAG) + Report Agent (LLM) | Full 4-agent chain functional |
| 7 | **LangGraph Integration** | Wire all agents into LangGraph StateGraph, test end-to-end pipeline | Complete autonomous pipeline |
| 8 | **Dashboard** | Build Streamlit UI — live feed, alerts, compliance gauge, analytics | Working web dashboard |
| 9 | **Database Integration** | PostgreSQL setup, violation logging, API endpoints (FastAPI) | Persistent data layer + REST API |
| 10 | **Testing & Evaluation** | End-to-end testing, precision/recall evaluation, ablation study | Evaluation metrics report |
| 11 | **Documentation** | Project report writing, literature survey finalization, diagrams | Draft project report |
| 12 | **Demo Preparation** | Final integration, presentation slides, demo rehearsal | Presentation-ready system |

---

## 10. EXPECTED OUTCOMES

1. A **trained YOLOv8 model** achieving ≥85% mAP@0.5 for helmet and vest detection on the held-out test set.
2. A **functional multi-agent system** (4 LangGraph agents) that autonomously processes violations through detection → risk assessment → regulatory lookup → report generation.
3. A **real-time web dashboard** (Streamlit) showing live video feed with violation overlays, compliance scoring, and analytics.
4. An **automated PDF compliance report generator** that cites exact BOCW Act sections for each violation.
5. A **comprehensive evaluation report** with precision, recall, F1-score per class, and an ablation study comparing:
   - YOLO-only baseline (detect + simple alert)
   - YOLO + rule-based risk scoring
   - Full agentic pipeline (YOLO + LangGraph + RAG + LLM)

---

## 11. FUTURE SCOPE

| Phase | Enhancement | Description |
|---|---|---|
| Phase 2 | Harness & fall detection | Extend YOLO model to detect harness usage and fall incidents |
| Phase 3 | Multi-language voice alerts | Real-time spoken warnings in Hindi, Tamil, Bengali for on-site workers |
| Phase 4 | Drone integration | Mount cameras on drones for large-site aerial monitoring |
| Phase 5 | Predictive analytics | Use historical violation data to predict WHERE and WHEN the next violation will occur |
| Phase 6 | Mobile application | Android app for site supervisors to monitor remotely |
| Phase 7 | Insurance integration | API for insurance companies to offer lower premiums to compliant sites |
| Phase 8 | Government dashboard | Scalable deployment for Labour Department nationwide monitoring |

---

## 12. REFERENCES

1. Redmon, J., & Farhadi, A. (2018). "YOLOv3: An Incremental Improvement." *arXiv:1804.02767*.
2. Jocher, G., et al. (2023). "Ultralytics YOLOv8." *GitHub: ultralytics/ultralytics*.
3. Nath, N. D., Behzadan, A. H., & Paal, S. G. (2020). "Deep learning for site safety: Real-time detection of personal protective equipment." *Automation in Construction*, 112, 103085.
4. Delhi, V. S. K., et al. (2020). "Detection of Personal Protective Equipment (PPE) Compliance on Construction Sites Using Computer Vision." *IEEE Access*, 8, 181929–181940.
5. Shi, J., et al. (2023). "SHEL5K: An Extended Dataset for Safety Helmet Detection." *Sensors*, 23(4), 2074.
6. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
7. Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*.
8. Building and Other Construction Workers (Regulation of Employment and Conditions of Service) Act, 1996. Government of India. *legislative.gov.in*.
9. BOCW Central Rules, 1998. Ministry of Labour and Employment. *legislative.gov.in*.
10. IS 3696 (Parts 1–3): Safety Code for Scaffolds and Ladders. Bureau of Indian Standards.
11. National Crime Records Bureau (NCRB). "Accidental Deaths & Suicides in India — Annual Report." *ncrb.gov.in*.
12. International Labour Organization (ILO). (2023). "World Statistic on Occupational Safety and Health." *ilo.org*.
13. Wu, J., et al. (2021). "Real-time Hard Hat Detection using YOLOv5." *arXiv:2106.07325*.
14. Wojke, N., Bewley, A., & Paulus, D. (2017). "Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT)." *IEEE ICIP 2017*.

---

## 13. APPENDIX

### A. Glossary of Terms

| Term | Definition |
|---|---|
| **PPE** | Personal Protective Equipment (helmets, vests, harnesses, gloves) |
| **BOCW Act** | Building and Other Construction Workers Act, 1996 |
| **YOLOv8** | You Only Look Once, version 8 — a real-time object detection model |
| **LangGraph** | A framework for building stateful, multi-agent AI applications |
| **RAG** | Retrieval-Augmented Generation — a technique to ground LLM outputs in factual documents |
| **mAP** | Mean Average Precision — standard metric for object detection accuracy |
| **DeepSORT** | Deep Simple Online and Realtime Tracking — algorithm for multi-object tracking |
| **ChromaDB** | An open-source vector database for storing and searching embeddings |
| **ONNX** | Open Neural Network Exchange — a format for optimizing and deploying ML models |
| **MLflow** | An open-source platform for managing the ML lifecycle |

### B. Tools & Accounts Required

| Tool | Purpose | Account Type |
|---|---|---|
| Google Colab | GPU training | Free / Pro (₹900/mo) |
| Roboflow | Dataset management | Free tier |
| GitHub | Version control | Free |
| Gemini API | LLM inference | Free tier (1,500 req/day) |
| MLflow | Experiment tracking | Local (no account needed) |
| Streamlit Cloud | Deployment (optional) | Free tier |

---

*— End of Project Requirement Report —*
