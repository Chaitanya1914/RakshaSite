# EVALUATION PROMPT FOR FABLE / ANY AI REVIEWER

> **Copy everything below this line and paste into Fable (or any AI model) for an independent evaluation.**
>
> "You are a senior AI/ML hackathon judge and B.Tech project evaluator at a top Indian engineering university. You have evaluated 500+ student projects and judged at SIH, Flipkart GRiD, and NASSCOM hackathons. Rate the following PRD on a scale of 1-10 across these dimensions: (1) Problem Authenticity — is this a real, urgent problem? (2) Technical Depth — does this go beyond a basic implementation? (3) Novelty/USP — how is this different from existing solutions? (4) Feasibility — can a 2nd-year B.Tech CSE student build this in 3 months? (5) Trending Skills Coverage — does this cover skills that matter in the 2026 job market? (6) Demo Impact — will the live demo impress judges/professors? (7) Future Scope & Commercial Viability — is this a dead-end project or a launchpad? Give a final weighted score out of 10 and list the top 3 strengths and top 3 weaknesses."

---

# RakshaSite — Autonomous Multi-Agent Construction Safety Compliance System
## Product Requirements Document v1.0

---

## 1. EXECUTIVE SUMMARY

**RakshaSite** is a real-time, AI-powered construction site safety monitoring system that combines Computer Vision, Multi-Agent AI, Generative AI, and Retrieval-Augmented Generation to autonomously detect PPE (Personal Protective Equipment) violations, assess risk severity, map each violation to the exact section of India's BOCW Act (1996), and generate natural-language compliance reports — all without human intervention.

It is not a helmet detector. It is an **autonomous regulatory compliance auditor for the physical world.**

---

## 2. PROBLEM STATEMENT

### 2.1 The Human Cost
- India records **38 construction worker deaths PER DAY** (NCRB data).
- Construction accounts for **24% of all occupational fatalities** in India.
- 80% of these deaths are attributable to falls, struck-by incidents, and PPE non-compliance — all PREVENTABLE.

### 2.2 The Enforcement Gap
- The Building and Other Construction Workers (BOCW) Act, 1996 mandates safety compliance.
- Violations carry penalties of up to ₹2,000/day fine and 3 months imprisonment.
- India has approximately **1,800 labour inspectors** to monitor **5 crore+ construction workers**.
- That's 1 inspector per 27,000 workers — enforcement is physically impossible without automation.

### 2.3 The Technology Gap
- Existing solutions (Intenseye, viAct) cost **$30,000-$50,000/year** — unaffordable for 99% of Indian construction companies.
- They detect PPE violations but do NOT map to Indian law (BOCW Act).
- They use traditional ML pipelines — no agentic reasoning, no autonomous report generation.
- **No Indian-market solution exists at an accessible price point.**

### 2.4 Target Users
- **Primary:** Construction site supervisors and safety officers at Indian infrastructure companies.
- **Secondary:** Labour department inspectors who need scalable monitoring tools.
- **Tertiary:** Insurance companies seeking data-driven risk assessment for construction sites.

---

## 3. COMPETITIVE ANALYSIS

| Product | What It Does | Price | Weakness | RakshaSite Advantage |
|---|---|---|---|---|
| **Intenseye** (USA) | CV-based workplace safety detection | $50,000/yr | No Indian law mapping. US OSHA focused. No agentic reasoning. | Maps to BOCW Act. Multi-agent reasoning. 100x cheaper. |
| **viAct** (Hong Kong) | Construction site CV monitoring | $30,000/yr | Traditional pipeline (detect → alert). No regulatory intelligence. | Autonomous agent chain: detect → reason → cite law → generate report. |
| **SafetyCulture (iAuditor)** | Digital safety checklists | $19/user/mo | Manual — humans still do the inspection. Zero AI detection. | Fully autonomous. No human in the detection loop. |
| **Indian market** | **No competitor exists** | — | — | First-mover advantage in India-specific construction safety AI. |

### Moat
No existing tool combines real-time CV detection with autonomous multi-agent reasoning that maps violations to a specific national law's section numbers and generates natural-language compliance reports. This is Spashta AI's architecture (dark pattern → CCPA law mapping) applied to physical safety.

---

## 4. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO INPUT LAYER                         │
│  IP Camera / Webcam / Uploaded Video → Frame Extraction      │
│  (OpenCV VideoCapture, 15-30 FPS)                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              PERCEPTION LAYER (Deep Learning + CV)           │
│                                                              │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │ YOLOv8n (Quantized) │    │ OpenCV Post-Processing      │ │
│  │ • Person detection   │    │ • Bounding box refinement   │ │
│  │ • Helmet detection   │───▶│ • Spatial zone mapping      │ │
│  │ • Vest detection     │    │ • Worker tracking (DeepSORT)│ │
│  │ • Harness detection  │    │ • Proximity calculation     │ │
│  └─────────────────────┘    └──────────────┬──────────────┘ │
└─────────────────────────────────────────────┼───────────────┘
                                              │
┌─────────────────────────────────────────────▼───────────────┐
│           INTELLIGENCE LAYER (Multi-Agent System)            │
│                      LangGraph Orchestration                 │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ DETECTION    │  │ RISK         │  │ COMPLIANCE       │  │
│  │ AGENT        │  │ ASSESSMENT   │  │ AGENT            │  │
│  │              │  │ AGENT        │  │                  │  │
│  │ Validates CV │  │ XGBoost      │  │ RAG lookup       │  │
│  │ outputs,     │──▶ risk model,  │──▶ against BOCW Act │  │
│  │ filters FPs, │  │ proximity +  │  │ + IS 3696.       │  │
│  │ tracks IDs   │  │ severity     │  │ Cites exact      │  │
│  │              │  │ scoring      │  │ section number.  │  │
│  └──────────────┘  └──────────────┘  └────────┬─────────┘  │
│                                                │             │
│                                    ┌───────────▼─────────┐  │
│                                    │ REPORT AGENT        │  │
│                                    │ (Generative AI)     │  │
│                                    │                     │  │
│                                    │ LLM generates       │  │
│                                    │ natural-language     │  │
│                                    │ incident report +   │  │
│                                    │ shift-end summary   │  │
│                                    └─────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   DATA & PERSISTENCE LAYER                   │
│                                                              │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ PostgreSQL      │  │ ChromaDB     │  │ MLflow         │ │
│  │ • Violation logs │  │ • BOCW Act   │  │ • Model        │ │
│  │ • Worker profiles│  │   embeddings │  │   versioning   │ │
│  │ • Site metadata  │  │ • IS 3696    │  │ • Experiment   │ │
│  │ • Analytics      │  │   standards  │  │   tracking     │ │
│  └─────────────────┘  └──────────────┘  └────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   PRESENTATION LAYER                         │
│                                                              │
│  Streamlit Dashboard:                                        │
│  • Live video feed with bounding boxes                       │
│  • Real-time violation alerts with BOCW Act citations        │
│  • Compliance score gauge (0-100)                            │
│  • Analytics: violation trends, heatmaps, worker stats       │
│  • PDF report download                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. COMPLETE TECH STACK — 14 TRENDING SKILLS MAPPED

| # | 2026 Job Market Skill | Technology Used | Where In System |
|---|---|---|---|
| 1 | **Computer Vision** | YOLOv8, OpenCV, DeepSORT | Perception Layer — detect workers, PPE, track across frames |
| 2 | **Deep Learning (CNN)** | YOLOv8 (CSPDarknet backbone) | Object detection model architecture |
| 3 | **Classical Machine Learning** | XGBoost / Random Forest | Risk Assessment Agent — severity scoring from features |
| 4 | **Generative AI** | Gemini Flash / Llama 3 via Ollama | Report Agent — generates natural-language incident narratives and compliance summaries |
| 5 | **Agentic AI** | LangGraph | Autonomous multi-step reasoning pipeline: detect → assess → cite → report |
| 6 | **Multi-Agent Systems** | LangGraph StateGraph | 4 specialized agents with defined roles, tools, and handoffs |
| 7 | **RAG (Retrieval-Augmented Generation)** | LangChain + ChromaDB | Compliance Agent retrieves relevant BOCW Act sections before generating citations |
| 8 | **NLP / Transformers** | Sentence Transformers (all-MiniLM-L6-v2) + spaCy NER | Embedding legal text + extracting section numbers from regulations |
| 9 | **Vector Databases** | ChromaDB | Stores BOCW Act and IS 3696 standard embeddings for semantic search |
| 10 | **Prompt Engineering** | Custom system prompts per agent | Each agent has a tunable, versioned system prompt defining its role and output schema |
| 11 | **Edge AI / Model Optimization** | ONNX Runtime, INT8 quantization | YOLOv8 exported to ONNX INT8 for edge deployment (Jetson Orin Nano) |
| 12 | **MLOps** | MLflow | Experiment tracking, model versioning, hyperparameter logging during YOLO training |
| 13 | **Data Engineering** | Custom ETL pipeline | Camera feed → frame extraction → batch inference → structured DB writes → analytics |
| 14 | **Full-Stack Development** | Streamlit + FastAPI + PostgreSQL | End-to-end web application with real-time streaming, REST APIs, and persistent storage |

---

## 6. AGENT SPECIFICATIONS

### Agent 1: Detection Agent
- **Role:** Validate raw YOLOv8 outputs, filter false positives, assign worker tracking IDs via DeepSORT.
- **Input:** Raw YOLO bounding boxes + confidence scores + frame metadata.
- **Logic:** If confidence < 0.6, discard. If same worker detected across 5+ consecutive frames with consistent violation, confirm as true positive.
- **Output JSON:**
```json
{
  "worker_id": "W-003",
  "frame_timestamp": "2026-09-15T14:23:07",
  "violations": ["NO_HELMET"],
  "confidence": 0.89,
  "bbox": [120, 45, 280, 310],
  "zone": "elevated_platform"
}
```

### Agent 2: Risk Assessment Agent
- **Role:** Calculate a severity score (0-100) based on violation type, worker location, proximity to hazards, and historical patterns.
- **Model:** XGBoost classifier trained on synthetic risk data.
- **Features:** `violation_type`, `zone_danger_level`, `worker_proximity_to_edge`, `time_of_day`, `consecutive_violation_count`, `weather_conditions`.
- **Output:** Risk score (0-100) + severity label (LOW / MEDIUM / HIGH / CRITICAL).

### Agent 3: Compliance Agent (RAG-Powered)
- **Role:** Map each validated violation to the specific BOCW Act section using semantic search over embedded regulations.
- **RAG Pipeline:**
  1. Convert violation description to embedding using Sentence Transformer.
  2. Query ChromaDB for top-3 most relevant BOCW Act sections.
  3. Return exact section number, quoted text, and applicable penalty.
- **Knowledge Base Contents:**
  - BOCW Act 1996 (full text, chunked by section)
  - IS 3696: Safety Code for Scaffolds (Parts 1-3)
  - IS 7069: Safety Code for Handling and Storage of Building Materials
  - BOCW Central Rules 1998
- **Output:**
```json
{
  "regulation": "BOCW Act 1996",
  "section": "Section 38(1)(a)",
  "quoted_text": "Every employer shall provide and maintain, for the safety of building workers, such protective equipment as may be necessary",
  "penalty": "Fine up to ₹2,000 and/or imprisonment up to 3 months",
  "relevance_score": 0.94
}
```

### Agent 4: Report Agent (Generative AI)
- **Role:** Consume outputs from all three upstream agents and generate a human-readable, natural-language compliance report.
- **LLM:** Gemini 1.5 Flash (API) or Llama 3 8B (local via Ollama).
- **Output Types:**
  1. **Instant Alert:** One-line violation summary displayed on dashboard.
  2. **Incident Report:** Detailed PDF with screenshot, violation details, regulation citation, recommended action.
  3. **Shift-End Summary:** Aggregated report covering all violations in the past 8 hours, compliance score, trend analysis.
- **Prompt Template (Versioned, Tunable):**
```
You are a construction safety compliance officer. Based on the following
detection data and regulatory match, generate a professional incident report.

Detection: {detection_agent_output}
Risk Assessment: {risk_agent_output}
Regulatory Match: {compliance_agent_output}

Generate:
1. A one-paragraph incident narrative
2. The exact regulatory citation
3. A recommended immediate action
4. A severity classification

Use formal, professional language suitable for a legal compliance document.
```

---

## 7. DATA PIPELINE

### 7.1 Training Data (Pre-Existing, Free)

| Dataset | Images | Labels | Format | Source |
|---|---|---|---|---|
| SHEL5K | 5,000 | Helmet / No Helmet | YOLO | Academic, free download |
| Hard Hat Workers | 5,000+ | Helmet, Head, Person | YOLO-ready | Kaggle |
| Pictor-v3 | 1,330 | Helmet, Vest, Worker | COCO/YOLO | Academic, free |
| Roboflow Construction Safety | Multiple sets | Pre-annotated | YOLOv8 export | Roboflow free tier |
| **Total available** | **12,000+** | **Multi-class** | **Ready to train** | — |

### 7.2 YOLOv8 Training Configuration
```yaml
model: yolov8n.pt  # Nano variant for edge deployment
epochs: 100
imgsz: 640
batch: 16
device: 0  # Google Colab T4 GPU
optimizer: AdamW
lr0: 0.001
augment: true  # Mosaic, MixUp, HSV shifts, perspective warp
```

### 7.3 Model Optimization for Edge
```python
# Export to ONNX with INT8 quantization
model.export(format="onnx", int8=True, dynamic=True)
# Achieves ~25 FPS on Jetson Orin Nano vs ~15 FPS without quantization
```

### 7.4 Regulatory Knowledge Base Preparation
```python
# Chunk BOCW Act PDF by section
# Embed each chunk using Sentence Transformers
from sentence_transformers import SentenceTransformer
from chromadb import Client

model = SentenceTransformer('all-MiniLM-L6-v2')
chunks = parse_bocw_act_pdf("bocw_act_1996.pdf")  # ~85 sections
embeddings = model.encode(chunks)
collection.add(documents=chunks, embeddings=embeddings)
```

---

## 8. FRONTEND SPECIFICATIONS

### 8.1 Live Monitoring View
- Split-screen: live video feed (left) + agent reasoning chain (right).
- Bounding boxes: Green = compliant worker, Red = violation detected.
- Badge counter showing active violations count.

### 8.2 Violation Detail Panel
- Per-violation: violation type, confidence score, BOCW Act section, penalty quote, screenshot crop, recommended action.

### 8.3 Compliance Score Gauge
- Real-time 0-100 score for the current shift.
- Formula: `100 - (weighted_violation_count / total_workers_detected × 100)`
- Color-coded: Green (80-100), Yellow (50-79), Red (0-49).

### 8.4 Analytics Dashboard
- Violation frequency by type (bar chart)
- Violation heatmap by camera zone (spatial overlay)
- Compliance score trend over days/weeks (line chart)
- Repeat offender list (table with worker IDs and violation counts)

---

## 9. API CONTRACTS

```
POST /api/v1/analyze
Body: { frame_b64, camera_id, site_id, timestamp }
→ { violations: [{ type, confidence, bbox, risk_score, bocw_section, penalty }] }

GET /api/v1/report/{shift_id}
→ Full shift compliance report (JSON or PDF)

GET /api/v1/analytics/{site_id}
→ { compliance_score, violation_counts, trend_data, top_offenders }

GET /api/v1/regulations
→ List of all BOCW Act sections with paraphrased descriptions
```

---

## 10. MVP SCOPE (3-Month Semester Build)

### IN:
- YOLOv8 model trained on helmet + vest detection (2 violation types)
- Detection Agent + Risk Assessment Agent + Compliance Agent + Report Agent (all 4 agents)
- RAG over BOCW Act (full text embedded)
- Streamlit dashboard with live webcam feed
- Real-time violation alerts with regulation citations
- Shift-end PDF report generation
- PostgreSQL violation logging
- MLflow experiment tracking
- Demo on live webcam with physical hard hat / vest

### OUT (Future Scope):
- Harness detection (requires specialized dataset)
- Drone-based monitoring
- Mobile app
- Multi-site management
- Insurance API integration
- Predictive violation forecasting
- Multi-language voice alerts (Hindi, Tamil, Bengali)
- RERA compliance integration for real estate

---

## 11. SUCCESS METRICS

| Metric | Target |
|---|---|
| YOLOv8 mAP@0.5 on test set | ≥ 85% |
| False positive rate | < 10% |
| End-to-end latency (frame → alert) | < 3 seconds |
| BOCW section citation accuracy | ≥ 90% (manual validation on 50 test cases) |
| System uptime during 30-min demo | 100% |

---

## 12. RISK MATRIX

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| YOLO model doesn't generalize to webcam (trained on dataset images) | Medium | High | Heavy augmentation (brightness, blur, perspective). Test on webcam EARLY in Week 2. |
| LangGraph agent chain is too slow (>5s latency) | Medium | Medium | Use Gemini Flash (fastest) or local Ollama. Cache frequent regulation lookups. |
| False positives on webcam (detects "no helmet" on a person with a hat) | Medium | High | Confidence threshold ≥ 0.7. Detection Agent validates across 5+ consecutive frames before confirming. |
| ChromaDB returns wrong BOCW section | Low | Medium | Manual validation of top-20 most common queries. Curate regulation chunks carefully. |
| Professor asks "how is this different from just YOLO + if-else?" | High | High | Demo the full agent chain: show the LangGraph state graph, the RAG retrieval, the generated report. The system REASONS, it doesn't just classify. |

---

## 13. DEVELOPMENT TIMELINE (3 Months)

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Dataset download, merge, clean. Setup MLflow. | Unified YOLOv8-format dataset ready |
| 2 | Train YOLOv8 on Colab. Test on webcam. | Trained model with mAP ≥ 85% |
| 3 | ONNX export + quantization. DeepSORT integration. | Edge-ready model, worker tracking working |
| 4 | Embed BOCW Act into ChromaDB. Build RAG pipeline. | Compliance Agent returning correct sections |
| 5 | Build XGBoost risk scoring model. | Risk Assessment Agent functional |
| 6 | Build LangGraph agent chain. Connect all 4 agents. | End-to-end pipeline working |
| 7 | Report Agent: LLM-powered report generation. | PDF reports generating correctly |
| 8 | Streamlit dashboard: live feed, alerts, analytics. | Full UI functional |
| 9 | PostgreSQL integration. API endpoints. | Data persistence + REST API |
| 10 | Integration testing. Held-out scenario testing. | Bug-free end-to-end system |
| 11 | Documentation, ablation study, evaluation chapter. | Academic report ready |
| 12 | Final demo prep, presentation slides. | Presentation-ready |

---

## 14. TEAM COMPOSITION (Suggested)

| Role | Responsibility |
|---|---|
| **ML Engineer (You)** | YOLOv8 training, ONNX optimization, XGBoost risk model, MLflow |
| **Backend Engineer** | FastAPI, PostgreSQL, ChromaDB, LangGraph agent chain |
| **Frontend Engineer** | Streamlit dashboard, real-time video streaming, PDF generation |
| **Research Lead** | BOCW Act analysis, dataset curation, evaluation metrics, documentation |

*For a solo project: prioritize ML + Backend in months 1-2, Frontend + Docs in month 3.*

---

## 15. ETHICAL CONSIDERATIONS

1. **No worker identification.** The system tracks "Worker #3" by bounding box, not by face. No facial recognition, no biometric data, no privacy violation.
2. **Framing:** Every output says "matches BOCW Act guideline" — never "this site is breaking the law." Informational, not prosecutorial.
3. **No punitive worker tracking.** The system monitors COMPLIANCE, not individual workers. Repeat offender data is for safety training purposes, not termination decisions.
4. **Consent:** In real deployment, workers and site management must be informed that AI monitoring is active. Signage at site entrance.

---

## 16. COST ANALYSIS

### Development Phase
| Item | Cost |
|---|---|
| Google Colab Pro (2 months) | ₹1,800 |
| Demo props (hard hat + vest) | ₹350 |
| Gemini API (if needed) | ₹500-800 |
| Miscellaneous | ₹500 |
| **Total** | **₹3,000 - ₹3,500** |

### Real-World Deployment (1 Site)
| Item | Cost |
|---|---|
| One-time setup (cameras, edge device, installation) | ₹35,000 - ₹45,000 |
| Monthly running (cloud, API, internet) | ₹4,000 - ₹5,000/month |
| Annual cost | ~₹85,000 - ₹1,00,000/year |

### Competitor Pricing
| Competitor | Annual Cost |
|---|---|
| Intenseye | ₹42,00,000/year ($50K) |
| viAct | ₹25,00,000/year ($30K) |
| **RakshaSite** | **₹1,00,000/year** |

**RakshaSite is 25-42x cheaper than any global competitor.**

---

*End of PRD v1.0*
