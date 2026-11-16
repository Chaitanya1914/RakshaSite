# Project: RakshaSite — Production & Demo Delivery

## Architecture
RakshaSite is an autonomous AI construction-safety and regulatory compliance auditing system.
- **Perception Layer**: YOLOv8n PPE detector (`models/best.pt`) detecting 10 classes, normalized to canonical PPE classes (`helmet`, `no_helmet`, `vest`, `no_vest`, `person`, `machinery`, `vehicle`).
- **Tracking Layer**: DeepSORT tracker with Kalman filtering and MobileNet embeddings, enforcing a 5-frame streak confirmation gate to eliminate transient false detections.
- **Multi-Agent Intelligence Layer (LangGraph StateGraph)**:
  1. *Detection Agent*: Spatial containment verification (PPE within person bounding box, area ratio ≥1%) and confidence filtering.
  2. *Risk Assessment Agent*: Non-linear risk scoring combining hazard proximity, zone multipliers (scaffolding, excavation, machinery), worker density, and an XGBoost model (`models/risk_xgboost.json`, $R^2=0.9998$).
  3. *Compliance Agent*: Hybrid statutory retrieval using persistent ChromaDB vector store (`./chroma_db`) indexing the Building and Other Construction Workers (BOCW) Act 1996 and Central Rules 1998, with statutory fallback rules.
  4. *Report Agent*: Anti-hallucination citation validation guard (`validate_section_citation`), structured markdown logging, and ReportLab PDF compilation (`reports/safety_audit.pdf`).
- **Presentation & API Layer**:
  - FastAPI backend (`backend/api.py`) exposing REST endpoints and WebSocket stream.
  - Lightweight Executive Demo Viewer (`GET /`, `GET /demo`) served directly by FastAPI without touching `frontend/`.
  - Documented REST & WebSocket API contract for Ashmeet's future React dashboard (`frontend/`).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Zero-Bloat Dataset Fetch | `scripts/fetch_data.py` + `scripts/manifest.json` with 3-tier resolution (checksum, local mirror, public upstream) | M1 | ORIGINAL_REQUEST §R1 |
| F2 | Fresh Clone Verification | Reproducibility harness cloning repo to temp folder and verifying fetch + demo | M1 | ORIGINAL_REQUEST §R1, §R4b |
| F3 | Real Validation Test Cases | `tests/test_cases.json` regenerated from real photographs; 151 pipeline tests pass | M2 | ORIGINAL_REQUEST §R4 |
| F4 | Honest PPE Metrics Evaluation | Evaluation of `models/best.pt` on real held-out set, logged in `models/real_val_metrics.json` | M2 | ORIGINAL_REQUEST §R3 |
| F5 | Robust One-Command Demo | `demo_runner.py` producing video, snapshot, PDF, and audit log with real violations | M3 | ORIGINAL_REQUEST §R2 |
| F6 | BOCW Statutory Citations | Verified citation mapping (helmet -> Sec 38/Rule 46, vest -> Sec 38/40/Rule 45) | M3 | ORIGINAL_REQUEST §R2 |
| F7 | Offline Fallback Guarantee | 100% offline pipeline execution without internet or LLM API keys | M3 | ORIGINAL_REQUEST §R2 |
| F8 | Backend Demo Viewer | Single-page executive dashboard served by FastAPI at `/` without touching `frontend/` | M3 | ORIGINAL_REQUEST §R2 |
| F9 | Frontend API Contract | Standardized API specification for Ashmeet's React dashboard | M3 | ORIGINAL_REQUEST §R2 |
| F10 | Real-Data Ablation Study | 3-tier benchmark on real images/video frames saved to `reports/ablation_study.json` | M4 | ORIGINAL_REQUEST §R4 |
| F11 | Arshi Documentation Deliverables | Formal Report updates, 10-slide deck, dataset docs, 3-minute viva demo script | M4 | ORIGINAL_REQUEST §R4, §R5 |
| F12 | Task List Closeout & Clean Commits | Close items in `CHAITANYA_TASKS.md` & `ARSHI_TASKS.md`; logical commits <25MB | M4 | ORIGINAL_REQUEST §R4, §R4b |
| F13 | Full Pytest Regression | ≥170 existing tests + new fetch/demo tests passing (100% pass rate) | M5 | ORIGINAL_REQUEST Acceptance |
| F14 | Comprehensive Status Report | Markdown report backed by real artifacts, demo script, and publish commands | M5 | ORIGINAL_REQUEST §R5, §R4b |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Dataset & Weights Distribution | `scripts/fetch_data.py`, `scripts/manifest.json`, mirror creation, fresh-clone test | none | DONE |
| M2 | Real-Image Test Cases & PPE Metrics | `tests/test_cases.json` real photos, `models/best.pt` sweeps (conf/iou/imgsz/TTA) & fine-tuning experiments, `models/real_val_metrics.json`, `models/detection_experiments.json` | M1 | DONE |
| M3 | One-Command Demo & Backend Viewer | `demo_runner.py`, `backend/api.py` viewer endpoints, `docs/FRONTEND_API_CONTRACT.md` | M1 | DONE |
| M4 | Real Ablation & Task Closeout | `reports/ablation_study.json`, Arshi docs (`team/`, `docs/`), task lists, clean git commits | M2, M3 | DONE |
| M5 | E2E Testing & Final Status Report | Automated E2E verification suite, `STATUS_REPORT.md`, publish commands, Sentinel handoff | M4 | DONE |

## Interface Contracts
### `scripts/fetch_data.py` CLI & Manifest
- Options: `--mirror <dir>`, `--create-mirror <dir>`, `--verify-only`
- Manifest schema:
  - `version`: semantic version
  - `assets`: map of `{ asset_name: { target_file / target_dir, expected_sha256, size_bytes, archive_name, upstream_url } }`
- Exit codes: `0` on verified success, non-zero on failure.

### Backend Demo Endpoints (`backend/api.py`)
- `GET /` and `GET /demo` -> HTML response (Executive Demo Viewer)
- `GET /api/v1/demo/results` -> JSON (`reports/demo_audit_log.json`)
- `GET /api/v1/demo/video` -> `video/mp4` (`reports/demo_annotated_output.mp4`)
- `GET /api/v1/demo/snapshot` -> `image/jpeg` (`reports/demo_snapshot_violation.jpg`)
- `GET /api/v1/demo/report/pdf` -> `application/pdf` (`reports/safety_audit.pdf`)
- `Mount /reports` -> `StaticFiles(directory="reports")`

### Real Test Cases Schema (`tests/test_cases.json`)
- Array of 50 test case objects:
  - `case_id`: `TC_001` .. `TC_050`
  - `image_file`: valid relative path to existing real image
  - `label_file`: valid relative path to existing real label
  - `is_violation`: boolean
  - `expected_risk_result`: object with `calculated_score` (0–100) and `severity_label` (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
  - `applicable_bocw_sections`: list of string citations (must contain `"Section 38"` for violations, empty for compliant)

### Real Ablation Study Schema (`reports/ablation_study.json`)
- Dictionary with keys: `yolo_only`, `yolo_plus_rules`, `full_agent_pipeline`
- Each containing: `precision`, `recall`, `f1_score`, `false_positive_rate`, `severity_accuracy`, `section_citation_accuracy`
- Backward compatible with `tests/test_agents.py` assertion keys.

## Code Layout
- `scripts/`: Distribution and setup utilities (`fetch_data.py`, `manifest.json`).
- `backend/`: FastAPI application (`api.py`, `schemas.py`, demo viewer templates/static).
- `src/`:
  - `src/detection/`: YOLOv8 detector, DeepSORT tracker, hallucination filters.
  - `src/agents/`: Detection, Risk (XGBoost), Compliance (BOCW RAG), Report agents and pipeline.
  - `src/rag/`: BOCW statutory parser, ChromaDB embedder, statutory knowledge base.
- `models/`: Model weights (`best.pt`, `best.onnx`, `risk_xgboost.json`, `real_val_metrics.json`, `training_plots/`).
- `data/`:
  - `data/raw/`: Demo video, sample photos, construction-ppe benchmark dataset.
  - `data/unified/`: Unified train/val labels.
- `reports/`: Demo outputs (`demo_annotated_output.mp4`, `demo_snapshot_violation.jpg`, `safety_audit.pdf`, `demo_audit_log.json`, `ablation_study.json`).
- `tests/`: Pytest suite (`test_pipeline.py`, `test_agents.py`, `test_detection.py`, `test_rag.py`, `test_cases.json`).
- `docs/`: Technical and academic documentation (`FRONTEND_API_CONTRACT.md`, `PRESENTATION_SLIDES.md`, `DATASET_DOCUMENTATION.md`, `DEMO_SCRIPT.md`).
- `team/`: Role guidelines, task trackers, formal academic project report.
- `frontend/`: Ashmeet's React dashboard directory (STRICTLY UNTOUCHED).

## Chaitanya & Arshi High-Quality Standards (User Directive)
- Sejal's (backend/database) and Ashmeet's (frontend) areas are strictly out of scope — touch them only where integration strictly requires.
- EVERY component owned by Chaitanya and Arshi must be demo-grade:
  1. Function-by-function audit: `src/detection/`, `src/agents/`, `src/rag/`, `knowledge_base/`, data tooling, `scratch/`, Arshi's docs/test cases for correct behavior, robust error handling (missing files, empty frames, no GPU, no network, no LLM key), docstrings, type hints, zero dead/placeholder/stub code.
  2. Public function unit test coverage: Edge cases included; target ≥85% line coverage for `src/detection`, `src/agents`, `src/rag` measured with `pytest-cov`.
  3. Professional demo output: Annotated video with labelled boxes, track IDs, risk colors, HUD; PDF with clean layout, BOCW citations, penalties, snapshots, summary stats.
  4. Arshi deliverables complete: Dataset datasheet/documentation, BOCW knowledge base, 50 real-image test cases, research docs, per-task evidence checklist in status report.
  5. Independent review scoring against this rubric before completion.
  6. Constraints: Zero remote side-effects, zero files > 25MB committed.

## CRITICAL GIT POLICY OVERRIDE (User Directive 2026-11-05T19:12:48Z)
- **ZERO GIT STATE MUTATION**: Do NOT run `git commit`, `git push`, `git merge`, `git rebase`, `git stash`, branch creation/checkout/reset, tags, or releases/uploads.
- All code, test, documentation, and configuration changes MUST remain as uncommitted working-tree modifications only.
- Read-only git commands (`git status`, `git diff`, `git log`, `git ls-files`) are permitted.
- Fresh-clone reproducibility checking must NOT rely on committing; instead, test harnesses copy working files (excluding gitignored models, data binaries, .venv) into a sandbox directory.
- Updated acceptance criterion: "No file > 25 MB among files that would be tracked (not gitignored) in the working tree".
- The final status report must provide a structured, logical commit plan (grouped file lists + commit messages) and exact commands for the user to review and run.
