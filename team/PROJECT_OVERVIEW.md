# RakshaSite — Complete Project Overview
## For All Team Members

## 1. What Are We Building?
An AI system that watches construction site video, detects workers without helmets/vests using YOLOv8, then passes data through 4 AI agents (LangGraph) that assess risk (XGBoost), find the violated Indian law (RAG over BOCW Act via ChromaDB), and auto-generate compliance reports (Gemini/Ollama LLM). Everything displayed on a React dashboard connected to a FastAPI backend.

## 2. Why Does This Matter?
- 38 construction worker deaths per day in India (NCRB)
- 80% preventable with PPE compliance
- Only 1,800 inspectors for 5 crore workers
- No affordable India-specific AI solution exists
- Competitors (Intenseye $50K/yr, viAct $30K/yr) are expensive and don't map to Indian law

## 3. System Architecture
```text
Video Input → YOLO Detection → Detection Agent → Risk Agent → Compliance Agent (RAG) → Report Agent (LLM) → Dashboard + Database
```

## 4. Tech Stack (Explain Each)
- **Python**: Core language used for backend, ML, and AI integration
- **YOLOv8**: Object detection model that finds people, helmets, and vests in video frames
- **OpenCV**: Library for video processing and frame extraction
- **DeepSORT**: Tracks individual workers across multiple frames to avoid duplicate counts
- **LangGraph**: Orchestrates the workflow of the 4 AI agents
- **XGBoost**: Machine learning model that calculates a risk severity score based on multiple factors
- **ChromaDB**: Vector database storing BOCW Act sections for quick semantic search
- **Sentence Transformers**: Converts text into mathematical vectors for the semantic search in RAG
- **Gemini/Ollama**: Large Language Model (LLM) that writes natural language compliance reports
- **FastAPI**: Python web framework for serving REST API endpoints to the frontend
- **React + Vite**: Frontend web framework and bundler for the dashboard
- **Tailwind CSS**: Utility-first CSS framework for styling the dashboard components
- **Recharts**: Library for adding interactive charts on the dashboard
- **SQLite/PostgreSQL**: Relational database for storing violation records and history
- **SQLAlchemy**: Python Object-Relational Mapper (ORM) to interact with the database
- **MLflow**: Tool for tracking machine learning experiments and model versions
- **ONNX Runtime**: Optimizes the YOLO model for faster inference speeds
- **ReportLab**: Library for generating PDF compliance reports from data
- **pytest**: Framework for writing and running automated tests

## 5. Project Directory Structure
```
rakshasite/
├── README.md (project overview and setup instructions)
├── main.py (entry point to run the application)
├── requirements.txt (list of Python dependencies)
├── .env.example (template for environment variables)
├── .gitignore (files to ignore in git)
├── configs/config.yaml (configuration settings for the project)
├── src/
│   ├── detection/ (YOLO + tracking code - Chaitanya)
│   ├── agents/ (4 AI agents logic - Chaitanya)
│   ├── rag/ (BOCW Act search logic - Chaitanya + Arshi)
│   ├── database/ (violation storage operations - Sejal)
│   └── utils/ (logging and helper functions)
├── backend/ (FastAPI API server - Sejal)
├── frontend/ (React dashboard application - Ashmeet)
├── data/ (training datasets and sample videos - Arshi)
├── models/ (trained YOLO model weights - Chaitanya)
├── knowledge_base/ (BOCW Act PDFs and docs - Arshi)
├── reports/ (generated PDF reports)
├── notebooks/ (Jupyter notebooks for model training/experimentation)
├── tests/ (test suite for the codebase)
└── team/ (this folder - team documentation)
```

## 6. Team Roles Summary
| Member | Role | What They Own | Skills They'll Learn |
|---|---|---|---|
| Chaitanya | ML Lead + Integration | Owns detection, agents, RAG | Already skilled in these areas |
| Sejal | Backend Developer | Owns database, FastAPI API | Learning Python backend development |
| Ashmeet | Frontend Developer | Owns React dashboard | Learning React frontend development |
| Arshi | Data & Research Lead | Owns datasets, BOCW knowledge base, docs | Learning data preparation and research |

## 7. How the 4-Agent Pipeline Works (Explain Like I'm a 2nd Year Student)
- **Detection Agent**: Takes raw YOLO bounding boxes as input. It filters out false positives by ensuring a violation is seen in 5 consecutive frames before outputting a validated violation.
- **Risk Agent**: Takes the validated violation. Runs it through an XGBoost model to calculate the severity, and outputs a risk score (0-100) and a label (LOW/MEDIUM/HIGH/CRITICAL).
- **Compliance Agent**: Takes the violation type. Searches ChromaDB for the matching section in the BOCW Act, and outputs the exact law (e.g., Section 38(1)(a)) along with penalty text.
- **Report Agent**: Takes all the gathered data (violation, risk, law). Uses an LLM to generate a natural language incident report, and outputs an alert message and a PDF report.

## 8. Data We're Using
| Dataset | Source | Images | What It Contains | Download |
|---|---|---|---|---|
| Hard Hat Workers | Kaggle | 5000+ | helmet, head, person | [kaggle.com](https://www.kaggle.com/) |
| Safety Helmet | Kaggle | 5000 | helmet, no_helmet | [kaggle.com](https://www.kaggle.com/) |
| Pictor-v3 | GitHub | 1330 | person, helmet, vest | [github.com](https://github.com/pictor-v3) |
| Construction Safety | Roboflow | varies | PPE classes | [roboflow.com](https://roboflow.com/) |
| BOCW Act 1996 | legislative.gov.in | PDF | Indian safety law | [legislative.gov.in](https://legislative.gov.in/) |

## 9. Demo Strategy
- Use a YouTube construction site video as input (not live webcam) to ensure consistent results.
- Record a screen showing the dashboard processing the video in real-time.
- Keep a live webcam and a hard hat as a backup in case the professor asks for a live, interactive demo.

## 10. Timeline (3 Months)
*(Note: A detailed week-by-week table will be planned and added here based on team availability and academic schedule.)*

## 11. GitHub Workflow
- **Repo structure**: We have one `main` branch for production-ready code, and 4 personal branches (`feat-chaitanya`, `feat-sejal`, `feat-ashmeet`, `feat-arshi`) for individual work.
- **Daily push rule**: Everyone must push their code to their branch at the end of every work day, even if it's incomplete.
- **Pull Request process**: When a feature is done, create a Pull Request (PR) to merge your branch into `main`.
- **Review**: Chaitanya will review all PRs and merge them into `main`.
- **Useful Git Commands**:
  - `git clone <repo_url>` (Download project)
  - `git checkout -b <branch_name>` (Create your branch)
  - `git add .` (Stage all changes)
  - `git commit -m "Your message"` (Save changes locally)
  - `git push origin <branch_name>` (Upload to GitHub)

## 12. Communication
- **WhatsApp group**: For daily quick updates, questions, and coordination.
- **Weekly 15-min standup**: Every Sunday.
- **Standup questions**: Each person answers: 
  1. What did I finish this week? 
  2. What's next for next week? 
  3. Am I stuck on anything?

## 13. Grading Criteria (What Professors Look For)
- **Problem authenticity**: Solving a real-world problem (construction deaths).
- **Technical depth**: Incorporating 14 trending skills (AI, LLMs, Vector DBs, React, etc.).
- **Working demo**: A functional application that can be shown live or via video.
- **Documentation quality**: Clear, well-written code and project documents.
- **Individual contribution**: GitHub commit history proves everyone did their part.

## 14. USP (Why We're Different)
- The only system combining Computer Vision + multi-agent reasoning + Indian law mapping.
- 25-42x cheaper than enterprise competitors like Intenseye or viAct.
- Packs 14 highly sought-after job-market skills into one comprehensive project.
- The first India-specific construction safety AI tailored for local laws and conditions.
