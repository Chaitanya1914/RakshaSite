# Chaitanya — ML Lead & Integration Lead

## Your Role
You are the technical backbone of this project. You own:
- YOLOv8 model training and optimization
- All 4 LangGraph agents
- RAG pipeline (ChromaDB + embeddings)
- Integration of everyone's work
- Code review and merging PRs
- Final demo preparation

## Files You Own
- `src/detection/yolo_trainer.py`: Script to train YOLOv8 model.
- `src/detection/detector.py`: Inference script using the trained YOLOv8 model.
- `src/detection/tracker.py`: Script implementing DeepSORT for tracking workers.
- `src/detection/utils.py`: Utility functions for detection.
- `src/agents/pipeline.py`: Main LangGraph StateGraph connecting all agents.
- `src/agents/detection_agent.py`: Agent for handling raw detections and rules.
- `src/agents/risk_agent.py`: Agent for assessing risk severity using XGBoost.
- `src/agents/compliance_agent.py`: Agent for mapping violations to BOCW Act sections using RAG.
- `src/agents/report_agent.py`: Agent for summarizing findings into final reports using LLMs.
- `src/rag/embedder.py`: Script to generate embeddings for the BOCW Act sections and store them in ChromaDB.

## Your Task Breakdown (Week by Week)

### Week 1-2: Data & Model Training
- [ ] Clone the repo, set up virtual environment
- [ ] Download datasets (Arshi will help prepare data, but you merge and validate)
- [ ] Run the data cleaning/merge script
- [ ] Train YOLOv8n on your RTX 3060 locally
- [ ] Commands to run:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  yolo detect train data=data/unified/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 device=0
  ```
- [ ] Log experiments with MLflow
- [ ] Target: mAP@0.5 >= 85%
- [ ] If accuracy is low, check class distribution, add augmentation, try yolov8s.pt

### Week 3: Model Optimization & Tracking
- [ ] Export model to ONNX with INT8 quantization
- [ ] Integrate DeepSORT for worker tracking
- [ ] Test on YouTube demo video
- [ ] Validate that hallucination filters work (min area, person containment, confidence threshold)

### Week 4: RAG Pipeline
- [ ] Arshi should have BOCW Act sections ready by now
- [ ] Embed sections into ChromaDB using Sentence Transformers
- [ ] Test retrieval accuracy: query each violation type and verify correct section returned
- [ ] Set up the hybrid retrieval (RAG + fallback rules)
- [ ] Manual validation of every violation-to-section mapping

### Week 5-6: Agent Pipeline
- [ ] Build Detection Agent (src/agents/detection_agent.py)
- [ ] Build Risk Agent with XGBoost (src/agents/risk_agent.py)
- [ ] Build Compliance Agent with RAG (src/agents/compliance_agent.py)
- [ ] Build Report Agent with Ollama/Gemini (src/agents/report_agent.py)
- [ ] Wire everything in LangGraph StateGraph (src/agents/pipeline.py)
- [ ] Test end-to-end: video frame → agent chain → report
- [ ] Include LangGraph learning resources: official docs, tutorials

### Week 7-8: Integration
- [ ] Connect your agent pipeline to Sejal's FastAPI backend
- [ ] Make sure WebSocket streaming works for Ashmeet's frontend
- [ ] Help Sejal debug API issues
- [ ] Help Ashmeet connect frontend to backend

### Week 9-10: Testing & Ablation
- [ ] Run the full test suite (pytest)
- [ ] Run ablation study: YOLO-only vs YOLO+rules vs full agent pipeline
- [ ] Generate comparison table with precision, recall, F1, FPR, severity accuracy, section citation accuracy
- [ ] This ablation study is your KILLER slide for presentation

### Week 11-12: Demo & Documentation
- [ ] Download YouTube construction site video for demo
- [ ] Record screen capture of dashboard processing the video (OBS Studio)
- [ ] Keep live webcam + hard hat as backup
- [ ] Review Arshi's documentation and project report
- [ ] Prepare presentation slides
- [ ] Final integration testing

## Your Git Workflow
```bash
# Your branch
git checkout -b chaitanya/ml-pipeline

# Daily workflow
git add .
git commit -m "feat: train YOLOv8 - mAP 87%"
git push origin chaitanya/ml-pipeline

# Reviewing others' PRs
git fetch origin
git checkout sejal/backend-api
# Review code, then merge via GitHub PR interface

# Merging to main
git checkout main
git merge chaitanya/ml-pipeline
git push origin main
```

## Your Responsibilities as Team Lead
1. Review and merge everyone's Pull Requests
2. If someone is stuck for 2+ days, pair program with them
3. Make architecture decisions
4. Ensure the integration works end-to-end
5. Track progress via GitHub commit history

## Key Decisions You'll Make
- YOLOv8n vs YOLOv8s (if accuracy is low, switch to small)
- Gemini API vs Ollama (RTX 3060 can run Ollama easily)
- SQLite vs PostgreSQL (use SQLite for now, mention PostgreSQL as future scope)
- How to handle edge cases in agent pipeline

## Resources to Learn
- LangGraph: https://langchain-ai.github.io/langgraph/
- ChromaDB: https://docs.trychroma.com/
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- DeepSORT: https://github.com/levan92/deep_sort_realtime
