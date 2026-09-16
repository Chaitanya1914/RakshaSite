import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any
from .schemas import AnalyzeRequest, AnalyzeResponse, AnalyticsResponse, ViolationResponse

app = FastAPI(title="RakshaSite API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_frame(request: AnalyzeRequest):
    # Dummy implementation for pipeline processing
    # In reality, this would decode base64, run YOLO, pass through agent pipeline
    if not request.frame_b64:
        raise HTTPException(status_code=400, detail="Empty frame")
    
    violation = ViolationResponse(
        worker_id="w-001",
        violation_type="No Helmet",
        confidence=0.92,
        risk_score=8,
        risk_label="HIGH",
        bocw_section="Section 38",
        penalty="Rs. 500",
        bbox=[100, 150, 200, 300]
    )
    
    return AnalyzeResponse(
        violations=[violation],
        compliance_score=75.5,
        alert_messages=["Worker detected without helmet in elevated zone."]
    )

@app.get("/api/v1/report/{shift_id}")
async def get_report(shift_id: str):
    return {
        "shift_id": shift_id,
        "total_violations": 12,
        "compliance_rating": "Fair",
        "details": [
            {"type": "No Harness", "count": 2},
            {"type": "No Helmet", "count": 10}
        ]
    }

@app.get("/api/v1/analytics/{camera_id}", response_model=AnalyticsResponse)
async def get_analytics(camera_id: str):
    return AnalyticsResponse(
        compliance_score=82.0,
        violation_counts={"No Helmet": 10, "No Harness": 2},
        trend_data=[{"date": "2026-09-14", "score": 80.0}, {"date": "2026-09-15", "score": 82.0}],
        top_offenders=["w-001", "w-045"]
    )

@app.get("/api/v1/regulations")
async def get_regulations():
    return {
        "Section 38": "Requires hard hats/helmets for all workers in active zones.",
        "Section 39": "Requires safety harnesses for work at heights above 2 meters.",
        "Section 41": "Requires protective footwear to prevent injuries from falling objects."
    }

@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Expecting JSON string with frame data
            request_data = json.loads(data)
            # Process frame (simulated)
            await asyncio.sleep(0.1)
            response = {
                "annotated_frame": request_data.get("frame_b64", ""),
                "alerts": ["Alert: Missing helmet detected"],
                "timestamp": request_data.get("timestamp", "unknown")
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
