from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class AnalyzeRequest(BaseModel):
    frame_b64: str = Field(..., description="Base64 encoded image frame")
    camera_id: str = Field(..., description="ID of the camera")
    timestamp: str = Field(..., description="Timestamp of the frame")

class ViolationResponse(BaseModel):
    worker_id: Optional[str] = None
    violation_type: str = Field(..., description="Type of violation")
    confidence: float = Field(..., description="Confidence score")
    risk_score: int = Field(..., description="Risk score out of 10")
    risk_label: str = Field(..., description="Risk label (e.g., HIGH, CRITICAL)")
    bocw_section: str = Field(..., description="Relevant BOCW section")
    penalty: str = Field(..., description="Applicable penalty")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")

class AnalyzeResponse(BaseModel):
    violations: List[ViolationResponse] = Field(default_factory=list)
    compliance_score: float = Field(..., description="Overall compliance score")
    alert_messages: List[str] = Field(default_factory=list)

class AnalyticsResponse(BaseModel):
    compliance_score: float = Field(..., description="Overall compliance score")
    violation_counts: Dict[str, int] = Field(..., description="Counts per violation type")
    trend_data: List[Dict[str, Any]] = Field(..., description="Trend data over time")
    top_offenders: List[str] = Field(..., description="List of top offenders")
