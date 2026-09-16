from typing import List, Optional, TypedDict

class SafetyState(TypedDict):
    raw_detections: List[dict]
    frame_timestamp: str
    camera_id: str
    validated_violations: List[dict]
    risk_assessments: List[dict]
    regulations: List[dict]
    report: Optional[str]
    alert_message: Optional[str]
    compliance_score: Optional[float]
