import logging
from typing import Any, Dict
from src.agents.state import SafetyState

logger = logging.getLogger(__name__)

def detection_agent(state: SafetyState) -> SafetyState:
    logger.info(f"Running detection agent on {len(state.get('raw_detections', []))} detections")
    
    validated_violations = []
    
    for det in state.get("raw_detections", []):
        confidence = det.get("confidence", 0.0)
        class_name = det.get("class_name", "")
        
        logger.debug(f"Evaluating detection: {class_name} with confidence {confidence}")
        
        if confidence >= 0.7 and class_name in ["no_helmet", "no_vest"]:
            logger.info(f"Validated violation found: {class_name}")
            validated_violations.append({
                "worker_id": det.get("worker_id", "unknown"),
                "violation_type": class_name,
                "confidence": confidence,
                "bbox": det.get("bbox", []),
                "zone": det.get("zone", "general")
            })
            
    return {**state, "validated_violations": validated_violations}
