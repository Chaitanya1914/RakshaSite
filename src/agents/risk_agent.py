import logging
from typing import Any, Dict, List
from src.agents.state import SafetyState

logger = logging.getLogger(__name__)

class RiskScorer:
    def __init__(self):
        self.model = None
        try:
            import xgboost as xgb
            # self.model = xgb.Booster()
            # self.model.load_model("path/to/model.json")
            # For this scaffolding, we fallback immediately if no model
        except ImportError:
            logger.warning("xgboost not found, falling back to rule-based scoring")
            
    def _rule_based_score(self, violation: Dict[str, Any]) -> float:
        score = 0.0
        
        violation_type = violation.get("violation_type", "")
        if violation_type == "no_helmet":
            score += 40
        elif violation_type == "no_vest":
            score += 30
            
        zone = violation.get("zone", "general")
        if zone == "restricted":
            score += 40
        elif zone == "active_machinery":
            score += 50
            
        return min(score, 100.0)
        
    def score(self, violation: Dict[str, Any]) -> Dict[str, Any]:
        if self.model:
            # dummy xgboost score
            score_val = 50.0
        else:
            score_val = self._rule_based_score(violation)
            
        label = "LOW"
        if score_val >= 80:
            label = "CRITICAL"
        elif score_val >= 60:
            label = "HIGH"
        elif score_val >= 40:
            label = "MEDIUM"
            
        return {
            "score": score_val,
            "label": label
        }

def risk_agent(state: SafetyState) -> SafetyState:
    logger.info(f"Running risk agent on {len(state.get('validated_violations', []))} violations")
    
    scorer = RiskScorer()
    risk_assessments = []
    
    for violation in state.get("validated_violations", []):
        risk = scorer.score(violation)
        
        assessment = {**violation, "risk_score": risk["score"], "risk_label": risk["label"]}
        risk_assessments.append(assessment)
        
    return {**state, "risk_assessments": risk_assessments}
