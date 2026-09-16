import logging
from typing import Any, Dict, List
from src.agents.state import SafetyState

logger = logging.getLogger(__name__)

FALLBACK_RULES = {
    "no_helmet": {
        "section_id": "Section 38(1)(a)",
        "quoted_text": "Every person engaged in any manufacturing process shall be provided with and shall wear suitable protective equipment including head protection.",
        "penalty": "Fine up to 5000 units"
    },
    "no_vest": {
        "section_id": "Section 38(1)(a)",
        "quoted_text": "High visibility clothing must be worn where vehicles or active machinery are in operation.",
        "penalty": "Fine up to 3000 units"
    }
}

class ComplianceChecker:
    def __init__(self):
        self.collection = None
        try:
            import chromadb
            # self.client = chromadb.Client()
            # self.collection = self.client.get_collection("safety_regulations")
        except ImportError:
            logger.warning("chromadb not found, will use fallback rules")
            
    def check(self, violation_type: str) -> Dict[str, Any]:
        confidence = 0.0
        if self.collection:
            # simulated RAG search
            # results = self.collection.query(query_texts=[violation_type], n_results=1)
            # if results and confidence >= 0.75:
            #     return result
            pass
            
        # Fallback
        if violation_type in FALLBACK_RULES:
            rule = FALLBACK_RULES[violation_type]
            return {
                "section_id": rule["section_id"],
                "quoted_text": rule["quoted_text"],
                "penalty": rule["penalty"],
                "confidence": 1.0,
                "source": "FALLBACK"
            }
        
        return {
            "section_id": "Unknown",
            "quoted_text": "No specific regulation found.",
            "penalty": "None",
            "confidence": 0.0,
            "source": "FALLBACK"
        }


def compliance_agent(state: SafetyState) -> SafetyState:
    logger.info("Running compliance agent")
    
    checker = ComplianceChecker()
    regulations = []
    
    for assessment in state.get("risk_assessments", []):
        reg = checker.check(assessment.get("violation_type", ""))
        regulations.append({**assessment, "regulation": reg})
        
    score = max(100.0 - (len(regulations) * 10), 0.0)
    
    return {**state, "regulations": regulations, "compliance_score": score}
