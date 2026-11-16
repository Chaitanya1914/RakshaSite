import logging
from typing import Any, Dict, List, Optional
from src.agents.state import SafetyState

logger = logging.getLogger(__name__)

FALLBACK_RULES: Dict[str, Dict[str, str]] = {
    "no_helmet": {
        "section_id": "Section 38(1)(a) & BOCW Rule 46 (IS 2925:1984)",
        "quoted_text": (
            "Every building worker exposed to falling objects or head injury hazards shall be provided with "
            "and shall wear an industrial safety helmet conforming to IS 2925:1984."
        ),
        "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.",
    },
    "no_vest": {
        "section_id": "Section 38(1)(a), Section 40(2)(r) & BOCW Rule 45",
        "quoted_text": (
            "High-visibility reflective safety vests must be worn by all workers on active construction sites "
            "and where vehicles or active machinery are in operation."
        ),
        "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.",
    },
    "no_harness": {
        "section_id": "Section 39, Section 40(2)(o) & BOCW Rules 41, 175–185 (IS 3521 / IS 3696)",
        "quoted_text": (
            "Full-body fall-arrest safety harnesses anchored to lifelines (IS 3521) and guarded scaffolding "
            "(IS 3696) are mandatory for work at heights exceeding 2 meters."
        ),
        "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months (up to 6 months for repeat offence).",
    },
    "no_gloves": {
        "section_id": "Section 40(2)(r) & BOCW Rule 49 (Hand Protection)",
        "quoted_text": (
            "Suitable protective industrial gloves shall be provided to workers handling sharp materials, "
            "reinforcement bars, concreting, or electrical equipment."
        ),
        "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.",
    },
    "no_boots": {
        "section_id": "Section 40(2)(r) & BOCW Rule 49 (Foot Protection - IS 15298)",
        "quoted_text": (
            "Protective steel-toe safety footwear conforming to IS 15298 shall be worn by workers to prevent "
            "crushing and puncture injuries."
        ),
        "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.",
    },
}


class ComplianceChecker:
    _shared_embedder = None

    def __init__(self, use_rag: bool = True, persist_directory: str = "./chroma_db"):
        self.use_rag = use_rag
        self.embedder = None
        if use_rag:
            if ComplianceChecker._shared_embedder is not None:
                self.embedder = ComplianceChecker._shared_embedder
            else:
                try:
                    from src.rag.embedder import BOCWEmbedder

                    ComplianceChecker._shared_embedder = BOCWEmbedder(
                        persist_directory=persist_directory,
                        collection_name="bocw_regulations",
                    )
                    ComplianceChecker._shared_embedder.ensure_populated()
                    self.embedder = ComplianceChecker._shared_embedder
                except Exception as e:
                    logger.warning(f"ChromaDB RAG unavailable ({e}), falling back to deterministic statutory rules")
                    self.embedder = None

    def check(self, violation_type: str) -> Dict[str, Any]:
        norm_type = violation_type.strip().lower().replace(" ", "_")

        # 1. Primary path: ChromaDB Hybrid RAG search
        if self.embedder is not None:
            try:
                rag_res = self.embedder.retrieve_best_regulation(norm_type)
                if rag_res and float(rag_res.get("confidence", 0.0)) >= 0.65 and rag_res.get("section_id") != "Unknown":
                    return rag_res
            except Exception as e:
                logger.warning(f"ChromaDB query failed ({e}), switching to fallback rules")

        # 2. Secondary path: Deterministic BOCW Act fallback rules
        if norm_type in FALLBACK_RULES:
            rule = FALLBACK_RULES[norm_type]
            return {
                "section_id": rule["section_id"],
                "quoted_text": rule["quoted_text"],
                "penalty": rule["penalty"],
                "confidence": 1.0,
                "source": "FALLBACK",
            }

        return {
            "section_id": "Unknown",
            "quoted_text": "No specific regulation found.",
            "penalty": "None",
            "confidence": 0.0,
            "source": "FALLBACK",
        }


def compliance_agent(state: SafetyState) -> SafetyState:
    logger.info("Running compliance agent")

    checker = ComplianceChecker()
    regulations: List[Dict[str, Any]] = []

    penalty_deduction = 0.0
    for assessment in state.get("risk_assessments", []):
        v_type = assessment.get("violation_type", "")
        reg = checker.check(v_type)
        regulations.append({**assessment, "regulation": reg})

        # Deduct compliance score weighted by risk severity
        r_label = assessment.get("risk_label", "MEDIUM")
        if r_label == "CRITICAL":
            penalty_deduction += 20.0
        elif r_label == "HIGH":
            penalty_deduction += 15.0
        else:
            penalty_deduction += 10.0

    score = max(100.0 - penalty_deduction, 0.0)
    return {**state, "regulations": regulations, "compliance_score": score}
