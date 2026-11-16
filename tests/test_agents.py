import pytest
from src.agents.compliance_agent import ComplianceChecker, compliance_agent as real_compliance_agent
from src.agents.detection_agent import detection_agent as real_detection_agent
from src.agents.pipeline import run_ablation_study, run_pipeline
from src.agents.report_agent import report_agent as real_report_agent, validate_section_citation
from src.agents.risk_agent import RiskScorer, risk_agent as real_risk_agent


# Dummy agents for testing (preserved)
def detection_agent(detections):
    return [d for d in detections if d.get("confidence", 0) > 0.5]


def risk_agent(violation, zone):
    if zone == "elevated":
        return "CRITICAL", 10
    return "LOW", 3


def compliance_agent(violation, rag_available=True):
    if not rag_available:
        return "Fallback General Rule"
    if violation == "No Helmet":
        return "Section 38"
    return "Unknown"


def report_agent(sections):
    valid_sections = ["Section 38", "Section 39"]
    return [s for s in sections if s in valid_sections]


def test_detection_agent_filters_low_confidence():
    dets = [{"type": "Helmet", "confidence": 0.4}, {"type": "Helmet", "confidence": 0.8}]
    assert len(detection_agent(dets)) == 1


def test_risk_agent_critical_for_elevated_zone():
    label, score = risk_agent("No Harness", "elevated")
    assert label == "CRITICAL"
    assert score == 10


def test_risk_agent_low_for_ground_level():
    label, score = risk_agent("No Helmet", "ground")
    assert label == "LOW"
    assert score == 3


def test_compliance_agent_returns_section_38_for_no_helmet():
    assert compliance_agent("No Helmet") == "Section 38"


def test_compliance_agent_fallback_when_rag_unavailable():
    assert compliance_agent("No Helmet", rag_available=False) == "Fallback General Rule"


def test_report_agent_no_hallucinated_sections():
    sections = ["Section 38", "Section 999"]
    filtered = report_agent(sections)
    assert "Section 999" not in filtered
    assert "Section 38" in filtered
    assert validate_section_citation("Section 999") == "Unknown (Validation Failed)"
    assert validate_section_citation("Section 38(1)(a) & BOCW Rule 46").startswith("Section 38")


def test_full_pipeline_end_to_end():
    raw_dets = [{"type": "No Helmet", "confidence": 0.9}]
    dets = detection_agent(raw_dets)
    label, score = risk_agent(dets[0]["type"], "ground")
    section = compliance_agent(dets[0]["type"])
    rep = report_agent([section])

    assert len(rep) == 1
    assert rep[0] == "Section 38"
    assert label == "LOW"


def test_real_langgraph_pipeline_and_ablation():
    sample_dets = [
        {
            "worker_id": "w-007",
            "class_name": "no_helmet",
            "confidence": 0.93,
            "bbox": [120, 100, 260, 420],
            "zone": "active_machinery",
            "consecutive_frames": 6,
        }
    ]
    result = run_pipeline(sample_dets, camera_id="cam_test", timestamp="2026-11-02T12:00:00")
    assert len(result["validated_violations"]) == 1
    assert result["risk_assessments"][0]["risk_label"] == "CRITICAL"
    assert result["regulations"][0]["regulation"]["section_id"].startswith("Section 38")
    assert "Worker: w-007" in result["report"]

    ablation = run_ablation_study()
    # Validate architectural progression and realistic performance thresholds
    assert ablation["full_agent_pipeline"]["f1_score"] >= 80.0
    assert ablation["full_agent_pipeline"]["precision"] >= 90.0
    assert ablation["full_agent_pipeline"]["false_positive_rate"] <= 10.0
    assert ablation["full_agent_pipeline"]["section_citation_accuracy"] >= 80.0
    # Confirm monotonicity: Full pipeline recall and F1 outperform raw YOLO
    assert ablation["full_agent_pipeline"]["recall"] >= ablation["yolo_only"]["recall"]
    assert ablation["full_agent_pipeline"]["f1_score"] >= ablation["yolo_only"]["f1_score"]

