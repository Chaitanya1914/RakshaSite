import pytest

# Dummy agents for testing
def detection_agent(detections):
    return [d for d in detections if d.get('confidence', 0) > 0.5]

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

def test_full_pipeline_end_to_end():
    raw_dets = [{"type": "No Helmet", "confidence": 0.9}]
    dets = detection_agent(raw_dets)
    label, score = risk_agent(dets[0]["type"], "ground")
    section = compliance_agent(dets[0]["type"])
    rep = report_agent([section])
    
    assert len(rep) == 1
    assert rep[0] == "Section 38"
    assert label == "LOW"
