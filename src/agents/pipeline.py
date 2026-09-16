from langgraph.graph import StateGraph, END
from src.agents.state import SafetyState
from src.agents.detection_agent import detection_agent
from src.agents.risk_agent import risk_agent
from src.agents.compliance_agent import compliance_agent
from src.agents.report_agent import report_agent
from typing import List, Dict, Any

def should_check_risk(state: SafetyState) -> str:
    if state.get("validated_violations"):
        return "risk_agent"
    return END

def should_check_compliance(state: SafetyState) -> str:
    assessments = state.get("risk_assessments", [])
    for assessment in assessments:
        if assessment.get("risk_label") in ["MEDIUM", "HIGH", "CRITICAL"]:
            return "compliance_agent"
    return END

def build_graph():
    workflow = StateGraph(SafetyState)
    
    workflow.add_node("detection_agent", detection_agent)
    workflow.add_node("risk_agent", risk_agent)
    workflow.add_node("compliance_agent", compliance_agent)
    workflow.add_node("report_agent", report_agent)
    
    workflow.set_entry_point("detection_agent")
    
    workflow.add_conditional_edges("detection_agent", should_check_risk)
    workflow.add_conditional_edges("risk_agent", should_check_compliance)
    
    workflow.add_edge("compliance_agent", "report_agent")
    workflow.add_edge("report_agent", END)
    
    return workflow.compile()

def run_pipeline(detections: List[Dict[str, Any]], camera_id: str = "cam_01", timestamp: str = "") -> SafetyState:
    graph = build_graph()
    
    initial_state: SafetyState = {
        "raw_detections": detections,
        "frame_timestamp": timestamp,
        "camera_id": camera_id,
        "validated_violations": [],
        "risk_assessments": [],
        "regulations": [],
        "report": None,
        "alert_message": None,
        "compliance_score": None
    }
    
    result = graph.invoke(initial_state)
    return result

def get_graph_visualization():
    graph = build_graph()
    return graph.get_graph().draw_mermaid()
