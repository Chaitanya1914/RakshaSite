import logging
import re
from typing import Any, Dict
from src.agents.state import SafetyState

logger = logging.getLogger(__name__)

def report_agent(state: SafetyState) -> SafetyState:
    logger.info("Running report agent")
    
    alert_parts = []
    report_lines = ["# Safety Violation Report", ""]
    
    for reg_item in state.get("regulations", []):
        v_type = reg_item.get("violation_type", "Unknown")
        worker = reg_item.get("worker_id", "Unknown")
        label = reg_item.get("risk_label", "LOW")
        reg = reg_item.get("regulation", {})
        
        alert_parts.append(f"{label} risk: Worker {worker} ({v_type})")
        
        section = reg.get('section_id', 'N/A')
        # Validate that the LLM/Fallback didn't hallucinate invalid sections
        # Basic pattern validation logic for sections
        if not re.match(r'^Section \d+.*', section) and section != 'Unknown':
            section = "Unknown (Validation Failed)"
            
        report_lines.append(f"## Worker: {worker} | Violation: {v_type} | Risk: {label}")
        report_lines.append(f"- **Zone**: {reg_item.get('zone', 'Unknown')}")
        report_lines.append(f"- **Confidence**: {reg_item.get('confidence', 0.0):.2f}")
        report_lines.append(f"- **Regulation**: {section}")
        report_lines.append(f"- **Rule Text**: {reg.get('quoted_text', 'N/A')}")
        report_lines.append(f"- **Potential Penalty**: {reg.get('penalty', 'N/A')}")
        report_lines.append("")
        
    alert_message = " | ".join(alert_parts)
    if not alert_message:
        alert_message = "No violations detected."
        report_lines.append("No violations detected in this frame.")
        
    report = "\n".join(report_lines)
    
    return {**state, "alert_message": alert_message, "report": report}
