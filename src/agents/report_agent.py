import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.agents.state import SafetyState

logger = logging.getLogger(__name__)

# Valid BOCW Act / Rule citation pattern to prevent LLM hallucination
VALID_SECTION_PATTERN = re.compile(r"^(Section|Rule)\s+\d+.*", re.IGNORECASE)


def validate_section_citation(section: str) -> str:
    """
    Ensure that a cited statutory section is a valid BOCW Act / Central Rule
    citation and not an LLM hallucination (e.g. 'Section 999').
    """
    if not section or section == "Unknown":
        return "Unknown"
    if not VALID_SECTION_PATTERN.match(section):
        return "Unknown (Validation Failed)"
    # Check that section number is within valid BOCW Act range (Sections 1-64) or Rules (1-252)
    m = re.match(r"^(Section|Rule)\s+(\d+)", section, re.IGNORECASE)
    if m:
        kind = m.group(1).lower()
        num = int(m.group(2))
        if kind == "section" and not (1 <= num <= 64):
            return "Unknown (Validation Failed)"
        if kind == "rule" and not (1 <= num <= 252):
            return "Unknown (Validation Failed)"
    return section


def _try_llm_executive_summary(regulations: List[Dict[str, Any]], compliance_score: float) -> Optional[str]:
    """
    Optionally generate an executive natural-language summary via Google Gemini API
    if `GEMINI_API_KEY` is configured in the environment.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_") or not regulations:
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.2,
            timeout=10,
        )
        items_text = "\n".join(
            f"- Worker {r.get('worker_id')}: {r.get('violation_type')} in {r.get('zone')} "
            f"(Risk: {r.get('risk_label')}, Law: {r.get('regulation', {}).get('section_id')})"
            for r in regulations
        )
        prompt = (
            "You are RakshaSite, an autonomous Indian construction safety compliance auditor. "
            "Write a concise 2-sentence executive action summary citing the exact BOCW Act sections below. "
            "Do not invent any section numbers.\n\n"
            f"Site Compliance Score: {compliance_score:.1f}/100\n"
            f"Violations:\n{items_text}"
        )
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", None)
        return str(content).strip() if content else None
    except Exception as e:
        logger.debug(f"LLM executive summary skipped ({e}); using deterministic report template.")
        return None


def generate_pdf_report(
    state: SafetyState,
    output_path: str = "reports/safety_audit.pdf",
) -> str:
    """
    Generate a printable PDF compliance inspection report using ReportLab.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(out_file), pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>RakshaSite — Autonomous Safety Compliance Audit Report</b>", styles["Title"]))
    elements.append(Spacer(1, 8))

    cam_id = state.get("camera_id", "cam_01")
    ts = state.get("frame_timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comp_score = state.get("compliance_score")
    score_val = float(comp_score) if comp_score is not None else 100.0

    meta_text = (
        f"<b>Camera / Feed ID:</b> {cam_id} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Timestamp:</b> {ts} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Site Compliance Score:</b> {score_val:.1f} / 100.0"
    )
    elements.append(Paragraph(meta_text, styles["Normal"]))
    elements.append(Spacer(1, 14))

    regulations = state.get("regulations", [])
    if not regulations:
        elements.append(Paragraph("No PPE or statutory safety violations detected in this inspection window.", styles["Normal"]))
    else:
        table_data = [["Worker ID", "Violation", "Zone", "Risk Tier", "BOCW Act Citation", "Statutory Penalty"]]
        for item in regulations:
            reg = item.get("regulation", {})
            section = validate_section_citation(reg.get("section_id", "N/A"))
            table_data.append(
                [
                    str(item.get("worker_id", "Unknown")),
                    str(item.get("violation_type", "Unknown")),
                    str(item.get("zone", "general")),
                    f"{item.get('risk_label', 'LOW')} ({item.get('risk_score', 0):.0f})",
                    section[:38],
                    str(reg.get("penalty", "N/A"))[:42],
                ]
            )

        tbl = Table(table_data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f8fafc")]),
                ]
            )
        )
        elements.append(tbl)

    doc.build(elements)
    logger.info(f"Generated PDF safety audit report at {out_file}")
    return str(out_file)


def report_agent(state: SafetyState) -> SafetyState:
    logger.info("Running report agent")

    alert_parts = []
    report_lines = ["# Safety Violation Report", ""]

    regulations = state.get("regulations", [])
    comp_score = state.get("compliance_score")
    score_val = float(comp_score) if comp_score is not None else 100.0

    llm_summary = _try_llm_executive_summary(regulations, score_val)
    if llm_summary:
        report_lines.append(f"> **Executive Summary**: {llm_summary}")
        report_lines.append("")

    for reg_item in regulations:
        v_type = reg_item.get("violation_type", "Unknown")
        worker = reg_item.get("worker_id", "Unknown")
        label = reg_item.get("risk_label", "LOW")
        reg = reg_item.get("regulation", {})

        alert_parts.append(f"{label} risk: Worker {worker} ({v_type})")

        raw_section = reg.get("section_id", "N/A")
        section = validate_section_citation(raw_section)

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
