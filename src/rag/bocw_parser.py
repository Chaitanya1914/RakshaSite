import re
from pathlib import Path
from typing import Any, Dict, List

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def extract_sections(file_path: str) -> Dict[str, str]:
    """
    Extract statutory sections from a PDF, Markdown (.md), or text (.txt) file.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        return {}

    text = ""
    if path_obj.suffix.lower() == ".pdf":
        if pdfplumber is None:
            raise ImportError("pdfplumber is required to parse PDF files.")
        with pdfplumber.open(str(path_obj)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    else:
        text = path_obj.read_text(encoding="utf-8")

    sections: Dict[str, str] = {}
    pattern = re.compile(r"((?:Section|Rule)\s+\d+.*?)(?=(?:###\s*)?(?:Section|Rule)\s+\d+|$)", re.DOTALL | re.IGNORECASE)
    matches = pattern.finditer(text)

    for i, match in enumerate(matches):
        section_content = match.group(1).strip()
        num_match = re.search(r"(Section|Rule)\s+(\d+)", section_content, re.IGNORECASE)
        if num_match:
            prefix = num_match.group(1).capitalize()
            section_id = f"{prefix}_{num_match.group(2)}"
        else:
            section_id = f"Section_Unknown_{i}"

        sections[section_id] = section_content

    return sections


def chunk_section(section_text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(section_text):
        end = start + max_chars
        chunks.append(section_text[start:end])
        start += step
    return chunks


def get_manual_bocw_sections() -> Dict[str, Dict[str, Any]]:
    """
    Return the curated dictionary of Indian construction safety laws:
    - Building and Other Construction Workers (BOCW) Act, 1996
    - BOCW Central Rules, 1998
    - Bureau of Indian Standards (IS 2925, IS 3521, IS 3696)
    """
    return {
        "Section_38": {
            "section_citation": "Section 38(1)(a) & BOCW Rule 46 (IS 2925:1984)",
            "content": (
                "Section 38 (Safety Committee & Safety Officers) read with BOCW Central Rule 46: "
                "Employer must provide and enforce adequate Personal Protective Equipment (PPE) including "
                "industrial safety helmets conforming to IS 2925:1984 for all building workers exposed to "
                "falling objects, head injury hazards, or electrical shock."
            ),
            "tags": ["safety", "ppe", "helmet", "hardhat", "head_protection", "equipment", "is_2925"],
            "violation_types": ["no_helmet", "NO_HELMET", "No Helmet", "missing_ppe"],
            "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months, or both (Rs. 100/day for continuing contravention).",
        },
        "Section_40_Vest": {
            "section_citation": "Section 38(1)(a), Section 40(2)(r) & BOCW Rule 45",
            "content": (
                "Section 40(2)(r) read with BOCW Central Rule 45 & Rule 54: High-visibility reflective safety vests "
                "(fluorescent protective apparel with retro-reflective strips) must be provided to and worn by all "
                "building workers operating on construction sites, especially near moving vehicles, cranes, or active machinery."
            ),
            "tags": ["vest", "high_visibility", "reflective", "apparel", "ppe", "machinery"],
            "violation_types": ["no_vest", "NO_VEST", "No Vest"],
            "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.",
        },
        "Section_39": {
            "section_citation": "Section 39, Section 40(2)(o) & BOCW Rules 41, 175–185 (IS 3521 / IS 3696)",
            "content": (
                "Section 39 & Section 40(2)(o) read with BOCW Central Rules 41 & 175–185: Fall protection mechanisms, "
                "guarded scaffolding (IS 3696), safety nets, and full-body safety harnesses anchored to lifelines (IS 3521) "
                "are mandatory for all building workers working at heights or elevations above 2 meters."
            ),
            "tags": ["height", "fall_protection", "harness", "scaffolding", "elevated", "is_3521", "is_3696"],
            "violation_types": ["no_harness", "NO_HARNESS", "No Harness", "fall_hazard"],
            "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months (up to 6 months for repeat offence under Section 47(2)).",
        },
        "Section_40_Gloves": {
            "section_citation": "Section 40(2)(r) & BOCW Rule 49 (Hand Protection)",
            "content": (
                "Section 40(2)(r) read with BOCW Central Rule 49: Employers must provide suitable protective industrial "
                "gloves (leather, insulated, or chemical-resistant) to building workers handling sharp materials, reinforcement "
                "bars, concreting, welding, or electrical cables."
            ),
            "tags": ["gloves", "hand_protection", "ppe", "rule_49"],
            "violation_types": ["no_gloves", "NO_GLOVES", "No Gloves"],
            "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.",
        },
        "Section_40_Boots": {
            "section_citation": "Section 40(2)(r) & BOCW Rule 49 (Foot Protection - IS 15298)",
            "content": (
                "Section 40(2)(r) read with BOCW Central Rule 49: Every building worker engaged in construction, "
                "excavation, demolition, or heavy material handling must wear protective steel-toe safety boots "
                "conforming to IS 15298 to prevent crushing, puncture, and slip injuries."
            ),
            "tags": ["boots", "footwear", "shoes", "foot_protection", "ppe", "rule_49"],
            "violation_types": ["no_boots", "NO_BOOTS", "No Boots"],
            "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.",
        },
        "Section_44": {
            "section_citation": "Section 44 (BOCW Act, 1996)",
            "content": (
                "Section 44 — Responsibility of Employers: An employer shall be responsible for providing constant "
                "and adequate supervision of any building or other construction work in his establishment to ensure "
                "compliance with safety provisions and to take all practical steps necessary to prevent accidents."
            ),
            "tags": ["employer_duty", "environment", "reporting", "supervision"],
            "violation_types": ["unsafe_environment", "failure_to_report"],
            "penalty": "Section 47 & Section 50: Fine up to Rs. 2,000, imprisonment up to 3 months, and potential cancellation of registration.",
        },
        "Section_45": {
            "section_citation": "Section 45 (BOCW Act, 1996)",
            "content": (
                "Section 45 — Responsibility of Contractors, Architects, and Project Engineers: Ensures that site "
                "supervisors and contractors maintain safe operating zones and protect workers' rights to safe working conditions."
            ),
            "tags": ["worker_rights", "refusal", "unsafe_conditions", "contractor_duty"],
            "violation_types": ["worker_coercion", "retaliation"],
            "penalty": "Section 47 & Section 50: Penalty up to Rs. 2,000 and/or imprisonment up to 3 months.",
        },
        "Section_47": {
            "section_citation": "Section 47(1) & Section 47(2) (BOCW Act, 1996)",
            "content": (
                "Section 47 — Penalties for Contravention of Safety Provisions: Whoever contravenes the provisions of "
                "any safety rules made under Section 40 shall be punishable with imprisonment up to 3 months, or fine "
                "up to Rs. 2,000, or both, plus Rs. 100/day for continuing contravention. Repeat offences carry up to "
                "6 months imprisonment."
            ),
            "tags": ["penalty", "general_contravention", "fine", "imprisonment"],
            "violation_types": ["general_violation"],
            "penalty": "Fine extending up to Rs. 2,000 or imprisonment up to 3 months, or both (up to 6 months for repeat violation).",
        },
    }
