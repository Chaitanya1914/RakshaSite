import re
from typing import Dict, List, Any
import pdfplumber

def extract_sections(pdf_path: str) -> Dict[str, str]:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    sections = {}
    pattern = re.compile(r'(Section \d+.*?)(?=Section \d+|$)', re.DOTALL | re.IGNORECASE)
    matches = pattern.finditer(text)
    
    for i, match in enumerate(matches):
        section_content = match.group(1).strip()
        num_match = re.search(r'Section (\d+)', section_content, re.IGNORECASE)
        if num_match:
            section_id = f"Section_{num_match.group(1)}"
        else:
            section_id = f"Section_Unknown_{i}"
        
        sections[section_id] = section_content
        
    return sections

def chunk_section(section_text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(section_text):
        end = start + max_chars
        chunks.append(section_text[start:end])
        start += (max_chars - overlap)
    return chunks

def get_manual_bocw_sections() -> Dict[str, Dict[str, Any]]:
    return {
        "Section_38": {
            "content": "Safety measures. Employer must provide adequate safety equipment including helmets, safety belts, and other PPE to workers.",
            "tags": ["safety", "ppe", "helmet", "equipment"],
            "violation_types": ["no_helmet", "missing_ppe"],
            "penalty": "Fine up to Rs. 1000 or imprisonment up to 3 months, or both."
        },
        "Section_39": {
            "content": "Safety at height. Fall protection mechanisms, scaffolding, and harnesses are mandatory for workers working above 2 meters.",
            "tags": ["height", "fall_protection", "harness", "scaffolding"],
            "violation_types": ["no_harness", "fall_hazard"],
            "penalty": "Fine up to Rs. 2000 or imprisonment up to 6 months."
        },
        "Section_44": {
            "content": "Duties of employers. The employer is responsible for providing a safe working environment and reporting any major accidents.",
            "tags": ["employer_duty", "environment", "reporting"],
            "violation_types": ["unsafe_environment", "failure_to_report"],
            "penalty": "Cancellation of registration and fine."
        },
        "Section_45": {
            "content": "Rights of workers. Workers have the right to refuse work in unsafe conditions and report unsafe practices without retaliation.",
            "tags": ["worker_rights", "refusal", "unsafe_conditions"],
            "violation_types": ["worker_coercion", "retaliation"],
            "penalty": "Penalty of Rs. 5000."
        },
        "Section_47": {
            "content": "Penalties. General penalties for contravention of provisions of the Act where no specific penalty is provided.",
            "tags": ["penalty", "general_contravention"],
            "violation_types": ["general_violation"],
            "penalty": "Fine extending up to Rs. 1000 or imprisonment up to 3 months, or both."
        }
    }
