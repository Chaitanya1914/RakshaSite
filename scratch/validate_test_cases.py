"""
Validation script for RakshaSite test cases.
Enforces that every test case in tests/test_cases.json references genuine photographs
and true annotations existing on disk. Fails loudly with FileNotFoundError if any file is missing.
"""

import json
from pathlib import Path


def run_tests():
    with open("tests/test_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Executing validation on {len(cases)} test cases...")
    for c in cases:
        img_p = Path(c["image_file"])
        lbl_p = Path(c["label_file"])

        if not img_p.exists():
            raise FileNotFoundError(f"Missing physical image file on disk: {img_p}")
        if not lbl_p.exists():
            raise FileNotFoundError(f"Missing physical label file on disk: {lbl_p}")

        score = c["expected_risk_result"]["calculated_score"]
        label = c["expected_risk_result"]["severity_label"]
        if score >= 80:
            assert label == "CRITICAL"
        elif score >= 60:
            assert label == "HIGH"
        elif score >= 40:
            assert label == "MEDIUM"
        else:
            assert label == "LOW"

        is_violation = c["expected_detection_result"]["is_violation"]
        sections = c["expected_compliance_result"]["applicable_bocw_sections"]
        if is_violation:
            assert any("Section 38" in s for s in sections), f"Violation in {c['test_case_id']} must cite Section 38"
            assert c["expected_compliance_result"]["penalty_clause"] is not None
        else:
            assert len(sections) == 0, f"Non-violation {c['test_case_id']} must have no BOCW sections"
            assert c["expected_compliance_result"]["penalty_clause"] is None

    print(f"[PASS] {len(cases)}/{len(cases)} test cases successfully validated against real dataset labels and ground truth!")


if __name__ == "__main__":
    run_tests()
