"""
tests/test_demo_runner.py — Unit & Integration Test Suite for RakshaSite Demo Pipeline
Verifies artifact generation, BOCW statutory citations, HUD drawing, CLI arguments, and offline fallback.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from demo_runner import draw_executive_hud, run_demo


def test_demo_runner_output_artifacts(tmp_path):
    """Verify that run_demo generates all 4 required artifacts with non-empty sizes."""
    out_dir = tmp_path / "test_reports"
    res = run_demo(
        video_source="data/raw/hardhat_demo.mp4",
        sample_image="data/raw/samples/construction-safety.jpg",
        max_frames=5,
        start_frame=120,
        output_dir=str(out_dir),
    )
    assert res is not None
    assert "frames_processed" in res
    assert res["frames_processed"] > 0

    video_file = out_dir / "demo_annotated_output.mp4"
    assert video_file.exists()
    assert video_file.stat().st_size > 1000

    snapshot_file = out_dir / "demo_snapshot_violation.jpg"
    assert snapshot_file.exists()
    assert snapshot_file.stat().st_size > 1000

    pdf_file = out_dir / "safety_audit.pdf"
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 500
    pdf_bytes = pdf_file.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-"), "Generated report must be valid PDF"

    audit_file = out_dir / "demo_audit_log.json"
    assert audit_file.exists()
    assert audit_file.stat().st_size > 100
    with open(audit_file, "r", encoding="utf-8") as f:
        log_json = json.load(f)
    assert log_json["frames_processed"] == 5
    assert "sample_agent_output" in log_json


def test_demo_audit_log_bocw_citations():
    """Verify that every violation cited in demo_audit_log.json matches BOCW Act statutes."""
    audit_path = Path("reports/demo_audit_log.json")
    assert audit_path.exists(), "demo_audit_log.json must exist"

    with open(audit_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    agent_out = log_data.get("sample_agent_output", {})
    assert "compliance_score" in agent_out
    regulations = agent_out.get("regulations_cited", [])
    assert len(regulations) > 0, "Expected at least 1 cited statutory violation in demo audit log"

    for r in regulations:
        assert "worker_id" in r
        assert "violation_type" in r
        sec = r.get("bocw_section", "")
        assert "Section" in sec or "Rule" in sec, f"Invalid BOCW section citation: {sec}"
        pen = r.get("penalty", "")
        assert "Section 47" in pen, f"Penalty must cite Section 47, got: {pen}"


def test_executive_hud_drawing():
    """Verify draw_executive_hud renders bounding boxes, compliance banner, and statutory ticker."""
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    sample_dets = [
        {"class_name": "no_helmet", "confidence": 0.92, "bbox": [100, 100, 200, 250]},
        {"class_name": "helmet", "confidence": 0.88, "bbox": [50, 50, 90, 90]},
    ]
    sample_state = {
        "compliance_score": 75.0,
        "regulations": [
            {
                "worker_id": "w-01",
                "violation_type": "no_helmet",
                "risk_label": "CRITICAL",
                "risk_score": 85.0,
                "regulation": {
                    "section_id": "Section 38(1)(a) & Rule 46",
                    "penalty": "Section 47: Fine up to Rs. 2,000",
                },
            }
        ],
    }

    sample_workers = [
        {
            "worker_id": "w-01",
            "bbox": (100, 100, 200, 250),
            "violations": ["NO_HELMET"],
            "is_compliant": False,
        }
    ]

    out = draw_executive_hud(
        dummy_frame,
        sample_dets,
        sample_state,
        frame_idx=10,
        active_workers=1,
        violating_workers=1,
        tracked_workers=sample_workers,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == dummy_frame.shape
    # Ensure pixels in top HUD banner are non-zero
    assert np.any(out[:56, :, :] > 0)
    # Ensure pixels in bottom ticker are non-zero
    assert np.any(out[-52:, :, :] > 0)


def test_demo_runner_cli_interface():
    """Verify demo_runner CLI runs with --video, --start-frame, and --max-frames, returning exit 0 and valid JSON."""
    cmd = [
        sys.executable,
        "demo_runner.py",
        "--video",
        "data/raw/hardhat_demo.mp4",
        "--start-frame",
        "120",
        "--max-frames",
        "2",
        "--conf",
        "0.25",
        "--offline",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"demo_runner CLI failed with stderr: {proc.stderr}"
    assert "frames_processed" in proc.stdout
    # Verify output is valid JSON
    start_idx = proc.stdout.find("{")
    end_idx = proc.stdout.rfind("}")
    assert start_idx != -1 and end_idx != -1, f"No JSON found in stdout: {proc.stdout}"
    out_json = json.loads(proc.stdout[start_idx : end_idx + 1])
    assert out_json["frames_processed"] == 2


def test_offline_fallback_zero_api_keys(tmp_path, monkeypatch):
    """Verify pipeline executes 100% offline with zero LLM API keys and HF_HUB_OFFLINE set."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    out_dir = tmp_path / "offline_reports"
    res = run_demo(
        video_source="data/raw/hardhat_demo.mp4",
        sample_image="data/raw/samples/construction-safety.jpg",
        max_frames=2,
        start_frame=120,
        output_dir=str(out_dir),
        offline=True,
    )
    assert res is not None
    assert (out_dir / "safety_audit.pdf").exists()
    assert (out_dir / "demo_audit_log.json").exists()

    with open(out_dir / "demo_audit_log.json", "r", encoding="utf-8") as f:
        log_data = json.load(f)
    regs = log_data["sample_agent_output"]["regulations_cited"]
    assert len(regs) > 0
    # Statutory citation should match Section 38
    assert any("Section 38" in r["bocw_section"] for r in regs)
