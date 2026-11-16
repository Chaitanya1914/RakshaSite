"""
tests/test_api_demo.py — Test Suite for Backend Demo Viewer, Endpoints & Regression
Verifies root HTML viewer, static reports mounting, demo REST endpoints, graceful missing file handling, and existing API routes.
"""

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.api import app, REPORTS_DIR

client = TestClient(app)


def test_executive_demo_viewer_endpoints():
    """Verify that GET / and GET /demo return the executive demo dashboard."""
    for path in ["/", "/demo"]:
        response = client.get(path)
        assert response.status_code == 200, f"Expected 200 on {path}, got {response.status_code}"
        assert "text/html" in response.headers.get("content-type", "")
        html = response.text
        assert "RakshaSite Executive Demo Viewer" in html
        assert "BOCW Act 1996" in html
        assert "<video" in html
        assert "<img" in html
        assert "demo_annotated_output.mp4" in html
        assert "demo_snapshot_violation.jpg" in html
        assert "safety_audit.pdf" in html


def test_static_reports_mounting():
    """Verify that reports/ directory is mounted and serves artifacts."""
    # Check JSON audit log
    resp_json = client.get("/reports/demo_audit_log.json")
    assert resp_json.status_code == 200
    assert "application/json" in resp_json.headers.get("content-type", "")
    data = resp_json.json()
    assert "video_source" in data
    assert "sample_agent_output" in data

    # Check snapshot image
    resp_img = client.get("/reports/demo_snapshot_violation.jpg")
    assert resp_img.status_code == 200
    assert "image" in resp_img.headers.get("content-type", "")
    assert len(resp_img.content) > 1000

    # Check PDF report
    resp_pdf = client.get("/reports/safety_audit.pdf")
    assert resp_pdf.status_code == 200
    assert "pdf" in resp_pdf.headers.get("content-type", "") or "octet-stream" in resp_pdf.headers.get("content-type", "")
    assert len(resp_pdf.content) > 500

    # Check non-existent file returns 404
    resp_404 = client.get("/reports/non_existent_file_xyz123.mp4")
    assert resp_404.status_code == 404


def test_demo_status_endpoint():
    """Verify GET /api/v1/demo/status reports artifact presence and readiness."""
    response = client.get("/api/v1/demo/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "artifacts" in data
    assert "pipeline_ready" in data
    assert "video" in data["artifacts"]
    assert "snapshot" in data["artifacts"]
    assert "pdf_report" in data["artifacts"]
    assert "audit_log" in data["artifacts"]


def test_demo_summary_endpoint():
    """Verify GET /api/v1/demo/summary aggregates executive compliance metrics."""
    response = client.get("/api/v1/demo/summary")
    assert response.status_code == 200
    data = response.json()
    assert "compliance_score" in data
    assert "compliance_rating" in data
    assert "overall_risk_tier" in data
    assert "total_violations" in data
    assert "bocw_sections_cited" in data
    assert isinstance(data["bocw_sections_cited"], list)


def test_demo_audit_log_endpoint():
    """Verify GET /api/v1/demo/audit-log returns the complete audit log data."""
    response = client.get("/api/v1/demo/audit-log")
    assert response.status_code == 200
    data = response.json()
    assert "sample_agent_output" in data
    assert "compliance_score" in data["sample_agent_output"]


def test_demo_endpoints_graceful_missing_files(tmp_path, monkeypatch):
    """Verify demo endpoints handle missing audit log cleanly without 500 Internal Server Error."""
    # Temporarily point REPORTS_DIR to an empty temp directory
    empty_dir = tmp_path / "empty_reports"
    empty_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("backend.api.REPORTS_DIR", empty_dir)

    resp_summary = client.get("/api/v1/demo/summary")
    assert resp_summary.status_code == 200
    data = resp_summary.json()
    assert data["compliance_score"] == 100.0
    assert data["overall_risk_tier"] == "SAFE"

    resp_status = client.get("/api/v1/demo/status")
    assert resp_status.status_code == 200
    assert resp_status.json()["status"] == "pending_execution"

    resp_log = client.get("/api/v1/demo/audit-log")
    assert resp_log.status_code == 404
    assert "detail" in resp_log.json()


def test_existing_api_endpoints_regression():
    """Verify that existing backend endpoints continue to function without regressions."""
    # Test GET /api/v1/regulations
    resp_regs = client.get("/api/v1/regulations")
    assert resp_regs.status_code == 200
    regs_data = resp_regs.json()
    assert len(regs_data) > 0

    # Test GET /api/v1/report/{shift_id}
    resp_rep = client.get("/api/v1/report/shift_123")
    assert resp_rep.status_code == 200
    assert resp_rep.json()["shift_id"] == "shift_123"

    # Test GET /api/v1/analytics/{camera_id}
    resp_ana = client.get("/api/v1/analytics/cam_01")
    assert resp_ana.status_code == 200
    assert "compliance_score" in resp_ana.json()
