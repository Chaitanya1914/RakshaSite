"""
tests/test_fresh_clone_reproducibility.py — Fresh clone reproducibility & git size audit.

Validates that:
1. Zero git blobs across the repository history exceed 25 MB (26,214,400 bytes).
2. Zero working tree files (excluding .git and .venv) exceed 25 MB.
3. In a fresh clone sandbox, scripts/fetch_data.py --mirror successfully restores
   all required weights, demo media, and validation photos, and passes SHA-256 verification.
4. Standalone scripts/verify_fresh_clone.py executes cleanly with exit code 0.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import pytest

from scripts.verify_fresh_clone import (
    audit_git_history_for_large_blobs,
    audit_working_tree_files,
    compute_sha256,
    run_fresh_clone_verification,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_no_git_blobs_exceed_25mb():
    """Verify that no blob across any commit in git history exceeds 25 MB (26,214,400 bytes)."""
    violations = audit_git_history_for_large_blobs(REPO_ROOT)
    assert len(violations) == 0, f"Found {len(violations)} git blobs > 25MB: {violations}"


def test_no_working_tree_files_exceed_25mb():
    """Verify that no working tree file (excluding .git and .venv) exceeds 25 MB."""
    violations = audit_working_tree_files(REPO_ROOT)
    assert len(violations) == 0, f"Found {len(violations)} workspace files > 25MB: {violations}"


def test_fresh_clone_fetch_and_verify():
    """
    Simulates a fresh clone in an isolated temporary sandbox, executes fetch_data.py
    with local mirror, and verifies all assets pass SHA-256 integrity checks.
    """
    mirror_path = REPO_ROOT / "dist" / "asset_mirror"
    assert mirror_path.exists(), f"Mirror directory missing at {mirror_path}"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_clone:
        clone_root = pathlib.Path(temp_clone).resolve()

        # Clone repository into temp sandbox without changing repository state
        clone_res = subprocess.run(
            ["git", "clone", str(REPO_ROOT), str(clone_root)],
            capture_output=True,
            text=True,
        )
        assert clone_res.returncode == 0, f"git clone failed: {clone_res.stderr}"

        # Copy uncommitted working files (e.g. scripts/) into sandbox
        shutil.copytree(REPO_ROOT / "scripts", clone_root / "scripts", dirs_exist_ok=True)

        # Confirm that before fetch, weights and video do NOT exist in fresh clone
        assert not (clone_root / "models" / "best.pt").exists()
        assert not (clone_root / "data" / "raw" / "hardhat_demo.mp4").exists()

        # Run fetch_data.py with mirror inside clone
        fetch_res = subprocess.run(
            [sys.executable, "scripts/fetch_data.py", "--mirror", str(mirror_path)],
            cwd=str(clone_root),
            capture_output=True,
            text=True,
        )
        assert fetch_res.returncode == 0, f"fetch_data.py failed:\n{fetch_res.stderr}\n{fetch_res.stdout}"

        # Verify weights
        pt_path = clone_root / "models" / "best.pt"
        assert pt_path.is_file()
        assert compute_sha256(pt_path) == "48fc9bfe7c62ab039052f54bb536935dfebb5f4ee07f73df0895f8a8e6ac6a0e"
        assert pt_path.stat().st_size == 6226211


        # Verify demo video
        mp4_path = clone_root / "data" / "raw" / "hardhat_demo.mp4"
        assert mp4_path.is_file()
        assert compute_sha256(mp4_path) == "e4ea71aa6ffc460c430debe677e9cecea6e2a1712d45ce26d48ace0a1aa76a83"
        assert mp4_path.stat().st_size == 4143534

        # Verify samples
        sample_img = clone_root / "data" / "raw" / "samples" / "construction-safety.jpg"
        assert sample_img.is_file()
        assert compute_sha256(sample_img) == "8f1c6bbb151cea7762f12c37ae1d41742da8b54d99aba980ebf82d06b2a8b150"

        # Verify validation images count
        val_imgs = list((clone_root / "data" / "unified" / "val" / "images").glob("*.*"))
        assert len(val_imgs) == 50

        # Verify --verify-only succeeds inside fresh clone
        verify_res = subprocess.run(
            [sys.executable, "scripts/fetch_data.py", "--verify-only"],
            cwd=str(clone_root),
            capture_output=True,
            text=True,
        )
        assert verify_res.returncode == 0, f"verify-only failed in clone:\n{verify_res.stderr}"


def test_standalone_verifier_script():
    """Verify that scripts/verify_fresh_clone.py executes cleanly with exit code 0."""
    exit_code = run_fresh_clone_verification(run_demo=False, quiet=True)
    assert exit_code == 0, "run_fresh_clone_verification failed"
