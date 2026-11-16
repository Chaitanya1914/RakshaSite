"""
tests/test_fetch_script.py — Unit and integration tests for scripts/fetch_data.py.

Verifies manifest schema, deterministic hashing, CLI argument handling,
corrupt checksum detection, missing file handling, Zip Slip security guard,
idempotent skip logic, and local mirror creation/unpacking flows.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "scripts" / "manifest.json"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_data.py"


def test_manifest_schema_and_required_assets():
    """Verify scripts/manifest.json adheres to schema and defines core required assets."""
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "version" in data
    assert "assets" in data
    assets = data["assets"]

    required_keys = ["weights", "demo_assets", "val_images"]
    for k in required_keys:
        assert k in assets, f"Missing required asset key '{k}' in manifest"
        item = assets[k]
        assert "target_file" in item or "target_dir" in item
        assert "archive_name" in item
        assert item["archive_name"].endswith(".zip")
        assert "size_bytes" in item
        assert item["size_bytes"] > 0
        if "target_file" in item:
            assert "expected_sha256" in item or "target_sha256" in item
            expected_sha = item.get("expected_sha256") or item.get("target_sha256")
            assert len(expected_sha) == 64


def test_compute_sha256_known_content():
    """Verify deterministic SHA256 calculation on arbitrary byte streams."""
    from scripts.fetch_data import compute_sha256

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"RakshaSite Safety Framework 2026")
        temp_name = tf.name

    try:
        expected = hashlib.sha256(b"RakshaSite Safety Framework 2026").hexdigest()
        actual = compute_sha256(pathlib.Path(temp_name))
        assert actual == expected
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass


def test_compute_sha256_missing_file_raises_error():
    """Verify compute_sha256 raises FileNotFoundError on missing file."""
    from scripts.fetch_data import compute_sha256

    missing_path = pathlib.Path("nonexistent_path_to_never_exist_98765.dat")
    with pytest.raises(FileNotFoundError):
        compute_sha256(missing_path)


def test_cli_help_flag():
    """Verify python scripts/fetch_data.py --help exits 0 with usage text."""
    res = subprocess.run(
        [sys.executable, str(FETCH_SCRIPT), "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "usage:" in res.stdout.lower() or "options:" in res.stdout.lower()


def test_verify_only_succeeds_on_valid_assets():
    """Verify that --verify-only succeeds (exit code 0) when all target assets are in place."""
    res = subprocess.run(
        [sys.executable, str(FETCH_SCRIPT), "--verify-only"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"--verify-only failed:\n{res.stderr}\n{res.stdout}"


def test_verify_detects_corrupted_checksum():
    """Verify that tampering with an asset file causes verification to fail with mismatch report."""
    from scripts.fetch_data import verify_asset

    with tempfile.TemporaryDirectory() as td:
        dummy_file = pathlib.Path(td) / "dummy.pt"
        dummy_file.write_bytes(b"corrupted content")

        legit_hash = hashlib.sha256(b"legit content").hexdigest()
        asset_spec = {
            "target_file": str(dummy_file),
            "expected_sha256": legit_hash,
            "size_bytes": 17,
        }

        is_valid, msg = verify_asset(dummy_file, asset_spec)
        assert is_valid is False
        assert "mismatch" in msg.lower() or "checksum" in msg.lower()


def test_verify_detects_missing_file():
    """Verify that missing asset returns False and reports file does not exist."""
    from scripts.fetch_data import verify_asset

    missing_path = pathlib.Path("nonexistent_directory_987/missing_model.pt")
    asset_spec = {
        "target_file": str(missing_path),
        "expected_sha256": "0" * 64,
        "size_bytes": 100,
    }

    is_valid, msg = verify_asset(missing_path, asset_spec)
    assert is_valid is False
    assert "missing" in msg.lower() or "not found" in msg.lower()


def test_create_mirror_and_unpack_flow():
    """Integration test verifying --create-mirror bundles assets and --mirror extracts them."""
    with tempfile.TemporaryDirectory() as td:
        sandbox = pathlib.Path(td)
        mirror_dir = sandbox / "mirror"
        mirror_dir.mkdir()
        target_dir = sandbox / "dest"
        target_dir.mkdir()

        # Create mock asset
        mock_data = b"MOCK_YOLO_WEIGHTS_VERSION_1_DETERMINISTIC"
        mock_hash = hashlib.sha256(mock_data).hexdigest()

        # Create zip in mirror
        zip_path = mirror_dir / "test_weights.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("models/best.pt", mock_data)

        mock_manifest = {
            "version": "1.0.0",
            "assets": {
                "weights": {
                    "target_file": "models/best.pt",
                    "expected_sha256": mock_hash,
                    "size_bytes": len(mock_data),
                    "archive_name": "test_weights.zip",
                }
            },
        }

        from scripts.fetch_data import fetch_all_assets

        success = fetch_all_assets(
            manifest_data=mock_manifest,
            base_dir=target_dir,
            mirror_dir=mirror_dir,
            quiet=True,
        )
        assert success is True

        extracted_file = target_dir / "models" / "best.pt"
        assert extracted_file.exists()
        assert extracted_file.read_bytes() == mock_data


def test_idempotent_skip_when_already_verified():
    """Verify fetch skips extraction when target file already matches expected hash."""
    from scripts.fetch_data import fetch_all_assets

    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        target_file = base / "models" / "best.pt"
        target_file.parent.mkdir(parents=True)
        content = b"ALREADY_PRESENT_VALID_CONTENT_2026"
        target_file.write_bytes(content)
        expected_hash = hashlib.sha256(content).hexdigest()

        mock_manifest = {
            "assets": {
                "weights": {
                    "target_file": "models/best.pt",
                    "expected_sha256": expected_hash,
                    "size_bytes": len(content),
                    "archive_name": "weights.zip",
                }
            }
        }

        mtime_before = target_file.stat().st_mtime

        # Calling fetch without mirror succeeds instantly due to idempotent check
        success = fetch_all_assets(mock_manifest, base_dir=base, mirror_dir=None, quiet=True)
        assert success is True
        assert target_file.stat().st_mtime == mtime_before


def test_nonexistent_mirror_directory_error_handling():
    """Verify that passing an invalid mirror directory reports an error cleanly without crashing."""
    res = subprocess.run(
        [sys.executable, str(FETCH_SCRIPT), "--mirror", "Z:\\invalid_nonexistent_directory_999"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "not exist" in res.stderr.lower() or "not found" in res.stderr.lower()


def test_zip_slip_prevention():
    """Verify that safe_extract_zip refuses to extract archive members with path traversal."""
    from scripts.fetch_data import safe_extract_zip

    with tempfile.TemporaryDirectory() as td:
        sandbox = pathlib.Path(td)
        malicious_zip = sandbox / "malicious.zip"
        dest_dir = sandbox / "extracted"
        dest_dir.mkdir()

        # Build malicious zip containing path traversal
        with zipfile.ZipFile(malicious_zip, "w") as zf:
            zf.writestr("../../escape_sandbox.txt", b"malicious content")

        with pytest.raises(PermissionError) as exc_info:
            safe_extract_zip(malicious_zip, dest_dir)

        assert "security violation" in str(exc_info.value).lower() or "directory traversal" in str(exc_info.value).lower()
        # Verify nothing escaped outside dest_dir
        escaped_file = sandbox / "escape_sandbox.txt"
        assert not escaped_file.exists()


def test_deterministic_zip_hashes():
    """Verify that create_mirror_archive generates identical SHA-256 digests across repeated runs."""
    from scripts.fetch_data import create_mirror_archive

    with tempfile.TemporaryDirectory() as td:
        sandbox = pathlib.Path(td)
        src_file = sandbox / "source.bin"
        src_file.write_bytes(b"REPRODUCIBLE_BINARY_STREAM_FOR_DETERMINISM_TEST")

        size1, sha1 = create_mirror_archive(
            output_dir=sandbox / "run1",
            archive_name="test.zip",
            file_tuples=[(src_file, "data/source.bin")],
        )

        size2, sha2 = create_mirror_archive(
            output_dir=sandbox / "run2",
            archive_name="test.zip",
            file_tuples=[(src_file, "data/source.bin")],
        )

        assert size1 == size2
        assert sha1 == sha2
        assert len(sha1) == 64


def test_alias_mapping_resolution():
    """Verify alias mapping allows fetching by alias name like demo_video."""
    from scripts.fetch_data import ALIAS_MAP

    assert "demo_video" in ALIAS_MAP
    assert ALIAS_MAP["demo_video"] == "demo_assets"
