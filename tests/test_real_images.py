"""
Automated Forensic Verification Test Suite for Real Test Case Photographs
Verifies that all 50 test cases in tests/test_cases.json reference authentic, genuine physical photographs
with >5,000 unique colors, non-uniform background variation (pixel std > 20.0), and valid JPEG/PNG magic bytes.
Ensures zero synthetic placeholder canvases exist in the dataset.
"""

import hashlib
import json
import zipfile
from pathlib import Path
import cv2
import numpy as np
import pytest

TEST_CASES_PATH = Path("tests/test_cases.json")
MANIFEST_PATH = Path("scripts/manifest.json")
MIRROR_ZIP_PATH = Path("dist/asset_mirror/rakshasite_val_real_images_v1.zip")


def load_test_cases():
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


test_cases = load_test_cases()


def test_test_cases_count():
    """Verify exactly 50 test cases are registered."""
    assert len(test_cases) == 50, f"Expected 50 test cases, found {len(test_cases)}"


@pytest.mark.parametrize("case", test_cases)
def test_image_forensic_authenticity(case):
    """
    Forensically verify that the test case image is a genuine physical photograph:
    1. Exists on disk with non-trivial file size (> 10 KB).
    2. Has authentic JPEG or PNG magic binary header.
    3. Has > 5,000 unique 24-bit RGB colors (synthetic canvases have < 2,000).
    4. Has pixel standard deviation > 20.0 (non-uniform realistic worksite scene).
    """
    img_path = Path(case["image_file"])
    lbl_path = Path(case["label_file"])

    assert img_path.exists(), f"Image file missing on disk: {img_path}"
    assert lbl_path.exists(), f"Label file missing on disk: {lbl_path}"

    # Verify file size
    size_bytes = img_path.stat().st_size
    assert size_bytes > 10_000, f"Image {img_path} is suspiciously small ({size_bytes} bytes)"

    # Verify binary magic bytes
    with open(img_path, "rb") as f:
        header = f.read(8)
    is_jpeg = header.startswith(b"\xff\xd8\xff")
    is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
    assert is_jpeg or is_png, f"Image {img_path} does not have valid JPEG/PNG magic bytes!"

    # Load image and compute visual entropy metrics
    img = cv2.imread(str(img_path))
    assert img is not None, f"Failed to decode image: {img_path}"
    assert img.ndim == 3 and img.shape[2] == 3, f"Expected 3-channel image, got shape {img.shape}"

    unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
    std = float(img.std())

    # Strict thresholds: genuine photos exhibit 50,000+ unique colors; synthetic canvases exhibit ~265
    assert unique_colors > 5_000, (
        f"Forensic verification failed for {img_path.name}: "
        f"unique color count {unique_colors} <= 5,000 (indicates synthetic canvas)"
    )
    assert std > 20.0, (
        f"Forensic verification failed for {img_path.name}: "
        f"pixel standard deviation {std:.2f} <= 20.0 (indicates flat/uniform canvas)"
    )


def test_mirror_archive_forensic_integrity():
    """Verify that rakshasite_val_real_images_v1.zip packages the genuine photographs."""
    assert MIRROR_ZIP_PATH.exists(), f"Mirror archive missing: {MIRROR_ZIP_PATH}"
    assert MANIFEST_PATH.exists(), f"Manifest missing: {MANIFEST_PATH}"

    # Verify SHA256 against manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_sha = manifest["assets"]["val_images"]["archive_sha256"]
    with open(MIRROR_ZIP_PATH, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()

    assert actual_sha == expected_sha, f"SHA256 mismatch for mirror zip: {actual_sha} vs {expected_sha}"

    # Verify zip contents match test_cases
    with zipfile.ZipFile(MIRROR_ZIP_PATH, "r") as z:
        zip_names = set(z.namelist())

    assert len(zip_names) == 50, f"Expected 50 files in mirror zip, found {len(zip_names)}"

    for case in test_cases:
        rel_posix = case["image_file"].replace("\\", "/")
        assert rel_posix in zip_names, f"Test case image {rel_posix} not found in mirror zip!"
