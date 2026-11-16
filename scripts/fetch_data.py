#!/usr/bin/env python3
"""
scripts/fetch_data.py — RakshaSite Asset & Dataset Fetcher & Mirror Manager.

Zero-bloat distribution tool for trained model checkpoints, demo video,
sample photographs, and validation datasets.
Standard library ONLY: argparse, hashlib, json, os, pathlib, shutil, sys, time, urllib.error, urllib.request, zipfile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

CHUNK_SIZE = 64 * 1024  # 64 KB streaming buffer
DETERMINISTIC_ZIP_TIME = (2026, 1, 1, 0, 0, 0)

ALIAS_MAP = {
    "demo_video": "demo_assets",
    "demo": "demo_assets",
    "samples": "demo_assets",
}


def compute_sha256(path: Union[Path, str]) -> str:
    """Compute SHA-256 hex digest of a file in streaming 64KB blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest().lower()


def format_bytes(n_bytes: int) -> str:
    """Format byte counts into human-readable strings."""
    val = float(n_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(val) < 1024.0 or unit == "GB":
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} GB"


def print_progress_bar(
    downloaded: int,
    total: Optional[int],
    start_time: float,
    prefix: str = "",
) -> None:
    """Render a clean standard-library terminal progress indicator."""
    elapsed = max(time.time() - start_time, 0.001)
    rate = downloaded / elapsed
    rate_str = f"{format_bytes(int(rate))}/s"

    if total and total > 0:
        pct = min(100.0, (downloaded / total) * 100.0)
        bar_len = 24
        filled = int(bar_len * downloaded / total)
        bar = "=" * filled + (">" if filled < bar_len else "") + " " * (bar_len - filled - (1 if filled < bar_len else 0))
        msg = f"\r{prefix} [{bar}] {pct:5.1f}% ({format_bytes(downloaded)} / {format_bytes(total)}) {rate_str}"
    else:
        msg = f"\r{prefix} ({format_bytes(downloaded)} downloaded) {rate_str}"

    sys.stdout.write(msg)
    sys.stdout.flush()


def download_stream(
    url: str,
    dest_path: Path,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    quiet: bool = False,
) -> None:
    """
    Download a URL streamingly with progressive hashing and atomic rename.
    Guarantees no corrupted partial files remain on disk if interrupted.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp_{os.getpid()}_{int(time.time())}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RakshaSite-AssetFetcher/1.0 (Mozilla/5.0)"},
    )

    start_time = time.time()
    hasher = hashlib.sha256()
    downloaded = 0

    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", 0)) or expected_size
            while chunk := resp.read(CHUNK_SIZE):
                out_f.write(chunk)
                hasher.update(chunk)
                downloaded += len(chunk)
                if not quiet:
                    print_progress_bar(downloaded, total_size, start_time, prefix=f"  [STREAM] {dest_path.name}")
        if not quiet:
            sys.stdout.write("\n")

        actual_sha256 = hasher.hexdigest().lower()
        if expected_sha256 and actual_sha256 != expected_sha256.lower():
            raise ValueError(
                f"Checksum mismatch for {dest_path.name}!\n"
                f"  Expected: {expected_sha256}\n"
                f"  Actual:   {actual_sha256}"
            )

        # Atomic rename
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)

    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        if not quiet:
            sys.stdout.write("\n")
        raise


def safe_extract_zip(zip_path: Path, extract_root: Path) -> List[Path]:
    """
    Extract a zip file safely, guarding against directory traversal ('Zip Slip').
    Returns a list of extracted Path objects.
    """
    extract_root = extract_root.resolve()
    extracted_paths: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_target = (extract_root / member.filename).resolve()
            if not str(member_target).startswith(str(extract_root)):
                raise PermissionError(
                    f"Security violation: Zip member '{member.filename}' attempts directory traversal outside {extract_root}"
                )
        zf.extractall(extract_root)
        for member in zf.infolist():
            extracted_paths.append((extract_root / member.filename).resolve())
    return extracted_paths


def verify_asset(
    target_or_root: Any,
    asset_spec_or_name: Any = None,
    asset_info: Optional[Dict[str, Any]] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Verify whether an asset is valid on disk.
    Supports dual signatures:
      1) verify_asset(file_path: Path, asset_spec: dict)
      2) verify_asset(repo_root: Path, asset_name: str, asset_info: dict)
    """
    spec: Dict[str, Any]
    target_path: Path

    # Signature 2: verify_asset(repo_root, asset_name, asset_info)
    if isinstance(asset_spec_or_name, str) and isinstance(asset_info, dict):
        repo_root = Path(target_or_root)
        spec = asset_info
        rel = spec.get("target_file") or spec.get("target_path") or spec.get("target_dir")
        if not rel:
            return False, f"Unknown target in spec for '{asset_spec_or_name}'"
        target_path = repo_root / rel
        base_dir = repo_root

    # Signature 1: verify_asset(target_path, asset_spec)
    elif isinstance(asset_spec_or_name, dict):
        spec = asset_spec_or_name
        target_path = Path(target_or_root)
        if not target_path.is_absolute() and base_dir:
            target_path = base_dir / target_path

    # Signature 3: verify_asset(asset_spec, base_dir=...)
    elif isinstance(target_or_root, dict):
        spec = target_or_root
        rel = spec.get("target_file") or spec.get("target_path") or spec.get("target_dir")
        if not rel:
            return False, "No target path specified in spec"
        target_path = (base_dir or Path(".")).resolve() / rel

    else:
        return False, "Invalid arguments passed to verify_asset"

    # Directory asset verification
    is_dir_asset = (
        "target_dir" in spec
        or spec.get("target_type") == "directory"
        or (target_path.exists() and target_path.is_dir())
    )

    if is_dir_asset:
        if not target_path.is_dir():
            return False, f"Missing directory: {target_path}"
        expected_count = spec.get("file_count") or spec.get("total_files")
        if expected_count is not None:
            files = [p for p in target_path.rglob("*.*") if p.is_file()]
            if len(files) < int(expected_count):
                return (
                    False,
                    f"Insufficient files in {target_path}: found {len(files)}, expected {expected_count}",
                )
        return True, f"Verified directory {target_path} ({len(list(target_path.rglob('*.*')))} files)"

    # Single file asset verification
    if not target_path.is_file():
        return False, f"Missing file: {target_path}"

    expected_sha = spec.get("expected_sha256") or spec.get("target_sha256") or spec.get("file_sha256")
    if expected_sha:
        actual_sha = compute_sha256(target_path)
        if actual_sha.lower() != expected_sha.lower():
            return (
                False,
                f"Checksum mismatch for {target_path.name}: expected {expected_sha}, got {actual_sha}",
            )

    expected_size = spec.get("size_bytes") or spec.get("target_size_bytes") or spec.get("file_size_bytes")
    if expected_size is not None:
        actual_size = target_path.stat().st_size
        if actual_size != int(expected_size):
            return (
                False,
                f"Size mismatch for {target_path.name}: expected {expected_size}, got {actual_size}",
            )

    # Check auxiliary files if declared
    aux_files = spec.get("auxiliary_files")
    if aux_files and isinstance(aux_files, list):
        root = base_dir or Path(".")
        for aux in aux_files:
            aux_p = Path(aux)
            if not aux_p.is_absolute():
                aux_p = root / aux_p
            if not aux_p.is_file():
                return False, f"Missing auxiliary file: {aux}"

    return True, f"Verified {target_path} ({format_bytes(target_path.stat().st_size)})"


def resolve_asset_archive(
    repo_root: Path,
    asset_name: str,
    asset_info: Dict[str, Any],
    mirror_dir: Optional[Path],
    download_dir: Path,
    quiet: bool = False,
) -> Path:
    """
    Resolve archive source using 3-tier precedence:
    1. Direct file in mirror directory (Tier 2)
    2. Public upstream URL (Tier 3)
    """
    archive_name = asset_info.get("archive_name")
    if not archive_name:
        raise ValueError(f"No archive_name specified for asset '{asset_name}'")

    expected_archive_sha = asset_info.get("archive_sha256")

    # Tier 2: Local Mirror
    if mirror_dir:
        mirror_path = Path(mirror_dir)
        if not mirror_path.exists():
            raise FileNotFoundError(f"Mirror directory does not exist: {mirror_path}")
        mirror_archive = mirror_path / archive_name
        if mirror_archive.is_file():
            if not quiet:
                print(f"  -> Found in local mirror: {mirror_archive}")
            if expected_archive_sha:
                sha = compute_sha256(mirror_archive)
                if sha.lower() != expected_archive_sha.lower():
                    raise ValueError(
                        f"Mirror archive {archive_name} checksum invalid: {sha} != {expected_archive_sha}"
                    )
            return mirror_archive
        else:
            raise FileNotFoundError(
                f"Mirror directory specified at {mirror_path}, but archive '{archive_name}' was not found in it."
            )

    # Tier 3: Upstream URL
    upstream_url = asset_info.get("upstream_url")
    if not upstream_url:
        raise FileNotFoundError(
            f"No local mirror archive found and no upstream_url configured for '{asset_name}'"
        )

    dest_archive = download_dir / archive_name
    if not quiet:
        print(f"  -> Downloading from upstream: {upstream_url}")
    download_stream(
        upstream_url,
        dest_archive,
        expected_sha256=expected_archive_sha,
        expected_size=asset_info.get("archive_size_bytes"),
        quiet=quiet,
    )
    return dest_archive


def fetch_asset(
    repo_root: Path,
    asset_name: str,
    asset_info: Dict[str, Any],
    mirror_dir: Optional[Path],
    force: bool = False,
    quiet: bool = False,
) -> bool:
    """Orchestrate verification, retrieval, extraction, and validation for one asset."""
    if not quiet:
        print(f"\n[ASSET] Processing: {asset_name} ({asset_info.get('description', '')})")

    # Tier 1: Check if already present & valid
    if not force:
        ok, msg = verify_asset(repo_root, asset_name, asset_info, base_dir=repo_root)
        if ok:
            if not quiet:
                print(f"  [ALREADY SATISFIED] {msg}")
            return True

    # Resolve archive from mirror or upstream
    download_dir = repo_root / "dist" / "downloads"
    archive_path = resolve_asset_archive(
        repo_root=repo_root,
        asset_name=asset_name,
        asset_info=asset_info,
        mirror_dir=mirror_dir,
        download_dir=download_dir,
        quiet=quiet,
    )

    # Safe extraction
    if not quiet:
        print(f"  -> Extracting {archive_path.name} to {repo_root}...")
    safe_extract_zip(archive_path, repo_root)

    # Handle flat extracted file if necessary
    target_rel = asset_info.get("target_file") or asset_info.get("target_path")
    if target_rel:
        target_path = repo_root / target_rel
        if not target_path.exists():
            flat_candidate = repo_root / Path(target_rel).name
            if flat_candidate.is_file():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(flat_candidate), str(target_path))

    # Post-extraction verification
    ok, msg = verify_asset(repo_root, asset_name, asset_info, base_dir=repo_root)
    if not ok:
        raise RuntimeError(f"Post-extraction verification failed for {asset_name}: {msg}")
    if not quiet:
        print(f"  [SUCCESS] {msg}")
    return True


def fetch_all_assets(
    manifest_data: Dict[str, Any],
    base_dir: Path,
    mirror_dir: Optional[Path] = None,
    quiet: bool = False,
    force: bool = False,
    asset_filter: Optional[List[str]] = None,
) -> bool:
    """
    Fetch and verify all or filtered assets declared in manifest_data.
    Returns True if all required/requested assets succeed, False otherwise.
    """
    assets: Dict[str, Any] = manifest_data.get("assets", {})
    if not assets:
        if not quiet:
            print("[WARN] Manifest contains no assets.", file=sys.stderr)
        return True

    target_keys: List[str]
    if asset_filter and asset_filter != ["all"]:
        # Resolve any aliases
        resolved_keys: List[str] = []
        for k in asset_filter:
            mapped = ALIAS_MAP.get(k, k)
            if mapped in assets:
                resolved_keys.append(mapped)
            else:
                if not quiet:
                    print(f"[ERROR] Unknown asset requested: {k}", file=sys.stderr)
                return False
        target_keys = resolved_keys
    else:
        # Default: process all required non-optional assets
        target_keys = [k for k, v in assets.items() if not v.get("optional", False)]

    all_success = True
    for key in target_keys:
        info = assets[key]
        try:
            fetch_asset(
                repo_root=base_dir,
                asset_name=key,
                asset_info=info,
                mirror_dir=mirror_dir,
                force=force,
                quiet=quiet,
            )
        except Exception as exc:
            if not quiet:
                print(f"  [ERROR] Failed to fetch {key}: {exc}", file=sys.stderr)
            all_success = False

    return all_success


def create_mirror_archive(
    output_dir: Path,
    archive_name: str,
    file_tuples: List[Tuple[Path, str]],
) -> Tuple[int, str]:
    """
    Create a zip archive with deterministic timestamps (2026-01-01 00:00:00)
    to guarantee bit-for-bit identical SHA-256 hashes across different systems.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_zip = output_dir / archive_name

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, arc_name in sorted(file_tuples, key=lambda x: str(x[1]).replace("\\", "/")):
            arc_posix = str(arc_name).replace("\\", "/")
            zinfo = zipfile.ZipInfo(arc_posix, DETERMINISTIC_ZIP_TIME)
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zinfo.external_attr = 0o644 << 16  # standard file attributes
            with open(file_path, "rb") as fp:
                zf.writestr(zinfo, fp.read())

    size = out_zip.stat().st_size
    sha = compute_sha256(out_zip)
    return size, sha


def create_mirror_routine(
    repo_root: Path,
    mirror_dir: Path,
    manifest_path: Path,
    quiet: bool = False,
) -> None:
    """
    Inspect workspace assets, package them into standard zip archives,
    calculate SHA-256 hashes, and output scripts/manifest.json.
    """
    if not quiet:
        print(f"\n[MIRROR-CREATE] Generating mirror distribution archives in: {mirror_dir}")
    mirror_dir.mkdir(parents=True, exist_ok=True)

    # 1. Weights
    pt_path = repo_root / "models" / "best.pt"
    if not pt_path.is_file():
        raise FileNotFoundError(f"Missing required model weights at: {pt_path}")
    w_size, w_sha = create_mirror_archive(
        mirror_dir,
        "rakshasite_weights_v1.zip",
        [(pt_path, "models/best.pt")],
    )
    if not quiet:
        print(f"  [CREATED] rakshasite_weights_v1.zip ({format_bytes(w_size)}) SHA256: {w_sha}")

    # 2. Demo Assets
    mp4_path = repo_root / "data" / "raw" / "hardhat_demo.mp4"
    if not mp4_path.is_file():
        raise FileNotFoundError(f"Missing demo video at: {mp4_path}")
    demo_files = [(mp4_path, "data/raw/hardhat_demo.mp4")]
    samples_dir = repo_root / "data" / "raw" / "samples"
    sample_rel_paths: List[str] = []
    if samples_dir.is_dir():
        for p in samples_dir.glob("*.*"):
            if p.is_file():
                rel = f"data/raw/samples/{p.name}"
                demo_files.append((p, rel))
                sample_rel_paths.append(rel)
    d_size, d_sha = create_mirror_archive(mirror_dir, "rakshasite_demo_assets_v1.zip", demo_files)
    if not quiet:
        print(f"  [CREATED] rakshasite_demo_assets_v1.zip ({format_bytes(d_size)}) SHA256: {d_sha}")

    # 3. Validation Real Images
    val_img_dir = repo_root / "data" / "unified" / "val" / "images"
    val_files = [(p, f"data/unified/val/images/{p.name}") for p in val_img_dir.glob("*.*") if p.is_file()]
    if not val_files:
        raise FileNotFoundError(f"No validation images found in: {val_img_dir}")
    v_size, v_sha = create_mirror_archive(mirror_dir, "rakshasite_val_real_images_v1.zip", val_files)
    if not quiet:
        print(f"  [CREATED] rakshasite_val_real_images_v1.zip ({format_bytes(v_size)}, {len(val_files)} files) SHA256: {v_sha}")

    # Assemble manifest data
    pt_sha = compute_sha256(pt_path)
    mp4_sha = compute_sha256(mp4_path)

    manifest_data: Dict[str, Any] = {
        "version": "1.0.0",
        "description": "RakshaSite asset and weights distribution manifest",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assets": {
            "weights": {
                "description": "Trained YOLOv8n PPE Detection Model Checkpoint (100 Epochs, 10 Classes)",
                "archive_name": "rakshasite_weights_v1.zip",
                "archive_sha256": w_sha,
                "archive_size_bytes": w_size,
                "target_file": "models/best.pt",
                "target_sha256": pt_sha,
                "target_size_bytes": pt_path.stat().st_size,
                "expected_sha256": pt_sha,
                "size_bytes": pt_path.stat().st_size,
                "upstream_url": "https://github.com/Chaitanya1914/RakshaSite/releases/download/v1.0.0/rakshasite_weights_v1.zip",
                "required": True,
            },
            "demo_assets": {
                "description": "Real Construction PPE Video Stream and Sample Still Photographs",
                "archive_name": "rakshasite_demo_assets_v1.zip",
                "archive_sha256": d_sha,
                "archive_size_bytes": d_size,
                "target_file": "data/raw/hardhat_demo.mp4",
                "target_sha256": mp4_sha,
                "target_size_bytes": mp4_path.stat().st_size,
                "expected_sha256": mp4_sha,
                "size_bytes": mp4_path.stat().st_size,
                "auxiliary_files": sorted(sample_rel_paths),
                "upstream_url": "https://github.com/Chaitanya1914/RakshaSite/releases/download/v1.0.0/rakshasite_demo_assets_v1.zip",
                "required": True,
            },
            "val_images": {
                "description": "Validation Photographic Test Images Matching tests/test_cases.json",
                "archive_name": "rakshasite_val_real_images_v1.zip",
                "archive_sha256": v_sha,
                "archive_size_bytes": v_size,
                "target_dir": "data/unified/val/images",
                "file_count": len(val_files),
                "total_files": len(val_files),
                "size_bytes": v_size,
                "upstream_url": "https://github.com/Chaitanya1914/RakshaSite/releases/download/v1.0.0/rakshasite_val_real_images_v1.zip",
                "required": True,
            },
            "construction_ppe_benchmark": {
                "description": "Ultralytics 1000-Image Construction-PPE Benchmark Dataset (CC BY 4.0)",
                "archive_name": "construction-ppe.zip",
                "target_dir": "data/raw/construction-ppe",
                "upstream_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip",
                "optional": True,
                "required": False,
            },
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    if not quiet:
        print(f"  [SAVED] Manifest written to: {manifest_path}")

    # Mirror copy of manifest
    with open(mirror_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    if not quiet:
        print(f"  [SAVED] Mirror manifest copy written to: {mirror_dir / 'manifest.json'}")


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for scripts/fetch_data.py."""
    parser = argparse.ArgumentParser(
        description="RakshaSite Zero-Bloat Asset Fetcher & Mirror Manager"
    )
    parser.add_argument(
        "--mirror",
        type=Path,
        default=None,
        help="Path to local mirror directory containing .zip archives",
    )
    parser.add_argument(
        "--create-mirror",
        type=Path,
        default=None,
        help="Package workspace assets into mirror archives and generate manifest.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing assets against manifest.json without downloading or extracting",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Custom manifest.json path (default: scripts/manifest.json)",
    )
    parser.add_argument(
        "--asset",
        default="all",
        help="Target asset name (default: all)",
    )
    parser.add_argument(
        "--assets",
        default=None,
        help="Comma-separated list of target asset names (e.g. weights,demo_assets)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch/re-extract even if files already exist and match hashes",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error logging output",
    )

    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = args.manifest or (repo_root / "scripts" / "manifest.json")

    # Mode 1: Create Mirror
    if args.create_mirror:
        try:
            create_mirror_routine(repo_root, args.create_mirror, manifest_path, quiet=args.quiet)
            if not args.quiet:
                print("\n[SUCCESS] Mirror creation complete.")
            return 0
        except Exception as e:
            print(f"[ERROR] Failed to create mirror: {e}", file=sys.stderr)
            return 2

    # Verify manifest exists
    if not manifest_path.is_file():
        print(f"[ERROR] Manifest file not found: {manifest_path}", file=sys.stderr)
        return 3

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse manifest JSON: {e}", file=sys.stderr)
        return 3

    assets: Dict[str, Any] = manifest.get("assets", {})

    # Determine requested assets
    raw_filter: str = args.assets if args.assets is not None else args.asset
    if raw_filter == "all":
        filter_list = None
    else:
        filter_list = [s.strip() for s in raw_filter.split(",") if s.strip()]

    # Mode 2: Verify Only
    if args.verify_only:
        targets_to_check = filter_list if filter_list else list(assets.keys())
        if not args.quiet:
            print(f"\n[AUDIT] Auditing assets against manifest {manifest_path.name}...")
        all_ok = True
        for raw_name in targets_to_check:
            name = ALIAS_MAP.get(raw_name, raw_name)
            if name not in assets:
                print(f"  [FAIL] Unknown asset: {raw_name}", file=sys.stderr)
                all_ok = False
                continue
            info = assets[name]
            ok, msg = verify_asset(repo_root, name, info, base_dir=repo_root)
            status = "[OK]  " if ok else "[FAIL]"
            if not args.quiet or not ok:
                print(f"  {status} {name:20s}: {msg}")
            if not ok and not info.get("optional", False):
                all_ok = False
        return 0 if all_ok else 1

    # Mode 3: Fetch Assets
    mirror_dir = args.mirror or (
        Path(os.environ["RAKSHASITE_MIRROR_DIR"])
        if "RAKSHASITE_MIRROR_DIR" in os.environ
        else None
    )

    if mirror_dir and not Path(mirror_dir).exists():
        print(f"[ERROR] Mirror directory does not exist: {mirror_dir}", file=sys.stderr)
        return 2

    success = fetch_all_assets(
        manifest_data=manifest,
        base_dir=repo_root,
        mirror_dir=mirror_dir,
        quiet=args.quiet,
        force=args.force,
        asset_filter=filter_list,
    )

    if success:
        if not args.quiet:
            print("\n[DONE] All requested assets are verified and ready for RakshaSite execution.")
        return 0
    else:
        print("\n[FAIL] One or more assets failed to resolve.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
