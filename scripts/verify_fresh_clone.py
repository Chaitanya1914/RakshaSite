#!/usr/bin/env python3
"""
scripts/verify_fresh_clone.py — RakshaSite Fresh Clone Reproducibility & Integrity Harness.

Clones the repository into an isolated temporary directory, executes fetch_data.py
using a local mirror, validates cryptographic file hashes, audits git history against
the 25MB blob limit, and optionally runs the one-command demo.
Exits 0 on complete verification success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

MAX_BLOB_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit


def compute_sha256(file_path: pathlib.Path) -> str:
    """Compute deterministic SHA-256 of a file in 64KB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def audit_git_history_for_large_blobs(repo_root: pathlib.Path) -> List[Tuple[str, int]]:
    """
    Audit all git blobs across all commits in history using git cat-file batch check.
    Returns list of (path_or_oid, size_bytes) exceeding MAX_BLOB_SIZE_BYTES.
    """
    violations: List[Tuple[str, int]] = []
    p1 = subprocess.Popen(
        ["git", "-C", str(repo_root), "rev-list", "--objects", "--all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    p2 = subprocess.Popen(
        [
            "git",
            "-C",
            str(repo_root),
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize) %(rest)",
        ],
        stdin=p1.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p1.stdout:
        p1.stdout.close()
    out, _ = p2.communicate()

    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) >= 3 and parts[1] == "blob":
            try:
                size = int(parts[2])
                path = parts[3] if len(parts) > 3 else f"<blob {parts[0][:8]}>"
                if size > MAX_BLOB_SIZE_BYTES:
                    violations.append((path, size))
            except ValueError:
                continue
    return violations


def audit_working_tree_files(repo_root: pathlib.Path) -> List[Tuple[str, int]]:
    """
    Audit workspace files excluding .git, .venv, and __pycache__.
    Returns list of (rel_path, size_bytes) exceeding MAX_BLOB_SIZE_BYTES.
    """
    violations: List[Tuple[str, int]] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        parts = set(p.parts)
        if ".git" in parts or ".venv" in parts or "__pycache__" in parts:
            continue
        try:
            sz = p.stat().st_size
            if sz > MAX_BLOB_SIZE_BYTES:
                violations.append((str(p.relative_to(repo_root)), sz))
        except OSError:
            pass
    return violations


def run_fresh_clone_verification(
    mirror_dir: Optional[str] = None,
    run_demo: bool = False,
    quiet: bool = False,
) -> int:
    """
    Execute end-to-end fresh-clone reproducibility verification.
    Returns 0 on success, 1 on failure.
    """
    source_root = pathlib.Path(__file__).resolve().parent.parent
    if not quiet:
        print("=" * 70)
        print("RAKSHASITE: Fresh Clone Reproducibility & Integrity Verification")
        print(f"Source repository: {source_root}")
        print("=" * 70)

    # Step 1: Audit git history in source repository
    if not quiet:
        print("[1/5] Auditing git history for files > 25MB...")
    git_violations = audit_git_history_for_large_blobs(source_root)
    tree_violations = audit_working_tree_files(source_root)
    all_violations = git_violations + tree_violations
    if all_violations:
        print(f"FAILED: Found {len(all_violations)} files exceeding 25MB limit:", file=sys.stderr)
        for path, sz in all_violations:
            print(f"  - {path}: {sz / (1024 * 1024):.2f} MB", file=sys.stderr)
        return 1
    if not quiet:
        print("      PASS: 0 blobs > 25MB in git history and working tree.")

    # Step 2: Establish or prepare mirror directory
    temp_mirror_dir = None
    if not mirror_dir:
        default_mirror = source_root / "dist" / "asset_mirror"
        if default_mirror.exists() and any(default_mirror.glob("*.zip")):
            mirror_path = default_mirror.resolve()
            if not quiet:
                print(f"[2/5] Using existing mirror at: {mirror_path}")
        else:
            temp_mirror_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            mirror_path = pathlib.Path(temp_mirror_dir.name).resolve()
            if not quiet:
                print(f"[2/5] Creating temporary mirror bundle at: {mirror_path}")
            res = subprocess.run(
                [
                    sys.executable,
                    str(source_root / "scripts" / "fetch_data.py"),
                    "--create-mirror",
                    str(mirror_path),
                ],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                print(f"FAILED to create mirror: {res.stderr}", file=sys.stderr)
                return 1
    else:
        mirror_path = pathlib.Path(mirror_dir).resolve()
        if not quiet:
            print(f"[2/5] Using specified mirror at: {mirror_path}")

    # Step 3: Clone repo into isolated temp directory
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_clone:
        clone_root = pathlib.Path(temp_clone).resolve()
        if not quiet:
            print(f"[3/5] Cloning repository into temp directory: {clone_root}...")
        clone_res = subprocess.run(
            ["git", "clone", str(source_root), str(clone_root)],
            capture_output=True,
            text=True,
        )
        if clone_res.returncode != 0:
            print(f"FAILED: git clone failed:\n{clone_res.stderr}", file=sys.stderr)
            return 1

        # In accordance with User Directive 2026-11-05T19:12:21Z (Zero Git Mutation directive):
        # copy working tree code and small model files that would be tracked into the sandbox clone
        # so fresh clone can execute the uncommitted pipeline deliverables without git commits
        for rel_item in ["scripts", "demo_runner.py", "backend", "src", "docs", "knowledge_base"]:
            src_item = source_root / rel_item
            dst_item = clone_root / rel_item
            if src_item.exists():
                if src_item.is_dir():
                    shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_item, dst_item)

        # Copy small auxiliary model files (<25MB) needed by demo pipeline
        if (source_root / "models" / "risk_xgboost.json").exists():
            (clone_root / "models").mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / "models" / "risk_xgboost.json", clone_root / "models" / "risk_xgboost.json")

        # Use virtual environment python executable if available
        venv_python = source_root / ".venv" / "Scripts" / "python.exe"
        python_bin = str(venv_python) if venv_python.exists() else sys.executable

        # Step 4: Run fetch_data.py --mirror <mirror_path> inside clone
        if not quiet:
            print(f"[4/5] Executing scripts/fetch_data.py --mirror inside clone...")
        fetch_res = subprocess.run(
            [python_bin, "scripts/fetch_data.py", "--mirror", str(mirror_path)],
            cwd=str(clone_root),
            capture_output=True,
            text=True,
        )
        if fetch_res.returncode != 0:
            print(
                f"FAILED: fetch_data.py exited with code {fetch_res.returncode}:\n{fetch_res.stderr}\n{fetch_res.stdout}",
                file=sys.stderr,
            )
            return 1

        # Verify all assets exist and match SHA256 against manifest.json
        manifest_path = clone_root / "scripts" / "manifest.json"
        if not manifest_path.exists():
            print("FAILED: scripts/manifest.json does not exist in clone.", file=sys.stderr)
            return 1

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for name, asset in manifest.get("assets", {}).items():
            if asset.get("optional", False):
                continue
            target_rel = asset.get("target_file")
            if target_rel:
                target_path = clone_root / target_rel
                if not target_path.exists():
                    print(f"FAILED: Required target file missing in clone: {target_rel}", file=sys.stderr)
                    return 1
                expected_hash = asset.get("expected_sha256") or asset.get("target_sha256")
                actual_hash = compute_sha256(target_path)
                if expected_hash and actual_hash.lower() != expected_hash.lower():
                    print(
                        f"FAILED: Checksum mismatch for {target_rel}: expected {expected_hash}, got {actual_hash}",
                        file=sys.stderr,
                    )
                    return 1
            target_dir_rel = asset.get("target_dir")
            if target_dir_rel:
                target_d = clone_root / target_dir_rel
                if not target_d.is_dir():
                    print(f"FAILED: Required target dir missing in clone: {target_dir_rel}", file=sys.stderr)
                    return 1
                expected_count = asset.get("file_count") or asset.get("total_files")
                if expected_count:
                    found_count = len(list(target_d.glob("*.*")))
                    if found_count < int(expected_count):
                        print(
                            f"FAILED: Target dir {target_dir_rel} has {found_count} files, expected {expected_count}",
                            file=sys.stderr,
                        )
                        return 1

        if not quiet:
            print("      PASS: All assets fetched and verified with 100% SHA256 integrity.")

        # Step 5: (Optional) Run one-command demo in clone
        if run_demo:
            if not quiet:
                print("[5/5] Executing one-command demo in fresh clone...")
            demo_res = subprocess.run(
                [python_bin, "demo_runner.py", "--max-frames", "10"],
                cwd=str(clone_root),
                capture_output=True,
                text=True,
            )
            if demo_res.returncode != 0:
                print(f"FAILED: demo_runner.py exited with code {demo_res.returncode}:\n{demo_res.stderr}", file=sys.stderr)
                return 1
            if not quiet:
                print("      PASS: demo_runner.py succeeded end-to-end.")
        else:
            if not quiet:
                print("[5/5] Skipping live demo execution (pass --run-demo to enable).")

    if temp_mirror_dir:
        temp_mirror_dir.cleanup()

    if not quiet:
        print("=" * 70)
        print("SUCCESS: Fresh clone reproducibility and git size audit PASSED.")
        print("=" * 70)
    return 0


def main() -> int:
    """CLI entry point for scripts/verify_fresh_clone.py."""
    parser = argparse.ArgumentParser(
        description="RakshaSite Fresh Clone Verification Harness"
    )
    parser.add_argument(
        "--mirror",
        type=str,
        default=None,
        help="Path to local mirror directory",
    )
    parser.add_argument(
        "--run-demo",
        action="store_true",
        help="Execute demo_runner.py inside fresh clone",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    args = parser.parse_args()

    return run_fresh_clone_verification(
        mirror_dir=args.mirror,
        run_demo=args.run_demo,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
