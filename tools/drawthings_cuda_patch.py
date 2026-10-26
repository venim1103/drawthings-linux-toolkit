#!/usr/bin/env python3
"""Strict, isolated CUDA patch generation/application; never applies CPU patches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "DRAW_THINGS_PATCH" / "cuda"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def patch_state(source, manifest):
    expected = manifest["files"]
    if all(digest(source / relative) == digest(BUNDLE / "ccv" / relative) for relative in expected):
        return "applied"
    if all(digest(source / relative) == baseline for relative, baseline in expected.items()):
        return "pristine"
    raise ValueError("Patch files are mixed, modified, or from an unsupported revision; refusing fallback")


def apply_patch(source, manifest):
    state = patch_state(source, manifest)
    if state == "applied":
        return state
    environment = dict(os.environ)
    environment["GIT_CEILING_DIRECTORIES"] = str(source.parent)
    patch = BUNDLE / manifest["patch"]
    for options in (("--check",), ()):
        subprocess.run(["git", "apply", *options, str(patch)], cwd=source,
                       env=environment, check=True, capture_output=True, text=True)
    if patch_state(source, manifest) != "applied":
        raise ValueError("Applied files do not match CUDA snapshots")
    return "applied"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "apply", "verify", "generate"))
    parser.add_argument("--ccv-source", required=True, type=Path)
    args = parser.parse_args()
    source = args.ccv_source.resolve()
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    if not source.is_dir():
        raise ValueError("CCV source directory does not exist")
    state = patch_state(source, manifest)
    if args.action == "generate":
        if state != "pristine":
            raise ValueError("Regeneration requires pristine pinned base files")
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "a").mkdir()
            (workspace / "b").mkdir()
            for relative, baseline in manifest["files"].items():
                destination = workspace / "b" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(BUNDLE / "ccv" / relative, destination)
                if baseline is not None:
                    original = workspace / "a" / relative
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / relative, original)
            result = subprocess.run(["git", "diff", "--no-index", "--src-prefix=", "--dst-prefix=", "a", "b"],
                                    cwd=workspace, capture_output=True, check=False)
            if result.returncode != 1 or not result.stdout:
                raise ValueError("Failed to generate nonempty unified CUDA diff")
            output = BUNDLE / manifest["patch"]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(result.stdout)
    elif args.action == "apply":
        state = apply_patch(source, manifest)
    elif args.action == "verify":
        with tempfile.TemporaryDirectory() as temporary:
            clean = Path(temporary)
            for relative, baseline in manifest["files"].items():
                if baseline is not None:
                    if state != "pristine":
                        raise ValueError("Verification requires pristine pinned base files")
                    destination = clean / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / relative, destination)
            apply_patch(clean, manifest)
            before = {relative: digest(clean / relative) for relative in manifest["files"]}
            apply_patch(clean, manifest)
            after = {relative: digest(clean / relative) for relative in manifest["files"]}
            if before != after:
                raise ValueError("CUDA patch application is not idempotent")
        state = "verified_strict_and_idempotent"
    print(json.dumps({"result": "PASS", "profile": manifest["profile"], "state": state}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"error: {error}") from error