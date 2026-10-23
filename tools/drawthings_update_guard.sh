#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

VERIFY_SCRIPT="$SCRIPT_DIR/verify_drawthings_patch_bundle.sh"
SAFETY_SCRIPT="$SCRIPT_DIR/strict_drawthings_safety_check.sh"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") snapshot [workspace_root] [output_directory]
  $(basename "$0") before [workspace_root]
  $(basename "$0") after [workspace_root] [strict_safety_check options]

Commands:
  snapshot Save a read-only baseline, including rollback binaries and local diffs.
  before   Run strict patch verify and save an update baseline snapshot.
  after    Run full strict safety gate after upstream update/rebase.

Examples:
  bash tools/drawthings_update_guard.sh snapshot

  # Before updating draw-things-community
  bash tools/drawthings_update_guard.sh before

  # After upstream update/rebase
  bash tools/drawthings_update_guard.sh after --allow-dirty

  # Faster after-check (skip smoke build)
  bash tools/drawthings_update_guard.sh after --allow-dirty --skip-build
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

COMMAND="$1"
shift

ROOT="$DEFAULT_ROOT"

if [[ $# -gt 0 && "$1" != --* ]]; then
  ROOT="$1"
  shift
fi

PATCH_ROOT="$ROOT/DRAW_THINGS_PATCH"
PATCHES_DIR="$PATCH_ROOT/patches"
REPO_ROOT="$ROOT/draw-things-community"
BASELINE_FILE="$PATCHES_DIR/update_baseline_latest.env"

if [[ ! -d "$ROOT" ]]; then
  echo "error: workspace root not found: $ROOT" >&2
  exit 1
fi
if [[ ! -d "$PATCH_ROOT" ]]; then
  echo "error: patch root not found: $PATCH_ROOT" >&2
  exit 1
fi
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "error: draw-things-community repo not found: $REPO_ROOT" >&2
  exit 1
fi

save_snapshot() {
  local python_bin="${DT_PYTHON:-$ROOT/.venv/bin/python}"
  if [[ ! -x "$python_bin" ]]; then
  echo "error: Python interpreter not found: $python_bin" >&2
  exit 1
  fi
  if [[ $# -gt 1 ]]; then
  echo "error: snapshot accepts at most one output directory" >&2
  exit 1
  fi

  "$python_bin" - "$ROOT" "${1:-}" <<'PY'
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
timestamp = datetime.now(timezone.utc)
destination = (
  Path(sys.argv[2]).resolve() if sys.argv[2] else
  root / "output" / "upgrades" / "baselines" / timestamp.strftime("%Y%m%dT%H%M%S_%fZ")
)
destination.mkdir(parents=True, exist_ok=False)

def command(arguments):
  return subprocess.run(arguments, check=True, capture_output=True).stdout

def copy_file(source, relative):
  if not source.is_file():
    return
  target = destination / relative
  target.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(source, target)

repositories = {}
community = root / "draw-things-community"
for name, repository in (
  ("toolkit", root),
  ("community", community),
  ("s4nnc", community / ".build/checkouts/s4nnc"),
  ("ccv", community / ".build/checkouts/ccv"),
):
  if not (repository / ".git").exists():
    raise RuntimeError(f"Required repository missing: {repository}")
  prefix = ["git", "-C", str(repository)]
  status = command(prefix + ["status", "--porcelain=v1", "--untracked-files=all"])
  head = command(prefix + ["rev-parse", "HEAD"]).decode().strip()
  repositories[name] = {"path": str(repository), "head": head, "status": status.decode()}
  source_directory = destination / "source" / name
  source_directory.mkdir(parents=True)
  for label, arguments in (
    ("working.patch", ["diff", "--binary"]),
    ("staged.patch", ["diff", "--cached", "--binary"]),
  ):
    (source_directory / label).write_bytes(command(prefix + arguments))
  untracked = command(prefix + ["ls-files", "--others", "--exclude-standard", "-z"])
  for relative in filter(None, untracked.decode().split("\0")):
    copy_file(repository / relative, Path("source") / name / "untracked" / relative)

shutil.copytree(root / "DRAW_THINGS_PATCH", destination / "patch-bundle")
for filename in ("custom.json", "custom_lora.json"):
  copy_file(root / "dt-models" / filename, Path("catalogs") / filename)
for filename in ("model-converter", "model-quantizer", "lora-converter"):
  binary = community / ".build/release" / filename
  if not binary.is_file():
    raise RuntimeError(f"Required rollback binary missing: {binary}")
  copy_file(binary, Path("bin") / filename)
for filename in ("gRPCServerCLI", "drawthings-grpc"):
  binary = shutil.which(filename)
  if binary is None:
    raise RuntimeError(f"Required runtime binary missing: {filename}")
  copy_file(Path(binary), Path("bin") / filename)
for relative in (
  "Libraries/GRPC/Models/Sources/imageService/imageService.proto",
  "Libraries/DataModels/Sources/config.fbs",
):
  copy_file(community / relative, Path("schemas") / relative)
for configuration in list((root / "output").glob("q6p_canary_*/config*")):
  if configuration.suffix in {".bin", ".json"}:
    copy_file(configuration, Path("configurations") / configuration.relative_to(root / "output"))

models = []
for model in sorted((root / "dt-models").iterdir()):
  if model.is_file() and (".ckpt" in model.name or model.suffix == ".safetensors"):
    metadata = model.stat()
    models.append({"file": model.name, "bytes": metadata.st_size, "mtime_ns": metadata.st_mtime_ns})

files = []
for saved in sorted(destination.rglob("*")):
  if saved.is_file():
    digest = hashlib.sha256()
    with saved.open("rb") as stream:
      for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    files.append({"file": str(saved.relative_to(destination)), "bytes": saved.stat().st_size, "sha256": digest.hexdigest()})

manifest = {
  "created_at": timestamp.isoformat(),
  "workspace": str(root),
  "repositories": repositories,
  "declared_image": os.environ.get("DRAWTHINGS_IMAGE"),
  "container_image_digest": os.environ.get("DRAWTHINGS_IMAGE_DIGEST"),
  "swift_version": command(["swift", "--version"]).decode().strip(),
  "model_weights_copied": False,
  "models": models,
  "files": files,
}
(destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
print(f"Saved read-only upgrade snapshot: {destination}")
print(f"Verified file hashes: {len(files)}; model weights left in place: {len(models)}")
PY
}

save_baseline() {
  mkdir -p "$PATCHES_DIR"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  local toolkit_head dt_head s4nnc_head ccv_head dt_branch
  toolkit_head="$(git -C "$ROOT" rev-parse HEAD)"
  dt_head="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  dt_branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"

  s4nnc_head="unknown"
  if [[ -d "$REPO_ROOT/.build/checkouts/s4nnc/.git" ]]; then
    s4nnc_head="$(git -C "$REPO_ROOT/.build/checkouts/s4nnc" rev-parse HEAD)"
  fi

  ccv_head="unknown"
  if [[ -d "$REPO_ROOT/.build/checkouts/ccv/.git" ]]; then
    ccv_head="$(git -C "$REPO_ROOT/.build/checkouts/ccv" rev-parse HEAD)"
  fi

  cat > "$BASELINE_FILE" <<EOF
timestamp="$timestamp"
toolkit_head="$toolkit_head"
draw_things_branch="$dt_branch"
draw_things_head="$dt_head"
s4nnc_head="$s4nnc_head"
ccv_head="$ccv_head"
EOF

  echo "==> Saved baseline: $BASELINE_FILE"
  echo "    timestamp:          $timestamp"
  echo "    draw-things branch: $dt_branch"
  echo "    draw-things head:   $dt_head"
}

show_baseline() {
  if [[ -f "$BASELINE_FILE" ]]; then
    echo "==> Existing baseline"
    # shellcheck disable=SC1090
    source "$BASELINE_FILE"
    echo "    timestamp:          ${timestamp:-unknown}"
    echo "    draw-things branch: ${draw_things_branch:-unknown}"
    echo "    draw-things head:   ${draw_things_head:-unknown}"
    echo "    s4nnc head:         ${s4nnc_head:-unknown}"
    echo "    ccv head:           ${ccv_head:-unknown}"
  else
    echo "==> No prior baseline found at $BASELINE_FILE"
  fi
}

case "$COMMAND" in
  snapshot)
    save_snapshot "$@"
    ;;
  before)
    echo "==> BEFORE update guard"
    bash "$VERIFY_SCRIPT" "$ROOT"
    save_baseline
    echo "==> Ready for upstream update/rebase."
    ;;
  after)
    echo "==> AFTER update guard"
    show_baseline
    bash "$SAFETY_SCRIPT" "$ROOT" "$@"
    echo "==> After-update safety checks passed."
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "error: unknown command '$COMMAND'" >&2
    usage
    exit 1
    ;;
esac
