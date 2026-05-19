# Repo Migration Guide

Last updated: 2026-05-19

This guide helps you create a clean Git repository from this workspace that includes devcontainer setup, patching system, helper tools, and Python workflows.

## 1) Suggested new repo scope

Include these paths first:

- .devcontainer/
- tools/
- DRAW_THINGS_PATCH/
- requirements.txt
- requirements-cuda.txt
- requirements-drawthings-tools.txt
- WORKSPACE_MANUAL.md
- REPO_MIGRATION_GUIDE.md

Do not commit by default:

- draw-things-community/.build/
- .venv/
- dt-models/
- output/
- ComfyUI/temp artifacts
- __pycache__/

## 2) Suggested repository layout

    your-repo/
      .devcontainer/
      tools/
      DRAW_THINGS_PATCH/
        patches/
      docs/
      requirements.txt
      requirements-cuda.txt
      requirements-drawthings-tools.txt
      WORKSPACE_MANUAL.md
      REPO_MIGRATION_GUIDE.md
      .gitignore
      README.md

## 3) Suggested .gitignore baseline

    .venv/
    __pycache__/
    *.pyc
    .cache/
    output/
    dt-models/
    draw-things-community/.build/
    draw-things-community/.swiftpm/

Adjust as needed if you decide to version some generated files.

## 4) Bootstrap steps for the new repo

1. Create repo and copy selected files.
2. Commit baseline docs and scripts.
3. Validate patching flow:

    bash tools/apply_drawthings_quant_patch.sh

4. Validate patch regeneration flow:

    bash tools/generate_drawthings_quant_patches.sh

5. Rebuild quantizer in draw-things-community clone.
6. Run one end-to-end test command and record output in docs.

## 5) Versioning strategy for patching

Recommended:

- Keep both formats:
  - Unified patches under DRAW_THINGS_PATCH/patches
  - Snapshot fallback files under DRAW_THINGS_PATCH
- Regenerate patch files whenever patched sources change.
- Keep patch updates in dedicated commits, separate from unrelated script changes.

## 6) Suggested commit plan

Commit 1:
- Add devcontainer, requirements, and base tooling scripts.

Commit 2:
- Add patching system and DRAW_THINGS_PATCH assets.

Commit 3:
- Add model and generation helper scripts.

Commit 4:
- Add manuals and troubleshooting docs.

Commit 5:
- Finalize metadata, release notes, and repository hygiene.

## 7) Optional hardening before public release

- Add CI checks:
  - Shell syntax checks for tools/*.sh
  - Python import smoke checks for tools/*.py
- Add script-level help checks in CI.
- Add a single bootstrap command script for new contributors.
- Add license and contribution notes.

## 8) Practical note about draw-things-community dependency

This workspace uses a local clone of draw-things-community and applies local patches into that tree and selected checkouts.

If you want maximal reproducibility in your own repo, choose one of these:

- Keep current patch-apply model (fast to maintain).
- Fork upstream dependencies and pin to your fork revisions (more stable long-term).

The current scripts support the first model out of the box.
