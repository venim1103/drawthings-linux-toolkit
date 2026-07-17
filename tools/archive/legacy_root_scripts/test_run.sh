#!/bin/bash
set -euo pipefail

MODEL="${1:-10_e_v1}"

cd /workspaces/drawthings-linux-toolkit
source .venv/bin/activate

DT_HOST=127.0.0.1:7861 \
DT_MODEL="${MODEL}" \
DT_WIDTH=384 \
DT_HEIGHT=704 \
DT_STEPS=8 \
DT_TEST_ONE_FRAME=1 \
DT_FPS_ID=5 \
DT_ALLOW_PREVIEW_FALLBACK=0 \
./tools/dt_generate_video.sh \
	"a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" \
	"blurry, distorted, low quality, artifacts"