#!/bin/bash
source .venv/bin/activate && DT_HOST=127.0.0.1:7861 DT_MODEL=$1 DT_WIDTH=384 DT_HEIGHT=704 DT_STEPS=8 DT_TEST_ONE_FRAME=1 DT_FPS_ID=5 DT_GUIDANCE=1.0 DT_HIRES_FIX=false ./tools/dt_generate_video.sh "a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" "blurry, distorted, low quality, artifacts"
