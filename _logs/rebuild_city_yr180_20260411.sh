#!/usr/bin/env bash
# Rebuild city single-state yr180 end-to-end.
# Previous 20260411_1553 run built an empty-bioenvelope blend because
# BV2_DATA_BUNDLE_ROOT was unset, so resolve_bioenvelope_ply_path couldn't
# locate the generated-state PLYs. We now set it explicitly and rebuild.
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
REPO_ROOT="D:/2026 Arboreal Futures/urban-futures"
LOG_PATH="${REPO_ROOT}/_logs/bV2/city_yr180_rebuild_${TS}.log"
STDOUT_LOG="${REPO_ROOT}/_logs/bV2/city_yr180_rebuild_${TS}.stdout.log"

mkdir -p "${REPO_ROOT}/_logs/bV2"

export BV2_SITE="city"
export BV2_MODE="single_state"
export BV2_YEAR="180"
export BV2_CAMERA_NAME="city-yr180-hero-image"
export BV2_SAVE_BLEND="1"
export BV2_RENDER_EXRS="1"
export BV2_UPLOAD_TO_MEDIAFLUX="1"
export BV2_VALIDATE_STRICT="1"
export BV2_SIM_ROOT="4.9"
export BV2_RENDER_TAG="8k64s"
export BV2_RES_X="7680"
export BV2_RES_Y="4320"
export BV2_SAMPLES="64"
export BV2_LOG_PATH="${LOG_PATH}"
export BV2_OUTPUT_TIMESTAMP="${TS}"
export BV2_DATA_BUNDLE_ROOT="D:\\2026 Arboreal Futures\\urban-futures\\_data-refactored\\model-outputs\\generated-states\\4.9\\output"
# Expose repo + venv site-packages so the bV2 headless Blender finds pandas/vtk
export PYTHONPATH="${REPO_ROOT};${REPO_ROOT}/.venv/Lib/site-packages"

BLENDER="C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
SCRIPT="${REPO_ROOT}/_futureSim_refactored/blender/blenderv2/bV2_build_scene.py"

echo "[rebuild_city_yr180] TS=${TS}"
echo "[rebuild_city_yr180] LOG=${LOG_PATH}"

"${BLENDER}" \
    --background \
    --python "${SCRIPT}" \
    >"${STDOUT_LOG}" 2>&1

echo "[rebuild_city_yr180] done TS=${TS}"
