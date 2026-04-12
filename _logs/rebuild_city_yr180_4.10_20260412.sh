#!/usr/bin/env bash
# Rebuild city single-state yr180 end-to-end against sim_root 4.10.
# Mirrors rebuild_city_yr180_20260411.sh (which targeted 4.9) but points
# BV2_SIM_ROOT and BV2_DATA_BUNDLE_ROOT at the freshly downloaded 4.10 data.
#
# --factory-startup is deliberately OMITTED so Blender 3.11's USER_SITE can
# pick up pandas/vtk from %APPDATA%/Python/Python311/site-packages.
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
REPO_ROOT="D:/2026 Arboreal Futures/urban-futures"
LOG_PATH="${REPO_ROOT}/_logs/bV2/city_yr180_rebuild_4.10_${TS}.log"
STDOUT_LOG="${REPO_ROOT}/_logs/bV2/city_yr180_rebuild_4.10_${TS}.stdout.log"

mkdir -p "${REPO_ROOT}/_logs/bV2"

export BV2_SITE="city"
export BV2_MODE="single_state"
export BV2_YEAR="180"
export BV2_CAMERA_NAME="city-yr180-hero-image"
export BV2_SAVE_BLEND="1"
export BV2_RENDER_EXRS="1"
export BV2_UPLOAD_TO_MEDIAFLUX="1"
export BV2_VALIDATE_STRICT="1"
export BV2_SIM_ROOT="4.10"
export BV2_RENDER_TAG="8k64s"
export BV2_RES_X="7680"
export BV2_RES_Y="4320"
export BV2_SAMPLES="64"
export BV2_LOG_PATH="${LOG_PATH}"
export BV2_OUTPUT_TIMESTAMP="${TS}"
export BV2_DATA_BUNDLE_ROOT="D:\\2026 Arboreal Futures\\urban-futures\\_data-refactored\\model-outputs\\generated-states\\4.10\\output"
# Expose repo + venv site-packages so the bV2 headless Blender finds pandas/vtk
export PYTHONPATH="${REPO_ROOT};${REPO_ROOT}/.venv/Lib/site-packages"

BLENDER="C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
SCRIPT="${REPO_ROOT}/_futureSim_refactored/blender/blenderv2/bV2_build_scene.py"

echo "[rebuild_city_yr180_4.10] TS=${TS}"
echo "[rebuild_city_yr180_4.10] LOG=${LOG_PATH}"

"${BLENDER}" \
    --background \
    --python "${SCRIPT}" \
    >"${STDOUT_LOG}" 2>&1

echo "[rebuild_city_yr180_4.10] done TS=${TS}"
