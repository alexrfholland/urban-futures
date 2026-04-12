#!/usr/bin/env bash
# Rebuild timeline (time-slice) EXRs end-to-end against sim_root 4.10 for all
# three sites: city, trimmed-parade, uni. Mirrors rebuild_city_yr180_4.10_20260412.sh
# but runs in BV2_MODE=timeline and iterates over per-site (camera, site) pairs.
#
# --factory-startup is deliberately OMITTED so Blender 3.11's USER_SITE can
# pick up pandas/vtk from %APPDATA%/Python/Python311/site-packages.
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
REPO_ROOT="D:/2026 Arboreal Futures/urban-futures"

mkdir -p "${REPO_ROOT}/_logs/bV2"

BLENDER="C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
SCRIPT="${REPO_ROOT}/_futureSim_refactored/blender/blenderv2/bV2_build_scene.py"

# Expose repo + venv site-packages so the bV2 headless Blender finds pandas/vtk
export PYTHONPATH="${REPO_ROOT};${REPO_ROOT}/.venv/Lib/site-packages"

# Shared env (unchanged across sites)
export BV2_MODE="timeline"
export BV2_SAVE_BLEND="1"
export BV2_RENDER_EXRS="1"
export BV2_UPLOAD_TO_MEDIAFLUX="1"
export BV2_VALIDATE_STRICT="1"
export BV2_SIM_ROOT="4.10"
export BV2_RENDER_TAG="8k64s"
export BV2_RES_X="7680"
export BV2_RES_Y="4320"
export BV2_SAMPLES="64"
export BV2_OUTPUT_TIMESTAMP="${TS}"
export BV2_DATA_BUNDLE_ROOT="D:\\2026 Arboreal Futures\\urban-futures\\_data-refactored\\model-outputs\\generated-states\\4.10\\output"

# Per-site (site, camera) pairs
SITE_SPECS=(
    "city|city - camera - time slice - zoom"
    "trimmed-parade|parade - camera - time slice - zoom"
    "uni|uni - camera - time slice - zoom"
)

for spec in "${SITE_SPECS[@]}"; do
    SITE="${spec%%|*}"
    CAMERA="${spec#*|}"

    LOG_PATH="${REPO_ROOT}/_logs/bV2/${SITE}_timeline_4.10_${TS}.log"
    STDOUT_LOG="${REPO_ROOT}/_logs/bV2/${SITE}_timeline_4.10_${TS}.stdout.log"

    export BV2_SITE="${SITE}"
    export BV2_CAMERA_NAME="${CAMERA}"
    export BV2_LOG_PATH="${LOG_PATH}"
    unset BV2_YEAR || true

    echo "[rebuild_timelines_4.10] TS=${TS} SITE=${SITE} CAMERA=${CAMERA}"
    echo "[rebuild_timelines_4.10] LOG=${LOG_PATH}"

    "${BLENDER}" \
        --background \
        --python "${SCRIPT}" \
        >"${STDOUT_LOG}" 2>&1

    echo "[rebuild_timelines_4.10] done SITE=${SITE}"
done

echo "[rebuild_timelines_4.10] all sites done TS=${TS}"
