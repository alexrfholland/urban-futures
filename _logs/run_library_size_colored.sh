#!/usr/bin/env bash
# Ad-hoc: size-colored silhouette PNG for each per-tree library EXR.
# Parses size label from the asset token filename.

set -euo pipefail

TS="$1"
REPO="d:/2026 Arboreal Futures/urban-futures"
BLENDER="/c/Program Files/Blender Foundation/Blender 4.2/blender.exe"
ASSET_FAMILY="20260407_232744_ply-library-exr-4sides-large-senescing-snag_el20_4k64s"
EXR_DIR="${REPO}/_data-refactored/blenderv2/output/_library/${ASSET_FAMILY}/exr"
OUT_BASE="${REPO}/_data-refactored/compositor/outputs/_library/${ASSET_FAMILY}"
SCRIPT="${REPO}/_futureSim_refactored/blender/compositor/scripts/render_library_size_colored.py"

SIZES_OUT="${OUT_BASE}/sizes__${TS}__library-reserve-tree"
mkdir -p "$SIZES_OUT"

TMP="/d/_mfr_tmp"
mkdir -p "$TMP"

EXRS=()
for ID in 11 12 13 14 15 16; do
  for VIEW in north south east west; do
    EXRS+=("precolonial.True_size.large_control.reserve-tree_id.${ID}_${VIEW}")
  done
done

N=${#EXRS[@]}
IDX=0
for STEM in "${EXRS[@]}"; do
  IDX=$((IDX+1))
  # Parse size label from stem: ..._size.<label>_control...
  SIZE_LABEL=$(echo "$STEM" | sed -n 's/.*size\.\([a-z]*\)_control.*/\1/p')
  if [ -z "$SIZE_LABEL" ]; then
    echo "[${IDX}/${N}] ${STEM}  SKIP: could not parse size"
    continue
  fi
  echo "==== [${IDX}/${N}] ${STEM}  size=${SIZE_LABEL} ===="
  EXR="${EXR_DIR}/${STEM}.exr"
  if [ ! -f "$EXR" ]; then
    echo "  MISSING EXR: $EXR"
    continue
  fi

  rm -f "${TMP}"/*.png || true
  export COMPOSITOR_LIBRARY_EXR="$EXR"
  export COMPOSITOR_OUTPUT_DIR="$TMP"
  export COMPOSITOR_SIZE_LABEL="$SIZE_LABEL"

  "$BLENDER" --background --factory-startup --python "$SCRIPT" >/dev/null 2>&1 || {
    echo "  BLENDER FAILED"
    continue
  }

  if [ -f "${TMP}/size_colored.png" ]; then
    mv "${TMP}/size_colored.png" "${SIZES_OUT}/${STEM}__size_colored_${SIZE_LABEL}.png"
    echo "  moved 1"
  else
    echo "  no size_colored.png produced"
  fi
done

echo ""
echo "##### DONE #####"
echo "Sizes outputs: $(ls -1 "$SIZES_OUT" | wc -l)"
