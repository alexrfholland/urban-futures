#!/usr/bin/env bash
# Pass-2 library extraction: AO + resource framebuffers from 24 library EXRs.
# Writes to short temp dir then bash-moves to canonical long-path location.

set -euo pipefail

TS="$1"   # e.g. 20260411_2216
REPO="d:/2026 Arboreal Futures/urban-futures"
BLENDER="/c/Program Files/Blender Foundation/Blender 4.2/blender.exe"
ASSET_FAMILY="20260407_232744_ply-library-exr-4sides-large-senescing-snag_el20_4k64s"
EXR_DIR="${REPO}/_data-refactored/blenderv2/output/_library/${ASSET_FAMILY}/exr"
OUT_BASE="${REPO}/_data-refactored/compositor/outputs/_library/${ASSET_FAMILY}"
SCRIPT="${REPO}/_futureSim_refactored/blender/compositor/scripts/render_library_raw_framebuffers.py"

AO_OUT="${OUT_BASE}/ao__${TS}__library-reserve-tree"
RESOURCES_OUT="${OUT_BASE}/resources__${TS}__library-reserve-tree"
mkdir -p "$AO_OUT" "$RESOURCES_OUT"

TMP="/d/_mfr_tmp"
mkdir -p "$TMP"

# Build list of all 24 EXR stems
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
  echo ""
  echo "==== [${IDX}/${N}] ${STEM} ===="
  EXR="${EXR_DIR}/${STEM}.exr"
  if [ ! -f "$EXR" ]; then
    echo "MISSING EXR: $EXR"
    continue
  fi

  rm -f "${TMP}"/*.png || true

  export COMPOSITOR_LIBRARY_EXR="$EXR"
  export COMPOSITOR_OUTPUT_DIR="$TMP"

  "$BLENDER" --background --factory-startup --python "$SCRIPT" >/dev/null 2>&1 || {
    echo "BLENDER FAILED for $STEM"
    continue
  }

  # Move PNGs to canonical locations: AO.png → ao folder, resource_* → resources folder
  local_moved_ao=0
  local_moved_res=0
  for PNG in "${TMP}"/*.png; do
    [ -f "$PNG" ] || continue
    BASE=$(basename "$PNG")
    if [ "$BASE" = "AO.png" ]; then
      mv "$PNG" "${AO_OUT}/${STEM}__AO.png"
      local_moved_ao=$((local_moved_ao+1))
    elif [[ "$BASE" == resource_* ]]; then
      mv "$PNG" "${RESOURCES_OUT}/${STEM}__${BASE}"
      local_moved_res=$((local_moved_res+1))
    else
      # unexpected leftover; just discard
      rm "$PNG"
    fi
  done
  echo "  ao=${local_moved_ao}  resources=${local_moved_res}"
done

echo ""
echo "##### DONE #####"
echo "AO outputs:        $(ls -1 "$AO_OUT" | wc -l)"
echo "Resource outputs:  $(ls -1 "$RESOURCES_OUT" | wc -l)"
