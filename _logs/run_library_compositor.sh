#!/usr/bin/env bash
# Ad-hoc library compositor run for:
#   precolonial.True + size.large + control.reserve-tree, all 4 views
#   families: mist + depth_outliner
# Writes to short temp dir (Blender 4.2 Python MAX_PATH), then bash-moves to
# canonical _data-refactored/compositor/outputs/_library/... location.

set -euo pipefail

TS="$1"   # e.g. 20260411_2016
REPO="d:/2026 Arboreal Futures/urban-futures"
BLENDER="/c/Program Files/Blender Foundation/Blender 4.2/blender.exe"
ASSET_FAMILY="20260407_232744_ply-library-exr-4sides-large-senescing-snag_el20_4k64s"
EXR_DIR="${REPO}/_data-refactored/blenderv2/output/_library/${ASSET_FAMILY}/exr"
OUT_BASE="${REPO}/_data-refactored/compositor/outputs/_library/${ASSET_FAMILY}"
MIST_BLEND="${REPO}/_data-refactored/compositor/temp_blends/template_instantiations/compositor_mist__library_reserve-tree.blend"
DEPTH_BLEND="${REPO}/_data-refactored/compositor/temp_blends/template_instantiations/compositor_depth_outliner__library_reserve-tree.blend"
MIST_SCRIPT="${REPO}/_futureSim_refactored/blender/compositor/scripts/render_edge_lab_current_mist.py"
DEPTH_SCRIPT="${REPO}/_futureSim_refactored/blender/compositor/scripts/render_edge_lab_current_depth_outliner.py"

MIST_OUT="${OUT_BASE}/mist__${TS}__library-reserve-tree"
DEPTH_OUT="${OUT_BASE}/depth-outliner__${TS}__library-reserve-tree"
mkdir -p "$MIST_OUT" "$DEPTH_OUT"

TMP="/d/_mfr_tmp"
mkdir -p "$TMP"

# Build list of all 24 EXRs (ids 11-16, views north/south/east/west)
EXRS=()
for ID in 11 12 13 14 15 16; do
  for VIEW in north south east west; do
    EXRS+=("precolonial.True_size.large_control.reserve-tree_id.${ID}_${VIEW}")
  done
done

run_family () {
  local FAMILY="$1"   # mist | depth
  local BLEND="$2"
  local SCRIPT="$3"
  local ENV_VAR="$4"  # COMPOSITOR_MIST_EXR or COMPOSITOR_DEPTH_EXR
  local OUT_DIR="$5"

  local N=${#EXRS[@]}
  local IDX=0
  for STEM in "${EXRS[@]}"; do
    IDX=$((IDX+1))
    echo ""
    echo "==== [${FAMILY} ${IDX}/${N}] ${STEM} ===="
    local EXR="${EXR_DIR}/${STEM}.exr"
    if [ ! -f "$EXR" ]; then
      echo "MISSING EXR: $EXR"
      continue
    fi

    # Clear temp dir
    rm -f "${TMP}"/*.png || true

    # Run Blender
    export COMPOSITOR_BLEND_PATH="$BLEND"
    export COMPOSITOR_OUTPUT_DIR="$TMP"
    export COMPOSITOR_SCENE_NAME="Current"
    if [ "$ENV_VAR" = "MIST" ]; then
      export COMPOSITOR_MIST_EXR="$EXR"
      unset COMPOSITOR_DEPTH_EXR || true
    else
      export COMPOSITOR_DEPTH_EXR="$EXR"
      unset COMPOSITOR_MIST_EXR || true
    fi

    "$BLENDER" --background --factory-startup --python "$SCRIPT" >/dev/null 2>&1 || {
      echo "BLENDER FAILED for $STEM"
      continue
    }

    # Move each PNG from temp into the canonical out dir with tree+view prefix
    local MOVED=0
    for PNG in "${TMP}"/*.png; do
      [ -f "$PNG" ] || continue
      local BASE
      BASE=$(basename "$PNG")
      local DEST="${OUT_DIR}/${STEM}__${BASE}"
      mv "$PNG" "$DEST"
      MOVED=$((MOVED+1))
    done
    echo "  moved ${MOVED} PNG(s) -> ${OUT_DIR}"
  done
}

echo "##### MIST FAMILY #####"
run_family "mist" "$MIST_BLEND" "$MIST_SCRIPT" "MIST" "$MIST_OUT"

echo ""
echo "##### DEPTH_OUTLINER FAMILY #####"
run_family "depth" "$DEPTH_BLEND" "$DEPTH_SCRIPT" "DEPTH" "$DEPTH_OUT"

echo ""
echo "##### DONE #####"
echo "Mist outputs:"
ls -1 "$MIST_OUT" | wc -l
echo "Depth outputs:"
ls -1 "$DEPTH_OUT" | wc -l
