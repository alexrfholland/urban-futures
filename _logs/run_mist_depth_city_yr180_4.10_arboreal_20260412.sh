#!/usr/bin/env bash
# Re-render mist + depth_outliner for 4.10 city_single-state_yr180 with
# the arboreal mask fix (compositor_mist.blend IDMask index 0 -> 3).
#
# 3 states only: pathway (positive_state), priority (positive_priority_state),
#                trending (trending_state).
#
# Output layout:
#   mist__<TS>/
#     pathway__mist_kirsch_thin.png
#     pathway__mist_kirsch_fine.png
#     pathway__mist_kirsch_extra_thin.png
#     priority__mist_kirsch_thin.png
#     ... (9 PNGs total)
#   depth_outliner__<TS>/
#     pathway__depth_outliner.png
#     priority__depth_outliner.png
#     trending__depth_outliner.png
#     (3 PNGs total)
set -uo pipefail

REPO="/d/2026 Arboreal Futures/urban-futures"
BLENDER="/c/Program Files/Blender Foundation/Blender 4.2/blender.exe"
SCRIPTS="${REPO}/_futureSim_refactored/blender/compositor/scripts"
CANONICAL="${REPO}/_futureSim_refactored/blender/compositor/canonical_templates"
TEMP_BLENDS="${REPO}/_data-refactored/compositor/temp_blends/template_instantiations"
OUT_BASE="${REPO}/_data-refactored/compositor/outputs/4.10/city_single-state_yr180"
EXR_BASE="${REPO}/_data-refactored/blenderv2/output/4.10/city_single-state_yr180"
SITE_KEY="city_single-state_yr180"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO}/_logs/compositor"
BATCH_LOG="${LOG_DIR}/city_yr180_mist_depth_arboreal_${TS}.log"

mkdir -p "$LOG_DIR" "$TEMP_BLENDS" "$OUT_BASE"
: > "$BATCH_LOG"

log() { echo "$@" | tee -a "$BATCH_LOG"; }
stamp() { date +%H:%M:%S; }

# label | exr_suffix
STATE_DEFS=(
  "pathway|positive_state"
  "priority|positive_priority_state"
  "trending|trending_state"
)

# Fail-fast: all 3 EXRs must exist
for SD in "${STATE_DEFS[@]}"; do
  suffix="${SD##*|}"
  f="${EXR_BASE}/${SITE_KEY}__${suffix}__8k64s.exr"
  if [ ! -f "$f" ]; then
    log "MISSING EXR: $f"
    exit 2
  fi
done

FAIL=0

run_family() {
  # $1 = family tag (mist | depth_outliner)
  # $2 = canonical blend filename
  # $3 = runner script name
  # $4 = env var that selects the EXR for that family
  local family="$1"
  local canonical="$2"
  local runner="$3"
  local exr_env="$4"

  log "=== [$(stamp)] ${family} ==="
  local wc="${TEMP_BLENDS}/compositor_${family}__${SITE_KEY}__4.10__${TS}.blend"
  cp -f "${CANONICAL}/${canonical}" "$wc"

  local out_dir="${OUT_BASE}/${family}__${TS}"
  mkdir -p "$out_dir"

  for SD in "${STATE_DEFS[@]}"; do
    local label="${SD%%|*}"
    local suffix="${SD##*|}"
    local exr="${EXR_BASE}/${SITE_KEY}__${suffix}__8k64s.exr"

    # Use a short tmp dir to dodge Windows MAX_PATH + collision issues
    local tmp_dir="${out_dir}/_tmp_${label}"
    mkdir -p "$tmp_dir"
    rm -f "${tmp_dir}"/*.png 2>/dev/null || true

    local per_log="${LOG_DIR}/city_yr180_${family}_${label}_${TS}.log"
    log "  [$(stamp)] ${family} ${label} <- ${suffix}"
    (
      export COMPOSITOR_BLEND_PATH="$wc"
      export COMPOSITOR_OUTPUT_DIR="$tmp_dir"
      export COMPOSITOR_SCENE_NAME="Current"
      export "${exr_env}=${exr}"
      "$BLENDER" --background --factory-startup \
        --python "${SCRIPTS}/${runner}" \
        >"$per_log" 2>&1
    )
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
      log "    FAILED EXIT=${exit_code}"
      tail -20 "$per_log" | tee -a "$BATCH_LOG"
      FAIL=$((FAIL+1))
      continue
    fi

    # Move PNGs with state prefix
    local moved=0
    for png in "${tmp_dir}"/*.png; do
      [ -f "$png" ] || continue
      local base
      base="$(basename "$png")"
      mv "$png" "${out_dir}/${label}__${base}"
      moved=$((moved+1))
    done
    rmdir "$tmp_dir" 2>/dev/null || true
    log "    moved ${moved} PNGs (prefix '${label}__')"
  done

  local total
  total=$(find "$out_dir" -name "*.png" -type f 2>/dev/null | wc -l)
  log "  ${family} total PNGs: ${total}"
}

run_family "mist"           "compositor_mist.blend"           "render_edge_lab_current_mist.py"           "COMPOSITOR_MIST_EXR"
run_family "depth_outliner" "compositor_depth_outliner.blend" "render_edge_lab_current_depth_outliner.py" "COMPOSITOR_DEPTH_EXR"

log "=== [$(stamp)] DONE fail=${FAIL} ==="
exit "$FAIL"
