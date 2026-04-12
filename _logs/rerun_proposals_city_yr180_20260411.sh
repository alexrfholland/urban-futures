#!/usr/bin/env bash
# Rerun the 3 proposal families for city_single-state_yr180, using the 7-slot
# working copies from the 20260411_2130 batch (the canonical templates are
# stale — they only have 5 slots, missing recruit-smalls and
# deploy-structure-fallen-logs). Writes the missing 2 PNGs per state into the
# existing proposal__20260411_234840 output dirs and then re-uploads them.
set -uo pipefail

REPO="/d/2026 Arboreal Futures/urban-futures"
BLENDER="/c/Program Files/Blender Foundation/Blender 4.2/blender.exe"
SCRIPTS="${REPO}/_futureSim_refactored/blender/compositor/scripts"
TEMP_BLENDS="${REPO}/_data-refactored/compositor/temp_blends/template_instantiations"
OUT_BASE="${REPO}/_data-refactored/compositor/outputs/4.9/city_single-state_yr180"
EXR_BASE="${REPO}/_data-refactored/blenderv2/output/4.9/city_single-state_yr180"
SITE_KEY="city_single-state_yr180"
ORIG_TS="20260411_234840"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO}/_logs/compositor"
BATCH_LOG="${LOG_DIR}/city_yr180_proposals_rerun_${TS}.log"
PY="${REPO}/.venv/Scripts/python.exe"
mkdir -p "$LOG_DIR"
: > "$BATCH_LOG"

log() { echo "$@" | tee -a "$BATCH_LOG"; }
stamp() { date +%H:%M:%S; }

# Use 2130-batch working copies (known-good, 7 slots)
PO_WC="${TEMP_BLENDS}/compositor_proposal-only__batch__20260411_2130.blend"
POL_WC="${TEMP_BLENDS}/compositor_proposal-outline__batch__20260411_2130.blend"
PCD_WC="${TEMP_BLENDS}/compositor_proposal-colored-depth__batch_fixed__20260411_2130.blend"

for f in "$PO_WC" "$POL_WC" "$PCD_WC"; do
  if [ ! -f "$f" ]; then
    log "MISSING WORKING COPY: $f"
    exit 2
  fi
done

STATE_DEFS=(
  "existing_condition_positive|existing_positive"
  "existing_condition_trending|existing_trending"
  "positive_state|pathway"
  "positive_priority_state|priority"
  "trending_state|trending"
  "bioenvelope_positive|bio_positive"
  "bioenvelope_trending|bio_trending"
)

FAIL=0

run_proposal_family() {
  local family="$1"
  local wc="$2"
  local scene_name="$3"
  local out_node="$4"

  log "=== [$(stamp)] ${family} ==="
  local out_dir="${OUT_BASE}/${family}__${ORIG_TS}"
  mkdir -p "$out_dir"
  local idx=0
  for SD in "${STATE_DEFS[@]}"; do
    idx=$((idx+1))
    local state="${SD%%|*}"
    local prefix="${SD##*|}"
    local exr="${EXR_BASE}/${SITE_KEY}__${state}__8k64s.exr"
    local per_log="${LOG_DIR}/city_yr180_${family}_${prefix}_rerun_${TS}.log"
    log "  [$(stamp)] [${idx}/7] ${family} ${state}"
    (
      export COMPOSITOR_BLEND_PATH="$wc"
      export COMPOSITOR_OUTPUT_DIR="$out_dir"
      export COMPOSITOR_EXR="$exr"
      export COMPOSITOR_OUTPUT_NODE_NAME="$out_node"
      export COMPOSITOR_SCENE_NAME="$scene_name"
      export COMPOSITOR_FILENAME_PREFIX="$prefix"
      "$BLENDER" --background --factory-startup \
        --python "${SCRIPTS}/_run_proposal_subcategory_20260411.py" \
        >"$per_log" 2>&1
    )
    local exit_code=$?
    local npng
    npng=$(find "$out_dir" -name "${prefix}__*.png" -type f 2>/dev/null | wc -l)
    log "    EXIT=$exit_code prefix_pngs=$npng"
    if [ "$exit_code" -ne 0 ] || [ "$npng" -ne 7 ]; then
      FAIL=$((FAIL+1))
      tail -10 "$per_log" | tee -a "$BATCH_LOG"
    fi
  done
}

run_proposal_family "proposal-only"          "$PO_WC"  "ProposalOnly"               "ProposalOnlyOutput"
run_proposal_family "proposal-outline"       "$POL_WC" "ProposalOutline"            "ProposalOutlineOutput"
run_proposal_family "proposal-colored-depth" "$PCD_WC" "ProposalColoredDepthOutlines" "ProposalColoredDepthOutput"

log "=== [$(stamp)] PROPOSAL RERUN DONE fail=${FAIL} ==="

# Re-upload the 3 proposal dirs (overwrite mode)
log "=== [$(stamp)] REUPLOAD PHASE ==="
cd "$REPO"
export PATH="${REPO}/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/jre/bin:$PATH"
export MEDIAFLUX_CLIENT_BIN_DIR="${REPO}/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/bin/windows"
for F in proposal-only proposal-outline proposal-colored-depth; do
  SRC="${OUT_BASE}/${F}__${ORIG_TS}"
  SUB="pipeline/4.9/compositor_pngs/${SITE_KEY}/${F}__${ORIG_TS}"
  log "  [$(stamp)] upload ${F}"
  log "    SRC=${SRC}"
  log "    DST=${SUB}"
  "$PY" -m mediafluxsync upload-project --create-parents --exclude-parent \
    --project-dir . "$SRC" "$SUB" >>"$BATCH_LOG" 2>&1
  EXIT=$?
  log "    UPLOAD EXIT=$EXIT"
  if [ "$EXIT" -ne 0 ]; then FAIL=$((FAIL+1)); fi
done

log "=== [$(stamp)] ALL DONE fail=${FAIL} ==="
exit "$FAIL"
