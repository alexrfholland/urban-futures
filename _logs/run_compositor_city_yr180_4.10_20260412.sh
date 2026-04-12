#!/usr/bin/env bash
# 4.10 compositor run for city_single-state_yr180 — FULL family set.
# Multi-EXR families (10): ao, normals, resources, base, shading, bioenvelope,
#                          sizes, mist, depth_outliner, proposals (masks)
# Per-state proposal sub-families (3): proposal-only, proposal-outline,
#                                      proposal-colored-depth
#
# Proposals use the 20260411_2130 batch working copies directly (7-slot, known
# good). Canonicals are stale (5 slots).
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
BATCH_LOG="${LOG_DIR}/city_yr180_compositor_4.10_batch_${TS}.log"
PY="${REPO}/.venv/Scripts/python.exe"

mkdir -p "$LOG_DIR" "$TEMP_BLENDS" "$OUT_BASE"
: > "$BATCH_LOG"

# 7 EXR inputs (view layers)
EXR_EXISTING_POS="${EXR_BASE}/${SITE_KEY}__existing_condition_positive__8k64s.exr"
EXR_EXISTING_TRN="${EXR_BASE}/${SITE_KEY}__existing_condition_trending__8k64s.exr"
EXR_PATHWAY="${EXR_BASE}/${SITE_KEY}__positive_state__8k64s.exr"
EXR_PRIORITY="${EXR_BASE}/${SITE_KEY}__positive_priority_state__8k64s.exr"
EXR_TRENDING="${EXR_BASE}/${SITE_KEY}__trending_state__8k64s.exr"
EXR_BIO_POS="${EXR_BASE}/${SITE_KEY}__bioenvelope_positive__8k64s.exr"
EXR_BIO_TRN="${EXR_BASE}/${SITE_KEY}__bioenvelope_trending__8k64s.exr"

# Fail-fast: all 7 EXRs must exist
for f in "$EXR_EXISTING_POS" "$EXR_EXISTING_TRN" "$EXR_PATHWAY" "$EXR_PRIORITY" "$EXR_TRENDING" "$EXR_BIO_POS" "$EXR_BIO_TRN"; do
  if [ ! -f "$f" ]; then
    echo "MISSING EXR: $f" | tee -a "$BATCH_LOG"
    exit 2
  fi
done

FAIL=0
log() { echo "$@" | tee -a "$BATCH_LOG"; }
stamp() { date +%H:%M:%S; }

copy_from_canonical() {
  # $1 = canonical blend filename
  # $2 = family tag
  local src="${CANONICAL}/$1"
  local dst="${TEMP_BLENDS}/compositor_$2__${SITE_KEY}__4.10__${TS}.blend"
  if [ ! -f "$src" ]; then
    log "MISSING CANONICAL: $src"
    return 1
  fi
  cp -f "$src" "$dst"
  echo "$dst"
}

# Generic multi-EXR runner wrapper.
# Usage: run_multi_exr FAMILY CANONICAL_BLEND RUNNER_SCRIPT
# Env vars relevant to the family must be exported beforehand via a subshell.
run_multi_exr() {
  local family="$1"
  local canonical="$2"
  local runner="$3"
  log "=== [$(stamp)] ${family} ==="
  local wc
  wc=$(copy_from_canonical "$canonical" "$family") || return
  local out_dir="${OUT_BASE}/${family}__${TS}"
  mkdir -p "$out_dir"
  local per_log="${LOG_DIR}/city_yr180_4.10_${family}_${TS}.log"
  (
    export COMPOSITOR_BLEND_PATH="$wc"
    export COMPOSITOR_OUTPUT_DIR="$out_dir"
    export COMPOSITOR_SCENE_NAME="Current"
    export COMPOSITOR_PATHWAY_EXR="$EXR_PATHWAY"
    export COMPOSITOR_PRIORITY_EXR="$EXR_PRIORITY"
    export COMPOSITOR_EXISTING_EXR="$EXR_EXISTING_POS"
    export COMPOSITOR_EXISTING_TRENDING_EXR="$EXR_EXISTING_TRN"
    export COMPOSITOR_TRENDING_EXR="$EXR_TRENDING"
    export COMPOSITOR_BIOENVELOPE_EXR="$EXR_BIO_POS"
    export COMPOSITOR_BIOENVELOPE_TRENDING_EXR="$EXR_BIO_TRN"
    "$BLENDER" --background --factory-startup \
      --python "${SCRIPTS}/${runner}" \
      >"$per_log" 2>&1
  )
  local exit_code=$?
  local npng
  npng=$(find "$out_dir" -name "*.png" -type f 2>/dev/null | wc -l)
  log "  ${family} EXIT=${exit_code} PNG_COUNT=${npng}"
  if [ "$exit_code" -ne 0 ]; then
    FAIL=$((FAIL+1))
    tail -12 "$per_log" | tee -a "$BATCH_LOG"
  fi
}

# ---------------------------------------------------------------
# Multi-EXR families
# ---------------------------------------------------------------
run_multi_exr "ao"             "compositor_ao.blend"             "render_edge_lab_current_core_outputs.py"
run_multi_exr "normals"        "compositor_normals.blend"        "render_edge_lab_current_core_outputs.py"
run_multi_exr "resources"      "compositor_resources.blend"      "render_edge_lab_current_core_outputs.py"
run_multi_exr "base"           "compositor_base.blend"           "render_edge_lab_current_base.py"
run_multi_exr "shading"        "compositor_shading.blend"        "render_edge_lab_current_shading.py"
run_multi_exr "bioenvelope"    "compositor_bioenvelope.blend"    "render_edge_lab_current_bioenvelopes.py"
run_multi_exr "sizes"          "compositor_sizes.blend"          "render_edge_lab_current_sizes.py"
run_multi_exr "mist"           "compositor_mist.blend"           "render_edge_lab_current_mist.py"
run_multi_exr "depth_outliner" "compositor_depth_outliner.blend" "render_edge_lab_current_depth_outliner.py"
run_multi_exr "proposals"      "compositor_proposal_masks.blend" "render_edge_lab_current_proposals.py"

# ---------------------------------------------------------------
# PROPOSAL sub-families — 2130-batch 7-slot working copies
# ---------------------------------------------------------------
PO_WC="${TEMP_BLENDS}/compositor_proposal-only__batch__20260411_2130.blend"
POL_WC="${TEMP_BLENDS}/compositor_proposal-outline__batch__20260411_2130.blend"
PCD_WC="${TEMP_BLENDS}/compositor_proposal-colored-depth__batch_fixed__20260411_2130.blend"

for f in "$PO_WC" "$POL_WC" "$PCD_WC"; do
  if [ ! -f "$f" ]; then
    log "MISSING PROPOSAL WORKING COPY: $f"
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

run_proposal_family() {
  local family="$1"
  local wc="$2"
  local scene_name="$3"
  local out_node="$4"

  log "=== [$(stamp)] ${family} ==="
  local out_dir="${OUT_BASE}/${family}__${TS}"
  mkdir -p "$out_dir"
  local idx=0
  for SD in "${STATE_DEFS[@]}"; do
    idx=$((idx+1))
    local state="${SD%%|*}"
    local prefix="${SD##*|}"
    local exr="${EXR_BASE}/${SITE_KEY}__${state}__8k64s.exr"
    local per_log="${LOG_DIR}/city_yr180_4.10_${family}_${prefix}_${TS}.log"
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
    if [ "$exit_code" -ne 0 ] || [ "$npng" -lt 5 ]; then
      FAIL=$((FAIL+1))
      tail -8 "$per_log" | tee -a "$BATCH_LOG"
    fi
  done
}

run_proposal_family "proposal-only"          "$PO_WC"  "ProposalOnly"                 "ProposalOnlyOutput"
run_proposal_family "proposal-outline"       "$POL_WC" "ProposalOutline"              "ProposalOutlineOutput"
run_proposal_family "proposal-colored-depth" "$PCD_WC" "ProposalColoredDepthOutlines" "ProposalColoredDepthOutput"

log "=== [$(stamp)] COMPOSITOR PHASE DONE fail=${FAIL} ==="

# ---------------------------------------------------------------
# UPLOAD to Mediaflux
# ---------------------------------------------------------------
log "=== [$(stamp)] UPLOAD PHASE ==="
cd "$REPO"
export PATH="${REPO}/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/jre/bin:$PATH"
export MEDIAFLUX_CLIENT_BIN_DIR="${REPO}/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/bin/windows"
FAMILIES=(
  ao
  normals
  resources
  base
  shading
  bioenvelope
  sizes
  mist
  depth_outliner
  proposals
  proposal-only
  proposal-outline
  proposal-colored-depth
)
for F in "${FAMILIES[@]}"; do
  SRC="${OUT_BASE}/${F}__${TS}"
  if [ ! -d "$SRC" ]; then
    log "  [$(stamp)] SKIP upload ${F} (no dir)"
    continue
  fi
  SUB="pipeline/4.10/compositor_pngs/${SITE_KEY}/${F}__${TS}"
  log "  [$(stamp)] upload ${F}"
  log "    SRC=${SRC}"
  log "    DST=${SUB}"
  "$PY" -m mediafluxsync upload-project --create-parents --exclude-parent \
    --project-dir . "$SRC" "$SUB" >>"$BATCH_LOG" 2>&1
  EXIT=$?
  log "    UPLOAD EXIT=$EXIT"
  if [ "$EXIT" -ne 0 ]; then FAIL=$((FAIL+1)); fi
done

log "=== [$(stamp)] ALL DONE total_fail=${FAIL} ==="
exit "$FAIL"
