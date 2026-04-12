#!/usr/bin/env bash
# Run all 6 compositor families (ao, shading, sizes, proposal-only,
# proposal-outline, proposal-colored-depth) on the city_single-state_yr180
# EXRs just rebuilt by rebuild_city_yr180_20260411.sh.
#
# Produces one output directory per family at
#   _data-refactored/compositor/outputs/4.9/city_single-state_yr180/<family>__<TS>/
# and then uploads each to Mediaflux at
#   pipeline/4.9/compositor_pngs/city_single-state_yr180/<family>__<TS>/
set -uo pipefail

REPO="/d/2026 Arboreal Futures/urban-futures"
BLENDER="/c/Program Files/Blender Foundation/Blender 4.2/blender.exe"
SCRIPTS="${REPO}/_futureSim_refactored/blender/compositor/scripts"
CANONICAL="${REPO}/_futureSim_refactored/blender/compositor/canonical_templates"
TEMP_BLENDS="${REPO}/_data-refactored/compositor/temp_blends/template_instantiations"
OUT_BASE="${REPO}/_data-refactored/compositor/outputs/4.9/city_single-state_yr180"
EXR_BASE="${REPO}/_data-refactored/blenderv2/output/4.9/city_single-state_yr180"
SITE_KEY="city_single-state_yr180"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO}/_logs/compositor"
BATCH_LOG="${LOG_DIR}/city_yr180_compositor_batch_${TS}.log"
PY="${REPO}/.venv/Scripts/python.exe"

mkdir -p "$LOG_DIR" "$TEMP_BLENDS" "$OUT_BASE"
: > "$BATCH_LOG"

# EXR inputs
EXR_EXISTING_POS="${EXR_BASE}/${SITE_KEY}__existing_condition_positive__8k64s.exr"
EXR_EXISTING_TRN="${EXR_BASE}/${SITE_KEY}__existing_condition_trending__8k64s.exr"
EXR_PATHWAY="${EXR_BASE}/${SITE_KEY}__positive_state__8k64s.exr"
EXR_PRIORITY="${EXR_BASE}/${SITE_KEY}__positive_priority_state__8k64s.exr"
EXR_TRENDING="${EXR_BASE}/${SITE_KEY}__trending_state__8k64s.exr"
EXR_BIO_POS="${EXR_BASE}/${SITE_KEY}__bioenvelope_positive__8k64s.exr"
EXR_BIO_TRN="${EXR_BASE}/${SITE_KEY}__bioenvelope_trending__8k64s.exr"

# Fail-fast check that all 7 EXRs exist
for f in "$EXR_EXISTING_POS" "$EXR_EXISTING_TRN" "$EXR_PATHWAY" "$EXR_PRIORITY" "$EXR_TRENDING" "$EXR_BIO_POS" "$EXR_BIO_TRN"; do
  if [ ! -f "$f" ]; then
    echo "MISSING EXR: $f" | tee -a "$BATCH_LOG"
    exit 2
  fi
done

FAIL=0
log() { echo "$@" | tee -a "$BATCH_LOG"; }
stamp() { date +%H:%M:%S; }

copy_working_copy() {
  # $1 = canonical blend name (no dir)
  # $2 = working-copy family tag
  local src="${CANONICAL}/$1"
  local dst="${TEMP_BLENDS}/compositor_$2__${SITE_KEY}__${TS}.blend"
  if [ ! -f "$src" ]; then
    log "MISSING CANONICAL: $src"
    return 1
  fi
  cp -f "$src" "$dst"
  echo "$dst"
}

# ---------------------------------------------------------------
# 1. AO
# ---------------------------------------------------------------
log "=== [$(stamp)] AO ==="
AO_WC=$(copy_working_copy "compositor_ao.blend" "ao")
AO_OUT="${OUT_BASE}/ao__${TS}"
mkdir -p "$AO_OUT"
AO_LOG="${LOG_DIR}/city_yr180_ao_${TS}.log"
(
  export COMPOSITOR_BLEND_PATH="$AO_WC"
  export COMPOSITOR_OUTPUT_DIR="$AO_OUT"
  export COMPOSITOR_SCENE_NAME="Current"
  export COMPOSITOR_RENDER_FAMILIES="ao"
  export COMPOSITOR_PATHWAY_EXR="$EXR_PATHWAY"
  export COMPOSITOR_PRIORITY_EXR="$EXR_PRIORITY"
  export COMPOSITOR_EXISTING_EXR="$EXR_EXISTING_POS"
  export COMPOSITOR_TRENDING_EXR="$EXR_TRENDING"
  "$BLENDER" --background --factory-startup \
    --python "${SCRIPTS}/render_edge_lab_current_core_outputs.py" \
    >"$AO_LOG" 2>&1
)
EXIT=$?
NPNG=$(find "$AO_OUT" -name "*.png" -type f 2>/dev/null | wc -l)
log "  AO EXIT=$EXIT PNG_COUNT=$NPNG"
if [ "$EXIT" -ne 0 ]; then FAIL=$((FAIL+1)); tail -8 "$AO_LOG" | tee -a "$BATCH_LOG"; fi

# ---------------------------------------------------------------
# 2. SHADING
# ---------------------------------------------------------------
log "=== [$(stamp)] SHADING ==="
SH_WC=$(copy_working_copy "compositor_shading.blend" "shading")
SH_OUT="${OUT_BASE}/shading__${TS}"
mkdir -p "$SH_OUT"
SH_LOG="${LOG_DIR}/city_yr180_shading_${TS}.log"
(
  export COMPOSITOR_BLEND_PATH="$SH_WC"
  export COMPOSITOR_OUTPUT_DIR="$SH_OUT"
  export COMPOSITOR_SCENE_NAME="Current"
  export COMPOSITOR_PATHWAY_EXR="$EXR_PATHWAY"
  export COMPOSITOR_PRIORITY_EXR="$EXR_PRIORITY"
  export COMPOSITOR_EXISTING_EXR="$EXR_EXISTING_POS"
  export COMPOSITOR_EXISTING_TRENDING_EXR="$EXR_EXISTING_TRN"
  export COMPOSITOR_BIOENVELOPE_EXR="$EXR_BIO_POS"
  export COMPOSITOR_BIOENVELOPE_TRENDING_EXR="$EXR_BIO_TRN"
  "$BLENDER" --background --factory-startup \
    --python "${SCRIPTS}/render_edge_lab_current_shading.py" \
    >"$SH_LOG" 2>&1
)
EXIT=$?
NPNG=$(find "$SH_OUT" -name "*.png" -type f 2>/dev/null | wc -l)
log "  SHADING EXIT=$EXIT PNG_COUNT=$NPNG"
if [ "$EXIT" -ne 0 ]; then FAIL=$((FAIL+1)); tail -8 "$SH_LOG" | tee -a "$BATCH_LOG"; fi

# ---------------------------------------------------------------
# 3. SIZES
# ---------------------------------------------------------------
log "=== [$(stamp)] SIZES ==="
SZ_WC=$(copy_working_copy "compositor_sizes.blend" "sizes")
SZ_OUT="${OUT_BASE}/sizes__${TS}"
mkdir -p "$SZ_OUT"
SZ_LOG="${LOG_DIR}/city_yr180_sizes_${TS}.log"
(
  export COMPOSITOR_BLEND_PATH="$SZ_WC"
  export COMPOSITOR_OUTPUT_DIR="$SZ_OUT"
  export COMPOSITOR_SCENE_NAME="Current"
  export COMPOSITOR_PATHWAY_EXR="$EXR_PATHWAY"
  export COMPOSITOR_PRIORITY_EXR="$EXR_PRIORITY"
  export COMPOSITOR_EXISTING_EXR="$EXR_EXISTING_POS"
  export COMPOSITOR_TRENDING_EXR="$EXR_TRENDING"
  "$BLENDER" --background --factory-startup \
    --python "${SCRIPTS}/render_edge_lab_current_sizes.py" \
    >"$SZ_LOG" 2>&1
)
EXIT=$?
NPNG=$(find "$SZ_OUT" -name "*.png" -type f 2>/dev/null | wc -l)
log "  SIZES EXIT=$EXIT PNG_COUNT=$NPNG"
if [ "$EXIT" -ne 0 ]; then FAIL=$((FAIL+1)); tail -8 "$SZ_LOG" | tee -a "$BATCH_LOG"; fi

# ---------------------------------------------------------------
# 4. PROPOSAL families (3 blends × 7 states)
# ---------------------------------------------------------------
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
  local family="$1"      # proposal-only / proposal-outline / proposal-colored-depth
  local canonical="$2"   # canonical blend filename
  local scene_name="$3"
  local out_node="$4"
  local needs_fix="$5"   # 1 to run fixup script, 0 otherwise

  log "=== [$(stamp)] ${family} ==="
  local wc="${TEMP_BLENDS}/compositor_${family}__${SITE_KEY}__${TS}.blend"
  cp -f "${CANONICAL}/${canonical}" "$wc"
  if [ "$needs_fix" = "1" ]; then
    local fix_log="${LOG_DIR}/city_yr180_${family}_fix_${TS}.log"
    "$BLENDER" --background --factory-startup \
      --python "${SCRIPTS}/_fix_proposal_cd_output_node_20260411.py" \
      -- "$wc" >"$fix_log" 2>&1
    local fix_exit=$?
    log "  FIX ${family} EXIT=$fix_exit"
    if [ "$fix_exit" -ne 0 ]; then FAIL=$((FAIL+1)); tail -8 "$fix_log" | tee -a "$BATCH_LOG"; return; fi
  fi
  local out_dir="${OUT_BASE}/${family}__${TS}"
  mkdir -p "$out_dir"
  local idx=0
  for SD in "${STATE_DEFS[@]}"; do
    idx=$((idx+1))
    local state="${SD%%|*}"
    local prefix="${SD##*|}"
    local exr="${EXR_BASE}/${SITE_KEY}__${state}__8k64s.exr"
    local per_log="${LOG_DIR}/city_yr180_${family}_${prefix}_${TS}.log"
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
      tail -8 "$per_log" | tee -a "$BATCH_LOG"
    fi
  done
}

run_proposal_family "proposal-only"          "proposal_only_layers.blend"          "ProposalOnly"               "ProposalOnlyOutput"          "0"
run_proposal_family "proposal-outline"       "proposal_outline_layers.blend"       "ProposalOutline"            "ProposalOutlineOutput"       "0"
run_proposal_family "proposal-colored-depth" "proposal_colored_depth_outlines.blend" "ProposalColoredDepthOutlines" "ProposalColoredDepthOutput" "1"

log "=== [$(stamp)] COMPOSITOR PHASE DONE fail=${FAIL} ==="

# ---------------------------------------------------------------
# 5. UPLOAD to Mediaflux
# ---------------------------------------------------------------
log "=== [$(stamp)] UPLOAD PHASE ==="
cd "$REPO"
export PATH="${REPO}/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/jre/bin:$PATH"
export MEDIAFLUX_CLIENT_BIN_DIR="${REPO}/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/bin/windows"
FAMILIES=(ao shading sizes proposal-only proposal-outline proposal-colored-depth)
for F in "${FAMILIES[@]}"; do
  SRC="${OUT_BASE}/${F}__${TS}"
  SUB="pipeline/4.9/compositor_pngs/${SITE_KEY}/${F}__${TS}"
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
