#!/usr/bin/env bash
# Re-run the 21 proposal-colored-depth passes after fixing the OutputFile node.
# Reuses the same TS=20260411_2130 so the output dir name matches the other two
# proposal families. Uses the rebuilt working copy at
# compositor_proposal-colored-depth__batch_fixed__20260411_2130.blend.
set -uo pipefail

REPO="/d/2026 Arboreal Futures/urban-futures"
BLENDER="/c/Program Files/Blender Foundation/Blender 4.2/blender.exe"
SCRIPT="${REPO}/_futureSim_refactored/blender/compositor/scripts/_run_proposal_subcategory_20260411.py"
WC="${REPO}/_data-refactored/compositor/temp_blends/template_instantiations/compositor_proposal-colored-depth__batch_fixed__20260411_2130.blend"
TS="20260411_2130"
BLEND_KEY="proposal-colored-depth"
OUT_NODE="ProposalColoredDepthOutput"
SCENE_NAME="ProposalColoredDepthOutlines"
LOG_DIR="${REPO}/_logs/compositor"
mkdir -p "$LOG_DIR"
BATCH_LOG="${LOG_DIR}/proposals_rerun_cd_${TS}.log"
echo "BATCH_TS=${TS}  WC=${WC}" | tee "$BATCH_LOG"

TIMELINES=(city_timeline parade_timeline uni_timeline)
STATE_DEFS=(
  "existing_condition_positive|existing_positive"
  "existing_condition_trending|existing_trending"
  "positive_state|pathway"
  "positive_priority_state|priority"
  "trending_state|trending"
  "bioenvelope_positive|bio_positive"
  "bioenvelope_trending|bio_trending"
)
N_TOTAL=$((${#TIMELINES[@]} * ${#STATE_DEFS[@]}))
IDX=0
FAIL=0
for T in "${TIMELINES[@]}"; do
  OUT_DIR="${REPO}/_data-refactored/compositor/outputs/4.9/${T}/${BLEND_KEY}__${TS}"
  mkdir -p "$OUT_DIR"
  EXR_BASE="${REPO}/_data-refactored/blenderv2/output/4.9/${T}"
  for SD in "${STATE_DEFS[@]}"; do
    IDX=$((IDX+1))
    STATE="${SD%%|*}"
    PREFIX="${SD##*|}"
    EXR="${EXR_BASE}/${T}__${STATE}__8k64s.exr"
    STAMP=$(date +%H:%M:%S)
    echo "[$STAMP] [${IDX}/${N_TOTAL}] ${T} ${STATE} prefix=${PREFIX}" | tee -a "$BATCH_LOG"
    if [ ! -f "$EXR" ]; then
      echo "  MISSING EXR: $EXR" | tee -a "$BATCH_LOG"
      FAIL=$((FAIL+1))
      continue
    fi
    PER_LOG="${LOG_DIR}/proposals_rerun_${BLEND_KEY}_${T}_${PREFIX}_${TS}.log"
    export COMPOSITOR_BLEND_PATH="$WC"
    export COMPOSITOR_OUTPUT_DIR="$OUT_DIR"
    export COMPOSITOR_EXR="$EXR"
    export COMPOSITOR_OUTPUT_NODE_NAME="$OUT_NODE"
    export COMPOSITOR_SCENE_NAME="$SCENE_NAME"
    export COMPOSITOR_FILENAME_PREFIX="$PREFIX"
    "$BLENDER" --background --factory-startup --python "$SCRIPT" >"$PER_LOG" 2>&1
    EXIT=$?
    NPNG=$(find "$OUT_DIR" -name "${PREFIX}__*.png" -type f 2>/dev/null | wc -l)
    echo "  EXIT=$EXIT PNG_COUNT_for_this_prefix=$NPNG" | tee -a "$BATCH_LOG"
    if [ "$EXIT" -ne 0 ] || [ "$NPNG" -ne 7 ]; then
      FAIL=$((FAIL+1))
      echo "  -- tail $PER_LOG --" | tee -a "$BATCH_LOG"
      tail -6 "$PER_LOG" | tee -a "$BATCH_LOG"
    fi
  done
done
echo "=== DONE total=${N_TOTAL} fail=${FAIL} ===" | tee -a "$BATCH_LOG"
