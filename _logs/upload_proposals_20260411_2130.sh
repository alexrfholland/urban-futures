#!/usr/bin/env bash
# Upload all 9 proposal output dirs to Mediaflux.
set -uo pipefail
REPO="/d/2026 Arboreal Futures/urban-futures"
cd "$REPO"
export PATH="${REPO}/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/jre/bin:$PATH"
export MEDIAFLUX_CLIENT_BIN_DIR="${REPO}/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/bin/windows"
TS="20260411_2130"
BLEND_KEYS=(proposal-only proposal-outline proposal-colored-depth)
TIMELINES=(city_timeline parade_timeline uni_timeline)
LOG_DIR="${REPO}/_logs/compositor"
BATCH_LOG="${LOG_DIR}/upload_proposals_${TS}.log"
: > "$BATCH_LOG"
PY=".venv/Scripts/python.exe"
FAIL=0
for BK in "${BLEND_KEYS[@]}"; do
  for T in "${TIMELINES[@]}"; do
    SRC="${REPO}/_data-refactored/compositor/outputs/4.9/${T}/${BK}__${TS}"
    SUB="pipeline/4.9/compositor_pngs/${T}/${BK}__${TS}"
    STAMP=$(date +%H:%M:%S)
    echo "[$STAMP] uploading ${BK} ${T}" | tee -a "$BATCH_LOG"
    echo "  SRC=${SRC}" | tee -a "$BATCH_LOG"
    echo "  DST=${SUB}" | tee -a "$BATCH_LOG"
    "$PY" -m mediafluxsync upload-project --create-parents --exclude-parent --project-dir . "$SRC" "$SUB" >>"$BATCH_LOG" 2>&1
    EXIT=$?
    echo "  EXIT=$EXIT" | tee -a "$BATCH_LOG"
    if [ "$EXIT" -ne 0 ]; then
      FAIL=$((FAIL+1))
    fi
  done
done
echo "=== UPLOAD DONE fail=${FAIL} ===" | tee -a "$BATCH_LOG"
