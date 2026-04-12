#!/usr/bin/env bash
set -euo pipefail

TS="$1"
export PATH="/d/2026 Arboreal Futures/urban-futures/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/jre/bin:$PATH"
export MEDIAFLUX_CLIENT_BIN_DIR="D:/2026 Arboreal Futures/urban-futures/.tools/mediaflux-bin/unimelb-mf-clients-0.8.5/bin/windows"
cd "d:/2026 Arboreal Futures/urban-futures"

ASSET_FAMILY="20260407_232744_ply-library-exr-4sides-large-senescing-snag_el20_4k64s"
AO_LOCAL="./_data-refactored/compositor/outputs/_library/${ASSET_FAMILY}/ao__${TS}__library-reserve-tree"
RES_LOCAL="./_data-refactored/compositor/outputs/_library/${ASSET_FAMILY}/resources__${TS}__library-reserve-tree"
AO_REMOTE="pipeline/_library/compositor_pngs/${ASSET_FAMILY}/ao__${TS}__library-reserve-tree"
RES_REMOTE="pipeline/_library/compositor_pngs/${ASSET_FAMILY}/resources__${TS}__library-reserve-tree"

echo "###### 1/2: AO upload ######"
./.venv/Scripts/python.exe -m mediafluxsync upload-project \
  "$AO_LOCAL" "$AO_REMOTE" \
  --project-dir . --create-parents --exclude-parent
echo "AO exit: $?"

echo ""
echo "###### 2/2: Resources upload ######"
./.venv/Scripts/python.exe -m mediafluxsync upload-project \
  "$RES_LOCAL" "$RES_REMOTE" \
  --project-dir . --create-parents --exclude-parent
echo "Resources exit: $?"

echo ""
echo "###### DONE ######"
