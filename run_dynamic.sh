#!/bin/sh
set -eu
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.causaflux_env/bin/python"
if [ ! -x "$PYTHON" ] || ! "$PYTHON" -c 'import causaflux, torch' >/dev/null 2>&1; then
  bash "$ROOT_DIR/scripts/setup_mac.sh"
fi
OUTPUT_DIR=${CAUSAFLUX_OUTPUT:-causaflux_v0.2_output}
CONFIG_FILE=${CAUSAFLUX_CONFIG:-configs/demo_v0.2.yaml}
"$PYTHON" -m causaflux.cli dynamic-run --config "$CONFIG_FILE" --output "$OUTPUT_DIR"
REPORT="$ROOT_DIR/$OUTPUT_DIR/report/index.html"
echo "Dynamic-core report: $REPORT"
if [ "$(uname -s)" = "Darwin" ] && command -v open >/dev/null 2>&1; then
  open "$REPORT" >/dev/null 2>&1 || true
fi
