#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT="${1:-$ROOT/demo_outputs/cancer_quickstart}"
PYTHON="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "CausaFlux environment not found; preparing it first."
  bash "$ROOT/scripts/setup_mac.sh"
fi
bash "$ROOT/scripts/run_staged.sh" "$PYTHON" "$ROOT/demos/cancer_quickstart/config.yaml" "$OUTPUT"
echo "Cancer quickstart report: $OUTPUT/report/index.html"
