#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT="${1:-$ROOT/demo_outputs/neurobiology_quickstart}"
PYTHON="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "CausaFlux environment not found; preparing it first."
  bash "$ROOT/scripts/setup_mac.sh"
fi
"$PYTHON" -m causaflux.cli neuro-run --config "$ROOT/demos/neurobiology_quickstart/config.yaml" --output "$OUTPUT"
"$PYTHON" -m causaflux.cli neuro-validate --input "$OUTPUT"
echo "Neurobiology quickstart report: $OUTPUT/neurobiology_report.html"
