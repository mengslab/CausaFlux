#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUTPUT="${1:-$ROOT/demo_outputs/dynamic_model_benchmark}"
PYTHON="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}/bin/python"
if [[ ! -x "$PYTHON" ]]; then bash "$ROOT/scripts/setup_mac.sh"; fi
"$PYTHON" -m causaflux.cli dynamic-benchmark-run \
  --output "$OUTPUT" \
  --epochs 12 --patience 4 --hidden-dim 40 --replicates-per-history 3 \
  --require-gate
"$PYTHON" -m causaflux.cli dynamic-benchmark-validate --input "$OUTPUT"
printf 'Dynamic benchmark report: %s\n' "$OUTPUT/report/index.html"
