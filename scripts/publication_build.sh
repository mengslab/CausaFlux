#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${1:-python3}"
OUTPUT="${2:-causaflux_v1.7.0_output}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
for group in core spatial therapeutics biomarkers active_learning neurobiology; do
  "$PYTHON_BIN" -u scripts/stage_publication_graphics.py --output "$OUTPUT" --worker-group "$group"
done
"$PYTHON_BIN" -u scripts/stage_publication_graphics.py --output "$OUTPUT" --finalize-only
