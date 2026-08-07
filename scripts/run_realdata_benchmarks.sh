#!/usr/bin/env bash
set -euo pipefail
PYTHON="$1"
OUTPUT="${2:-causaflux_v1.7.0_realdata}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
"$PYTHON" -m causaflux.cli benchmark-preflight --output "$OUTPUT/preflight"
"$PYTHON" -m causaflux.cli benchmark-report --output "$OUTPUT" --project-root "$ROOT"
"$PYTHON" -m causaflux.cli benchmark-validate --input "$OUTPUT"
