#!/usr/bin/env bash
set -euo pipefail
PYTHON="$1"
OUTPUT="${2:-causaflux_v1.7.0_validation}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
"$PYTHON" -m causaflux.cli validation-list
"$PYTHON" -m causaflux.cli validation-run --snapshots benchmarks/snapshots/sea_ad --output "$OUTPUT" --bootstrap "${CAUSAFLUX_VALIDATION_BOOTSTRAP:-500}"
"$PYTHON" -m causaflux.cli validation-validate --input "$OUTPUT"
