#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUTPUT="${1:-demo_outputs/biological_validation}"
PYTHON="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3 || command -v python)"; fi
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m causaflux.cli validation-run --snapshots benchmarks/snapshots/sea_ad --output "$OUTPUT" --bootstrap 100
