#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
PYTHON="$ROOT/.causaflux_env/bin/python"
OUTPUT="${1:-causaflux_v2.0.0_release}"
"$PYTHON" -m causaflux.cli v2-validate --input "$OUTPUT" --require-prospectively-validated
