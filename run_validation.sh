#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
ENV_DIR="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}"; export CAUSAFLUX_ENV="$ENV_DIR"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then bash scripts/setup_mac.sh; else "$ENV_DIR/bin/python" -m pip install --quiet -e .; fi
OUTPUT="${1:-causaflux_v1.7.0_validation}"
bash scripts/run_biological_validation.sh "$ENV_DIR/bin/python" "$OUTPUT"
REPORT="$ROOT/$OUTPUT/reports/index.html"
if [[ -f "$REPORT" ]]; then command -v open >/dev/null && open "$REPORT" || true; fi
