#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}/bin/python"
[[ -x "$PY" ]] || { echo "Run sh run.sh once to create the environment"; exit 1; }
"$PY" -m causaflux.cli benchmark-report --output demo_outputs/realdata_registry --project-root "$ROOT"
