#!/usr/bin/env bash
# CausaFlux v2.0.0 — one-command stable software release/evidence workflow.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [ ! -x .causaflux_env/bin/python ]; then bash scripts/setup_mac.sh; fi
PYTHON="$ROOT/.causaflux_env/bin/python"
OUTPUT="${1:-causaflux_v2.0.0_release}"
"$PYTHON" -m causaflux.cli v2-run --output "$OUTPUT"
"$PYTHON" -m causaflux.cli v2-validate --input "$OUTPUT"
printf '\nCausaFlux v2.0.0 software release bundle complete.\nReport: %s/report/index.html\n\nIMPORTANT: the report itself determines whether the real prospectively validated claim is eligible.\n' "$OUTPUT"
if command -v open >/dev/null 2>&1 && [ -f "$OUTPUT/report/index.html" ]; then open "$OUTPUT/report/index.html" >/dev/null 2>&1 || true; fi
