#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [ ! -x .causaflux_env/bin/python ]; then bash scripts/setup_mac.sh; fi
source .causaflux_env/bin/activate
OUTPUT="${1:-causaflux_v1.8.0_prospective_loop}"
causaflux prospective-run --output "$OUTPUT" --require-gate
causaflux prospective-validate --input "$OUTPUT" --require-gate
printf 'CausaFlux v1.8.0 prospective loop complete: %s\n' "$OUTPUT"
if command -v open >/dev/null 2>&1 && [ -f "$OUTPUT/report/index.html" ]; then
  open "$OUTPUT/report/index.html" >/dev/null 2>&1 || true
fi
