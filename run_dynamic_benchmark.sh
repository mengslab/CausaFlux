#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
ENV_DIR="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}"; export CAUSAFLUX_ENV="$ENV_DIR"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  bash scripts/setup_mac.sh
else
  "$ENV_DIR/bin/python" -m pip install --quiet -e .
fi
OUTPUT="${1:-causaflux_v1.7.0_dynamic_benchmark}"
EPOCHS="${CAUSAFLUX_DYNAMIC_EPOCHS:-28}"
REPLICATES="${CAUSAFLUX_DYNAMIC_REPLICATES:-4}"
"$ENV_DIR/bin/causaflux" dynamic-benchmark-run \
  --output "$OUTPUT" \
  --epochs "$EPOCHS" \
  --replicates-per-history "$REPLICATES" \
  --require-gate
"$ENV_DIR/bin/causaflux" dynamic-benchmark-validate --input "$OUTPUT"
REPORT="$ROOT/$OUTPUT/report/index.html"
if [[ -f "$REPORT" ]]; then command -v open >/dev/null && open "$REPORT" || true; fi
printf 'CausaFlux v1.7.0 dynamic benchmark complete: %s\n' "$OUTPUT"
