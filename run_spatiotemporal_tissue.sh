#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
OUTPUT="${1:-causaflux_v1.7.0_spatiotemporal_tissue}"
ENV_DIR="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then bash scripts/setup_mac.sh; fi
"$ENV_DIR/bin/causaflux" spatiotemporal-tissue-run --output "$OUTPUT" \
  --donors "${CAUSAFLUX_ST_DONORS:-12}" --sections-per-donor "${CAUSAFLUX_ST_SECTIONS:-2}" \
  --cells-per-section "${CAUSAFLUX_ST_CELLS:-36}" --bootstrap "${CAUSAFLUX_ST_BOOTSTRAP:-100}" --require-gate
"$ENV_DIR/bin/causaflux" spatiotemporal-tissue-validate --input "$OUTPUT"
printf 'CausaFlux v1.7.0 spatiotemporal digital tissue complete: %s\n' "$OUTPUT"
