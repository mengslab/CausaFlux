#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUTPUT="${1:-demo_outputs/spatiotemporal_digital_tissue}"
ENV_DIR="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then bash scripts/setup_mac.sh; fi
"$ENV_DIR/bin/causaflux" spatiotemporal-tissue-run --output "$OUTPUT" --donors 12 --sections-per-donor 2 --cells-per-section 30 --bootstrap 40 --require-gate
"$ENV_DIR/bin/causaflux" spatiotemporal-tissue-validate --input "$OUTPUT"
