#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUTPUT="${1:-demo_outputs/multimodal_dynamic_state}"
ENV_DIR="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then bash scripts/setup_mac.sh; fi
"$ENV_DIR/bin/causaflux" multimodal-dynamic-run --output "$OUTPUT" --epochs "${CAUSAFLUX_MM_EPOCHS:-18}" --replicates-per-history "${CAUSAFLUX_MM_REPLICATES:-3}" --bootstrap "${CAUSAFLUX_MM_BOOTSTRAP:-30}" --require-gate
"$ENV_DIR/bin/causaflux" multimodal-dynamic-validate --input "$OUTPUT"
