#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
OUTPUT="${1:-demo_outputs/intervention_generalization}"
ENV_DIR="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then bash scripts/setup_mac.sh; fi
"$ENV_DIR/bin/causaflux" intervention-generalization-run --output "$OUTPUT" --replicates "${CAUSAFLUX_IG_REPLICATES:-3}" --bootstrap "${CAUSAFLUX_IG_BOOTSTRAP:-30}" --require-gate
"$ENV_DIR/bin/causaflux" intervention-generalization-validate --input "$OUTPUT"
