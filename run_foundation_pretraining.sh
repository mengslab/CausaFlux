#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [ ! -x .causaflux_env/bin/python ]; then bash scripts/setup_mac.sh; fi
source .causaflux_env/bin/activate
OUTPUT="${1:-causaflux_v1.7.0_foundation_pretraining}"
causaflux foundation-pretrain-run --output "$OUTPUT" --samples 900 --components 12 --seed 170 --require-gate
printf 'CausaFlux v1.7.0 foundation pretraining complete: %s\n' "$OUTPUT"
