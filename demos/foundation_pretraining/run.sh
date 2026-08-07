#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUTPUT="${1:-demo_outputs/foundation_pretraining}"
causaflux foundation-pretrain-run --output "$OUTPUT" --samples 900 --components 12 --seed 170 --require-gate
