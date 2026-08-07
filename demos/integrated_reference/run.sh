#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
bash "$ROOT/run_synthetic_smoke.sh" "${1:-causaflux_v1.4.0_output}"
