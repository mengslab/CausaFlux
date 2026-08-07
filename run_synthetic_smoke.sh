#!/usr/bin/env bash
# Re-exec with Bash when the user starts this file with: sh run.sh
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
ENV_DIR="${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}"
export CAUSAFLUX_ENV="$ENV_DIR"

python_is_supported() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)
PY
}

if [[ -x "$ENV_DIR/bin/python" ]] && ! python_is_supported "$ENV_DIR/bin/python"; then
  echo "Removing incompatible CausaFlux environment: $($ENV_DIR/bin/python --version 2>&1)"
  rm -rf "$ENV_DIR"
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Preparing a compatible Python 3.10-3.12 environment..."
  bash "$ROOT/scripts/setup_mac.sh"
else
  echo "Using existing CausaFlux environment: $($ENV_DIR/bin/python --version 2>&1)"
  "$ENV_DIR/bin/python" -m pip install --quiet -e .
fi

PYTHON="$ENV_DIR/bin/python"
if ! python_is_supported "$PYTHON"; then
  echo "ERROR: CausaFlux requires Python 3.10, 3.11, or 3.12; found $($PYTHON --version 2>&1)." >&2
  exit 1
fi

OUTPUT="${1:-causaflux_v1.7.0_output}"
bash scripts/run_staged.sh "$PYTHON" configs/cancer_closed_loop_v1.7.0.yaml "$OUTPUT"
REPORT="$ROOT/$OUTPUT/report/index.html"
if [[ -f "$REPORT" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "$REPORT"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$REPORT"
  fi
fi
