#!/usr/bin/env bash
# Launch the optional interactive CausaFlux v2.0.0 Streamlit UI.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [ ! -x .causaflux_env/bin/python ]; then bash scripts/setup_mac.sh; fi
PYTHON="$ROOT/.causaflux_env/bin/python"
OUTPUT="${1:-causaflux_v2.0.0_release}"
if ! "$PYTHON" -c 'import streamlit' >/dev/null 2>&1; then
  echo "Installing the optional local UI dependency..."
  "$PYTHON" -m pip install -e '.[app]'
fi
export CAUSAFLUX_OUTPUT="$OUTPUT"
exec "$PYTHON" -m causaflux.cli ui --output "$OUTPUT"
