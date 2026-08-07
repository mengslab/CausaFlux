#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [[ ! -x .causaflux_env/bin/python ]]; then
  bash scripts/setup_mac.sh
fi
if ! .causaflux_env/bin/python -c 'import streamlit' >/dev/null 2>&1; then
  .causaflux_env/bin/python -m pip install -e ".[app]"
fi
exec .causaflux_env/bin/python -m streamlit run app/streamlit_app.py
