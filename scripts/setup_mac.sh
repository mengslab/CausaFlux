#!/usr/bin/env bash
# Re-exec with Bash when invoked through /bin/sh.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
ENV_DIR="${CAUSAFLUX_ENV:-$ROOT_DIR/.causaflux_env}"

version_is_supported() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)
PY
}

resolve_executable() {
  local candidate="$1"
  if command -v "$candidate" >/dev/null 2>&1; then
    command -v "$candidate"
    return 0
  fi
  if [[ -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  return 1
}

select_python() {
  local candidate resolved

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if resolved="$(resolve_executable "$PYTHON_BIN")" && version_is_supported "$resolved"; then
      SELECTED_PYTHON="$resolved"
      return 0
    fi
    echo "PYTHON_BIN=$PYTHON_BIN is unavailable or is not Python 3.10-3.12." >&2
    return 1
  fi

  # Intel macOS has the most reliable PyTorch 2.2 wheel coverage with Python 3.11.
  if [[ "$OS_NAME" == "Darwin" && "$ARCH_NAME" == "x86_64" ]]; then
    CANDIDATES=(
      python3.11 python3.12 python3.10
      /usr/local/bin/python3.11 /usr/local/bin/python3.12 /usr/local/bin/python3.10
      /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.10
      python3
    )
  else
    CANDIDATES=(
      python3.12 python3.11 python3.10
      /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10
      /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3.10
      python3
    )
  fi

  for candidate in "${CANDIDATES[@]}"; do
    if resolved="$(resolve_executable "$candidate")" && version_is_supported "$resolved"; then
      SELECTED_PYTHON="$resolved"
      return 0
    fi
  done
  return 1
}

find_conda() {
  local candidate
  CANDIDATES=(
    "${CONDA_EXE:-}"
    conda
    "$HOME/anaconda3/bin/conda"
    "$HOME/miniconda3/bin/conda"
    "/opt/anaconda3/bin/conda"
    "/usr/local/anaconda3/bin/conda"
    "/opt/miniconda3/bin/conda"
    "/usr/local/miniconda3/bin/conda"
  )
  for candidate in "${CANDIDATES[@]}"; do
    [[ -z "$candidate" ]] && continue
    if CONDA_BIN="$(resolve_executable "$candidate")"; then
      return 0
    fi
  done
  return 1
}

create_environment() {
  SELECTED_PYTHON=""
  if select_python; then
    echo "Creating CausaFlux environment with $($SELECTED_PYTHON --version 2>&1)"
    "$SELECTED_PYTHON" -m venv "$ENV_DIR"
    return 0
  fi

  if find_conda; then
    local conda_python="3.12"
    if [[ "$OS_NAME" == "Darwin" && "$ARCH_NAME" == "x86_64" ]]; then
      conda_python="3.11"
    fi
    echo "No compatible standalone Python was found."
    echo "Creating a local Conda environment with Python $conda_python in: $ENV_DIR"
    "$CONDA_BIN" create --yes --prefix "$ENV_DIR" "python=$conda_python" pip
    return 0
  fi

  cat >&2 <<'MSG'
CausaFlux v2.0.0 requires Python 3.10, 3.11, or 3.12.
The active Python 3.13 installation cannot be used with this release.
Install Python 3.11/3.12 or Anaconda/Miniconda, then rerun:

  sh run.sh
MSG
  return 1
}

if [[ -x "$ENV_DIR/bin/python" ]] && ! version_is_supported "$ENV_DIR/bin/python"; then
  echo "Removing incompatible environment: $($ENV_DIR/bin/python --version 2>&1)"
  rm -rf "$ENV_DIR"
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  create_environment
fi

PYTHON="$ENV_DIR/bin/python"
PIP=("$PYTHON" -m pip)
if ! version_is_supported "$PYTHON"; then
  echo "Installation stopped: $($PYTHON --version 2>&1) is unsupported." >&2
  exit 1
fi

echo "Using environment Python: $($PYTHON --version 2>&1)"
echo "System: $OS_NAME $ARCH_NAME"
"${PIP[@]}" install --upgrade "pip<27" setuptools wheel

if [[ "$OS_NAME" == "Darwin" && "$ARCH_NAME" == "x86_64" ]]; then
  echo "Intel Mac detected: installing the compatible PyTorch/NumPy set."
  "${PIP[@]}" install "numpy>=1.26,<2" "torch==2.2.2"
fi

"${PIP[@]}" install -e ".[dev]"

"$PYTHON" - <<'PY'
import platform, sys
import numpy as np
import torch
import anndata
import mudata
import causaflux
print("CausaFlux environment verification")
print("  CausaFlux:", causaflux.__version__)
print("  Python:", sys.version.split()[0])
print("  Architecture:", platform.machine())
print("  NumPy:", np.__version__)
print("  PyTorch:", torch.__version__)
print("  AnnData:", anndata.__version__)
print("  MuData:", mudata.__version__)
print("  MPS available:", bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available())
PY

echo "CausaFlux v2.0.0 installation is ready."
