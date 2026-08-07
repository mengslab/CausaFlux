#!/usr/bin/env bash
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
echo "CausaFlux v1.8.0 macOS diagnostic"
echo "  macOS: $(sw_vers -productVersion 2>/dev/null || echo unknown)"
echo "  architecture: $(uname -m)"
echo "  shell: ${SHELL:-unknown}"
echo "  conda: $(command -v conda 2>/dev/null || echo not-found)"
if [[ -x .causaflux_env/bin/python ]]; then
  .causaflux_env/bin/python - <<'PY'
import platform, sys
print("  environment Python:", sys.version.split()[0])
print("  environment architecture:", platform.machine())
for package in ("numpy", "pandas", "torch", "sklearn", "networkx", "causaflux"):
    try:
        module = __import__(package)
        print(f"  {package}:", getattr(module, "__version__", "installed"))
    except Exception as exc:
        print(f"  {package} import error:", exc)
PY
else
  echo "  environment: not created"
fi
