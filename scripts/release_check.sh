#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${1:-${CAUSAFLUX_ENV:-$ROOT/.causaflux_env}/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3 || command -v python)"; fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONWARNINGS=ignore
"$PYTHON" -m compileall -q src scripts tests

# One fresh process per test module avoids native scientific-library state accumulation.
for test_file in tests/test_*.py; do
  "$PYTHON" -m pytest -q "$test_file"
done

"$PYTHON" scripts/verify_prospective_loop.py prospective_loop_reference
"$PYTHON" scripts/verify_foundation_pretraining.py foundation_pretraining_reference
"$PYTHON" scripts/verify_dynamic_benchmark.py dynamic_benchmark_reference
"$PYTHON" scripts/verify_multimodal_dynamic.py multimodal_dynamic_reference
"$PYTHON" scripts/verify_intervention_generalization.py intervention_generalization_reference
"$PYTHON" scripts/verify_spatiotemporal_tissue.py spatiotemporal_tissue_reference
"$PYTHON" scripts/verify_release.py reference_demo
"$PYTHON" scripts/verify_publication_graphics.py reference_demo >/dev/null
"$PYTHON" scripts/verify_realdata_release.py realdata_reference
"$PYTHON" scripts/verify_biological_validation.py biological_validation_reference
"$PYTHON" -m causaflux.cli foundation-pretrain-validate --input foundation_pretraining_reference >/dev/null
"$PYTHON" -m causaflux.cli dynamic-benchmark-validate --input dynamic_benchmark_reference >/dev/null
"$PYTHON" -m causaflux.cli multimodal-dynamic-validate --input multimodal_dynamic_reference >/dev/null
"$PYTHON" -m causaflux.cli spatiotemporal-tissue-validate --input spatiotemporal_tissue_reference >/dev/null
"$PYTHON" -m causaflux.cli validation-validate --input biological_validation_reference >/dev/null
"$PYTHON" -m causaflux.cli benchmark-validate --input realdata_reference >/dev/null
"$PYTHON" -m causaflux.cli platform-validate --input reference_demo >/dev/null
bash -n run.sh run_prospective_loop.sh run_dynamic_benchmark.sh scripts/*.sh demos/*/run.sh
printf 'CausaFlux v1.8.0 prospective-experimental-loop release checks passed.\n'
