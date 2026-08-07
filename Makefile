.PHONY: setup test verify verify-dynamic verify-prospective release-check demo app clean

setup:
	bash scripts/setup_mac.sh

test:
	$${CAUSAFLUX_ENV:-.causaflux_env}/bin/python -m pytest

verify:
	$${CAUSAFLUX_ENV:-.causaflux_env}/bin/python scripts/verify_release.py reference_demo

verify-dynamic:
	$${CAUSAFLUX_ENV:-.causaflux_env}/bin/python scripts/verify_dynamic_benchmark.py dynamic_benchmark_reference

verify-prospective:
	$${CAUSAFLUX_ENV:-.causaflux_env}/bin/python scripts/verify_prospective_loop.py prospective_loop_reference

release-check:
	bash scripts/release_check.sh

demo:
	sh run.sh

app:
	bash scripts/run_app.sh

clean:
	rm -rf .pytest_cache build dist src/*.egg-info src/causaflux/__pycache__ tests/__pycache__ scripts/__pycache__
