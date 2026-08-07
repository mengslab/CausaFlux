# CausaFlux v1.4.0 API Reference

## Platform API

```python
from causaflux import (
    PLATFORM_VERSION,
    build_artifact_manifest,
    environment_snapshot,
    finalize_research_platform,
    get_demo_registry,
    platform_doctor,
    validate_research_platform,
)
```

- `get_demo_registry()` returns packaged `DemoSpec` objects.
- `environment_snapshot()` reports Python, system, and dependency versions.
- `build_artifact_manifest(output_dir)` returns file sizes and SHA-256 hashes.
- `finalize_research_platform(output_dir)` writes cards, provenance, validation, and the platform report.
- `validate_research_platform(output_dir)` returns a `PlatformValidationReport`.
- `platform_doctor()` checks Python and package readiness.

## Domain API

Major public interfaces are exported from `causaflux`:

- `run_causal_experiment`
- `generate_multimodal_mudata`, `read_multimodal`, `validate_multimodal`
- `build_spatial_heterograph`, `validate_spatial_graph`
- `run_counterfactual_therapeutics`
- `run_causal_biomarkers`
- `run_closed_loop_experimentation`
- `run_neurobiology_configuration`

## CLI

Use `causaflux --help` for the full command list. Platform commands are `version`, `doctor`, `demo-list`, `demo-run`, and `platform-validate`. Domain validators are available for multimodal, spatial, therapeutic, biomarker, experiment, and neurobiology outputs.

## Publication graphics API

```python
from causaflux import (
    EXPORT_PROFILES,
    apply_publication_style,
    export_figure,
    rebuild_reference_figures,
    validate_publication_bundle,
)
```

- `EXPORT_PROFILES` provides Nature and Cell single- and double-column dimensions.
- `apply_publication_style(profile)` applies centralized typography, line weights, and color-safe palettes.
- `export_figure(...)` writes PNG, LZW TIFF, SVG, PDF, panel source data, and a figure manifest.
- `rebuild_reference_figures(output_dir)` regenerates the complete publication bundle.
- `validate_publication_bundle(output_dir)` checks formats, dimensions, hashes, source data, and visual baselines.

CLI equivalents are `publication-build` and `publication-validate`.

## Biological validation API (v1.4.0)

```python
from causaflux import (
    load_hypothesis_registry,
    run_biological_validation,
    write_biological_validation,
    validate_biological_validation,
)

hypotheses = load_hypothesis_registry()
run = run_biological_validation("benchmarks/snapshots/sea_ad", n_boot=500)
write_biological_validation(run, "validation_output")
assert validate_biological_validation("validation_output")["valid"]
```

CLI commands: `validation-list`, `validation-preregister`, `validation-run`, and `validation-validate`.

## Dynamic model benchmark API (v1.4.0)

```python
from causaflux import DynamicBenchmarkConfig, run_dynamic_benchmark

status = run_dynamic_benchmark(
    "dynamic_benchmark",
    DynamicBenchmarkConfig(epochs=28, replicates_per_history=4),
)
```

External dataset workflow:

```python
from causaflux import load_external_benchmark_npz, run_dynamic_benchmark

data = load_external_benchmark_npz("my_longitudinal_dataset.npz")
status = run_dynamic_benchmark("external_result", data=data)
```

Primary functions:

- `generate_dynamic_benchmark_data`
- `make_split`
- `run_dynamic_benchmark`
- `validate_dynamic_benchmark`
- `save_external_benchmark_npz`
- `load_external_benchmark_npz`
- `validate_external_benchmark_data`
- `external_benchmark_contract`

## Multimodal dynamic state API (v1.4.0)

```python
from causaflux import (
    MultimodalDynamicConfig,
    generate_multimodal_dynamic_data,
    run_multimodal_dynamic_benchmark,
    validate_multimodal_dynamic_benchmark,
    save_external_multimodal_npz,
    load_external_multimodal_npz,
)
```

`MultimodalDynamicConfig` controls the longitudinal context window, encoder/fusion latent size, modality-dropout probability, optimization, donor-bootstrap replicates, and device.

`run_multimodal_dynamic_benchmark()` writes the model comparison, cross-modal decoder metrics, uncertainty coverage, missingness sensitivity, split audit, exit gate, figures, source data, model checkpoints, model/dataset cards, and artifact hashes.

`save_external_multimodal_npz()` and `load_external_multimodal_npz()` implement the v1.4 external benchmark contract without requiring core model changes.

## Spatiotemporal digital tissue (v1.7.0)

Python API:

```python
from causaflux import (
    SpatiotemporalTissueConfig,
    generate_spatiotemporal_tissue_data,
    run_spatiotemporal_tissue_benchmark,
    validate_spatiotemporal_tissue,
)

cfg = SpatiotemporalTissueConfig()
data = generate_spatiotemporal_tissue_data(cfg)
run_spatiotemporal_tissue_benchmark("spatiotemporal_tissue", cfg, data=data, require_gate=True)
```

CLI:

```bash
causaflux spatiotemporal-tissue-run --output spatiotemporal_tissue --require-gate
causaflux spatiotemporal-tissue-validate --input spatiotemporal_tissue
causaflux spatiotemporal-tissue-export-fixture --output tissue_fixture.npz
causaflux nicheformer-adapter-show
```
