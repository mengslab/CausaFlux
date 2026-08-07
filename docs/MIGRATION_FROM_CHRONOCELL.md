# Migration from ChronoCell to CausaFlux

The project was renamed beginning with version 0.4.0.

| Former interface | CausaFlux v0.4.0 |
|---|---|
| `ChronoCell_v0.3.1.zip` | `CausaFlux_v0.4.0.zip` |
| Python package `chronocell` | Python package `causaflux` |
| CLI `chronocell` | CLI `causaflux` |
| `CHRONOCELL_CONFIG` | `CAUSAFLUX_CONFIG` |
| `CHRONOCELL_OUTPUT` | `CAUSAFLUX_OUTPUT` |
| `chronocell_v..._output` | `causaflux_v..._output` |
| model class `ChronoCell` | model class `CausaFlux` |
| model config `ChronoCellConfig` | `CausaFluxConfig` |

The earlier `ChronoDataset` class remains available as an internal compatibility
alias. New code should use `CausaFluxDataset`.

Configuration concepts and prior causal outputs are retained, but v0.4.0 state models
use fused multimodal features by default.
