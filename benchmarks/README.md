# CausaFlux real-data benchmark registry

This directory defines six benchmark families with discovery cohorts, independent validation cohorts, access classes, repository adapters, citations, license records, and leakage controls.

## Contents

- `manifests/`: accession-pinned benchmark definitions
- `licenses/`: source-specific access and license matrix
- `citations/`: BibTeX references
- `snapshots/sea_ad/`: two unchanged public metadata workbooks
- `adapter_capabilities.csv`: supported repository adapter modes
- `independent_validation_cohorts.csv`: discovery/validation separation

The registry is intentionally non-destructive. Run `causaflux benchmark-plan` to create an access plan. It does not download data.

## Dynamic model benchmark fixture

`fixtures/dynamic_benchmark_fixture_v1.4.0.npz` follows the external dynamic benchmark contract and contains the deterministic software-validation trajectories used by v1.4.0. It is not a biological dataset. The fixture exists to verify model implementations, split policies, metrics, uncertainty calibration, and the performance gate.
