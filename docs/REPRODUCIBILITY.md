# Reproducibility and Provenance

Every completed v1.0 platform run contains:

```text
run_manifest.json
experiment_config.yaml
stage_status.json
cards/
provenance/environment.json
provenance/artifact_manifest.csv
provenance/provenance_summary.json
provenance/platform_validation.json
provenance/platform_validation.csv
```

`artifact_manifest.csv` records relative path, artifact category, byte size, and SHA-256 hash. The manifest excludes its own provenance directory to avoid recursive hashing.

`environment.json` records Python, platform, architecture, package versions, and numerical-thread environment variables.

To refresh provenance after an approved extension:

```bash
causaflux platform-validate --input <output> --refresh
```

For publication or collaboration, preserve the exact configuration, full output directory, package archive checksum, and external raw-data accession or immutable storage identifier.
