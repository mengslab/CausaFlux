# CausaFlux v1.9.0 — Real-World Data Hub

v1.9 connects the virtual-cell layer to the accession-pinned real-data framework introduced in prior releases.

The packaged registry covers cancer spatial, cancer molecular, perturbational, neural/glial, AMP-AD molecular, and neurophysiology benchmark families. Large and controlled datasets are not redistributed.

## Evidence classes

CausaFlux keeps separate labels for:

- accession-ready metadata;
- real observational replication;
- real perturbational validation;
- real prospective validation.

An observational association is never promoted to a causal or prospective claim simply because it is used by the same application.

## User dataset contracts

`causaflux realworld-register` freezes the local source path, content SHA-256, modality list, longitudinal/perturbational/spatial/prospective flags and analysis-relevant column mappings. This provides a reproducible bridge from LIMS/ELN or exported data into later model training and prospective cycles.
