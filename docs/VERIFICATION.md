# CausaFlux v1.7.0 Verification

The release verifier checks four layers.

## Package integrity

- framework and version consistency;
- required source, configuration, documentation, and output files;
- ZIP integrity and clean-extraction import behavior.

## Multimodal integrity

- standard H5MU root hierarchy;
- five aligned AnnData modalities;
- `obsm/spatial` with two coordinate dimensions;
- unique observation and fused-feature identifiers;
- finite modality benchmarks.

## Spatial graph integrity

- all six node types are present;
- every edge references an exported node;
- proximity and ligand–receptor edge types are distinct;
- distances and communication scores are nonnegative;
- all niche labels belong to the declared vocabulary;
- communication-circuit intervals contain the point estimates;
- PyG-compatible metadata contains node and edge type declarations;
- report assets and GraphML exports exist.

## Statistical integrity

- no donor overlap in held-out folds;
- state probabilities sum to one;
- bootstrap fits complete successfully;
- transition intervals contain their means;
- ensemble mutual information is nonnegative.

Run:

```bash
.causaflux_env/bin/python -m pytest
.causaflux_env/bin/python scripts/verify_release.py causaflux_v1.4.0_output
```

## v1.4.0 counterfactual therapeutic checks

The release verifier additionally requires:

- unique intervention and regimen identifiers;
- gene, drug, combination, sequence, and timing prediction tables;
- resistance probabilities and normal-cell toxicity bounded in `[0, 1]`;
- ordered donor-bootstrap intervals;
- at least one successful therapeutic bootstrap replicate per regimen;
- zero donor overlap in the therapeutic surrogate split manifest;
- unique integer therapeutic ranks beginning at one;
- report references to all therapeutic figures;
- a passing `therapeutic_qc.json` result.

## v1.4.0 causal biomarker checks

The verifier additionally requires:

- all candidate measurements evaluated only before the terminal outcome time;
- complete ranking, time-course, bootstrap, panel, and assay-manifest tables;
- biomarker scores bounded in `[0, 1]`;
- ordered donor-bootstrap score intervals;
- unique ranks beginning at one;
- association AUC values bounded in `[0.5, 1]` after direction orientation;
- leave-one-donor-out panel AUC values bounded in `[0.5, 1]`;
- held-out donor labels equal to the donor represented in each panel prediction;
- at least one completed donor-bootstrap replicate;
- report references to all four biomarker figures;
- a passing `biomarker_qc.json` result.

## v1.4.0 closed-loop experimentation checks

The verifier additionally requires:

- four experiment classes: CRISPR, drug, imaging, and sampling time;
- unique experiment identifiers;
- nonnegative expected information gain;
- ordered information-gain uncertainty intervals;
- bootstrap batch-selection probabilities bounded in `[0, 1]`;
- a selected first-round batch that does not exceed its budget;
- normalized hypothesis probabilities after every update;
- nonincreasing entropy in the bundled software-only demonstration;
- exclusion of completed first-round experiments from the second-round ranking;
- complete outcome templates with controls, uncertainty fields, QC requirements, and decision rules;
- report references to all five closed-loop figures;
- a passing `closed_loop_qc.json` result.


## v1.7.0 spatiotemporal digital-tissue checks

The v1.6 verifier additionally requires:

- time-varying node and typed heterogeneous-edge exports;
- explicit regulatory and organelle edge tables;
- zero tissue-section overlap between train and test in both primary regimes;
- zero donor overlap for the held-out-donor regime;
- learned communication-edge gate diagnostics;
- a positive section-cluster bootstrap lower bound for multivariate state-RMSE improvement;
- tissue-outcome RMSE improvement over the cell-intrinsic model in both regimes;
- a Nicheformer adapter contract with an external-checkpoint boundary;
- spatial-interference and intrinsic/neighborhood decomposition tables;
- publication figure bundles and source data;
- SHA-256 verification of all benchmark artifacts.
