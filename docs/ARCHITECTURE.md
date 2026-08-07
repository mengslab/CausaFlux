# CausaFlux v1.4.0 Architecture

```text
Longitudinal causal table ───────────────┐
                                         ├─ shared row_id alignment
RNA / ATAC / protein / mutation / drug ─┤
                                         ├─ MuData + obsm["spatial"]
Spatial coordinates ─────────────────────┘
        │
        ├─ multimodal donor-held-out baselines
        ├─ calibrated state probabilities and uncertainty
        ├─ temporal disease-transition model
        ├─ editable causal DAG and treatment effects
        ├─ multicellular spatial heterograph
        │     ├─ typed cell nodes and proximity edges
        │     ├─ niches and ligand–receptor circuits
        │     └─ GraphML / PyG-compatible exports
        ├─ counterfactual therapeutic engine
        │     ├─ intervention catalog
        │     ├─ gene and drug predictions
        │     ├─ simultaneous combinations
        │     ├─ ordered treatment sequences
        │     ├─ timing-window search
        │     ├─ normal-cell toxicity model
        │     ├─ donor-bootstrap intervals
        │     └─ Pareto and uncertainty-adjusted ranking
        ├─ causal biomarker engine
        │     ├─ feature-by-time early warning
        │     ├─ donor-stability analysis
        │     ├─ causal-path proximity and evidence
        │     ├─ assayability and redundancy
        │     ├─ donor-bootstrap score intervals
        │     └─ compact held-donor biomarker panels
        └─ closed-loop experiment engine
              ├─ explicit competing hypotheses and priors
              ├─ CRISPR, drug, imaging, and sampling candidates
              ├─ expected information gain and uncertainty
              ├─ budget, capacity, and diversity constraints
              ├─ outcome ingestion and posterior updates
              └─ next-round recommendation and batch export
```

## Design principles

1. Observation identity is explicit and shared across every modality.
2. Donors, not cells, define validation and bootstrap units.
3. Physical proximity and molecular communication are separate edge types.
4. Intervention events alter named state variables rather than opaque treatment labels.
5. Benefit and normal-cell toxicity are modeled separately.
6. Timing, simultaneity, and order are distinct regimen properties.
7. Every graph, state change, counterfactual, and ranking is exportable as an auditable table.
8. Predictive association, temporal precedence, causal proximity, and perturbational support remain separate evidence axes.
9. Synthetic demonstrations are clearly separated from biological and clinical evidence.
10. Information value and therapeutic value are reported separately before batch selection.
11. Completed experiments are excluded before the next round is ranked.

## Core modules

- `causaflux.multimodal`: aligned MuData model and fusion diagnostics.
- `causaflux.spatial`: coordinates, heterograph, niches, communication circuits.
- `causaflux.uncertainty`: calibration, donor bootstrap, model disagreement.
- `causaflux.causal_models`: transitions, DAGs, and causal effects.
- `causaflux.active_learning`: hypotheses, expected information gain, constrained batch selection, posterior updating, outcome templates, and figures.
- `causaflux.biomarkers`: early-warning analysis, causal-proximity ranking, donor bootstrap, assay manifest, and compact panels.
- `causaflux.therapeutics`: intervention schema, regimen generation, counterfactual state
  updates, response prediction, bootstrap uncertainty, Pareto ranking, and figures.
- `causaflux.causal_workflow`: stage-aware analytical workflow and resume boundaries.
- `causaflux.staged_workflow`: isolated causal, therapeutic, biomarker, closed-loop, and report stages.
- `scripts/run_staged.sh`: one-command seven-process production orchestrator.
- `causaflux._mudata_compat`: standards-aligned H5MU writer for constrained environments.

## Therapeutic data flow

```text
state at decision time
   ↓
explicit intervention events
   ↓
dose × timing × onset × persistence
   ↓
bounded mechanistic state changes
   ↓
donor-audited resistance surrogate
   ↓
benefit + immune response + tumor viability
   ↓
normal-cell vulnerability + extrapolation
   ↓
donor bootstrap + Pareto ranking
```

## Future adapter boundary

The reference engine is deliberately transparent. Future models can replace individual
components with time-varying causal estimators, controlled differential equations,
PK/PD compartments, graph-conditioned predictors, chemical foundation models, or
Bayesian optimization while preserving the same intervention, regimen, prediction,
and uncertainty contracts.


## File-backed stage boundaries

The production launcher intentionally does not keep every numerical library in one
Python interpreter. Stage 1 serializes the exact integrated feature table and all
donor-held-out baseline outputs. Stage 2 reconstructs MuData from the aligned CSV
bundle and adds spatial coordinates, graph outputs, and transitions. Stages 3 and 4
fit causal and therapeutic models in fresh processes. Stage 5 computes causal biomarker
rankings and compact panels from serialized data. Stage 6 computes experiment recommendations,
selects a constrained batch, performs the software-only posterior demonstration, and reranks the
next round. Stage 7 reads only exported artifacts to render final figures, manifests, model cards,
and the HTML report.

## v0.9 neurobiology layer

The neurobiology configuration is implemented in `causaflux.neurobiology` and executes as a separate file-backed stage. It produces donor-held-out neural–glial state probabilities, degeneration-risk predictions, imaging–electrophysiology alignments, APOE-stratified summaries, cell-type driver scores, and bootstrap transition intervals. The stage writes to `output/neurobiology/`; final report assembly adds `report/neurobiology.html` and a summary section to the integrated report.

## v1.0 platform layer

The scientific workflow is followed by publication and platform-validation stages. `causaflux.platform` writes dataset cards, a platform model card, a demo registry, an environment snapshot, an artifact SHA-256 manifest, structured validation tables, and `report/platform.html`. The platform layer reads serialized scientific artifacts and does not alter fitted model outputs.

The integrated launcher therefore separates numerical execution from release governance:

```text
Scientific stages 1–7 → report stage 8 → publication stage 9 → provenance and validation stage 10
```

This boundary makes it possible to revalidate or repackage an approved run without refitting models.

## v1.4.0 publication layer

A tenth, process-isolated publication stage reads serialized scientific outputs and rebuilds every panel through `causaflux.visualization.publication`. Figure generation is separated into bounded domain groups to avoid retaining Matplotlib, TIFF, HDF5, and numerical-library state across the whole run.

```text
Scientific stages 1–7
    → report assembly 8
    → publication graphics 9
    → provenance/platform validation 10
```

Each figure bundle contains editable SVG/PDF, 600-dpi PNG/TIFF, panel-level CSV source data, a deterministic layout where applicable, and a JSON figure manifest. The publication layer does not alter statistical results.

## v1.4.0 dynamic benchmark layer

The dynamic benchmark is isolated from the disease-specific workflow and uses one common forecasting interface:

```text
context observations + context interventions + irregular times
                         +
              known future intervention schedule
                         ↓
              multi-step future state forecast
              heteroscedastic uncertainty
              terminal fate probabilities
```

Static and dynamic models implement the same `ForecastModel` contract. Split logic operates before tensor standardization. Validation-only temperature scaling calibrates each model's predicted heteroscedastic variance. Complete perturbation histories are never divided across train, validation, and test.

The factorized CausaFlux model contains stable identity, context, reversible adaptation, and accumulated commitment states. A time-aware contextual branch supports multi-step trajectory decoding, while the factorized latent state supports fate prediction.

## v1.4.0 multimodal dynamic-state layer

The v1.4 layer sits after the v1.3 generalization benchmark and before any foundation-pretraining work.

```text
RNA ─────────────┐
Imaging ─────────┤
Reporter ────────┤   modality-specific encoders
Phosphoproteome ─┤            │
Metabolome ──────┤            ▼
Lipidome ────────┘     PoE or MoE fusion
                              │
                         temporal GRU
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
      destructive state   late-omics      latent context
                           decoders        + donor/cohort
```

PoE treats each observed modality as a latent Gaussian expert; missing experts contribute zero precision. MoE learns availability-aware gating weights. Training performs complete-modality dropout. Cross-modal decoders predict final RNA, phosphoproteomic, metabolomic and lipidomic state.

The primary release gate is incremental prediction from early imaging/reporters beyond three prespecified static baselines. Foundation pretraining remains blocked after a synthetic pass.
