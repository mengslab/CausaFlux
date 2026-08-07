# CausaFlux v1.4.0 Multimodal Dynamic State Model

## Scientific question

Can early live-imaging and pathway-reporter histories improve prediction of a later destructive state beyond baseline covariates, the latest RNA state, and static multimodal fusion?

## Modalities

The v1.4 benchmark contract has separate tensors and observation masks for RNA, imaging, pathway reporters, phosphoproteomics, metabolomics and lipidomics. Modalities remain separate until their encoders produce latent distributions or embeddings.

## Modality-specific encoders

`ModalityEncoder` maps each modality into a shared latent dimension. Product-of-experts (PoE) encoders emit a mean and log variance; mixture-of-experts (MoE) encoders emit a deterministic latent embedding. A missing modality contributes zero observation precision or receives a masked gate weight.

## Fusion

### Product of experts

PoE combines observed modality distributions by precision, with a unit-Gaussian prior. Highly certain observed modalities contribute more precision. Unobserved modalities do not contribute.

### Mixture of experts

MoE learns a gating score for each available modality at each time point. Missing modalities are assigned effectively zero probability before normalization.

## Modality dropout

During training, complete modalities are randomly dropped at the trajectory-batch level. At least one modality remains available. This forces the model to operate under incomplete assay panels rather than relying on a single privileged modality.

## Cross-modal decoders

The dynamic latent state predicts the final RNA, phosphoproteomic, metabolomic and lipidomic state. Validation uses normalized RMSE, feature-wise correlation and validation-calibrated interval coverage at 50%, 80%, 90% and 95% nominal levels.

## Donor and cohort context

The reference PoE/MoE models include regularized donor and cohort embeddings in the final latent context. The primary benchmark is a perturbation-history holdout, so donor overlap is explicitly reported as intentional. A donor-holdout benchmark must map unseen donors to the unknown donor state rather than reusing a fitted donor embedding.

## Missing-not-at-random sensitivity

The benchmark evaluates four observation regimes:

1. observed acquisition pattern;
2. additional 20% MCAR loss;
3. destructive-state-dependent imaging/reporting loss;
4. assay-quality-dependent phosphoproteomic/metabolomic/lipidomic loss.

These are sensitivity analyses, not claims that the missingness mechanism is identified from the data.

## Exit gate

At least one model using the **temporal history of early imaging and reporters** must have lower validation-calibrated test log loss and Brier score than all three prespecified baselines:

- `BaselineCovariatesMLP`;
- `LatestRNAMLP`;
- `StaticMultimodalFusion`.

Its AUC must be non-inferior to the strongest baseline within 0.02. The synthetic gate verifies software behavior only. Foundation pretraining remains blocked until this gate is passed on a locked real longitudinal multimodal perturbation dataset.

## External contract

```bash
causaflux multimodal-dynamic-export-fixture --output multimodal_dynamic_fixture_v1.4.0.npz
causaflux multimodal-dynamic-run --data-npz my_multimodal_longitudinal.npz --output benchmark
causaflux multimodal-dynamic-validate --input benchmark
```

The external NPZ must contain explicit donor, cohort, perturbation-history, outcome, modality, mask and time axes. The model code does not need to be modified for a conforming dataset.
