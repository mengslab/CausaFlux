# CausaFlux v1.7.0 — Foundation Adapter and Pretraining

## Scope

This release adds external embedding contracts for scGPT, GET, Nicheformer, MrVI, ESM-2, MolFormer and DINOv2. Third-party checkpoints are not bundled. Scientific use requires user-managed execution, license compliance, exact model/checkpoint provenance and row-level alignment.

## Ten pretraining objectives

1. masked modality reconstruction;
2. cross-modal reconstruction;
3. temporal ordering;
4. future-state prediction;
5. intervention identification;
6. intervention-effect prediction;
7. graph-edge prediction;
8. recovery-versus-commitment discrimination;
9. time-to-fate prediction;
10. contrastive donor/tissue invariance.

## Evaluations

The benchmark reports frozen encoder, linear probe, parameter-efficient transfer, full fine-tuning and zero-shot transfer. Generalization is measured under donor, tissue and perturbation holdouts plus a prospective-time proxy split.

## Gate

The synthetic software gate passes only when the pretrained CausaFlux representation improves mean future-state and intervention-effect prediction over a same-capacity non-pretrained representation, with per-split future-state improvement and intervention-effect non-inferiority. PCA is reported as an unsupervised representation baseline but is not the release exit criterion.

Real foundation pretraining authorization remains blocked until the upstream real longitudinal, multimodal, intervention and spatiotemporal gates are satisfied. The synthetic reference is a software benchmark only.
