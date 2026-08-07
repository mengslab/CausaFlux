# CausaFlux Virtual Cell 2.0

CausaFlux 2.0 integrates five layers:

1. **Real-world longitudinal perturbation data** — time, dose, sequence, modality, donor and intervention histories are represented explicitly.
2. **AI-guided dynamic state model** — the retained factorized state architecture distinguishes cell identity, reversible adaptation, commitment and context and is augmented by multimodal, intervention, spatial and foundation-model components.
3. **Counterfactual virtual-cell simulation** — candidate interventions are compared as trajectories, not only terminal labels.
4. **Prospective experimental loop** — model freezes and predictions are locked before outcomes; Cycle 2 uses the updated model rather than reusing Cycle 1 predictions.
5. **Evidence-governed reporting** — release claims, uncertainty calibration, failures, negative results, external replication and prospective cycles are linked in a single auditable evidence ledger.

## Distribution shift

The `shift-calibration` command evaluates central 90% predictive-interval coverage overall and by a prespecified shift group. A release-level PASS requires both acceptable overall calibration and a minimum group coverage.

## Version semantics

`2.0.0` identifies the software release. The phrase `prospectively validated virtual cell` is a gated evidence claim. In an unevidenced installation the software may be stable and fully functional while the biological claim remains `NOT_YET_ELIGIBLE`.
