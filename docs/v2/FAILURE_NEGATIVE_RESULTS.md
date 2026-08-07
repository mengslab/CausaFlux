# Failure and negative-result policy

CausaFlux v2.0.0 treats technical failure and biological negative results as different evidence classes.

A **technical/assay failure** is excluded from posterior biological updating when its prespecified QC rule fails, but its attempted experiment and full cost remain in the efficiency ledger.

A **negative biological result** passes experimental QC but does not support the predicted effect. It is retained in prediction-versus-outcome evaluation, calibration, posterior updating, falsification summaries, and the public/internal evidence ledger as permitted by governance.

The release gate requires an explicit reporting-completeness attestation for real prospective studies. This prevents selective reporting of only model-confirming experiments.
