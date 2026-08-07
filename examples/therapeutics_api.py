"""Minimal CausaFlux v1.4.0 counterfactual-therapeutics example."""

from pathlib import Path

import pandas as pd

from causaflux import TherapeuticConfig, run_counterfactual_therapeutics
from causaflux.therapeutics import write_therapeutic_outputs

frame = pd.read_csv("examples/cancer_longitudinal_template.csv")
config = TherapeuticConfig(
    decision_time_hours=24,
    horizon_hours=168,
    timing_grid=(0, 24, 48, 72, 120),
    bootstrap=30,
    seed=31,
)
result = run_counterfactual_therapeutics(frame, config)
write_therapeutic_outputs(result, Path("therapeutic_output"), config)

print(result.predictions.head(10))
print(result.qc)
