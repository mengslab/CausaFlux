"""Minimal CausaFlux v1.4.0 closed-loop experiment-design example."""
from pathlib import Path

import pandas as pd
import yaml

from causaflux import ClosedLoopConfig, run_closed_loop_experimentation

project = Path(__file__).resolve().parents[1]
config = yaml.safe_load((project / "configs" / "cancer_closed_loop_v1.4.0.yaml").read_text())
root = project / "reference_demo"

result = run_closed_loop_experimentation(
    hypotheses_payload=config["closed_loop"]["hypotheses"],
    candidates_payload=None,
    config=ClosedLoopConfig(bootstrap=20, eig_samples=600, seed=31),
    therapeutic_predictions=pd.read_csv(root / "therapeutics" / "all_regimen_predictions.csv"),
    biomarkers=pd.read_csv(root / "biomarkers" / "causal_biomarker_ranking.csv"),
    biomarker_timecourse=pd.read_csv(root / "biomarkers" / "early_warning_timecourse.csv"),
    transition_uncertainty=pd.read_csv(root / "transitions" / "transition_bootstrap_intervals.csv"),
)

print(result.round1_ranking.head(10)[
    ["rank", "experiment_type", "experiment_name", "expected_information_gain_nats", "priority_score"]
])
print("\nSelected batch")
print(result.round1_batch[["batch_position", "experiment_type", "experiment_name", "cumulative_cost"]])
