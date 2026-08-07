"""Update CausaFlux hypotheses from completed experimental outcomes."""
from pathlib import Path

import pandas as pd

from causaflux import ClosedLoopConfig, update_closed_loop_from_observations

project = Path(__file__).resolve().parents[1]
active = project / "reference_demo" / "active_learning"
hypotheses = pd.read_csv(active / "hypothesis_priors.csv")
catalog = pd.read_csv(active / "experiment_catalog.csv")
observations = pd.read_csv(project / "examples" / "completed_experiments.csv")

posterior, ranking, batch = update_closed_loop_from_observations(
    hypotheses,
    catalog,
    observations,
    config=ClosedLoopConfig(simulate_demonstration_round=False),
)

print(posterior.tail(1).T)
print(ranking.head(10)[["rank", "experiment_type", "experiment_name", "priority_score"]])
print(batch[["batch_position", "experiment_type", "experiment_name", "cumulative_cost"]])
