"""Minimal CausaFlux v1.4.0 causal-biomarker example."""
from pathlib import Path

import pandas as pd
import yaml

from causaflux import BiomarkerConfig, run_causal_biomarkers, write_biomarker_outputs
from causaflux.causal_models import build_causal_graph

project = Path(__file__).resolve().parents[1]
config = yaml.safe_load((project / "configs" / "cancer_closed_loop_v1.4.0.yaml").read_text())
frame = pd.read_csv(project / "reference_demo" / "data" / "cancer_longitudinal.csv")
graph = build_causal_graph(config["causal_graph"]["nodes"], config["causal_graph"]["edges"])
payload = config["biomarkers"]

result = run_causal_biomarkers(
    frame,
    graph,
    payload["features"],
    BiomarkerConfig(bootstrap=20, top_panel_size=3, seed=31),
    assayability=payload["assayability"],
    metadata_overrides=payload["metadata_overrides"],
)
write_biomarker_outputs(result, project / "example_biomarker_output")
print(result.ranking.head())
print(result.panels)
