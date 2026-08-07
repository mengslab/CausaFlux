from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def _table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{_format_value(value)}</td></tr>"
        for label, value in rows
    )
    return f"<table>{body}</table>"


def generate_html_report(
    output_path: str | Path,
    experiment_name: str,
    dataset_summary: dict[str, Any],
    training_summary: dict[str, Any],
    evaluation_metrics: dict[str, Any],
    scenario_fates_path: str | Path,
    scenario_plot_path: str | Path,
    training_plot_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_root = output_path.parent
    scenario_fates = pd.read_csv(scenario_fates_path)
    fate_pivot = scenario_fates.pivot(
        index="scenario", columns="fate", values="mean_probability"
    ).reset_index()
    scenario_table = fate_pivot.to_html(index=False, float_format=lambda x: f"{x:.4f}")

    dataset_rows = [
        ("Trajectories", dataset_summary.get("n_trajectories")),
        ("Biological/replicate groups", dataset_summary.get("n_groups")),
        ("Observation features", dataset_summary.get("observation_dim")),
        ("Intervention channels", dataset_summary.get("intervention_dim")),
        ("Observed feature fraction", dataset_summary.get("observed_feature_fraction")),
    ]
    training_rows = [
        ("Device", training_summary.get("device")),
        ("Training trajectories", training_summary.get("n_train")),
        ("Validation trajectories", training_summary.get("n_val")),
        ("Test trajectories", training_summary.get("n_test")),
        ("Best epoch", training_summary.get("best_epoch")),
        ("Best validation loss", training_summary.get("best_val_loss")),
    ]
    evaluation_rows = [
        ("Standardized next-state RMSE", evaluation_metrics.get("next_state_rmse_standardized")),
        ("Gaussian NLL", evaluation_metrics.get("next_state_mean_gaussian_nll")),
        ("90% interval coverage", evaluation_metrics.get("predictive_interval_90_coverage")),
        ("Fate accuracy", evaluation_metrics.get("fate_accuracy")),
        ("Fate Brier score", evaluation_metrics.get("fate_brier_score")),
        ("Fate calibration error", evaluation_metrics.get("fate_expected_calibration_error")),
    ]

    training_image = Path(os.path.relpath(Path(training_plot_path).resolve(), report_root.resolve()))
    scenario_image = Path(os.path.relpath(Path(scenario_plot_path).resolve(), report_root.resolve()))
    payload = html.escape(
        json.dumps(
            {
                "dataset": dataset_summary,
                "training": training_summary,
                "evaluation": evaluation_metrics,
            },
            indent=2,
        )
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(experiment_name)} — CausaFlux v0.2</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1120px; margin: 0 auto; padding: 32px; line-height: 1.5; }}
h1, h2 {{ line-height: 1.2; }}
.hero {{ padding: 24px; border: 1px solid #8885; border-radius: 14px; margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }}
.card {{ border: 1px solid #8885; border-radius: 12px; padding: 18px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; border-bottom: 1px solid #8884; padding: 8px; vertical-align: top; }}
img {{ max-width: 100%; height: auto; border-radius: 8px; background: white; }}
code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
pre {{ overflow-x: auto; padding: 14px; border: 1px solid #8885; border-radius: 8px; }}
.note {{ opacity: 0.8; }}
</style>
</head>
<body>
<section class="hero">
<h1>{html.escape(experiment_name)}</h1>
<p><strong>CausaFlux v0.2 experiment report.</strong> This software prototype models irregular stress trajectories, missing measurements, probabilistic future states, terminal fate, and intervention schedules.</p>
<p class="note">Synthetic results demonstrate software behavior only. They are not experimental evidence or treatment recommendations.</p>
</section>
<div class="grid">
<section class="card"><h2>Dataset</h2>{_table(dataset_rows)}</section>
<section class="card"><h2>Training</h2>{_table(training_rows)}</section>
<section class="card"><h2>Evaluation</h2>{_table(evaluation_rows)}</section>
</div>
<section><h2>Training history</h2><img src="{training_image.as_posix()}" alt="Training history"></section>
<section><h2>Counterfactual scenario comparison</h2><img src="{scenario_image.as_posix()}" alt="Scenario comparison">{scenario_table}</section>
<section><h2>Machine-readable summary</h2><pre>{payload}</pre></section>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    return output_path
