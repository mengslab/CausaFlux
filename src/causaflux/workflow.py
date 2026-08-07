from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import load_dataset
from .evaluation import evaluate_checkpoint
from .model import CausaFluxConfig
from .plotting import plot_scenario_comparison, plot_simulation, plot_training_history
from .report import generate_html_report
from .simulation import InterventionEvent, build_intervention_schedule, simulate_with_uncertainty
from .synthetic import save_synthetic_upr
from .training import TrainingConfig, load_checkpoint, train_model
from .utils import ensure_dir, json_dump


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("experiment config must contain a YAML mapping")
    return payload


def _initial_state(dataset, configured: list[float] | None) -> np.ndarray:
    if configured is not None:
        state = np.asarray(configured, dtype=np.float32)
        if state.shape != (dataset.raw_observations.shape[-1],):
            raise ValueError("simulation initial_state length does not match observation dimension")
        return state
    values = dataset.raw_observations[:, 0]
    masks = dataset.observation_mask[:, 0]
    numerator = (values * masks).sum(axis=0)
    denominator = np.maximum(masks.sum(axis=0), 1.0)
    return (numerator / denominator).astype(np.float32)


def _events_from_payload(payload: list[dict[str, Any]]) -> list[InterventionEvent]:
    return [
        InterventionEvent(
            channel=item["channel"],
            value=float(item["value"]),
            start=float(item["start"]),
            stop=float(item["stop"]),
            shape=str(item.get("shape", "constant")),
            end_value=(None if item.get("end_value") is None else float(item["end_value"])),
            period=None if item.get("period") is None else float(item["period"]),
            duty_cycle=float(item.get("duty_cycle", 0.5)),
        )
        for item in payload
    ]


def run_experiment(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    device_override: str | None = None,
) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    config = load_experiment_config(config_path)
    experiment = config.get("experiment", {})
    experiment_name = str(experiment.get("name", "CausaFlux v0.2 experiment"))
    seed = int(experiment.get("seed", 7))
    if output_dir is None:
        output_dir = config.get("output", "causaflux_output")
    output_dir = ensure_dir(output_dir)
    data_dir = ensure_dir(output_dir / "data")
    run_dir = ensure_dir(output_dir / "run")
    scenario_root = ensure_dir(output_dir / "scenarios")
    report_dir = ensure_dir(output_dir / "report")

    shutil.copy2(config_path, output_dir / "experiment_config.yaml")
    data_config = config.get("data", {})
    mode = str(data_config.get("mode", "synthetic"))
    if mode == "synthetic":
        data_path = data_dir / "synthetic_upr_v0.2.npz"
        dataset = save_synthetic_upr(
            data_path,
            n_trajectories=int(data_config.get("n_trajectories", 384)),
            min_steps=int(data_config.get("min_steps", 8)),
            max_steps=int(data_config.get("max_steps", 16)),
            seed=seed,
            missing_feature_rate=float(data_config.get("missing_feature_rate", 0.08)),
            replicate_size=int(data_config.get("replicate_size", 4)),
        )
    elif mode in {"npz", "csv"}:
        source = Path(data_config["path"])
        if not source.is_absolute():
            source = (config_path.parent / source).resolve()
        dataset = load_dataset(source)
        data_path = data_dir / ("input_dataset.npz")
        dataset.to_npz(data_path)
    else:
        raise ValueError("data.mode must be synthetic, npz, or csv")

    if bool(data_config.get("export_long_csv", True)):
        dataset.to_long_csv(data_dir / "dataset_long.csv")
    dataset_summary = dataset.summary()
    json_dump(dataset_summary, data_dir / "dataset_summary.json")

    model_payload = dict(config.get("model", {}))
    model_payload.setdefault("observation_dim", dataset.raw_observations.shape[-1])
    model_payload.setdefault("intervention_dim", dataset.interventions.shape[-1])
    model_payload.setdefault("n_fates", len(dataset.fate_names))
    model_config = CausaFluxConfig(**model_payload)

    training_payload = dict(config.get("training", {}))
    training_payload.setdefault("seed", seed)
    if device_override is not None:
        training_payload["device"] = device_override
    training_config = TrainingConfig(**training_payload)
    result = train_model(data_path, run_dir, model_config, training_config)
    training_plot = run_dir / "training_curve.png"
    plot_training_history(run_dir / "history.csv", training_plot)
    metrics = evaluate_checkpoint(
        result["checkpoint"],
        data_path,
        run_dir / "evaluation",
        device=training_config.device,
    )

    model, standardizer, checkpoint, device = load_checkpoint(
        result["checkpoint"], training_config.device
    )
    simulation_config = config.get("simulation", {})
    final_time = float(simulation_config.get("final_time", 10.0))
    steps = int(simulation_config.get("steps", 18))
    times = np.linspace(0.0, final_time, steps, dtype=np.float32)
    initial_state = _initial_state(dataset, simulation_config.get("initial_state"))
    initial_mask = np.isfinite(initial_state).astype(np.float32)
    mc_samples = int(simulation_config.get("mc_samples", 24))
    scenarios = simulation_config.get("scenarios", [])
    if not scenarios:
        raise ValueError("simulation.scenarios must contain at least one scenario")

    scenario_frames: dict[str, pd.DataFrame] = {}
    fate_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        name = str(scenario["name"])
        scenario_dir = ensure_dir(scenario_root / name)
        events = _events_from_payload(scenario.get("events", []))
        schedule = build_intervention_schedule(
            times,
            model.config.intervention_dim,
            events,
            intervention_names=checkpoint["intervention_names"],
        )
        schedule_frame = pd.DataFrame({"time": times})
        for index, intervention_name in enumerate(checkpoint["intervention_names"]):
            schedule_frame[intervention_name] = schedule[:, index]
        schedule_frame.to_csv(scenario_dir / "schedule.csv", index=False)

        simulation = simulate_with_uncertainty(
            model,
            standardizer,
            initial_state,
            times,
            schedule,
            device,
            mc_samples=mc_samples,
            initial_observation_mask=initial_mask,
            sample_process_noise=bool(simulation_config.get("sample_process_noise", True)),
            seed=seed,
        )
        frame = pd.DataFrame({"time": times})
        for feature_index, feature_name in enumerate(checkpoint["feature_names"]):
            frame[f"{feature_name}_mean"] = simulation["trajectory_mean"][:, feature_index]
            frame[f"{feature_name}_std"] = simulation["trajectory_std"][:, feature_index]
            frame[f"{feature_name}_p05"] = simulation["trajectory_lower"][:, feature_index]
            frame[f"{feature_name}_p95"] = simulation["trajectory_upper"][:, feature_index]
            frame[f"{feature_name}_decoder_std"] = simulation["decoder_std_mean"][:, feature_index]
        frame.to_csv(scenario_dir / "simulation.csv", index=False)
        scenario_frames[name] = frame
        fate_frame = pd.DataFrame(
            {
                "fate": checkpoint["fate_names"],
                "mean_probability": simulation["fate_probability_mean"],
                "std_probability": simulation["fate_probability_std"],
            }
        )
        fate_frame.to_csv(scenario_dir / "fate_probabilities.csv", index=False)
        for row in fate_frame.to_dict(orient="records"):
            fate_rows.append({"scenario": name, **row})
        selected = simulation_config.get("plot_feature_indices")
        plot_simulation(
            times,
            simulation["trajectory_mean"],
            simulation["trajectory_lower"],
            simulation["trajectory_upper"],
            scenario_dir / "simulation.png",
            feature_names=checkpoint["feature_names"],
            selected=selected,
            title=f"{name} forecast",
        )

    combined_fates = pd.DataFrame(fate_rows)
    combined_fates_path = scenario_root / "scenario_fate_summary.csv"
    combined_fates.to_csv(combined_fates_path, index=False)
    comparison_plot = scenario_root / "scenario_comparison.png"
    plot_scenario_comparison(
        scenario_frames,
        combined_fates,
        comparison_plot,
        feature_names=checkpoint["feature_names"],
        selected_features=simulation_config.get("comparison_features"),
    )

    with (run_dir / "training_summary.json").open("r", encoding="utf-8") as handle:
        training_summary = json.load(handle)
    report_path = generate_html_report(
        report_dir / "index.html",
        experiment_name=experiment_name,
        dataset_summary=dataset_summary,
        training_summary=training_summary,
        evaluation_metrics=metrics,
        scenario_fates_path=combined_fates_path,
        scenario_plot_path=comparison_plot,
        training_plot_path=training_plot,
    )
    print(f"CausaFlux experiment complete: {output_dir}")
    print(f"HTML report: {report_path}")
    return {
        "output_dir": output_dir,
        "data": data_path,
        "checkpoint": Path(result["checkpoint"]),
        "report": report_path,
    }
