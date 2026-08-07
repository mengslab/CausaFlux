from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from .data import load_dataset
from .evaluation import evaluate_checkpoint
from .model import CausaFluxConfig
from .plotting import plot_simulation, plot_training_history
from .simulation import (
    InterventionEvent,
    build_intervention_schedule,
    events_from_csv,
    simulate_with_uncertainty,
)
from .synthetic import save_synthetic_upr
from .training import TrainingConfig, load_checkpoint, train_model
from .utils import ensure_dir
from .workflow import run_experiment
from .causal_workflow import run_causal_experiment
from .causal_models import build_causal_graph
from .biomarkers import BiomarkerConfig, run_causal_biomarkers, validate_biomarker_outputs, write_biomarker_outputs
from .active_learning import (
    ClosedLoopConfig,
    run_closed_loop_experimentation,
    update_closed_loop_from_observations,
    validate_closed_loop_outputs,
    write_closed_loop_outputs,
)
from .multimodal import read_csv_bundle, read_multimodal, validate_multimodal, write_csv_bundle, write_multimodal
from .therapeutics import (
    TherapeuticConfig,
    fit_therapeutic_model,
    intervention_catalog,
    predict_regimens,
    run_counterfactual_therapeutics,
    write_therapeutic_outputs,
)
from .neurobiology import (
    NeurobiologyConfig,
    run_neurobiology_configuration,
    validate_neurobiology_outputs,
    write_neurobiology_outputs,
    generate_neurobiology_report,
)
from .platform import (
    PLATFORM_VERSION,
    demo_registry_frame,
    finalize_research_platform,
    platform_doctor,
    validate_research_platform,
)
from .visualization.publication import (
    finalize_publication_inventory,
    rebuild_reference_figure_group,
    validate_publication_bundle,
    PUBLICATION_GROUPS,
)
from .realdata import (
    benchmark_registry_frame, accession_manifest_frame, build_download_plan,
    generate_realdata_reports, get_benchmark, preflight_benchmarks,
    validate_realdata_output, validate_realdata_registry,
)
from .biological_validation import (
    hypothesis_registry_frame, freeze_preregistration,
    run_and_write_biological_validation, validate_biological_validation,
)
from .dynamic_benchmark import (
    DynamicBenchmarkConfig,
    run_dynamic_benchmark,
    validate_dynamic_benchmark,
    load_external_benchmark_npz,
    save_external_benchmark_npz,
    generate_dynamic_benchmark_data,
    MODEL_ORDER as DYNAMIC_MODEL_ORDER,
)
from .multimodal_dynamic import (
    MultimodalDynamicConfig,
    run_multimodal_dynamic_benchmark,
    validate_multimodal_dynamic_benchmark,
    load_external_multimodal_npz,
    save_external_multimodal_npz,
    generate_multimodal_dynamic_data,
    MODEL_ORDER as MULTIMODAL_DYNAMIC_MODEL_ORDER,
)
from .intervention_generalization import (
    InterventionGeneralizationConfig,
    run_intervention_generalization_benchmark,
    validate_intervention_generalization,
    generate_intervention_generalization_data,
    save_external_intervention_npz,
    load_external_intervention_npz,
    adapter_registry_frame,
)
from .spatiotemporal_tissue import (
    SpatiotemporalTissueConfig,
    run_spatiotemporal_tissue_benchmark,
    validate_spatiotemporal_tissue,
    generate_spatiotemporal_tissue_data,
    save_external_spatiotemporal_npz,
    load_external_spatiotemporal_npz,
    nicheformer_adapter_spec,
)
from .foundation_pretraining import (
    FoundationPretrainingConfig, run_foundation_pretraining, validate_foundation_pretraining,
    generate_foundation_data, save_external_foundation_npz, load_external_foundation_npz,
    adapter_registry as foundation_adapter_registry, objective_registry_frame,
)
from .prospective_loop import (
    ProspectiveLoopConfig,
    run_prospective_loop,
    validate_prospective_loop,
    write_contract_bundle,
    ingest_external_cycle,
)
from .virtual_cell_release import run_virtual_cell_release
from .virtual_cell_validation import validate_virtual_cell_release
from .real_world_hub import UserDatasetContract, register_user_dataset, preview_tabular_dataset
from .v2_release import run_v2_release
from .v2_release_gate import validate_v2_output
from .longitudinal_realdata import convert_longitudinal_table, run_real_longitudinal_benchmark, write_public_dataset_bundle
from .shift_calibration import evaluate_shift_calibration_file
from .spatial import (
    SpatialGraphConfig,
    build_spatial_heterograph,
    plot_communication_circuits,
    plot_contact_heatmap,
    plot_heterograph_summary,
    plot_niche_composition,
    plot_spatial_atlas,
    validate_spatial_graph,
    write_spatial_graph_outputs,
)


def _load_yaml(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def command_generate(args) -> None:
    dataset = save_synthetic_upr(
        args.output,
        n_trajectories=args.n_trajectories,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        seed=args.seed,
        missing_feature_rate=args.missing_feature_rate,
    )
    if args.csv:
        dataset.to_long_csv(args.csv)
    print(json.dumps(dataset.summary(), indent=2))
    print(f"saved dataset to {args.output}")


def command_import_csv(args) -> None:
    dataset = load_dataset(args.input)
    dataset.to_npz(args.output)
    print(json.dumps(dataset.summary(), indent=2))
    print(f"converted {args.input} to {args.output}")


def command_export_csv(args) -> None:
    dataset = load_dataset(args.input)
    dataset.to_long_csv(args.output)
    print(f"exported {len(dataset)} trajectories to {args.output}")


def command_train(args) -> None:
    config_payload = _load_yaml(args.config)
    base = load_dataset(args.data)
    model_payload = dict(config_payload.get("model", {}))
    model_payload.setdefault("observation_dim", base.raw_observations.shape[-1])
    model_payload.setdefault("intervention_dim", base.interventions.shape[-1])
    model_payload.setdefault("n_fates", len(base.fate_names))
    model_config = CausaFluxConfig(**model_payload)
    train_payload = dict(config_payload.get("training", {}))
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "seed": args.seed,
        "split_mode": args.split_mode,
    }
    train_payload.update({key: value for key, value in overrides.items() if value is not None})
    training_config = TrainingConfig(**train_payload)
    result = train_model(args.data, args.output, model_config, training_config)
    plot_training_history(
        Path(args.output) / "history.csv",
        Path(args.output) / "training_curve.png",
    )
    print(f"best checkpoint: {result['checkpoint']}")


def command_evaluate(args) -> None:
    metrics = evaluate_checkpoint(args.checkpoint, args.data, args.output, args.device)
    print(json.dumps(metrics, indent=2))


def _built_in_events(name: str, final_time: float) -> list[InterventionEvent]:
    if name == "continuous_stress":
        return [InterventionEvent("ER_stress", 1.0, 0.0, final_time)]
    if name == "stress_recovery":
        return [InterventionEvent("ER_stress", 1.0, 0.0, final_time * 0.40)]
    if name == "pulsatile_stress":
        return [
            InterventionEvent(
                "ER_stress",
                1.0,
                0.0,
                final_time * 0.75,
                shape="pulse",
                period=max(final_time / 5.0, 0.5),
                duty_cycle=0.45,
            )
        ]
    if name == "ire1_inhibition":
        return [
            InterventionEvent("ER_stress", 1.0, 0.0, final_time * 0.55),
            InterventionEvent("IRE1_inhibition", 0.6, 0.0, final_time * 0.55),
        ]
    if name == "atf6_support":
        return [
            InterventionEvent("ER_stress", 1.0, 0.0, final_time * 0.55),
            InterventionEvent("ATF6_activation", 0.5, 0.0, final_time * 0.55),
        ]
    raise ValueError(f"unknown scenario: {name}")


def command_simulate(args) -> None:
    output_dir = ensure_dir(args.output)
    model, standardizer, checkpoint, device = load_checkpoint(args.checkpoint, args.device)
    times = np.linspace(0.0, args.final_time, args.steps, dtype=np.float32)
    events = (
        events_from_csv(args.schedule_csv)
        if args.schedule_csv
        else _built_in_events(args.scenario, args.final_time)
    )
    schedule = build_intervention_schedule(
        times,
        model.config.intervention_dim,
        events,
        intervention_names=checkpoint["intervention_names"],
    )
    if args.initial_state_csv:
        initial_frame = pd.read_csv(args.initial_state_csv)
        initial = initial_frame.iloc[0][checkpoint["feature_names"]].to_numpy(dtype=np.float32)
    else:
        initial = np.asarray(
            [0.12, 0.10, 0.10, 0.85, 0.85, 0.82, 0.08, 0.08, 0.85, 0.95, 0.90, 0.04],
            dtype=np.float32,
        )
        if len(initial) != model.config.observation_dim:
            initial = standardizer.mean.copy()
    result = simulate_with_uncertainty(
        model,
        standardizer,
        initial,
        times,
        schedule,
        device,
        mc_samples=args.mc_samples,
        seed=args.seed,
    )
    frame = pd.DataFrame({"time": times})
    for index, name in enumerate(checkpoint["feature_names"]):
        frame[f"{name}_mean"] = result["trajectory_mean"][:, index]
        frame[f"{name}_std"] = result["trajectory_std"][:, index]
        frame[f"{name}_p05"] = result["trajectory_lower"][:, index]
        frame[f"{name}_p95"] = result["trajectory_upper"][:, index]
    frame.to_csv(output_dir / "simulation.csv", index=False)
    schedule_frame = pd.DataFrame({"time": times})
    for index, name in enumerate(checkpoint["intervention_names"]):
        schedule_frame[name] = schedule[:, index]
    schedule_frame.to_csv(output_dir / "schedule.csv", index=False)
    fate_frame = pd.DataFrame(
        {
            "fate": checkpoint["fate_names"],
            "mean_probability": result["fate_probability_mean"],
            "std_probability": result["fate_probability_std"],
        }
    )
    fate_frame.to_csv(output_dir / "fate_probabilities.csv", index=False)
    plot_simulation(
        times,
        result["trajectory_mean"],
        result["trajectory_lower"],
        result["trajectory_upper"],
        output_dir / "simulation.png",
        feature_names=checkpoint["feature_names"],
        title=args.scenario if not args.schedule_csv else "custom schedule",
    )
    print(fate_frame.to_string(index=False))
    print(f"simulation outputs: {output_dir}")



def command_multimodal_validate(args) -> None:
    mdata = read_multimodal(args.input)
    print(json.dumps(validate_multimodal(mdata), indent=2, default=str))


def command_multimodal_export(args) -> None:
    mdata = read_multimodal(args.input)
    validate_multimodal(mdata)
    destination = write_csv_bundle(mdata, args.output)
    print(f"exported multimodal CSV bundle to {destination}")


def command_multimodal_import(args) -> None:
    mdata = read_csv_bundle(args.input)
    validate_multimodal(mdata)
    destination = write_multimodal(mdata, args.output)
    print(f"created MuData file: {destination}")


def command_spatial_build(args) -> None:
    frame = pd.read_csv(args.input)
    config = SpatialGraphConfig(
        seed=args.seed,
        k_neighbors=args.k_neighbors,
        max_distance=args.max_distance,
        neighborhood_radius=args.neighborhood_radius,
        communication_radius=args.communication_radius,
        bootstrap=args.bootstrap,
        export_graphml=not args.no_graphml,
    )
    result = build_spatial_heterograph(frame, config)
    output = ensure_dir(args.output)
    write_spatial_graph_outputs(result, output, export_graphml=config.export_graphml)
    plot_spatial_atlas(result.nodes, output / "spatial_atlas.png", args.representative_sample)
    plot_contact_heatmap(result.contact_enrichment, output / "contact_enrichment_heatmap.png")
    plot_communication_circuits(result.circuits, output / "communication_circuits.png")
    plot_heterograph_summary(result.nodes, result.communication_edges, output / "heterograph_summary.png")
    plot_niche_composition(result.niche_summary, output / "spatial_niche_composition.png")
    print(json.dumps(result.qc, indent=2))
    print(f"spatial graph outputs: {output}")


def command_spatial_validate(args) -> None:
    nodes = pd.read_csv(Path(args.input) / "graph_nodes.csv")
    spatial_edges = pd.read_csv(Path(args.input) / "spatial_edges.csv")
    communication_edges = pd.read_csv(Path(args.input) / "communication_edges.csv")
    report = validate_spatial_graph(nodes, spatial_edges, communication_edges, SpatialGraphConfig())
    print(json.dumps(report, indent=2))


def command_therapeutics_rank(args) -> None:
    frame = pd.read_csv(args.input)
    config = TherapeuticConfig(
        comparator=args.comparator,
        horizon_hours=args.horizon,
        timing_grid=tuple(args.timing_grid),
        default_start_hour=args.default_start,
        sequence_delay_hours=args.sequence_delay,
        bootstrap=args.bootstrap,
        max_reference_rows_per_donor=args.max_reference_rows_per_donor,
        seed=args.seed,
    )
    result = run_counterfactual_therapeutics(frame, config)
    output = ensure_dir(args.output)
    write_therapeutic_outputs(result, output, config)
    print(json.dumps(result.qc, indent=2))
    print(result.predictions.nsmallest(args.top_n, "rank")[
        ["rank", "regimen_category", "regimen_name", "resistance_risk_reduction", "normal_cell_toxicity", "uncertainty_adjusted_utility"]
    ].to_string(index=False))
    print(f"therapeutic outputs: {output}")


def command_therapeutics_predict(args) -> None:
    frame = pd.read_csv(args.input)
    catalog = intervention_catalog()
    lookup = catalog.set_index("intervention_id", drop=False)
    intervention_ids = list(args.interventions)
    missing = sorted(set(intervention_ids) - set(lookup.index))
    if missing:
        raise ValueError(f"Unknown intervention IDs: {missing}")
    starts = list(args.start_hours or [24.0] * len(intervention_ids))
    doses = list(args.doses or [1.0] * len(intervention_ids))
    if len(starts) == 1 and len(intervention_ids) > 1:
        starts *= len(intervention_ids)
    if len(doses) == 1 and len(intervention_ids) > 1:
        doses *= len(intervention_ids)
    if len(starts) != len(intervention_ids) or len(doses) != len(intervention_ids):
        raise ValueError("start-hours and doses must each have length 1 or match interventions")
    events = []
    for position, (intervention_id, start, dose) in enumerate(zip(intervention_ids, starts, doses), 1):
        spec = lookup.loc[intervention_id]
        events.append({
            "intervention_id": intervention_id,
            "intervention_name": str(spec["intervention_name"]),
            "intervention_type": str(spec["intervention_type"]),
            "mechanism": str(spec["mechanism"]),
            "start_hour": float(start),
            "dose": float(dose),
            "duration_hours": float(spec["default_duration_hours"]),
            "sequence_position": position,
        })
    regimen = pd.DataFrame([{
        "regimen_id": "CUSTOM__" + "__".join(intervention_ids),
        "regimen_name": " then ".join(event["intervention_name"] for event in events),
        "regimen_category": "custom",
        "n_events": len(events),
        "n_mechanisms": len({event["mechanism"] for event in events}),
        "mechanisms": ";".join(sorted({event["mechanism"] for event in events})),
        "first_start_hour": min(starts),
        "last_start_hour": max(starts),
        "complexity_penalty": max(0.0, 0.04 * (len(events) - 1)),
        "events_json": json.dumps(events, sort_keys=True),
    }])
    config = TherapeuticConfig(
        comparator=args.comparator,
        horizon_hours=args.horizon,
        bootstrap=1,
        seed=args.seed,
    )
    model = fit_therapeutic_model(frame, seed=args.seed)
    predictions, state_changes = predict_regimens(frame, model, catalog, regimen, config)
    output = ensure_dir(args.output)
    predictions.to_csv(output / "custom_regimen_prediction.csv", index=False)
    state_changes.to_csv(output / "custom_regimen_state_changes.csv", index=False)
    print(predictions.to_string(index=False))
    print(f"custom prediction outputs: {output}")


def command_biomarkers_rank(args) -> None:
    frame = pd.read_csv(args.input)
    payload = _load_yaml(args.config) if args.config else {}
    graph_payload = payload.get("causal_graph", {})
    graph = build_causal_graph(graph_payload.get("nodes", []), graph_payload.get("edges", []))
    biomarker_payload = payload.get("biomarkers", {})
    features = list(args.features or biomarker_payload.get("features", []))
    if not features:
        raise ValueError("Provide --features or a config with biomarkers.features")
    config = BiomarkerConfig(
        outcome_column=args.outcome,
        cell_type=args.cell_type,
        target_node=args.target_node,
        warning_auc_threshold=args.warning_auc_threshold,
        warning_stability_threshold=args.warning_stability_threshold,
        bootstrap=args.bootstrap,
        top_panel_size=args.top_panel_size,
        seed=args.seed,
    )
    result = run_causal_biomarkers(
        frame, graph, features, config,
        assayability=biomarker_payload.get("assayability", {}),
        metadata_overrides=biomarker_payload.get("metadata_overrides", {}),
    )
    output = ensure_dir(args.output)
    write_biomarker_outputs(result, output, write_plots=True)
    print(json.dumps(result.qc, indent=2))
    print(result.ranking.head(args.top_n)[[
        "rank", "biomarker", "selected_time_hours", "early_warning_lead_hours",
        "association_auc", "causal_proximity", "uncertainty_adjusted_score", "evidence_tier",
    ]].to_string(index=False))
    print(f"biomarker outputs: {output}")


def command_biomarkers_validate(args) -> None:
    report = validate_biomarker_outputs(args.input)
    print(json.dumps(report, indent=2))



def _closed_loop_config_from_payload(payload: dict, seed: int) -> ClosedLoopConfig:
    return ClosedLoopConfig(
        budget=float(payload.get("budget", 2.4)),
        batch_size=int(payload.get("batch_size", 4)),
        round2_budget=float(payload.get("round2_budget", 2.0)),
        round2_batch_size=int(payload.get("round2_batch_size", 3)),
        max_per_type=int(payload.get("max_per_type", 2)),
        require_type_coverage=bool(payload.get("require_type_coverage", True)),
        diversity_penalty=float(payload.get("diversity_penalty", 0.10)),
        information_gain_weight=float(payload.get("information_gain_weight", 0.40)),
        therapeutic_value_weight=float(payload.get("therapeutic_value_weight", 0.22)),
        biomarker_value_weight=float(payload.get("biomarker_value_weight", 0.13)),
        temporal_value_weight=float(payload.get("temporal_value_weight", 0.10)),
        feasibility_weight=float(payload.get("feasibility_weight", 0.15)),
        bootstrap=int(payload.get("bootstrap", 60)),
        eig_samples=int(payload.get("eig_samples", 1200)),
        seed=seed,
        simulate_demonstration_round=bool(payload.get("simulate_demonstration_round", True)),
        true_hypothesis=payload.get("true_hypothesis"),
    )


def command_experiments_rank(args) -> None:
    root = Path(args.input)
    payload = _load_yaml(args.config) if args.config else {}
    closed_loop_payload = payload.get("closed_loop", {})
    config = _closed_loop_config_from_payload(closed_loop_payload, args.seed)
    result = run_closed_loop_experimentation(
        hypotheses_payload=closed_loop_payload.get("hypotheses") or None,
        candidates_payload=closed_loop_payload.get("candidates") or None,
        config=config,
        therapeutic_predictions=pd.read_csv(root / "therapeutics" / "all_regimen_predictions.csv"),
        biomarkers=pd.read_csv(root / "biomarkers" / "causal_biomarker_ranking.csv"),
        biomarker_timecourse=pd.read_csv(root / "biomarkers" / "early_warning_timecourse.csv"),
        transition_uncertainty=pd.read_csv(root / "transitions" / "transition_bootstrap_intervals.csv"),
    )
    output = ensure_dir(args.output)
    write_closed_loop_outputs(result, output, write_plots=True)
    print(json.dumps(result.qc, indent=2))
    print(result.round1_ranking.head(args.top_n)[[
        "rank", "experiment_type", "experiment_name", "mechanism",
        "expected_information_gain_nats", "priority_score", "relative_cost",
        "bootstrap_batch_selection_probability",
    ]].to_string(index=False))
    print(f"closed-loop outputs: {output}")


def command_experiments_validate(args) -> None:
    print(json.dumps(validate_closed_loop_outputs(args.input), indent=2))


def command_experiments_update(args) -> None:
    root = Path(args.input)
    active = root / "active_learning" if (root / "active_learning").exists() else root
    hypotheses = pd.read_csv(active / "hypothesis_priors.csv")
    catalog = pd.read_csv(active / "experiment_catalog.csv")
    observations = pd.read_csv(args.observations)
    payload = _load_yaml(args.config) if args.config else {}
    config = _closed_loop_config_from_payload(payload.get("closed_loop", {}), args.seed)
    history, ranking, batch = update_closed_loop_from_observations(
        hypotheses,
        catalog,
        observations,
        config=config,
        therapeutic_predictions=pd.read_csv(root / "therapeutics" / "all_regimen_predictions.csv") if (root / "therapeutics" / "all_regimen_predictions.csv").exists() else pd.DataFrame(),
        biomarkers=pd.read_csv(root / "biomarkers" / "causal_biomarker_ranking.csv") if (root / "biomarkers" / "causal_biomarker_ranking.csv").exists() else pd.DataFrame(),
        biomarker_timecourse=pd.read_csv(root / "biomarkers" / "early_warning_timecourse.csv") if (root / "biomarkers" / "early_warning_timecourse.csv").exists() else pd.DataFrame(),
        transition_uncertainty=pd.read_csv(root / "transitions" / "transition_bootstrap_intervals.csv") if (root / "transitions" / "transition_bootstrap_intervals.csv").exists() else pd.DataFrame(),
    )
    output = ensure_dir(args.output)
    history.to_csv(output / "updated_hypothesis_posterior.csv", index=False)
    ranking.to_csv(output / "updated_experiment_recommendations.csv", index=False)
    batch.to_csv(output / "updated_selected_batch.csv", index=False)
    print(ranking.head(args.top_n)[[
        "rank", "experiment_type", "experiment_name", "expected_information_gain_nats", "priority_score"
    ]].to_string(index=False))
    print(f"updated closed-loop outputs: {output}")


def command_neuro_run(args) -> None:
    payload = _load_yaml(args.config).get("neurobiology", {})
    config = NeurobiologyConfig(
        n_donors=int(payload.get("n_donors", args.n_donors)),
        cells_per_type=int(payload.get("cells_per_type", args.cells_per_type)),
        times_days=tuple(float(value) for value in payload.get("times_days", [0, 7, 21, 42])),
        apoe4_fraction=float(payload.get("apoe4_fraction", 0.5)),
        bootstrap=int(payload.get("bootstrap", args.bootstrap)),
        seed=int(payload.get("seed", args.seed)),
        warning_time_days=float(payload.get("warning_time_days", 21.0)),
        terminal_time_days=float(payload.get("terminal_time_days", 42.0)),
    )
    result = run_neurobiology_configuration(config)
    output = ensure_dir(args.output)
    write_neurobiology_outputs(result, output, write_plots=True)
    validate_neurobiology_outputs(output)
    report = generate_neurobiology_report(output, output / "neurobiology_report.html")
    print(json.dumps(result.qc, indent=2))
    print(f"neurobiology report: {report}")


def command_neuro_validate(args) -> None:
    report = validate_neurobiology_outputs(args.input)
    print(json.dumps(report, indent=2))

def command_therapeutics_validate(args) -> None:
    root = Path(args.input)
    qc = json.loads((root / "therapeutic_qc.json").read_text())
    predictions = pd.read_csv(root / "all_regimen_predictions.csv")
    required_categories = {"gene", "drug", "combination", "sequence", "timing"}
    valid = bool(
        qc.get("valid")
        and required_categories.issubset(set(predictions["regimen_category"]))
        and predictions["counterfactual_resistance_probability"].between(0, 1).all()
        and predictions["normal_cell_toxicity"].between(0, 1).all()
        and predictions["utility_ci_low"].le(predictions["utility_ci_high"]).all()
    )
    report = {**qc, "cli_validation": valid}
    if not valid:
        raise SystemExit("Therapeutic output validation failed")
    print(json.dumps(report, indent=2))



def command_publication_build(args) -> None:
    root = Path(args.input)
    if args.group:
        inventory = rebuild_reference_figure_group(root, args.group)
    else:
        script = Path(__file__).resolve().parents[2] / "scripts" / "publication_build.sh"
        subprocess.run(["bash", str(script), sys.executable, str(root)], check=True)
        inventory = pd.read_csv(root / "publication_graphics" / "figure_inventory.csv")
    print(f"publication figure bundles: {len(inventory)}")


def command_publication_validate(args) -> None:
    report = validate_publication_bundle(args.input, check_hashes=not args.skip_hashes)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)

def command_version(args) -> None:
    from . import __version__
    print(f"CausaFlux {__version__}")


def command_platform_doctor(args) -> None:
    report = platform_doctor(args.project_root)
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


def command_platform_validate(args) -> None:
    if args.refresh:
        report = finalize_research_platform(args.input)
    else:
        report = validate_research_platform(args.input, verify_hashes=not args.skip_hashes)
    print(json.dumps(report.to_dict(), indent=2))
    if not report.valid:
        raise SystemExit(1)


def command_demo_list(args) -> None:
    frame = demo_registry_frame(Path(__file__).resolve().parents[2])
    if args.json:
        print(frame.to_json(orient="records", indent=2))
    else:
        print(frame[["demo_id", "domain", "title", "command"]].to_string(index=False))


def command_demo_run(args) -> None:
    project = Path(__file__).resolve().parents[2]
    registry = demo_registry_frame(project).set_index("demo_id")
    if args.demo_id not in registry.index:
        raise SystemExit(f"Unknown demo: {args.demo_id}")
    row = registry.loc[args.demo_id]
    if args.demo_id == "integrated_reference":
        command = ["bash", str(project / "run_synthetic_smoke.sh")]
        if args.output:
            command.append(args.output)
    else:
        script = project / "demos" / args.demo_id / "run.sh"
        command = ["bash", str(script)]
        if args.output:
            command.append(args.output)
    subprocess.run(command, cwd=project, check=True)


def command_run(args) -> None:
    project = Path(__file__).resolve().parents[2]
    output = args.output or "causaflux_v1.7.0_output"
    command = [
        "bash", str(project / "scripts" / "run_staged.sh"), sys.executable,
        str(Path(args.config).resolve()), str(output),
    ]
    subprocess.run(command, cwd=project, check=True)
    print(f"report: {Path(output) / 'report' / 'index.html'}")


def command_dynamic_run(args) -> None:
    result = run_experiment(args.config, args.output, args.device)
    print(f"report: {result['report']}")


def command_demo(args) -> None:
    project = Path(__file__).resolve().parents[2]
    config = project / "configs" / "cancer_closed_loop_v1.7.0.yaml"
    command = ["bash", str(project / "scripts" / "run_staged.sh"), sys.executable, str(config), str(args.output)]
    subprocess.run(command, cwd=project, check=True)


def command_dynamic_demo(args) -> None:
    config = Path(__file__).resolve().parents[2] / "configs" / "demo_v0.2.yaml"
    run_experiment(config, args.output, args.device)



def command_benchmark_list(args) -> None:
    frame = benchmark_registry_frame(args.manifest_dir)
    print(frame.to_string(index=False))


def command_benchmark_show(args) -> None:
    spec = get_benchmark(args.id, args.manifest_dir)
    payload = {
        "benchmark_id": spec.benchmark_id,
        "title": spec.title,
        "domain": spec.domain,
        "status": spec.status,
        "estimated_storage_gb": spec.estimated_storage_gb,
        "primary_question": spec.primary_question,
        "evaluation": spec.evaluation,
        "sources": [source.__dict__ for source in spec.sources],
    }
    print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def command_benchmark_preflight(args) -> None:
    registry = validate_realdata_registry(args.manifest_dir)
    checks = preflight_benchmarks(args.manifest_dir)
    if args.output:
        output = ensure_dir(args.output)
        checks.to_csv(output / "preflight_checks.csv", index=False)
        accession_manifest_frame(args.manifest_dir).to_csv(output / "accession_manifest.csv", index=False)
        (output / "registry_validation.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps(registry, indent=2))
    print(checks.to_string(index=False))
    if not registry["valid"]:
        raise SystemExit(1)


def command_benchmark_plan(args) -> None:
    ids = None if not args.id or args.id == ["all"] else args.id
    frame = build_download_plan(args.output, ids, metadata_only=not args.full, manifest_dir=args.manifest_dir)
    destination = Path(args.plan_csv) if args.plan_csv else Path(args.output) / "download_plan.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(frame.to_string(index=False))
    print(f"download plan: {destination}")


def command_benchmark_report(args) -> None:
    paths = generate_realdata_reports(args.output, project_root=args.project_root, manifest_dir=args.manifest_dir)
    result = validate_realdata_output(args.output)
    print(json.dumps(result, indent=2))
    print(f"report: {paths['report']}")
    if not result["valid"]:
        raise SystemExit(1)


def command_benchmark_validate(args) -> None:
    result = validate_realdata_output(args.input)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)



def command_dynamic_benchmark_run(args) -> None:
    cfg = DynamicBenchmarkConfig(
        seed=args.seed,
        replicates_per_history=args.replicates_per_history,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    models = args.models or DYNAMIC_MODEL_ORDER
    data = load_external_benchmark_npz(args.data_npz) if args.data_npz else None
    status = run_dynamic_benchmark(args.output, cfg, model_names=models, data=data)
    print(json.dumps(status, indent=2))
    if args.require_gate and status["gate"]["status"] != "PASS":
        raise SystemExit(2)




def command_dynamic_benchmark_export_fixture(args) -> None:
    cfg = DynamicBenchmarkConfig(
        seed=args.seed,
        replicates_per_history=args.replicates_per_history,
    )
    path = save_external_benchmark_npz(generate_dynamic_benchmark_data(cfg), args.output)
    print(f"dynamic benchmark fixture: {path}")

def command_dynamic_benchmark_validate(args) -> None:
    result = validate_dynamic_benchmark(args.input)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)

def command_multimodal_dynamic_run(args) -> None:
    cfg = MultimodalDynamicConfig(
        seed=args.seed,
        replicates_per_history=args.replicates_per_history,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        modality_dropout=args.modality_dropout,
        bootstrap_replicates=args.bootstrap,
        device=args.device,
    )
    models = args.models or MULTIMODAL_DYNAMIC_MODEL_ORDER
    data = load_external_multimodal_npz(args.data_npz) if args.data_npz else None
    status = run_multimodal_dynamic_benchmark(args.output, cfg, data=data, models=models, require_gate=args.require_gate)
    print(status["comparison"].to_string(index=False))
    print(json.dumps(status["gate"], indent=2))


def command_multimodal_dynamic_export_fixture(args) -> None:
    cfg = MultimodalDynamicConfig(seed=args.seed, replicates_per_history=args.replicates_per_history)
    path = save_external_multimodal_npz(generate_multimodal_dynamic_data(cfg), args.output)
    print(f"multimodal dynamic fixture: {path}")


def command_multimodal_dynamic_validate(args) -> None:
    result = validate_multimodal_dynamic_benchmark(args.input, verify_hashes=not args.skip_hashes)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


def _parse_external_adapters(values: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError("--external-adapter expects NAME=PATH")
        name, path = item.split("=", 1)
        result[name] = path
    return result


def command_intervention_generalization_run(args) -> None:
    cfg = InterventionGeneralizationConfig(
        seed=args.seed,
        replicates=args.replicates,
        bootstrap_replicates=args.bootstrap,
        conformal_alpha=args.conformal_alpha,
        ridge_alpha=args.ridge_alpha,
    )
    data = load_external_intervention_npz(args.data_npz) if args.data_npz else None
    status = run_intervention_generalization_benchmark(
        args.output, cfg, data=data,
        external_predictions=_parse_external_adapters(args.external_adapter),
        require_gate=args.require_gate,
    )
    print(json.dumps(status, indent=2, default=str))


def command_intervention_generalization_export_fixture(args) -> None:
    cfg = InterventionGeneralizationConfig(seed=args.seed, replicates=args.replicates)
    path = save_external_intervention_npz(generate_intervention_generalization_data(cfg), args.output)
    print(f"intervention generalization fixture: {path}")


def command_intervention_generalization_validate(args) -> None:
    result = validate_intervention_generalization(args.input, verify_hashes=not args.skip_hashes)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


def command_intervention_adapter_list(args) -> None:
    print(adapter_registry_frame().to_string(index=False))


def command_spatiotemporal_tissue_run(args) -> None:
    cfg = SpatiotemporalTissueConfig(
        seed=args.seed, n_donors=args.donors, sections_per_donor=args.sections_per_donor,
        cells_per_section=args.cells_per_section, k_neighbors=args.k_neighbors,
        ridge_alpha=args.ridge_alpha, gate_C=args.gate_c, bootstrap_replicates=args.bootstrap,
    )
    data = load_external_spatiotemporal_npz(args.data_npz) if args.data_npz else None
    status = run_spatiotemporal_tissue_benchmark(args.output, cfg, data=data, require_gate=args.require_gate)
    print(json.dumps(status, indent=2, default=str))


def command_spatiotemporal_tissue_export_fixture(args) -> None:
    cfg = SpatiotemporalTissueConfig(seed=args.seed, n_donors=args.donors, sections_per_donor=args.sections_per_donor, cells_per_section=args.cells_per_section)
    path = save_external_spatiotemporal_npz(generate_spatiotemporal_tissue_data(cfg), args.output)
    print(f"spatiotemporal digital-tissue fixture: {path}")


def command_spatiotemporal_tissue_validate(args) -> None:
    result = validate_spatiotemporal_tissue(args.input, verify_hashes=not args.skip_hashes)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


def command_nicheformer_adapter_show(args) -> None:
    print(json.dumps(nicheformer_adapter_spec(), indent=2))


def command_validation_list(args) -> None:
    frame = hypothesis_registry_frame(args.manifest_dir)
    print(frame[["hypothesis_id", "domain", "status", "title", "discovery_cohort", "replication_cohort"]].to_string(index=False))


def command_validation_preregister(args) -> None:
    path = freeze_preregistration(args.output, args.manifest_dir)
    print(f"preregistration lock: {path}")


def command_validation_run(args) -> None:
    project = Path(__file__).resolve().parents[2]
    snapshots = Path(args.snapshots) if args.snapshots else project / "benchmarks" / "snapshots" / "sea_ad"
    destination = run_and_write_biological_validation(snapshots, args.output, n_boot=args.bootstrap, seed=args.seed)
    result = validate_biological_validation(destination)
    print(json.dumps(result, indent=2))
    print(f"report: {Path(destination) / 'reports' / 'index.html'}")
    if not result["valid"]:
        raise SystemExit(1)


def command_validation_validate(args) -> None:
    result = validate_biological_validation(args.input, verify_hashes=not args.skip_hashes)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


def command_foundation_run(args) -> None:
    cfg=FoundationPretrainingConfig(seed=args.seed,n_samples=args.samples,n_components=args.components)
    data=load_external_foundation_npz(args.data_npz) if args.data_npz else None
    status=run_foundation_pretraining(args.output,cfg,data=data,require_gate=args.require_gate)
    print(json.dumps(status,indent=2))
    print(f"report: {Path(args.output)/'report'/'index.html'}")

def command_foundation_validate(args) -> None:
    status=validate_foundation_pretraining(args.input,verify_hashes=not args.skip_hashes)
    print(json.dumps(status,indent=2))
    if not status['valid']: raise SystemExit(1)

def command_foundation_export(args) -> None:
    cfg=FoundationPretrainingConfig(seed=args.seed,n_samples=args.samples,n_components=args.components)
    save_external_foundation_npz(generate_foundation_data(cfg),args.output)
    print(args.output)

def command_foundation_adapter_list(args) -> None:
    print(json.dumps(foundation_adapter_registry(),indent=2))

def command_foundation_objective_list(args) -> None:
    print(objective_registry_frame().to_string(index=False))



def command_prospective_run(args) -> None:
    payload = _load_yaml(args.config).get("prospective_loop", {}) if args.config else {}
    config = ProspectiveLoopConfig(
        seed=int(payload.get("seed", args.seed)),
        truth_hypothesis=str(payload.get("truth_hypothesis", args.truth_hypothesis)),
        max_cycles=int(payload.get("max_cycles", 3)),
        min_cycles=int(payload.get("min_cycles", 3)),
        experiments_per_cycle=int(payload.get("experiments_per_cycle", args.experiments_per_cycle)),
        cycle_budget=float(payload.get("cycle_budget", args.cycle_budget)),
        posterior_stop_threshold=float(payload.get("posterior_stop_threshold", 0.92)),
        min_expected_information_gain=float(payload.get("min_expected_information_gain", 0.015)),
        discovery_threshold=float(payload.get("discovery_threshold", 0.70)),
        recovery_threshold=float(payload.get("recovery_threshold", -0.55)),
        baseline_strategy=str(payload.get("baseline_strategy", "prespecified_non_ai_fixed_order")),
        require_independent_cycle3=bool(payload.get("require_independent_cycle3", True)),
        synthetic_failure_experiment=None if args.no_synthetic_failure else payload.get("synthetic_failure_experiment", args.synthetic_failure_experiment),
    )
    result = run_prospective_loop(args.output, config=config)
    validation = validate_prospective_loop(args.output, require_gate=args.require_gate)
    print(json.dumps(validation, indent=2))
    print(result.comparison.to_string(index=False))
    print(f"prospective report: {Path(args.output) / 'report' / 'index.html'}")


def command_prospective_validate(args) -> None:
    print(json.dumps(validate_prospective_loop(args.input, require_gate=args.require_gate), indent=2))


def command_prospective_contract_export(args) -> None:
    paths = write_contract_bundle(args.output)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


def command_prospective_ingest(args) -> None:
    manifest = ingest_external_cycle(args.cycle_dir, args.qc, args.outcomes)
    print(json.dumps(manifest, indent=2))


def command_virtual_cell_run(args) -> None:
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    manifest = run_virtual_cell_release(project_root, args.output)
    validation = validate_virtual_cell_release(args.output, require_real_prospective=args.require_real_prospective)
    print(json.dumps(validation, indent=2))
    print(f"virtual-cell report: {Path(args.output) / 'report' / 'index.html'}")
    if not validation['valid']:
        raise SystemExit(1)


def command_virtual_cell_validate(args) -> None:
    result = validate_virtual_cell_release(args.input, require_real_prospective=args.require_real_prospective)
    print(json.dumps(result, indent=2))
    if not result['valid']:
        raise SystemExit(1)


def command_realworld_register(args) -> None:
    modalities = tuple(item.strip() for item in args.modalities.split(',') if item.strip())
    contract = UserDatasetContract(
        dataset_id=args.dataset_id, path=args.input, data_class=args.data_class, modalities=modalities,
        longitudinal=args.longitudinal, perturbational=args.perturbational, spatial=args.spatial,
        prospective=args.prospective, outcome_available=args.outcome_available, donor_column=args.donor_column,
        time_column=args.time_column, intervention_column=args.intervention_column, outcome_column=args.outcome_column,
        notes=args.notes or '',
    )
    path = register_user_dataset(contract, args.output)
    payload = {'contract': str(path)}
    if args.preview:
        payload['preview'] = preview_tabular_dataset(args.input)
    print(json.dumps(payload, indent=2))



def command_v2_run(args) -> None:
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    manifest = run_v2_release(project_root, args.output, external_evidence_dir=args.external_evidence_dir)
    result = validate_v2_output(args.output, require_prospectively_validated=args.require_prospectively_validated)
    print(json.dumps(result, indent=2))
    print(f"v2 report: {Path(args.output) / 'report' / 'index.html'}")
    if not result['valid']:
        raise SystemExit(1)


def command_v2_validate(args) -> None:
    result = validate_v2_output(args.input, require_prospectively_validated=args.require_prospectively_validated)
    print(json.dumps(result, indent=2))
    if not result['valid']:
        raise SystemExit(1)


def command_longitudinal_convert(args) -> None:
    result = convert_longitudinal_table(args.input, args.output, args.manifest)
    print(json.dumps(result, indent=2))


def command_longitudinal_benchmark(args) -> None:
    models = [x.strip() for x in args.models.split(',') if x.strip()] if args.models else None
    result = run_real_longitudinal_benchmark(args.input, args.output, model_names=models, seed=args.seed)
    print(json.dumps(result, indent=2, default=str))


def command_longitudinal_contract(args) -> None:
    paths = write_public_dataset_bundle(args.output)
    print(json.dumps({k:str(v) for k,v in paths.items()}, indent=2))


def command_shift_calibration(args) -> None:
    result = evaluate_shift_calibration_file(args.input, args.output)
    print(json.dumps(result, indent=2))
    if args.require_gate and result.get('real_distribution_shift_gate') != 'PASS':
        raise SystemExit(1)


def command_ui(args) -> None:
    env = dict(**__import__('os').environ)
    if args.output:
        env['CAUSAFLUX_OUTPUT'] = args.output
    app = Path(__file__).with_name('ui_app.py')
    cmd = [sys.executable, '-m', 'streamlit', 'run', str(app), '--server.headless=false']
    if args.port:
        cmd.extend(['--server.port', str(args.port)])
    raise SystemExit(subprocess.call(cmd, env=env))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causaflux",
        description="CausaFlux v2.0.0 release-gated virtual cell with real longitudinal perturbation data, prospective evidence, calibrated uncertainty, and claim-linked reporting",
    )
    sub = parser.add_subparsers(dest="command", required=True)



    foundation_run = sub.add_parser("foundation-pretrain-run", help="run v1.7 multi-objective foundation pretraining and downstream transfer benchmark")
    foundation_run.add_argument("--output", default="causaflux_v1.7.0_foundation_pretraining")
    foundation_run.add_argument("--data-npz", help="external dataset following the v1.7 foundation NPZ contract")
    foundation_run.add_argument("--samples", type=int, default=900)
    foundation_run.add_argument("--components", type=int, default=12)
    foundation_run.add_argument("--seed", type=int, default=170)
    foundation_run.add_argument("--require-gate", action="store_true")
    foundation_run.set_defaults(func=command_foundation_run)

    foundation_validate = sub.add_parser("foundation-pretrain-validate", help="validate v1.7 adapters, objectives, transfer evaluations, gate, and hashes")
    foundation_validate.add_argument("--input", required=True)
    foundation_validate.add_argument("--skip-hashes", action="store_true")
    foundation_validate.set_defaults(func=command_foundation_validate)

    foundation_export = sub.add_parser("foundation-pretrain-export-fixture", help="write the deterministic v1.7 external foundation NPZ fixture")
    foundation_export.add_argument("--output", default="foundation_pretraining_fixture_v1.7.0.npz")
    foundation_export.add_argument("--samples", type=int, default=900)
    foundation_export.add_argument("--components", type=int, default=12)
    foundation_export.add_argument("--seed", type=int, default=170)
    foundation_export.set_defaults(func=command_foundation_export)

    foundation_adapters = sub.add_parser("foundation-adapter-list", help="list scGPT, GET, Nicheformer, MrVI and selected encoder contracts")
    foundation_adapters.set_defaults(func=command_foundation_adapter_list)
    foundation_objectives = sub.add_parser("foundation-objective-list", help="list the ten CausaFlux pretraining objectives")
    foundation_objectives.set_defaults(func=command_foundation_objective_list)

    dynamic_benchmark_run = sub.add_parser(
        "dynamic-benchmark-run",
        help="train and compare static, recurrent, transformer, Neural CDE, and PRESCIENT-style models",
    )
    dynamic_benchmark_run.add_argument("--output", default="causaflux_v1.7.0_dynamic_benchmark")
    dynamic_benchmark_run.add_argument("--data-npz", help="external benchmark dataset following the v1.7.0 NPZ contract")
    dynamic_benchmark_run.add_argument("--models", nargs="+", choices=DYNAMIC_MODEL_ORDER)
    dynamic_benchmark_run.add_argument("--epochs", type=int, default=28)
    dynamic_benchmark_run.add_argument("--patience", type=int, default=6)
    dynamic_benchmark_run.add_argument("--batch-size", type=int, default=32)
    dynamic_benchmark_run.add_argument("--hidden-dim", type=int, default=48)
    dynamic_benchmark_run.add_argument("--replicates-per-history", type=int, default=4)
    dynamic_benchmark_run.add_argument("--seed", type=int, default=130)
    dynamic_benchmark_run.add_argument("--device", default="cpu")
    dynamic_benchmark_run.add_argument("--require-gate", action="store_true")
    dynamic_benchmark_run.set_defaults(func=command_dynamic_benchmark_run)


    dynamic_benchmark_export = sub.add_parser(
        "dynamic-benchmark-export-fixture",
        help="write the deterministic dynamic software fixture using the external benchmark NPZ contract",
    )
    dynamic_benchmark_export.add_argument("--output", default="dynamic_benchmark_fixture_v1.7.0.npz")
    dynamic_benchmark_export.add_argument("--replicates-per-history", type=int, default=4)
    dynamic_benchmark_export.add_argument("--seed", type=int, default=130)
    dynamic_benchmark_export.set_defaults(func=command_dynamic_benchmark_export_fixture)

    dynamic_benchmark_validate = sub.add_parser(
        "dynamic-benchmark-validate",
        help="validate held-out-history metrics, uncertainty coverage, and the foundation-pretraining gate",
    )
    dynamic_benchmark_validate.add_argument("--input", required=True)
    dynamic_benchmark_validate.set_defaults(func=command_dynamic_benchmark_validate)

    multimodal_dynamic_run = sub.add_parser(
        "multimodal-dynamic-run",
        help="benchmark modality-specific encoders, PoE/MoE fusion, imaging forecasting, and missingness sensitivity",
    )
    multimodal_dynamic_run.add_argument("--output", default="causaflux_v1.7.0_multimodal_dynamic")
    multimodal_dynamic_run.add_argument("--data-npz", help="external dataset following the v1.7.0 multimodal NPZ contract")
    multimodal_dynamic_run.add_argument("--models", nargs="+", choices=MULTIMODAL_DYNAMIC_MODEL_ORDER)
    multimodal_dynamic_run.add_argument("--epochs", type=int, default=30)
    multimodal_dynamic_run.add_argument("--patience", type=int, default=7)
    multimodal_dynamic_run.add_argument("--batch-size", type=int, default=32)
    multimodal_dynamic_run.add_argument("--hidden-dim", type=int, default=48)
    multimodal_dynamic_run.add_argument("--latent-dim", type=int, default=12)
    multimodal_dynamic_run.add_argument("--modality-dropout", type=float, default=0.25)
    multimodal_dynamic_run.add_argument("--bootstrap", type=int, default=100)
    multimodal_dynamic_run.add_argument("--replicates-per-history", type=int, default=5)
    multimodal_dynamic_run.add_argument("--seed", type=int, default=140)
    multimodal_dynamic_run.add_argument("--device", default="cpu")
    multimodal_dynamic_run.add_argument("--require-gate", action="store_true")
    multimodal_dynamic_run.set_defaults(func=command_multimodal_dynamic_run)

    multimodal_dynamic_export = sub.add_parser(
        "multimodal-dynamic-export-fixture",
        help="write the deterministic v1.4 multimodal longitudinal NPZ fixture",
    )
    multimodal_dynamic_export.add_argument("--output", default="multimodal_dynamic_fixture_v1.7.0.npz")
    multimodal_dynamic_export.add_argument("--replicates-per-history", type=int, default=5)
    multimodal_dynamic_export.add_argument("--seed", type=int, default=140)
    multimodal_dynamic_export.set_defaults(func=command_multimodal_dynamic_export_fixture)

    multimodal_dynamic_validate = sub.add_parser(
        "multimodal-dynamic-validate",
        help="validate v1.4 multimodal benchmark outputs, exit gate, source data, and hashes",
    )
    multimodal_dynamic_validate.add_argument("--input", required=True)
    multimodal_dynamic_validate.add_argument("--skip-hashes", action="store_true")
    multimodal_dynamic_validate.set_defaults(func=command_multimodal_dynamic_validate)


    spatiotemporal_run = sub.add_parser(
        "spatiotemporal-tissue-run",
        help="benchmark time-varying heterogeneous tissue graphs and neighborhood-conditioned continuous dynamics",
    )
    spatiotemporal_run.add_argument("--output", default="causaflux_v1.7.0_spatiotemporal_tissue")
    spatiotemporal_run.add_argument("--data-npz", help="external dataset following the v1.7.0 spatiotemporal NPZ contract")
    spatiotemporal_run.add_argument("--donors", type=int, default=12)
    spatiotemporal_run.add_argument("--sections-per-donor", type=int, default=2)
    spatiotemporal_run.add_argument("--cells-per-section", type=int, default=36)
    spatiotemporal_run.add_argument("--k-neighbors", type=int, default=5)
    spatiotemporal_run.add_argument("--ridge-alpha", type=float, default=1.0)
    spatiotemporal_run.add_argument("--gate-c", type=float, default=2.0)
    spatiotemporal_run.add_argument("--bootstrap", type=int, default=100)
    spatiotemporal_run.add_argument("--seed", type=int, default=160)
    spatiotemporal_run.add_argument("--require-gate", action="store_true")
    spatiotemporal_run.set_defaults(func=command_spatiotemporal_tissue_run)

    spatiotemporal_export = sub.add_parser(
        "spatiotemporal-tissue-export-fixture",
        help="write the deterministic v1.6 spatiotemporal digital-tissue NPZ fixture",
    )
    spatiotemporal_export.add_argument("--output", default="spatiotemporal_tissue_fixture_v1.7.0.npz")
    spatiotemporal_export.add_argument("--donors", type=int, default=12)
    spatiotemporal_export.add_argument("--sections-per-donor", type=int, default=2)
    spatiotemporal_export.add_argument("--cells-per-section", type=int, default=36)
    spatiotemporal_export.add_argument("--seed", type=int, default=160)
    spatiotemporal_export.set_defaults(func=command_spatiotemporal_tissue_export_fixture)

    spatiotemporal_validate = sub.add_parser(
        "spatiotemporal-tissue-validate",
        help="validate v1.6 tissue graphs, split audits, neighborhood gate, outcomes, figures, and hashes",
    )
    spatiotemporal_validate.add_argument("--input", required=True)
    spatiotemporal_validate.add_argument("--skip-hashes", action="store_true")
    spatiotemporal_validate.set_defaults(func=command_spatiotemporal_tissue_validate)

    nicheformer_show = sub.add_parser("nicheformer-adapter-show", help="show the external Nicheformer embedding adapter contract")
    nicheformer_show.set_defaults(func=command_nicheformer_adapter_show)

    intervention_run = sub.add_parser(
        "intervention-generalization-run",
        help="benchmark unseen perturbation, dose, combination, and sequence generalization",
    )
    intervention_run.add_argument("--output", default="causaflux_v1.7.0_intervention_generalization")
    intervention_run.add_argument("--data-npz", help="external dataset following the v1.7.0 intervention NPZ contract")
    intervention_run.add_argument("--external-adapter", action="append", help="actual established-model prediction file as NAME=PATH; repeat for CPA/GEARS/TxPert/scGPT")
    intervention_run.add_argument("--replicates", type=int, default=5)
    intervention_run.add_argument("--bootstrap", type=int, default=100)
    intervention_run.add_argument("--conformal-alpha", type=float, default=0.10)
    intervention_run.add_argument("--ridge-alpha", type=float, default=2.0)
    intervention_run.add_argument("--seed", type=int, default=150)
    intervention_run.add_argument("--require-gate", action="store_true")
    intervention_run.set_defaults(func=command_intervention_generalization_run)

    intervention_export = sub.add_parser(
        "intervention-generalization-export-fixture",
        help="write the deterministic v1.5 intervention-generalization NPZ fixture",
    )
    intervention_export.add_argument("--output", default="intervention_generalization_fixture_v1.7.0.npz")
    intervention_export.add_argument("--replicates", type=int, default=5)
    intervention_export.add_argument("--seed", type=int, default=150)
    intervention_export.set_defaults(func=command_intervention_generalization_export_fixture)

    intervention_validate = sub.add_parser(
        "intervention-generalization-validate",
        help="validate v1.5 intervention metrics, uncertainty, support diagnostics, gates, and hashes",
    )
    intervention_validate.add_argument("--input", required=True)
    intervention_validate.add_argument("--skip-hashes", action="store_true")
    intervention_validate.set_defaults(func=command_intervention_generalization_validate)

    intervention_adapters = sub.add_parser(
        "intervention-adapter-list",
        help="list CPA, GEARS, TxPert and scGPT external adapter contracts",
    )
    intervention_adapters.set_defaults(func=command_intervention_adapter_list)

    validation_list = sub.add_parser("validation-list", help="list preregistered biological-validation hypotheses")
    validation_list.add_argument("--manifest-dir")
    validation_list.set_defaults(func=command_validation_list)

    validation_preregister = sub.add_parser("validation-preregister", help="freeze hypothesis manifests before analysis")
    validation_preregister.add_argument("--output", default="biological_validation/preregistration")
    validation_preregister.add_argument("--manifest-dir")
    validation_preregister.set_defaults(func=command_validation_preregister)

    validation_run = sub.add_parser("validation-run", help="run the public SEA-AD biological-validation benchmark")
    validation_run.add_argument("--snapshots")
    validation_run.add_argument("--output", default="causaflux_v1.7.0_validation")
    validation_run.add_argument("--bootstrap", type=int, default=500)
    validation_run.add_argument("--seed", type=int, default=120)
    validation_run.set_defaults(func=command_validation_run)

    validation_validate = sub.add_parser("validation-validate", help="validate preregistration, replication, evidence, and manuscript outputs")
    validation_validate.add_argument("--input", required=True)
    validation_validate.add_argument("--skip-hashes", action="store_true")
    validation_validate.set_defaults(func=command_validation_validate)

    benchmark_list = sub.add_parser("benchmark-list", help="list accession-pinned real-data benchmarks")
    benchmark_list.add_argument("--manifest-dir")
    benchmark_list.set_defaults(func=command_benchmark_list)

    benchmark_show = sub.add_parser("benchmark-show", help="show one benchmark manifest")
    benchmark_show.add_argument("--id", required=True)
    benchmark_show.add_argument("--manifest-dir")
    benchmark_show.set_defaults(func=command_benchmark_show)

    benchmark_preflight = sub.add_parser("benchmark-preflight", help="validate manifests, access requirements, licenses, and tools")
    benchmark_preflight.add_argument("--manifest-dir")
    benchmark_preflight.add_argument("--output")
    benchmark_preflight.set_defaults(func=command_benchmark_preflight)

    benchmark_plan = sub.add_parser("benchmark-plan", help="write a non-destructive download and access plan")
    benchmark_plan.add_argument("--id", nargs="+", default=["all"])
    benchmark_plan.add_argument("--output", default="realdata_benchmarks/data")
    benchmark_plan.add_argument("--plan-csv")
    benchmark_plan.add_argument("--manifest-dir")
    benchmark_plan.add_argument("--full", action="store_true", help="plan full assay downloads instead of metadata-only access")
    benchmark_plan.set_defaults(func=command_benchmark_plan)

    benchmark_report = sub.add_parser("benchmark-report", help="generate accession, licensing, and real-metadata benchmark reports")
    benchmark_report.add_argument("--output", default="causaflux_v1.7.0_realdata")
    benchmark_report.add_argument("--project-root")
    benchmark_report.add_argument("--manifest-dir")
    benchmark_report.set_defaults(func=command_benchmark_report)

    benchmark_validate = sub.add_parser("benchmark-validate", help="validate generated real-data benchmark reports")
    benchmark_validate.add_argument("--input", required=True)
    benchmark_validate.set_defaults(func=command_benchmark_validate)

    version = sub.add_parser("version", help="print the CausaFlux version")
    version.set_defaults(func=command_version)

    doctor = sub.add_parser("doctor", help="check Python, package files, and demo readiness")
    doctor.add_argument("--project-root")
    doctor.set_defaults(func=command_platform_doctor)

    platform_validate = sub.add_parser(
        "platform-validate", help="validate cross-domain outputs, provenance, reports, and hashes"
    )
    platform_validate.add_argument("--input", required=True)
    platform_validate.add_argument("--refresh", action="store_true", help="regenerate platform artifacts before validation")
    platform_validate.add_argument("--skip-hashes", action="store_true")
    platform_validate.set_defaults(func=command_platform_validate)

    publication_build = sub.add_parser(
        "publication-build", help="regenerate Nature/Cell publication figure bundles"
    )
    publication_build.add_argument("--input", required=True, help="CausaFlux analysis output directory")
    publication_build.add_argument("--group", choices=PUBLICATION_GROUPS)
    publication_build.set_defaults(func=command_publication_build)

    publication_validate = sub.add_parser(
        "publication-validate", help="validate vector/raster exports, source data, and visual baselines"
    )
    publication_validate.add_argument("--input", required=True)
    publication_validate.add_argument("--skip-hashes", action="store_true")
    publication_validate.set_defaults(func=command_publication_validate)

    demo_list = sub.add_parser("demo-list", help="list packaged cancer, neurobiology, and integrated demos")
    demo_list.add_argument("--json", action="store_true")
    demo_list.set_defaults(func=command_demo_list)

    demo_run = sub.add_parser("demo-run", help="run one packaged demo by ID")
    demo_run.add_argument("demo_id", choices=["spatiotemporal_digital_tissue", "intervention_generalization", "multimodal_dynamic_state", "dynamic_model_benchmark", "cancer_quickstart", "neurobiology_quickstart", "integrated_reference", "biological_validation", "realdata_registry"])
    demo_run.add_argument("--output")
    demo_run.set_defaults(func=command_demo_run)

    run = sub.add_parser("run", help="run the validated v1.0 cancer and neurobiology platform")
    run.add_argument("--config", default="configs/cancer_closed_loop_v1.7.0.yaml")
    run.add_argument("--output")
    run.set_defaults(func=command_run)

    dynamic_run = sub.add_parser("dynamic-run", help="run the preserved v0.2 dynamic virtual-cell workflow")
    dynamic_run.add_argument("--config", default="configs/demo_v0.2.yaml")
    dynamic_run.add_argument("--output")
    dynamic_run.add_argument("--device", help="override manifest device")
    dynamic_run.set_defaults(func=command_dynamic_run)

    generate = sub.add_parser("generate", help="generate a synthetic irregular-time UPR dataset")
    generate.add_argument("--output", default="data/synthetic_upr_v0.2.npz")
    generate.add_argument("--csv")
    generate.add_argument("--n-trajectories", type=int, default=512)
    generate.add_argument("--min-steps", type=int, default=8)
    generate.add_argument("--max-steps", type=int, default=16)
    generate.add_argument("--missing-feature-rate", type=float, default=0.08)
    generate.add_argument("--seed", type=int, default=7)
    generate.set_defaults(func=command_generate)

    import_csv = sub.add_parser("import-csv", help="convert a long-format CSV dataset to NPZ")
    import_csv.add_argument("--input", required=True)
    import_csv.add_argument("--output", required=True)
    import_csv.set_defaults(func=command_import_csv)

    export_csv = sub.add_parser("export-csv", help="export NPZ data to long-format CSV")
    export_csv.add_argument("--input", required=True)
    export_csv.add_argument("--output", required=True)
    export_csv.set_defaults(func=command_export_csv)

    mm_validate = sub.add_parser("multimodal-validate", help="validate an aligned CausaFlux H5MU file")
    mm_validate.add_argument("--input", required=True)
    mm_validate.set_defaults(func=command_multimodal_validate)

    mm_export = sub.add_parser("multimodal-export", help="export H5MU modalities as an aligned CSV bundle")
    mm_export.add_argument("--input", required=True)
    mm_export.add_argument("--output", required=True)
    mm_export.set_defaults(func=command_multimodal_export)

    mm_import = sub.add_parser("multimodal-import", help="build an H5MU file from an aligned CSV bundle")
    mm_import.add_argument("--input", required=True)
    mm_import.add_argument("--output", required=True)
    mm_import.set_defaults(func=command_multimodal_import)

    spatial_build = sub.add_parser("spatial-build", help="build a typed spatial heterograph from a causal observation CSV")
    spatial_build.add_argument("--input", required=True)
    spatial_build.add_argument("--output", required=True)
    spatial_build.add_argument("--seed", type=int, default=31)
    spatial_build.add_argument("--k-neighbors", type=int, default=8)
    spatial_build.add_argument("--max-distance", type=float, default=230.0)
    spatial_build.add_argument("--neighborhood-radius", type=float, default=180.0)
    spatial_build.add_argument("--communication-radius", type=float, default=190.0)
    spatial_build.add_argument("--bootstrap", type=int, default=50)
    spatial_build.add_argument("--representative-sample")
    spatial_build.add_argument("--no-graphml", action="store_true")
    spatial_build.set_defaults(func=command_spatial_build)

    spatial_validate = sub.add_parser("spatial-validate", help="validate exported CausaFlux spatial graph tables")
    spatial_validate.add_argument("--input", required=True)
    spatial_validate.set_defaults(func=command_spatial_validate)

    biomarkers_rank = sub.add_parser(
        "biomarkers-rank",
        help="rank early-warning and causal-proximity biomarkers and panels",
    )
    biomarkers_rank.add_argument("--input", required=True, help="causal longitudinal observation CSV")
    biomarkers_rank.add_argument("--output", required=True)
    biomarkers_rank.add_argument("--config", default="configs/cancer_closed_loop_v1.7.0.yaml")
    biomarkers_rank.add_argument("--features", nargs="+")
    biomarkers_rank.add_argument("--outcome", default="future_resistant")
    biomarkers_rank.add_argument("--cell-type", default="tumor")
    biomarkers_rank.add_argument("--target-node", default="stable_resistance")
    biomarkers_rank.add_argument("--warning-auc-threshold", type=float, default=0.65)
    biomarkers_rank.add_argument("--warning-stability-threshold", type=float, default=0.60)
    biomarkers_rank.add_argument("--bootstrap", type=int, default=80)
    biomarkers_rank.add_argument("--top-panel-size", type=int, default=3)
    biomarkers_rank.add_argument("--top-n", type=int, default=12)
    biomarkers_rank.add_argument("--seed", type=int, default=31)
    biomarkers_rank.set_defaults(func=command_biomarkers_rank)

    biomarkers_validate = sub.add_parser(
        "biomarkers-validate", help="validate exported causal biomarker outputs"
    )
    biomarkers_validate.add_argument("--input", required=True)
    biomarkers_validate.set_defaults(func=command_biomarkers_validate)

    experiments_rank = sub.add_parser(
        "experiments-rank",
        help="rank CRISPR, drug, imaging, and sampling-time experiments",
    )
    experiments_rank.add_argument("--input", required=True, help="CausaFlux analysis output directory")
    experiments_rank.add_argument("--output", required=True)
    experiments_rank.add_argument("--config", default="configs/cancer_closed_loop_v1.7.0.yaml")
    experiments_rank.add_argument("--top-n", type=int, default=12)
    experiments_rank.add_argument("--seed", type=int, default=31)
    experiments_rank.set_defaults(func=command_experiments_rank)

    experiments_validate = sub.add_parser(
        "experiments-validate", help="validate exported closed-loop experiment recommendations"
    )
    experiments_validate.add_argument("--input", required=True)
    experiments_validate.set_defaults(func=command_experiments_validate)

    experiments_update = sub.add_parser(
        "experiments-update", help="update mechanism probabilities from completed experiment outcomes"
    )
    experiments_update.add_argument("--input", required=True, help="CausaFlux output directory or active_learning directory")
    experiments_update.add_argument("--observations", required=True, help="CSV with experiment_id and observed_standardized_readout")
    experiments_update.add_argument("--output", required=True)
    experiments_update.add_argument("--config", default="configs/cancer_closed_loop_v1.7.0.yaml")
    experiments_update.add_argument("--top-n", type=int, default=12)
    experiments_update.add_argument("--seed", type=int, default=31)
    experiments_update.set_defaults(func=command_experiments_update)

    therapeutics_rank = sub.add_parser(
        "therapeutics-rank",
        help="rank gene, drug, combination, sequence, and timing counterfactuals",
    )
    therapeutics_rank.add_argument("--input", required=True, help="causal observation CSV")
    therapeutics_rank.add_argument("--output", required=True)
    therapeutics_rank.add_argument("--comparator", default="standard_therapy")
    therapeutics_rank.add_argument("--horizon", type=float, default=168.0)
    therapeutics_rank.add_argument("--timing-grid", type=float, nargs="+", default=[0, 24, 48, 72, 120])
    therapeutics_rank.add_argument("--default-start", type=float, default=12.0)
    therapeutics_rank.add_argument("--sequence-delay", type=float, default=12.0)
    therapeutics_rank.add_argument("--bootstrap", type=int, default=30)
    therapeutics_rank.add_argument("--max-reference-rows-per-donor", type=int, default=30)
    therapeutics_rank.add_argument("--top-n", type=int, default=12)
    therapeutics_rank.add_argument("--seed", type=int, default=31)
    therapeutics_rank.set_defaults(func=command_therapeutics_rank)

    therapeutics_predict = sub.add_parser(
        "therapeutics-predict",
        help="predict one custom ordered intervention regimen",
    )
    therapeutics_predict.add_argument("--input", required=True, help="causal observation CSV")
    therapeutics_predict.add_argument("--output", required=True)
    therapeutics_predict.add_argument("--interventions", nargs="+", required=True)
    therapeutics_predict.add_argument("--start-hours", type=float, nargs="+")
    therapeutics_predict.add_argument("--doses", type=float, nargs="+")
    therapeutics_predict.add_argument("--comparator", default="standard_therapy")
    therapeutics_predict.add_argument("--horizon", type=float, default=168.0)
    therapeutics_predict.add_argument("--seed", type=int, default=31)
    therapeutics_predict.set_defaults(func=command_therapeutics_predict)

    therapeutics_validate = sub.add_parser(
        "therapeutics-validate", help="validate exported therapeutic predictions"
    )
    therapeutics_validate.add_argument("--input", required=True)
    therapeutics_validate.set_defaults(func=command_therapeutics_validate)


    neuro_run = sub.add_parser(
        "neuro-run", help="run neural–glial trajectory, imaging and electrophysiology integration"
    )
    neuro_run.add_argument("--config", default="configs/neurobiology_v1.7.0.yaml")
    neuro_run.add_argument("--output", default="causaflux_v1.7.0_neuro_output")
    neuro_run.add_argument("--n-donors", type=int, default=8)
    neuro_run.add_argument("--cells-per-type", type=int, default=16)
    neuro_run.add_argument("--bootstrap", type=int, default=50)
    neuro_run.add_argument("--seed", type=int, default=47)
    neuro_run.set_defaults(func=command_neuro_run)

    neuro_validate = sub.add_parser(
        "neuro-validate", help="validate exported neural–glial configuration outputs"
    )
    neuro_validate.add_argument("--input", required=True)
    neuro_validate.set_defaults(func=command_neuro_validate)


    prospective_run = sub.add_parser(
        "prospective-run", help="run the v1.8 three-cycle prospectively locked experimental-loop benchmark"
    )
    prospective_run.add_argument("--output", default="causaflux_v1.8.0_prospective_loop")
    prospective_run.add_argument("--config", default="configs/prospective_loop_v1.8.0.yaml")
    prospective_run.add_argument("--seed", type=int, default=180)
    prospective_run.add_argument("--truth-hypothesis", default="H1_PROTEOSTASIS_UPSTREAM")
    prospective_run.add_argument("--experiments-per-cycle", type=int, default=2)
    prospective_run.add_argument("--cycle-budget", type=float, default=1.30)
    prospective_run.add_argument("--synthetic-failure-experiment", default="IMG_MITO_24H")
    prospective_run.add_argument("--no-synthetic-failure", action="store_true")
    prospective_run.add_argument("--require-gate", action="store_true")
    prospective_run.set_defaults(func=command_prospective_run)

    prospective_validate = sub.add_parser(
        "prospective-validate", help="validate v1.8 prospective locks, QC, costs, calibration and exit gate"
    )
    prospective_validate.add_argument("--input", required=True)
    prospective_validate.add_argument("--require-gate", action="store_true")
    prospective_validate.set_defaults(func=command_prospective_validate)

    prospective_contract = sub.add_parser(
        "prospective-contract-export", help="export LIMS/ELN, QC and outcome contracts for a real prospective study"
    )
    prospective_contract.add_argument("--output", required=True)
    prospective_contract.set_defaults(func=command_prospective_contract_export)

    prospective_ingest = sub.add_parser(
        "prospective-ingest", help="ingest real QC and outcomes against an already locked prospective cycle"
    )
    prospective_ingest.add_argument("--cycle-dir", required=True)
    prospective_ingest.add_argument("--qc", required=True)
    prospective_ingest.add_argument("--outcomes", required=True)
    prospective_ingest.set_defaults(func=command_prospective_ingest)

    virtual_run = sub.add_parser("virtual-cell-run", help="run the v1.9 integrated AI-guided virtual-cell release workflow")
    virtual_run.add_argument("--output", default="causaflux_v1.9.0_virtual_cell")
    virtual_run.add_argument("--project-root")
    virtual_run.add_argument("--require-real-prospective", action="store_true", help="fail unless real three-cycle prospective validation is authorized")
    virtual_run.set_defaults(func=command_virtual_cell_run)

    virtual_validate = sub.add_parser("virtual-cell-validate", help="validate v1.9 AI, real-world, figures, reports and prospective gates")
    virtual_validate.add_argument("--input", required=True)
    virtual_validate.add_argument("--require-real-prospective", action="store_true")
    virtual_validate.set_defaults(func=command_virtual_cell_validate)

    real_register = sub.add_parser("realworld-register", help="register a user real-world dataset with a SHA-256 provenance contract")
    real_register.add_argument("--input", required=True)
    real_register.add_argument("--dataset-id", required=True)
    real_register.add_argument("--output", default="real_world_user_data")
    real_register.add_argument("--data-class", default="experimental")
    real_register.add_argument("--modalities", default="rna")
    real_register.add_argument("--longitudinal", action="store_true")
    real_register.add_argument("--perturbational", action="store_true")
    real_register.add_argument("--spatial", action="store_true")
    real_register.add_argument("--prospective", action="store_true")
    real_register.add_argument("--outcome-available", action="store_true")
    real_register.add_argument("--donor-column")
    real_register.add_argument("--time-column")
    real_register.add_argument("--intervention-column")
    real_register.add_argument("--outcome-column")
    real_register.add_argument("--notes")
    real_register.add_argument("--preview", action="store_true")
    real_register.set_defaults(func=command_realworld_register)

    v2_run = sub.add_parser("v2-run", help="build the CausaFlux v2.0 release-evidence bundle and enforce the real prospective-validation gate")
    v2_run.add_argument("--output", default="causaflux_v2.0.0_release")
    v2_run.add_argument("--project-root")
    v2_run.add_argument("--external-evidence-dir", help="directory containing an external evidence_ledger.csv or evidence CSVs")
    v2_run.add_argument("--require-prospectively-validated", action="store_true", help="fail unless every real v2 evidence criterion passes")
    v2_run.set_defaults(func=command_v2_run)

    v2_validate = sub.add_parser("v2-validate", help="validate the v2 software bundle and optionally require the real prospective-validation claim")
    v2_validate.add_argument("--input", required=True)
    v2_validate.add_argument("--require-prospectively-validated", action="store_true")
    v2_validate.set_defaults(func=command_v2_validate)

    longitudinal_contract = sub.add_parser("longitudinal-contract", help="export public longitudinal-perturbation registry and CausaFlux table contract")
    longitudinal_contract.add_argument("--output", default="real_longitudinal_contract")
    longitudinal_contract.set_defaults(func=command_longitudinal_contract)

    longitudinal_convert = sub.add_parser("longitudinal-convert", help="convert a real longitudinal perturbation table to the CausaFlux dynamic benchmark NPZ")
    longitudinal_convert.add_argument("--input", required=True)
    longitudinal_convert.add_argument("--output", required=True)
    longitudinal_convert.add_argument("--manifest")
    longitudinal_convert.set_defaults(func=command_longitudinal_convert)

    longitudinal_benchmark = sub.add_parser("longitudinal-benchmark", help="train/evaluate CausaFlux on an actual longitudinal perturbation table")
    longitudinal_benchmark.add_argument("--input", required=True)
    longitudinal_benchmark.add_argument("--output", default="real_longitudinal_benchmark")
    longitudinal_benchmark.add_argument("--models", default="LatestStateMLP,HistorySummaryMLP,CausaFluxFactorizedGRU")
    longitudinal_benchmark.add_argument("--seed", type=int, default=200)
    longitudinal_benchmark.set_defaults(func=command_longitudinal_benchmark)

    shift_cal = sub.add_parser("shift-calibration", help="evaluate uncertainty calibration under a prespecified distribution shift")
    shift_cal.add_argument("--input", required=True)
    shift_cal.add_argument("--output", default="distribution_shift_calibration")
    shift_cal.add_argument("--require-gate", action="store_true")
    shift_cal.set_defaults(func=command_shift_calibration)

    ui = sub.add_parser("ui", help="launch the browser user interface")
    ui.add_argument("--output", default="causaflux_v2.0.0_release")
    ui.add_argument("--port", type=int)
    ui.set_defaults(func=command_ui)

    train = sub.add_parser("train", help="train CausaFlux")
    train.add_argument("--data", required=True)
    train.add_argument("--output", default="runs/upr_v0.2")
    train.add_argument("--config")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--device", help="auto, cpu, mps, or cuda")
    train.add_argument("--seed", type=int)
    train.add_argument("--split-mode", choices=["group", "random"])
    train.set_defaults(func=command_train)

    evaluate = sub.add_parser("evaluate", help="evaluate a saved model")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--output", default="evaluation")
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(func=command_evaluate)

    simulate = sub.add_parser("simulate", help="run a counterfactual intervention forecast")
    simulate.add_argument("--checkpoint", required=True)
    simulate.add_argument("--output", default="simulation")
    simulate.add_argument("--device", default="auto")
    simulate.add_argument("--final-time", type=float, default=10.0)
    simulate.add_argument("--steps", type=int, default=18)
    simulate.add_argument(
        "--scenario",
        choices=[
            "continuous_stress",
            "stress_recovery",
            "pulsatile_stress",
            "ire1_inhibition",
            "atf6_support",
        ],
        default="stress_recovery",
    )
    simulate.add_argument("--schedule-csv")
    simulate.add_argument("--initial-state-csv")
    simulate.add_argument("--mc-samples", type=int, default=30)
    simulate.add_argument("--seed", type=int, default=7)
    simulate.set_defaults(func=command_simulate)

    demo = sub.add_parser("demo", help="run the bundled v1.0 integrated demonstration")
    demo.add_argument("--output", default="causaflux_v1.7.0_output")
    demo.set_defaults(func=command_demo)

    dynamic_demo = sub.add_parser("dynamic-demo", help="run the bundled v0.2 dynamic UPR demo")
    dynamic_demo.add_argument("--output", default="causaflux_v0.2_output")
    dynamic_demo.add_argument("--device")
    dynamic_demo.set_defaults(func=command_dynamic_demo)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
