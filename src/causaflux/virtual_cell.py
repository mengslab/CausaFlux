"""CausaFlux v1.9.0 integrated AI-guided virtual-cell layer.

The v1.9 layer does not replace the validated estimators from earlier releases.
It combines their locked validation evidence into a reliability-weighted model
router, produces an interpretable virtual-cell trajectory, ranks interventions,
and propagates module disagreement plus prospective calibration into uncertainty.

The bundled reference is a software validation fixture. A biological claim of a
"prospectively validated virtual cell" requires a real, locked three-cycle study.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd

VIRTUAL_CELL_VERSION = "1.9.0"
STATE_NAMES = (
    "proteostasis_capacity",
    "mitochondrial_reserve",
    "inflammatory_dysfunction",
    "commitment_risk",
    "recovery_potential",
)


@dataclass(frozen=True)
class ModuleEvidence:
    module: str
    selected_model: str
    gate: str
    primary_metric: str
    primary_value: float
    reliability: float
    evidence_class: str
    prospective_relevance: str
    source: str


@dataclass(frozen=True)
class InterventionScenario:
    scenario_id: str
    label: str
    stress: float
    ire1_support: float = 0.0
    perk_relief: float = 0.0
    atf6_support: float = 0.0
    mitochondrial_support: float = 0.0
    anti_inflammatory: float = 0.0
    delayed_start_fraction: float = 0.0
    pulse_fraction: float = 1.0
    cost: float = 1.0


DEFAULT_SCENARIOS = (
    InterventionScenario("stress_only", "Stress only", stress=1.0, cost=0.25),
    InterventionScenario("ire1_timed", "Timed IRE1/XBP1 support", stress=1.0, ire1_support=0.75, delayed_start_fraction=0.18, cost=0.65),
    InterventionScenario("perk_relief", "PERK/ATF4 relief", stress=1.0, perk_relief=0.62, delayed_start_fraction=0.12, cost=0.58),
    InterventionScenario("atf6_support", "ATF6 adaptive support", stress=1.0, atf6_support=0.68, cost=0.54),
    InterventionScenario("mito_support", "Mitochondrial reserve support", stress=1.0, mitochondrial_support=0.74, cost=0.60),
    InterventionScenario("combo_recovery", "Combination recovery program", stress=1.0, ire1_support=0.50, atf6_support=0.40, mitochondrial_support=0.52, anti_inflammatory=0.35, delayed_start_fraction=0.10, cost=0.98),
    InterventionScenario("pulsed_stress", "Pulsed stress with recovery intervals", stress=0.90, pulse_fraction=0.52, cost=0.35),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _clip_reliability(value: float) -> float:
    return float(np.clip(value, 0.35, 0.98))


def load_module_evidence(project_root: str | Path) -> pd.DataFrame:
    """Summarize retained module evidence into normalized reliability weights."""
    root = Path(project_root)
    rows: list[ModuleEvidence] = []

    dyn = pd.read_csv(root / "dynamic_benchmark_reference" / "model_comparison.csv")
    best = dyn.sort_values(["calibrated_gaussian_nll", "trajectory_rmse"]).iloc[0]
    reliability = _clip_reliability(0.55 * (1.0 / (1.0 + float(best.trajectory_rmse))) + 0.45 * (1.0 / (1.0 + float(best.calibrated_gaussian_nll))))
    rows.append(ModuleEvidence("dynamic_state", str(best.model), "PASS", "trajectory_rmse", float(best.trajectory_rmse), reliability, "synthetic_prospective_proxy", "future-state dynamics", "dynamic_benchmark_reference/model_comparison.csv"))

    mm = pd.read_csv(root / "multimodal_dynamic_reference" / "model_comparison.csv")
    best = mm.sort_values(["test_log_loss", "destructive_score_rmse"]).iloc[0]
    reliability = _clip_reliability(0.58 * float(best.test_auc) + 0.42 * (1.0 / (1.0 + float(best.test_log_loss))))
    rows.append(ModuleEvidence("multimodal_fusion", str(best.model), "PASS", "test_auc", float(best.test_auc), reliability, "synthetic_multimodal", "early multimodal state inference", "multimodal_dynamic_reference/model_comparison.csv"))

    ig = pd.read_csv(root / "intervention_generalization_reference" / "model_comparison.csv")
    overall = ig[ig.holdout_type == "overall"].sort_values("rmse").iloc[0]
    reliability = _clip_reliability(1.0 / (1.0 + float(overall.rmse)))
    rows.append(ModuleEvidence("intervention_generalization", str(overall.model), "PASS", "overall_rmse", float(overall.rmse), reliability, "synthetic_intervention_holdout", "counterfactual intervention ranking", "intervention_generalization_reference/model_comparison.csv"))

    tissue = pd.read_csv(root / "spatiotemporal_tissue_reference" / "model_comparison.csv")
    cft = tissue[tissue.model == "CausaFluxSpatiotemporalGNN"]
    tissue_rmse = float(cft.state_rmse.mean())
    reliability = _clip_reliability(1.0 / (1.0 + tissue_rmse))
    rows.append(ModuleEvidence("spatiotemporal_context", "CausaFluxSpatiotemporalGNN", "PASS", "mean_state_rmse", tissue_rmse, reliability, "synthetic_tissue_holdout", "neighborhood-conditioned trajectory", "spatiotemporal_tissue_reference/model_comparison.csv"))

    foundation = pd.read_csv(root / "foundation_pretraining_reference" / "foundation_evaluation_matrix.csv")
    subset = foundation[(foundation.representation == "CausaFluxFoundation") & (foundation.evaluation == "linear_probe") & foundation.split.isin(["donor_holdout", "tissue_holdout", "perturbation_holdout"])]
    fr = float(subset.future_state_rmse.mean())
    reliability = _clip_reliability(1.0 / (1.0 + fr))
    rows.append(ModuleEvidence("foundation_representation", "CausaFluxFoundation", "PASS", "holdout_future_rmse", fr, reliability, "synthetic_pretraining_transfer", "foundation representation and transfer", "foundation_pretraining_reference/foundation_evaluation_matrix.csv"))

    prospective = pd.read_csv(root / "prospective_loop_reference" / "cycle_calibration.csv")
    evaluable = prospective[prospective.n_evaluable > 0]
    rmse = float(evaluable.prediction_rmse.mean())
    coverage_error = float(np.mean(np.abs(evaluable.interval_coverage_90 - 0.90)))
    reliability = _clip_reliability(0.60 * (1.0 / (1.0 + rmse)) + 0.40 * (1.0 - min(coverage_error, 1.0)))
    rows.append(ModuleEvidence("prospective_calibration", "LockedCycleCalibrator", "PASS", "mean_prediction_rmse", rmse, reliability, "synthetic_locked_three_cycle", "prospective calibration and uncertainty", "prospective_loop_reference/cycle_calibration.csv"))

    frame = pd.DataFrame([asdict(row) for row in rows])
    frame["normalized_weight"] = frame.reliability / frame.reliability.sum()
    return frame


def _pulse(time_fraction: float, pulse_fraction: float) -> float:
    if pulse_fraction >= 0.999:
        return 1.0
    # Four stress pulses across the trajectory.
    phase = (time_fraction * 4.0) % 1.0
    return 1.0 if phase <= pulse_fraction else 0.0


def _component_step(state: np.ndarray, scenario: InterventionScenario, tfrac: float, module: str) -> np.ndarray:
    """One transparent module-conditioned state derivative."""
    p, m, inflam, commit, recovery = state
    active = 1.0 if tfrac >= scenario.delayed_start_fraction else 0.0
    stress = scenario.stress * _pulse(tfrac, scenario.pulse_fraction)
    ire1 = active * scenario.ire1_support
    perk = active * scenario.perk_relief
    atf6 = active * scenario.atf6_support
    mito = active * scenario.mitochondrial_support
    anti = active * scenario.anti_inflammatory

    # Common mechanistic skeleton. Individual validated modules emphasize different terms.
    dp = 0.32 * ire1 + 0.28 * atf6 + 0.08 * recovery - 0.43 * stress - 0.10 * inflam
    dm = 0.34 * mito + 0.08 * recovery - 0.27 * stress - 0.12 * inflam
    di = 0.25 * stress + 0.17 * commit - 0.21 * ire1 - 0.18 * perk - 0.24 * anti - 0.08 * p
    dc = 0.17 * stress + 0.18 * inflam - 0.17 * p - 0.12 * m - 0.12 * perk
    dr = 0.12 * p + 0.08 * m - 0.38 * inflam - 0.44 * commit - 0.18 * stress + 0.13 * atf6 + 0.12 * anti + 0.12 * ire1 + 0.10 * mito

    if module == "dynamic_state":
        dp *= 1.08; dc *= 1.10; dr *= 1.10
    elif module == "multimodal_fusion":
        dm *= 1.12; di *= 1.09
    elif module == "intervention_generalization":
        dp += 0.08 * ire1; dm += 0.08 * mito; di -= 0.06 * (perk + anti)
    elif module == "spatiotemporal_context":
        di += 0.04 * stress * (1.0 - anti); dr -= 0.025 * inflam
    elif module == "foundation_representation":
        dp *= 0.98; dm *= 1.02; dr *= 1.04
    elif module == "prospective_calibration":
        dc *= 0.96; dr *= 0.98

    # Soft homeostatic return prevents unbounded states.
    home = np.array([0.72, 0.76, 0.16, 0.12, 0.76])
    derivative = np.array([dp, dm, di, dc, dr]) + 0.05 * (home - state)
    return derivative


def simulate_scenario(
    scenario: InterventionScenario,
    module_evidence: pd.DataFrame,
    *,
    final_time_hours: float = 72.0,
    steps: int = 37,
    initial_state: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ensemble trajectory and component trajectories for one scenario."""
    if initial_state is None:
        initial_state = np.array([0.78, 0.80, 0.12, 0.10, 0.82], dtype=float)
    times = np.linspace(0.0, final_time_hours, steps)
    dt = 1.0 / max(steps - 1, 1)
    components: dict[str, np.ndarray] = {}
    weights = dict(zip(module_evidence.module, module_evidence.normalized_weight))

    for module in module_evidence.module:
        traj = np.zeros((steps, len(STATE_NAMES)), dtype=float)
        traj[0] = initial_state
        for idx in range(1, steps):
            tfrac = float(times[idx - 1] / final_time_hours)
            derivative = _component_step(traj[idx - 1], scenario, tfrac, str(module))
            traj[idx] = np.clip(traj[idx - 1] + 2.6 * dt * derivative, 0.0, 1.0)
        components[str(module)] = traj

    stack = np.stack([components[str(module)] for module in module_evidence.module], axis=0)
    w = np.asarray([weights[str(module)] for module in module_evidence.module], dtype=float)[:, None, None]
    mean = np.sum(w * stack, axis=0)
    disagreement = np.sqrt(np.sum(w * (stack - mean[None, :, :]) ** 2, axis=0))

    # Prospective calibration inflates disagreement so intervals are not falsely narrow.
    calibration = module_evidence[module_evidence.module == "prospective_calibration"].iloc[0]
    inflation = 1.20 + min(float(calibration.primary_value), 1.0) * 0.35
    sd = np.clip(0.045 + inflation * disagreement, 0.04, 0.30)
    lower = np.clip(mean - 1.645 * sd, 0.0, 1.0)
    upper = np.clip(mean + 1.645 * sd, 0.0, 1.0)

    out = pd.DataFrame({"time_hours": times, "scenario_id": scenario.scenario_id, "scenario_label": scenario.label})
    for j, name in enumerate(STATE_NAMES):
        out[f"{name}_mean"] = mean[:, j]
        out[f"{name}_sd"] = sd[:, j]
        out[f"{name}_p05"] = lower[:, j]
        out[f"{name}_p95"] = upper[:, j]

    component_rows = []
    for module, traj in components.items():
        for idx, time in enumerate(times):
            row = {"module": module, "time_hours": time, "scenario_id": scenario.scenario_id}
            row.update({name: float(traj[idx, j]) for j, name in enumerate(STATE_NAMES)})
            component_rows.append(row)
    return out, pd.DataFrame(component_rows)


def _recommendation_row(scenario: InterventionScenario, trajectory: pd.DataFrame) -> dict[str, Any]:
    final = trajectory.iloc[-1]
    recovery = float(final["recovery_potential_mean"])
    inflam = float(final["inflammatory_dysfunction_mean"])
    commitment = float(final["commitment_risk_mean"])
    proteostasis = float(final["proteostasis_capacity_mean"])
    mito = float(final["mitochondrial_reserve_mean"])
    uncertainty = float(np.mean([final[f"{name}_sd"] for name in STATE_NAMES]))
    biological_utility = 0.34 * recovery + 0.18 * proteostasis + 0.14 * mito - 0.18 * inflam - 0.16 * commitment
    calibrated_utility = biological_utility - 0.22 * uncertainty
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_label": scenario.label,
        "cost": scenario.cost,
        "final_recovery_potential": recovery,
        "final_proteostasis_capacity": proteostasis,
        "final_mitochondrial_reserve": mito,
        "final_inflammatory_dysfunction": inflam,
        "final_commitment_risk": commitment,
        "mean_predictive_uncertainty": uncertainty,
        "biological_utility": biological_utility,
        "calibrated_utility": calibrated_utility,
        "utility_per_cost": calibrated_utility / max(scenario.cost, 1e-9),
    }


def run_virtual_cell_ensemble(
    project_root: str | Path,
    output_dir: str | Path,
    *,
    scenarios: tuple[InterventionScenario, ...] = DEFAULT_SCENARIOS,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    module_evidence = load_module_evidence(project_root)
    module_path = out / "ai_model_router.csv"
    module_evidence.to_csv(module_path, index=False)

    trajectories: list[pd.DataFrame] = []
    components: list[pd.DataFrame] = []
    recommendations: list[dict[str, Any]] = []
    for scenario in scenarios:
        trajectory, component = simulate_scenario(scenario, module_evidence)
        trajectories.append(trajectory)
        components.append(component)
        recommendations.append(_recommendation_row(scenario, trajectory))

    traj = pd.concat(trajectories, ignore_index=True)
    comp = pd.concat(components, ignore_index=True)
    reco = pd.DataFrame(recommendations).sort_values(["calibrated_utility", "utility_per_cost"], ascending=False).reset_index(drop=True)
    reco.insert(0, "rank", np.arange(1, len(reco) + 1))
    traj_path = out / "virtual_cell_trajectories.csv"
    comp_path = out / "module_component_trajectories.csv"
    reco_path = out / "ai_guided_intervention_ranking.csv"
    traj.to_csv(traj_path, index=False)
    comp.to_csv(comp_path, index=False)
    reco.to_csv(reco_path, index=False)

    top = reco.iloc[0]
    card = {
        "framework": "CausaFlux",
        "version": VIRTUAL_CELL_VERSION,
        "virtual_cell_id": "CFVC-REFERENCE-001",
        "modeling_mode": "reliability_weighted_validated_module_ensemble",
        "state_variables": list(STATE_NAMES),
        "selected_intervention": str(top.scenario_id),
        "selected_intervention_label": str(top.scenario_label),
        "selected_intervention_calibrated_utility": float(top.calibrated_utility),
        "selected_intervention_uncertainty": float(top.mean_predictive_uncertainty),
        "evidence_boundary": "Bundled trajectory is a software-reference virtual cell. It is not a biological prediction for a patient, animal, or cell line.",
        "prospective_claim_authorized": False,
    }
    card_path = out / "virtual_cell_card.json"
    card_path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return {"model_router": module_path, "trajectories": traj_path, "components": comp_path, "ranking": reco_path, "card": card_path}
