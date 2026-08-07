"""Prospective experimental loop for CausaFlux v1.8.0.

This module turns model-guided experimental design into an auditable prospective
workflow. Predictions are frozen before outcomes are generated or ingested,
assay QC is explicit, failed experiments still count against cost, Bayesian
posterior updates are versioned, and a prespecified non-AI strategy is evaluated
under the same synthetic outcome oracle.

The bundled reference run is a deterministic software-validation fixture. It is
not biological evidence and it does not claim that CausaFlux has prospectively
improved real experiments until external locked outcomes are ingested.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir, json_dump


PROSPECTIVE_VERSION = "1.8.0"
CONTRACT_VERSION = "causaflux-experiment-contract-1.0"
HYPOTHESIS_IDS = (
    "H1_PROTEOSTASIS_UPSTREAM",
    "H2_MITOCHONDRIAL_PARALLEL",
    "H3_ANTIGEN_EXCLUSION",
    "H4_ENHANCER_COMMITMENT",
)


@dataclass(frozen=True)
class ProspectiveLoopConfig:
    seed: int = 180
    truth_hypothesis: str = "H1_PROTEOSTASIS_UPSTREAM"
    max_cycles: int = 3
    min_cycles: int = 3
    experiments_per_cycle: int = 2
    cycle_budget: float = 1.30
    posterior_stop_threshold: float = 0.92
    min_expected_information_gain: float = 0.015
    discovery_threshold: float = 0.70
    recovery_threshold: float = -0.55
    interval_z: float = 1.6448536269514722  # central 90% interval
    failure_cost_fraction: float = 1.0
    baseline_strategy: str = "prespecified_non_ai_fixed_order"
    require_independent_cycle3: bool = True
    synthetic_failure_experiment: str | None = "IMG_MITO_24H"


@dataclass
class ProspectiveLoopResult:
    hypotheses: pd.DataFrame
    catalog: pd.DataFrame
    predictions: pd.DataFrame
    outcomes: pd.DataFrame
    qc: pd.DataFrame
    posterior_history: pd.DataFrame
    cost_ledger: pd.DataFrame
    calibration: pd.DataFrame
    baseline_results: pd.DataFrame
    comparison: pd.DataFrame
    gate: dict[str, Any]
    run_manifest: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n")
    return sha256_text(payload)


def hash_tree(root: str | Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if any(part in {"__pycache__", ".git", ".causaflux_env"} for part in path.parts):
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _entropy(probabilities: Sequence[float]) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if p.size else 0.0


def _normal_logpdf(value: float, means: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-8)
    return -0.5 * ((value - means) / sigma) ** 2 - math.log(sigma * math.sqrt(2.0 * math.pi))


def posterior_update(prior: Sequence[float], observation: float, means: Sequence[float], sigma: float) -> np.ndarray:
    prior = np.asarray(prior, dtype=float)
    prior = np.clip(prior, 1e-12, None)
    prior /= prior.sum()
    means = np.asarray(means, dtype=float)
    logw = np.log(prior) + _normal_logpdf(float(observation), means, sigma)
    logw -= float(np.max(logw))
    posterior = np.exp(logw)
    posterior /= posterior.sum()
    return posterior


def expected_information_gain(prior: Sequence[float], means: Sequence[float], sigma: float, seed: int, n: int = 2400) -> float:
    """Monte-Carlo mutual information I(H;Y) in nats."""
    prior = np.asarray(prior, dtype=float)
    prior = np.clip(prior, 1e-12, None)
    prior /= prior.sum()
    means = np.asarray(means, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(prior), size=max(300, int(n)), p=prior)
    y = rng.normal(means[idx], max(float(sigma), 1e-8))
    loglik = -0.5 * ((y[:, None] - means[None, :]) / sigma) ** 2 - math.log(
        sigma * math.sqrt(2.0 * math.pi)
    )
    logw = loglik + np.log(prior)[None, :]
    logw -= logw.max(axis=1, keepdims=True)
    post = np.exp(logw)
    post /= post.sum(axis=1, keepdims=True)
    ent = -np.sum(np.where(post > 0, post * np.log(post), 0.0), axis=1)
    return max(0.0, _entropy(prior) - float(np.mean(ent)))


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join([str(seed), *map(str, parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")


def default_hypotheses() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [HYPOTHESIS_IDS[0], "IRE1-XBP1 proteostasis is upstream of persistent tolerance", "IRE1-XBP1 proteostasis", 0.30],
            [HYPOTHESIS_IDS[1], "Mitochondrial reserve is a parallel driver of persistent tolerance", "Mitochondrial reserve", 0.25],
            [HYPOTHESIS_IDS[2], "Antigen-presentation failure drives immune-protected resistance", "Antigen presentation", 0.25],
            [HYPOTHESIS_IDS[3], "Enhancer plasticity controls commitment to stable resistance", "Enhancer plasticity", 0.20],
        ],
        columns=["hypothesis_id", "hypothesis", "mechanism", "prior_probability"],
    )


def default_experiment_catalog() -> pd.DataFrame:
    """Prespecified experiment universe for the synthetic prospective fixture.

    expected_readout columns are model predictions under each competing mechanism.
    The non-AI baseline order is frozen before any synthetic outcomes are sampled.
    """
    h1, h2, h3, h4 = HYPOTHESIS_IDS
    rows = [
        ("DRUG_IRE1I_TIMECOURSE", "IRE1 inhibitor dose-time course", "drug", "IRE1-XBP1 proteostasis", "IRE1A", "pharmacologic", 72, 0.42, 0.30, 0.08, 1, "A", "functional", True, -1.05, -0.28, -0.10, -0.36),
        ("CRISPR_XBP1_24H", "CRISPRi XBP1 at 24 h", "crispr", "IRE1-XBP1 proteostasis", "XBP1", "genetic", 72, 0.62, 0.32, 0.10, 5, "A", "genetic", True, -1.18, -0.32, -0.12, -0.43),
        ("IMG_ER_RECOVERY_18H", "Live ER recovery imaging at 18 h", "imaging", "IRE1-XBP1 proteostasis", "ER morphology", "imaging", 18, 0.25, 0.27, 0.06, 7, "A", "imaging", True, -0.90, -0.25, -0.08, -0.28),
        ("PROTEOMICS_XBP1_TARGETS", "Orthogonal XBP1-target proteomics", "sampling_time", "IRE1-XBP1 proteostasis", "XBP1 target proteins", "proteomics", 48, 0.46, 0.28, 0.07, 10, "A", "proteomics", True, -0.98, -0.24, -0.09, -0.31),
        ("DRUG_MITOI_TIMECOURSE", "Mitochondrial reserve inhibitor time course", "drug", "Mitochondrial reserve", "OXPHOS", "pharmacologic", 72, 0.44, 0.31, 0.09, 2, "B", "functional", True, -0.25, -1.06, -0.09, -0.23),
        ("CRISPR_NDUFS1_24H", "CRISPRi NDUFS1 at 24 h", "crispr", "Mitochondrial reserve", "NDUFS1", "genetic", 72, 0.64, 0.34, 0.11, 6, "B", "genetic", True, -0.24, -1.15, -0.08, -0.27),
        ("IMG_MITO_24H", "Mitochondrial membrane-potential imaging", "imaging", "Mitochondrial reserve", "mitochondria", "imaging", 24, 0.24, 0.29, 0.18, 8, "B", "imaging", True, -0.18, -0.91, -0.06, -0.18),
        ("CRISPR_B2M_TAP1_RESCUE", "CRISPRa B2M/TAP1 antigen-presentation rescue", "crispr", "Antigen presentation", "B2M;TAP1", "genetic", 96, 0.70, 0.36, 0.14, 3, "C", "genetic", False, -0.08, -0.07, -1.16, -0.15),
        ("FLOW_MHC1_48H", "Flow-cytometric MHC-I recovery", "sampling_time", "Antigen presentation", "MHC-I", "flow", 48, 0.31, 0.28, 0.08, 9, "C", "flow", False, -0.08, -0.06, -0.96, -0.12),
        ("CRISPR_EP300_24H", "CRISPRi EP300 enhancer-plasticity perturbation", "crispr", "Enhancer plasticity", "EP300", "genetic", 72, 0.66, 0.35, 0.12, 4, "D", "genetic", False, -0.29, -0.18, -0.09, -1.08),
        ("ATAC_COMMITMENT_36H", "ATAC-seq commitment-window sampling", "sampling_time", "Enhancer plasticity", "enhancer accessibility", "atac", 36, 0.48, 0.30, 0.09, 11, "D", "chromatin", False, -0.23, -0.15, -0.08, -0.94),
        ("UNTARGETED_LATE_OMICS", "Untargeted late-state multi-omics panel", "sampling_time", "Broad characterization", "multi-omic panel", "multiomics", 96, 0.78, 0.44, 0.15, 12, "E", "multiomics", False, -0.55, -0.52, -0.49, -0.50),
    ]
    columns = [
        "experiment_id", "experiment_name", "experiment_type", "mechanism", "target",
        "assay_type", "sample_time_hours", "estimated_cost", "measurement_noise",
        "technical_failure_probability", "baseline_order", "confirmation_group", "assay_family",
        "recovery_informative", f"expected_readout__{h1}", f"expected_readout__{h2}",
        f"expected_readout__{h3}", f"expected_readout__{h4}",
    ]
    return pd.DataFrame(rows, columns=columns)


def experiment_contract_schema() -> dict[str, Any]:
    required = [
        "contract_version", "study_id", "cycle_id", "experiment_id", "sample_id",
        "assay_type", "perturbation_type", "target", "dose", "dose_unit", "start_time",
        "duration", "time_unit", "sample_time", "replicate_id", "randomization_block",
        "expected_cost", "cost_unit", "model_freeze_id", "preregistration_id", "status",
    ]
    properties = {
        "contract_version": {"type": "string", "const": CONTRACT_VERSION},
        "study_id": {"type": "string"},
        "cycle_id": {"type": "integer", "minimum": 1},
        "experiment_id": {"type": "string"},
        "sample_id": {"type": "string"},
        "biospecimen_id": {"type": ["string", "null"]},
        "plate_id": {"type": ["string", "null"]},
        "well_id": {"type": ["string", "null"]},
        "instrument_run_id": {"type": ["string", "null"]},
        "assay_type": {"type": "string"},
        "perturbation_type": {"type": "string"},
        "target": {"type": "string"},
        "agent": {"type": ["string", "null"]},
        "dose": {"type": "number"},
        "dose_unit": {"type": "string"},
        "start_time": {"type": "number"},
        "duration": {"type": "number", "minimum": 0},
        "time_unit": {"type": "string"},
        "sample_time": {"type": "number", "minimum": 0},
        "batch_id": {"type": ["string", "null"]},
        "replicate_id": {"type": "string"},
        "randomization_block": {"type": "string"},
        "operator": {"type": ["string", "null"]},
        "protocol_uri": {"type": ["string", "null"]},
        "sample_manifest_uri": {"type": ["string", "null"]},
        "expected_cost": {"type": "number", "minimum": 0},
        "cost_unit": {"type": "string"},
        "model_freeze_id": {"type": "string"},
        "preregistration_id": {"type": "string"},
        "status": {"type": "string", "enum": ["planned", "started", "completed", "failed", "cancelled"]},
        "created_at_utc": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://causaflux.local/contracts/experiment-contract-1.0.schema.json",
        "title": "CausaFlux LIMS/ELN-compatible experiment contract",
        "type": "object",
        "additionalProperties": True,
        "required": required,
        "properties": properties,
    }


def qc_contract_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CausaFlux experimental QC ingestion contract",
        "type": "object",
        "required": ["cycle_id", "experiment_id", "sample_id", "assay_status", "qc_pass", "usable_for_primary_endpoint"],
        "properties": {
            "cycle_id": {"type": "integer", "minimum": 1},
            "experiment_id": {"type": "string"},
            "sample_id": {"type": "string"},
            "assay_status": {"type": "string", "enum": ["pass", "fail", "partial"]},
            "qc_metric": {"type": ["string", "null"]},
            "qc_value": {"type": ["number", "null"]},
            "qc_threshold": {"type": ["number", "null"]},
            "qc_pass": {"type": "boolean"},
            "failure_reason": {"type": ["string", "null"]},
            "usable_for_primary_endpoint": {"type": "boolean"},
        },
    }


def outcome_contract_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CausaFlux locked outcome ingestion contract",
        "type": "object",
        "required": ["cycle_id", "experiment_id", "sample_id", "outcome_name", "observed_value", "standard_error", "endpoint_role"],
        "properties": {
            "cycle_id": {"type": "integer", "minimum": 1},
            "experiment_id": {"type": "string"},
            "sample_id": {"type": "string"},
            "outcome_name": {"type": "string"},
            "observed_value": {"type": "number"},
            "standard_error": {"type": "number", "exclusiveMinimum": 0},
            "unit": {"type": "string"},
            "endpoint_role": {"type": "string", "enum": ["primary", "secondary"]},
            "blinded": {"type": "boolean"},
            "measurement_timestamp_utc": {"type": ["string", "null"]},
        },
    }


def write_contract_bundle(output_dir: str | Path) -> dict[str, Path]:
    out = ensure_dir(output_dir)
    schema_path = out / "experiment_contract.schema.json"
    qc_path = out / "experimental_qc.schema.json"
    outcome_path = out / "outcome_contract.schema.json"
    json_dump(experiment_contract_schema(), schema_path)
    json_dump(qc_contract_schema(), qc_path)
    json_dump(outcome_contract_schema(), outcome_path)

    pd.DataFrame(
        [{
            "contract_version": CONTRACT_VERSION,
            "study_id": "STUDY_ID",
            "cycle_id": 1,
            "experiment_id": "EXPERIMENT_ID",
            "sample_id": "SAMPLE_ID",
            "biospecimen_id": "",
            "plate_id": "",
            "well_id": "",
            "instrument_run_id": "",
            "assay_type": "assay",
            "perturbation_type": "drug_or_genetic_or_environmental",
            "target": "TARGET",
            "agent": "",
            "dose": 0.0,
            "dose_unit": "uM",
            "start_time": 0.0,
            "duration": 24.0,
            "time_unit": "hours",
            "sample_time": 24.0,
            "batch_id": "",
            "replicate_id": "R1",
            "randomization_block": "B1",
            "operator": "",
            "protocol_uri": "",
            "sample_manifest_uri": "",
            "expected_cost": 1.0,
            "cost_unit": "relative_cost",
            "model_freeze_id": "MODEL_FREEZE_ID",
            "preregistration_id": "PREREGISTRATION_ID",
            "status": "planned",
            "created_at_utc": "",
            "notes": "",
        }]
    ).to_csv(out / "experiment_contract_template.csv", index=False)

    pd.DataFrame([{
        "cycle_id": 1, "experiment_id": "EXPERIMENT_ID", "sample_id": "SAMPLE_ID",
        "assay_status": "pass", "qc_metric": "read_depth_or_assay_specific_metric",
        "qc_value": 1.0, "qc_threshold": 0.8, "qc_pass": True, "failure_reason": "",
        "usable_for_primary_endpoint": True,
    }]).to_csv(out / "experimental_qc_template.csv", index=False)

    pd.DataFrame([{
        "cycle_id": 1, "experiment_id": "EXPERIMENT_ID", "sample_id": "SAMPLE_ID",
        "outcome_name": "standardized_primary_readout", "observed_value": 0.0,
        "standard_error": 0.2, "unit": "standardized", "endpoint_role": "primary",
        "blinded": True, "measurement_timestamp_utc": "",
    }]).to_csv(out / "outcome_template.csv", index=False)

    (out / "ELN_TEMPLATE.md").write_text(
        "# CausaFlux prospective experiment ELN record\n\n"
        "- Study ID:\n- Cycle ID:\n- Experiment ID:\n- Model freeze ID:\n- Preregistration ID:\n"
        "- Hypothesis tested:\n- Prespecified primary endpoint:\n- Perturbation/dose/timing:\n"
        "- Randomization/blinding:\n- QC acceptance criteria:\n- Failure handling rule:\n"
        "- Raw-data URI:\n- Protocol URI:\n- Deviations from preregistration:\n- Outcome lock timestamp:\n",
        encoding="utf-8",
    )
    return {"experiment_schema": schema_path, "qc_schema": qc_path, "outcome_schema": outcome_path}


def _model_freeze_manifest(
    cycle: int,
    prior: np.ndarray,
    catalog: pd.DataFrame,
    code_root: Path,
    parent_freeze_id: str | None,
    config: ProspectiveLoopConfig,
) -> dict[str, Any]:
    prior_payload = {hid: float(prior[i]) for i, hid in enumerate(HYPOTHESIS_IDS)}
    code_hash = hash_tree(code_root)
    catalog_hash = sha256_frame(catalog)
    config_hash = sha256_text(json.dumps(config.__dict__, sort_keys=True))
    model_state_hash = sha256_text(json.dumps(prior_payload, sort_keys=True))
    freeze_id = f"CF180-C{cycle}-{sha256_text(code_hash + catalog_hash + model_state_hash)[:12]}"
    return {
        "framework": "CausaFlux",
        "version": PROSPECTIVE_VERSION,
        "cycle": int(cycle),
        "model_freeze_id": freeze_id,
        "parent_model_freeze_id": parent_freeze_id,
        "model_class": "Bayesian mechanism posterior + CausaFlux experiment utility engine",
        "model_state": prior_payload,
        "model_state_sha256": model_state_hash,
        "candidate_catalog_sha256": catalog_hash,
        "configuration_sha256": config_hash,
        "source_tree_sha256": code_hash,
        "training_or_evidence_boundary": "Synthetic software-validation fixture; no real prospective biological claim.",
        "created_at_utc": _utc_now(),
        "immutable_after_prediction_export": True,
    }


def _prediction_table(catalog: pd.DataFrame, prior: np.ndarray, cycle: int, completed: set[str], config: ProspectiveLoopConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    entropy_before = _entropy(prior)
    for _, row in catalog.iterrows():
        experiment_id = str(row["experiment_id"])
        if experiment_id in completed:
            continue
        means = row[[f"expected_readout__{hid}" for hid in HYPOTHESIS_IDS]].to_numpy(dtype=float)
        sigma = float(row["measurement_noise"])
        eig = expected_information_gain(prior, means, sigma, _stable_seed(config.seed, f"cycle{cycle}", experiment_id))
        pred_mean = float(np.dot(prior, means))
        variance = float(np.dot(prior, (means - pred_mean) ** 2) + sigma**2)
        pred_sd = math.sqrt(max(variance, 1e-12))
        # Probability of a biologically meaningful absolute response under the mixture.
        rng = np.random.default_rng(_stable_seed(config.seed, "prob", str(cycle), experiment_id))
        h = rng.choice(len(prior), size=3000, p=prior)
        draws = rng.normal(means[h], sigma)
        p_discovery = float(np.mean(np.abs(draws) >= config.discovery_threshold))
        p_recovery = float(np.mean(draws <= config.recovery_threshold))
        utility = (
            0.58 * eig
            + 0.20 * p_discovery
            + 0.12 * p_recovery
            + 0.10 * (1.0 - float(row["technical_failure_probability"]))
        ) / max(float(row["estimated_cost"]), 0.05)
        rows.append({
            **row.to_dict(),
            "cycle": cycle,
            "prior_entropy_nats": entropy_before,
            "predicted_mean": pred_mean,
            "predicted_sd": pred_sd,
            "prediction_interval_90_low": pred_mean - config.interval_z * pred_sd,
            "prediction_interval_90_high": pred_mean + config.interval_z * pred_sd,
            "predicted_discovery_probability": p_discovery,
            "predicted_recovery_probability": p_recovery,
            "expected_information_gain_nats": eig,
            "utility_per_cost": utility,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["utility_per_cost", "expected_information_gain_nats"], ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def _select_batch(predictions: pd.DataFrame, cycle: int, completed_rows: pd.DataFrame, config: ProspectiveLoopConfig) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    chosen: list[int] = []
    cost = 0.0
    used_groups: set[str] = set()
    previous_mechanisms = set(completed_rows.get("mechanism", pd.Series(dtype=str)).dropna().astype(str))
    previous_assays = set(completed_rows.get("assay_family", pd.Series(dtype=str)).dropna().astype(str))

    candidates = predictions.copy()
    if cycle == 3 and config.require_independent_cycle3 and previous_mechanisms:
        # Confirmation/falsification must be an orthogonal assay of a mechanism already tested.
        candidates["confirmation_priority"] = (
            candidates["mechanism"].astype(str).isin(previous_mechanisms).astype(int)
            + (~candidates["assay_family"].astype(str).isin(previous_assays)).astype(int)
        )
        candidates = candidates.sort_values(["confirmation_priority", "utility_per_cost"], ascending=False)

    for idx, row in candidates.iterrows():
        row_cost = float(row["estimated_cost"])
        if cost + row_cost > config.cycle_budget + 1e-12:
            continue
        group = str(row["confirmation_group"])
        if group in used_groups and len(chosen) + 1 < config.experiments_per_cycle:
            continue
        chosen.append(idx)
        used_groups.add(group)
        cost += row_cost
        if len(chosen) >= config.experiments_per_cycle:
            break
    selected = candidates.loc[chosen].copy() if chosen else candidates.head(0).copy()
    selected["selection_role"] = "model_guided"
    if cycle == 3 and not selected.empty:
        selected["selection_role"] = "independent_confirmation_or_falsification"
    return selected.reset_index(drop=True)


def _write_preregistration(
    cycle_dir: Path,
    freeze: Mapping[str, Any],
    predictions: pd.DataFrame,
    selected: pd.DataFrame,
    config: ProspectiveLoopConfig,
) -> dict[str, Any]:
    prereg = ensure_dir(cycle_dir / "preregistration")
    predictions_path = prereg / "preregistered_predictions.csv"
    selected_path = prereg / "selected_experiments.csv"
    predictions.to_csv(predictions_path, index=False)
    selected.to_csv(selected_path, index=False)
    prereg_id = f"PREREG-C{freeze['cycle']}-{sha256_file(predictions_path)[:12]}"
    lock = {
        "preregistration_id": prereg_id,
        "cycle": freeze["cycle"],
        "model_freeze_id": freeze["model_freeze_id"],
        "prediction_export_sha256": sha256_file(predictions_path),
        "selected_experiments_sha256": sha256_file(selected_path),
        "primary_endpoint": "observed_standardized_readout",
        "primary_metrics": ["prediction_rmse", "90pct_interval_coverage", "brier_discovery"],
        "failed_assay_rule": "Failed/primary-endpoint-unusable assays are excluded from posterior update and locked prediction scoring but remain in attempted-cost accounting. Replacement, if any, uses the next preregistered eligible ranking without inspecting failed outcomes.",
        "adaptive_stopping_rule": f"After at least {config.min_cycles} cycles, stop when posterior max >= {config.posterior_stop_threshold:.3f}, all remaining EIG < {config.min_expected_information_gain:.3f}, budget exhausted, or max_cycles reached.",
        "baseline_strategy": config.baseline_strategy,
        "locked_before_outcome_access": True,
        "created_at_utc": _utc_now(),
    }
    json_dump(lock, prereg / "prediction_lock.json")
    return lock


def _experiment_contract_rows(selected: pd.DataFrame, cycle: int, freeze_id: str, prereg_id: str) -> pd.DataFrame:
    rows = []
    for i, row in selected.iterrows():
        sample_id = f"C{cycle}-{row['experiment_id']}-R1"
        rows.append({
            "contract_version": CONTRACT_VERSION,
            "study_id": "CAUSAFLUX_PROSPECTIVE_REFERENCE",
            "cycle_id": cycle,
            "experiment_id": row["experiment_id"],
            "sample_id": sample_id,
            "biospecimen_id": f"BIO-C{cycle}-{i+1:02d}",
            "plate_id": f"PLATE-C{cycle}",
            "well_id": f"A{i+1}",
            "instrument_run_id": f"RUN-C{cycle}-{row['assay_family']}",
            "assay_type": row["assay_type"],
            "perturbation_type": row["experiment_type"],
            "target": row["target"],
            "agent": row["target"],
            "dose": 1.0,
            "dose_unit": "relative",
            "start_time": 0.0,
            "duration": float(row["sample_time_hours"]),
            "time_unit": "hours",
            "sample_time": float(row["sample_time_hours"]),
            "batch_id": f"BATCH-C{cycle}",
            "replicate_id": "R1",
            "randomization_block": f"BLOCK-C{cycle}",
            "operator": "synthetic_reference",
            "protocol_uri": "protocol://replace-with-eln-or-sop-uri",
            "sample_manifest_uri": "lims://replace-with-sample-manifest-uri",
            "expected_cost": float(row["estimated_cost"]),
            "cost_unit": "relative_cost",
            "model_freeze_id": freeze_id,
            "preregistration_id": prereg_id,
            "status": "planned",
            "created_at_utc": _utc_now(),
            "notes": "Synthetic prospective-loop fixture",
        })
    return pd.DataFrame(rows)


def _simulate_assay(selected: pd.DataFrame, cycle: int, config: ProspectiveLoopConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate locked synthetic QC and outcomes after preregistration.

    Outcomes are deterministic per experiment ID and truth, so the AI and baseline
    policies see the same potential outcome for a given experiment.
    """
    outcomes: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    truth_col = f"expected_readout__{config.truth_hypothesis}"
    for _, row in selected.iterrows():
        experiment_id = str(row["experiment_id"])
        sample_id = f"C{cycle}-{experiment_id}-R1"
        failure_seed = _stable_seed(config.seed, "qc", experiment_id)
        rng_failure = np.random.default_rng(failure_seed)
        forced = config.synthetic_failure_experiment == experiment_id
        failed = forced or bool(rng_failure.uniform() < float(row["technical_failure_probability"]) * 0.45)
        if failed:
            qc_rows.append({
                "cycle_id": cycle, "experiment_id": experiment_id, "sample_id": sample_id,
                "assay_status": "fail", "qc_metric": "synthetic_assay_quality", "qc_value": 0.42,
                "qc_threshold": 0.70, "qc_pass": False,
                "failure_reason": "prespecified synthetic technical failure" if forced else "synthetic technical failure",
                "usable_for_primary_endpoint": False,
            })
            continue
        qc_rows.append({
            "cycle_id": cycle, "experiment_id": experiment_id, "sample_id": sample_id,
            "assay_status": "pass", "qc_metric": "synthetic_assay_quality", "qc_value": 0.91,
            "qc_threshold": 0.70, "qc_pass": True, "failure_reason": "",
            "usable_for_primary_endpoint": True,
        })
        rng = np.random.default_rng(_stable_seed(config.seed, "outcome", experiment_id, config.truth_hypothesis))
        sigma = float(row["measurement_noise"])
        observed = float(rng.normal(float(row[truth_col]), sigma))
        outcomes.append({
            "cycle_id": cycle, "experiment_id": experiment_id, "sample_id": sample_id,
            "outcome_name": "observed_standardized_readout", "observed_value": observed,
            "standard_error": sigma, "unit": "standardized", "endpoint_role": "primary",
            "blinded": True, "measurement_timestamp_utc": _utc_now(),
        })
    return pd.DataFrame(qc_rows), pd.DataFrame(outcomes)


def ingest_experimental_qc(qc: pd.DataFrame, contract: pd.DataFrame) -> pd.DataFrame:
    required = {"cycle_id", "experiment_id", "sample_id", "assay_status", "qc_pass", "usable_for_primary_endpoint"}
    missing = required - set(qc.columns)
    if missing:
        raise ValueError(f"QC table missing required columns: {sorted(missing)}")
    known = set(contract["experiment_id"].astype(str))
    unknown = sorted(set(qc["experiment_id"].astype(str)) - known)
    if unknown:
        raise ValueError(f"QC references experiments outside the locked contract: {unknown}")
    result = qc.copy()
    result["qc_pass"] = result["qc_pass"].astype(bool)
    result["usable_for_primary_endpoint"] = result["usable_for_primary_endpoint"].astype(bool)
    result["locked_evaluation_eligible"] = result["qc_pass"] & result["usable_for_primary_endpoint"] & (result["assay_status"] == "pass")
    return result


def _locked_evaluate(predictions: pd.DataFrame, selected: pd.DataFrame, qc: pd.DataFrame, outcomes: pd.DataFrame, lock_path: Path, config: ProspectiveLoopConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    pred_path = lock_path.parent / "preregistered_predictions.csv"
    if sha256_file(pred_path) != lock["prediction_export_sha256"]:
        raise ValueError("preregistered prediction file hash no longer matches the lock")
    selected_ids = set(selected["experiment_id"].astype(str))
    pred = predictions[predictions["experiment_id"].astype(str).isin(selected_ids)].copy()
    eligible_ids = set(qc.loc[qc["locked_evaluation_eligible"], "experiment_id"].astype(str))
    out = outcomes[outcomes["experiment_id"].astype(str).isin(eligible_ids)].copy()
    eval_frame = pred.merge(out[["experiment_id", "observed_value", "standard_error"]], on="experiment_id", how="inner")
    if eval_frame.empty:
        metrics = {
            "n_attempted": int(len(selected)), "n_evaluable": 0, "n_failed": int(len(selected)),
            "prediction_rmse": None, "interval_coverage_90": None, "brier_discovery": None,
        }
        return eval_frame, metrics
    eval_frame["residual"] = eval_frame["observed_value"] - eval_frame["predicted_mean"]
    eval_frame["inside_90_interval"] = (
        (eval_frame["observed_value"] >= eval_frame["prediction_interval_90_low"])
        & (eval_frame["observed_value"] <= eval_frame["prediction_interval_90_high"])
    )
    eval_frame["observed_discovery"] = (eval_frame["observed_value"].abs() >= config.discovery_threshold).astype(int)
    eval_frame["observed_recovery"] = (eval_frame["observed_value"] <= config.recovery_threshold).astype(int)
    eval_frame["squared_discovery_error"] = (
        eval_frame["predicted_discovery_probability"] - eval_frame["observed_discovery"]
    ) ** 2
    metrics = {
        "n_attempted": int(len(selected)),
        "n_evaluable": int(len(eval_frame)),
        "n_failed": int(len(selected) - len(eval_frame)),
        "prediction_rmse": float(np.sqrt(np.mean(eval_frame["residual"] ** 2))),
        "interval_coverage_90": float(eval_frame["inside_90_interval"].mean()),
        "brier_discovery": float(eval_frame["squared_discovery_error"].mean()),
        "mean_predicted_discovery_probability": float(eval_frame["predicted_discovery_probability"].mean()),
        "observed_discovery_rate": float(eval_frame["observed_discovery"].mean()),
        "recovery_trajectories_identified": int(eval_frame["observed_recovery"].sum()),
    }
    return eval_frame, metrics


def _posterior_from_evaluable(prior: np.ndarray, catalog: pd.DataFrame, eval_frame: pd.DataFrame) -> np.ndarray:
    posterior = prior.copy()
    lookup = catalog.set_index("experiment_id")
    for _, outcome in eval_frame.iterrows():
        row = lookup.loc[str(outcome["experiment_id"])]
        means = row[[f"expected_readout__{hid}" for hid in HYPOTHESIS_IDS]].to_numpy(dtype=float)
        posterior = posterior_update(posterior, float(outcome["observed_value"]), means, float(outcome["standard_error"]))
    return posterior


def _adaptive_stop(cycle: int, posterior: np.ndarray, remaining_predictions: pd.DataFrame, cumulative_cost: float, config: ProspectiveLoopConfig) -> tuple[bool, str]:
    if cycle < config.min_cycles:
        return False, "minimum_required_cycles_not_reached"
    if cycle >= config.max_cycles:
        return True, "maximum_cycles_reached"
    if float(np.max(posterior)) >= config.posterior_stop_threshold:
        return True, "posterior_confidence_threshold_reached"
    if remaining_predictions.empty or float(remaining_predictions["expected_information_gain_nats"].max()) < config.min_expected_information_gain:
        return True, "remaining_information_gain_below_threshold"
    if cumulative_cost >= config.cycle_budget * config.max_cycles:
        return True, "total_budget_exhausted"
    return False, "continue"


def _baseline_select(catalog: pd.DataFrame, completed: set[str], cycle: int, config: ProspectiveLoopConfig) -> pd.DataFrame:
    remaining = catalog[~catalog["experiment_id"].astype(str).isin(completed)].sort_values("baseline_order")
    chosen = []
    cost = 0.0
    for _, row in remaining.iterrows():
        row_cost = float(row["estimated_cost"])
        if cost + row_cost <= config.cycle_budget + 1e-12:
            chosen.append(row)
            cost += row_cost
        if len(chosen) >= config.experiments_per_cycle:
            break
    return pd.DataFrame(chosen) if chosen else remaining.head(0).copy()


def _simulate_policy_baseline(catalog: pd.DataFrame, hypotheses: pd.DataFrame, config: ProspectiveLoopConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    prior = hypotheses["prior_probability"].to_numpy(dtype=float)
    prior /= prior.sum()
    entropy0 = _entropy(prior)
    completed: set[str] = set()
    records = []
    cumulative_cost = 0.0
    recovery_found = 0
    discoveries = 0
    for cycle in range(1, config.max_cycles + 1):
        selected = _baseline_select(catalog, completed, cycle, config)
        qc, outcomes = _simulate_assay(selected, cycle, config)
        contract = pd.DataFrame({"experiment_id": selected["experiment_id"]})
        qc = ingest_experimental_qc(qc, contract)
        eligible = set(qc.loc[qc["locked_evaluation_eligible"], "experiment_id"].astype(str))
        lookup = catalog.set_index("experiment_id")
        for _, row in selected.iterrows():
            experiment_id = str(row["experiment_id"])
            cumulative_cost += float(row["estimated_cost"])
            completed.add(experiment_id)
            outcome = outcomes.loc[outcomes["experiment_id"] == experiment_id]
            if experiment_id in eligible and not outcome.empty:
                value = float(outcome.iloc[0]["observed_value"])
                discoveries += int(abs(value) >= config.discovery_threshold)
                recovery_found += int(value <= config.recovery_threshold)
                means = lookup.loc[experiment_id, [f"expected_readout__{hid}" for hid in HYPOTHESIS_IDS]].to_numpy(dtype=float)
                prior = posterior_update(prior, value, means, float(outcome.iloc[0]["standard_error"]))
                status = "evaluable"
            else:
                value = np.nan
                status = "failed_qc"
            records.append({"strategy": config.baseline_strategy, "cycle": cycle, "experiment_id": experiment_id, "status": status, "observed_value": value, "cost": float(row["estimated_cost"]), "posterior_entropy_nats": _entropy(prior)})
    reduction = entropy0 - _entropy(prior)
    metrics = {
        "outcome_discovery": float(discoveries),
        "uncertainty_reduction_nats": float(reduction),
        "experiment_efficiency": float(reduction / max(cumulative_cost, 1e-8)),
        "recovery_trajectory_identification": float(recovery_found),
        "total_cost": float(cumulative_cost),
    }
    return pd.DataFrame(records), metrics


def _comparison_gate(ai_metrics: Mapping[str, float], baseline_metrics: Mapping[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    criteria = [
        ("outcome_discovery", 0.0),
        ("uncertainty_reduction_nats", 0.01),
        ("experiment_efficiency", 0.0),
        ("recovery_trajectory_identification", 0.0),
    ]
    rows = []
    for metric, margin in criteria:
        ai = float(ai_metrics[metric])
        baseline = float(baseline_metrics[metric])
        improvement = ai - baseline
        passed = bool(improvement > margin)
        rows.append({"metric": metric, "causaflux": ai, "non_ai_baseline": baseline, "absolute_improvement": improvement, "prespecified_margin": margin, "criterion_pass": passed})
    frame = pd.DataFrame(rows)
    passed_metrics = frame.loc[frame["criterion_pass"], "metric"].tolist()
    gate = {
        "framework": "CausaFlux",
        "version": PROSPECTIVE_VERSION,
        "software_gate": "PASS" if passed_metrics else "FAIL",
        "exit_criterion": "Model-guided selection must improve outcome discovery, uncertainty reduction, experiment efficiency, or recovery-trajectory identification relative to a prespecified non-AI strategy.",
        "passing_metrics": passed_metrics,
        "n_passing_metrics": len(passed_metrics),
        "synthetic_fixture": True,
        "real_prospective_claim_authorized": False,
        "authorization_boundary": "Reference result validates software logic only. A real claim requires three prospectively locked experimental cycles, with Cycle 3 independent confirmation or falsification.",
    }
    return frame, gate


def plot_posterior_history(history: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for hid in HYPOTHESIS_IDS:
        if hid in history:
            ax.plot(history["cycle"], history[hid], marker="o", label=hid)
    ax.set_xlabel("Prospective cycle")
    ax.set_ylabel("Posterior probability")
    ax.set_ylim(0, 1)
    ax.set_title("Cycle-to-cycle posterior update")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = Path(output_path)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_calibration(calibration: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    valid = calibration.dropna(subset=["mean_predicted_discovery_probability", "observed_discovery_rate"])
    if not valid.empty:
        ax.plot(valid["mean_predicted_discovery_probability"], valid["observed_discovery_rate"], marker="o")
        for _, row in valid.iterrows():
            ax.annotate(f"C{int(row['cycle'])}", (row["mean_predicted_discovery_probability"], row["observed_discovery_rate"]), xytext=(4, 4), textcoords="offset points")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean preregistered discovery probability")
    ax.set_ylabel("Observed discovery fraction")
    ax.set_title("Cycle-to-cycle calibration")
    fig.tight_layout()
    path = Path(output_path)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_strategy_comparison(comparison: pd.DataFrame, output_path: str | Path) -> Path:
    frame = comparison.set_index("metric")[["causaflux", "non_ai_baseline"]]
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    frame.plot(kind="bar", ax=ax)
    ax.set_ylabel("Prespecified metric value")
    ax.set_title("CausaFlux vs prespecified non-AI strategy")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = Path(output_path)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def generate_prospective_report(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    report_dir = ensure_dir(root / "report")
    comparison = pd.read_csv(root / "strategy_comparison.csv")
    calibration = pd.read_csv(root / "cycle_calibration.csv")
    costs = pd.read_csv(root / "experiment_cost_ledger.csv")
    posterior = pd.read_csv(root / "posterior_history.csv")
    gate = json.loads((root / "prospective_exit_gate.json").read_text(encoding="utf-8"))
    cycles = sorted(int(p.name.split("_")[1]) for p in root.glob("cycle_[0-9]*") if p.is_dir() and p.name.split("_")[1].isdigit())
    cycle_sections = []
    for cycle in cycles:
        cdir = root / f"cycle_{cycle}"
        eval_path = cdir / "evaluation" / "locked_metrics.json"
        metrics = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
        failed_path = cdir / "experiment" / "failed_assays.csv"
        failed = pd.read_csv(failed_path) if failed_path.exists() else pd.DataFrame()
        cycle_sections.append(
            f"<h3>Cycle {cycle}</h3><pre>{json.dumps(metrics, indent=2)}</pre>"
            + ("<p><b>Failed assays:</b></p>" + failed.to_html(index=False) if not failed.empty else "<p>No failed assays.</p>")
        )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>CausaFlux v1.8.0 Prospective Loop</title>
<style>body{{font-family:Arial,sans-serif;max-width:1180px;margin:28px auto;line-height:1.45}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ddd;padding:6px}}th{{background:#f3f3f3}}.ok{{border-left:5px solid #2b7a68;background:#f2faf7;padding:12px}}.warn{{border-left:5px solid #b8842f;background:#fff8e9;padding:12px}}code,pre{{background:#f7f7f7;padding:8px;overflow:auto}}</style></head><body>
<h1>CausaFlux v1.8.0 — Prospective Experimental Loop</h1>
<div class='ok'><b>Software exit gate: {gate['software_gate']}</b><br>Passing metrics: {', '.join(gate['passing_metrics']) or 'none'}</div>
<div class='warn'><b>Evidence boundary:</b> This bundled run is synthetic software validation. It does not constitute prospective biological validation.</div>
<h2>Required prospective sequence</h2><ol><li>Cycle 1: prediction → experiment → outcome → locked evaluation.</li><li>Cycle 2: posterior-updated model → new recommendation → new experiment → locked evaluation.</li><li>Cycle 3: independent confirmation or falsification.</li></ol>
<h2>CausaFlux vs non-AI strategy</h2>{comparison.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Cycle calibration</h2>{calibration.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Experiment-cost ledger</h2>{costs.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Posterior history</h2>{posterior.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Cycle audit</h2>{''.join(cycle_sections)}
<h2>Governance artifacts</h2><p>Each cycle contains a model freeze manifest, a SHA-256 locked preregistered prediction export, LIMS/ELN experiment contract, QC ingestion, failed-assay ledger, locked prediction-vs-outcome evaluation, and posterior update.</p>
</body></html>"""
    path = report_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def run_prospective_loop(output_dir: str | Path, config: ProspectiveLoopConfig | None = None, code_root: str | Path | None = None) -> ProspectiveLoopResult:
    config = config or ProspectiveLoopConfig()
    if config.truth_hypothesis not in HYPOTHESIS_IDS:
        raise ValueError(f"unknown truth_hypothesis: {config.truth_hypothesis}")
    root = ensure_dir(output_dir)
    contracts = ensure_dir(root / "contracts")
    write_contract_bundle(contracts)
    hypotheses = default_hypotheses()
    catalog = default_experiment_catalog()
    hypotheses.to_csv(root / "hypothesis_priors.csv", index=False)
    catalog.to_csv(root / "prespecified_experiment_catalog.csv", index=False)
    baseline_plan = catalog[["baseline_order", "experiment_id", "experiment_name", "estimated_cost"]].sort_values("baseline_order")
    baseline_plan.to_csv(root / "prespecified_non_ai_strategy.csv", index=False)

    code_root = Path(code_root) if code_root else Path(__file__).resolve().parent
    prior = hypotheses["prior_probability"].to_numpy(dtype=float)
    prior /= prior.sum()
    initial_entropy = _entropy(prior)
    completed: set[str] = set()
    completed_rows = pd.DataFrame()
    predictions_all: list[pd.DataFrame] = []
    outcomes_all: list[pd.DataFrame] = []
    qc_all: list[pd.DataFrame] = []
    posterior_rows = [{"cycle": 0, "model_freeze_id": "PRIOR", "posterior_entropy_nats": initial_entropy, **{hid: float(prior[i]) for i, hid in enumerate(HYPOTHESIS_IDS)}}]
    cost_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    parent_freeze: str | None = None
    stop_reason = "maximum_cycles_reached"

    for cycle in range(1, config.max_cycles + 1):
        cycle_dir = ensure_dir(root / f"cycle_{cycle}")
        freeze_dir = ensure_dir(cycle_dir / "model_freeze")
        freeze = _model_freeze_manifest(cycle, prior, catalog, code_root, parent_freeze, config)
        json_dump(freeze, freeze_dir / "model_freeze_manifest.json")
        predictions = _prediction_table(catalog, prior, cycle, completed, config)
        selected = _select_batch(predictions, cycle, completed_rows, config)
        if selected.empty:
            stop_reason = "no_eligible_experiments"
            break
        lock = _write_preregistration(cycle_dir, freeze, predictions, selected, config)
        contract = _experiment_contract_rows(selected, cycle, freeze["model_freeze_id"], lock["preregistration_id"])
        experiment_dir = ensure_dir(cycle_dir / "experiment")
        contract.to_csv(experiment_dir / "lims_eln_experiment_contract.csv", index=False)

        # Outcomes are generated only after prediction lock has been written.
        qc_raw, outcomes = _simulate_assay(selected, cycle, config)
        qc = ingest_experimental_qc(qc_raw, contract)
        qc.to_csv(experiment_dir / "experimental_qc_ingested.csv", index=False)
        outcomes.to_csv(experiment_dir / "locked_outcomes.csv", index=False)
        failed = qc.loc[~qc["locked_evaluation_eligible"]].copy()
        failed.to_csv(experiment_dir / "failed_assays.csv", index=False)

        eval_dir = ensure_dir(cycle_dir / "evaluation")
        eval_frame, metrics = _locked_evaluate(predictions, selected, qc, outcomes, cycle_dir / "preregistration" / "prediction_lock.json", config)
        eval_frame.to_csv(eval_dir / "prediction_vs_outcome.csv", index=False)
        json_dump(metrics, eval_dir / "locked_metrics.json")

        prior_before = prior.copy()
        prior = _posterior_from_evaluable(prior, catalog, eval_frame)
        model_update = {
            "cycle": cycle,
            "parent_model_freeze_id": freeze["model_freeze_id"],
            "prior": {hid: float(prior_before[i]) for i, hid in enumerate(HYPOTHESIS_IDS)},
            "posterior": {hid: float(prior[i]) for i, hid in enumerate(HYPOTHESIS_IDS)},
            "prior_entropy_nats": _entropy(prior_before),
            "posterior_entropy_nats": _entropy(prior),
            "entropy_reduction_nats": _entropy(prior_before) - _entropy(prior),
            "n_evaluable_outcomes": int(len(eval_frame)),
            "update_type": "Bayesian posterior model update; neural/foundation weights remain frozen during the cycle",
            "created_at_utc": _utc_now(),
        }
        update_dir = ensure_dir(cycle_dir / "model_update")
        json_dump(model_update, update_dir / "posterior_model_update.json")

        posterior_rows.append({"cycle": cycle, "model_freeze_id": freeze["model_freeze_id"], "posterior_entropy_nats": _entropy(prior), **{hid: float(prior[i]) for i, hid in enumerate(HYPOTHESIS_IDS)}})
        for _, row in selected.iterrows():
            expid = str(row["experiment_id"])
            qrow = qc.loc[qc["experiment_id"] == expid]
            status = "evaluable" if (not qrow.empty and bool(qrow.iloc[0]["locked_evaluation_eligible"])) else "failed_qc"
            cost_rows.append({
                "strategy": "CausaFlux_model_guided", "cycle": cycle, "experiment_id": expid,
                "planned_cost": float(row["estimated_cost"]), "attempted_cost": float(row["estimated_cost"]),
                "failed_assay_cost": float(row["estimated_cost"]) if status == "failed_qc" else 0.0,
                "evaluable_cost": float(row["estimated_cost"]) if status == "evaluable" else 0.0,
                "status": status,
            })
        calibration_rows.append({"cycle": cycle, **metrics})
        predictions_all.append(predictions.assign(model_freeze_id=freeze["model_freeze_id"], preregistration_id=lock["preregistration_id"]))
        if not outcomes.empty:
            outcomes_all.append(outcomes)
        qc_all.append(qc)
        completed.update(selected["experiment_id"].astype(str))
        completed_rows = pd.concat([completed_rows, selected], ignore_index=True)
        remaining = _prediction_table(catalog, prior, cycle + 1, completed, config)
        cumulative_cost = sum(r["attempted_cost"] for r in cost_rows)
        stop, stop_reason = _adaptive_stop(cycle, prior, remaining, cumulative_cost, config)
        json_dump({"cycle": cycle, "stop": stop, "reason": stop_reason, "posterior_max": float(np.max(prior)), "remaining_max_eig": None if remaining.empty else float(remaining["expected_information_gain_nats"].max()), "cumulative_cost": cumulative_cost}, cycle_dir / "adaptive_stopping.json")
        parent_freeze = freeze["model_freeze_id"]
        if stop:
            break

    predictions_frame = pd.concat(predictions_all, ignore_index=True) if predictions_all else pd.DataFrame()
    outcomes_frame = pd.concat(outcomes_all, ignore_index=True) if outcomes_all else pd.DataFrame()
    qc_frame = pd.concat(qc_all, ignore_index=True) if qc_all else pd.DataFrame()
    posterior_history = pd.DataFrame(posterior_rows)
    cost_ledger = pd.DataFrame(cost_rows)
    calibration = pd.DataFrame(calibration_rows)

    predictions_frame.to_csv(root / "all_preregistered_predictions.csv", index=False)
    outcomes_frame.to_csv(root / "all_locked_outcomes.csv", index=False)
    qc_frame.to_csv(root / "all_experimental_qc.csv", index=False)
    posterior_history.to_csv(root / "posterior_history.csv", index=False)
    cost_ledger.to_csv(root / "experiment_cost_ledger.csv", index=False)
    calibration.to_csv(root / "cycle_calibration.csv", index=False)

    baseline_results, baseline_metrics = _simulate_policy_baseline(catalog, hypotheses, config)
    baseline_results.to_csv(root / "non_ai_baseline_results.csv", index=False)

    evaluable_predictions = []
    for cycle in calibration["cycle"].astype(int).tolist() if not calibration.empty else []:
        path = root / f"cycle_{cycle}" / "evaluation" / "prediction_vs_outcome.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                evaluable_predictions.append(frame)
    all_eval = pd.concat(evaluable_predictions, ignore_index=True) if evaluable_predictions else pd.DataFrame()
    ai_cost = float(cost_ledger["attempted_cost"].sum()) if not cost_ledger.empty else 0.0
    ai_reduction = initial_entropy - float(posterior_history.iloc[-1]["posterior_entropy_nats"])
    ai_metrics = {
        "outcome_discovery": float(all_eval["observed_discovery"].sum()) if not all_eval.empty else 0.0,
        "uncertainty_reduction_nats": float(ai_reduction),
        "experiment_efficiency": float(ai_reduction / max(ai_cost, 1e-8)),
        "recovery_trajectory_identification": float(all_eval["observed_recovery"].sum()) if not all_eval.empty else 0.0,
        "total_cost": ai_cost,
    }
    comparison, gate = _comparison_gate(ai_metrics, baseline_metrics)
    comparison.to_csv(root / "strategy_comparison.csv", index=False)
    gate["completed_cycles"] = int(posterior_history["cycle"].max())
    gate["required_sequence_complete"] = bool(gate["completed_cycles"] >= 3)
    gate["cycle3_independent_confirmation_required"] = config.require_independent_cycle3
    cycle3_path = root / "cycle_3" / "preregistration" / "selected_experiments.csv"
    if cycle3_path.exists():
        c3 = pd.read_csv(cycle3_path)
        gate["cycle3_roles"] = sorted(c3["selection_role"].astype(str).unique().tolist()) if "selection_role" in c3 else []
        gate["cycle3_independent_confirmation_or_falsification"] = "independent_confirmation_or_falsification" in gate["cycle3_roles"]
    else:
        gate["cycle3_roles"] = []
        gate["cycle3_independent_confirmation_or_falsification"] = False
    gate["adaptive_stop_reason"] = stop_reason
    if not gate["required_sequence_complete"] or not gate["cycle3_independent_confirmation_or_falsification"]:
        gate["software_gate"] = "FAIL"
    json_dump(gate, root / "prospective_exit_gate.json")

    run_manifest = {
        "framework": "CausaFlux", "version": PROSPECTIVE_VERSION, "release": "Prospective Experimental Loop",
        "contracts_sha256": {p.name: sha256_file(p) for p in sorted(contracts.glob("*")) if p.is_file()},
        "hypothesis_priors_sha256": sha256_file(root / "hypothesis_priors.csv"),
        "experiment_catalog_sha256": sha256_file(root / "prespecified_experiment_catalog.csv"),
        "non_ai_strategy_sha256": sha256_file(root / "prespecified_non_ai_strategy.csv"),
        "completed_cycles": gate["completed_cycles"], "software_gate": gate["software_gate"],
        "synthetic_fixture": True, "created_at_utc": _utc_now(),
    }
    json_dump(run_manifest, root / "prospective_run_manifest.json")

    fig_dir = ensure_dir(root / "figures")
    plot_posterior_history(posterior_history, fig_dir / "posterior_by_cycle.png")
    plot_calibration(calibration, fig_dir / "cycle_calibration.png")
    plot_strategy_comparison(comparison, fig_dir / "strategy_comparison.png")
    generate_prospective_report(root)

    return ProspectiveLoopResult(
        hypotheses=hypotheses, catalog=catalog, predictions=predictions_frame, outcomes=outcomes_frame,
        qc=qc_frame, posterior_history=posterior_history, cost_ledger=cost_ledger,
        calibration=calibration, baseline_results=baseline_results, comparison=comparison,
        gate=gate, run_manifest=run_manifest,
    )


def validate_prospective_loop(output_dir: str | Path, require_gate: bool = False) -> dict[str, Any]:
    root = Path(output_dir)
    required = [
        "contracts/experiment_contract.schema.json", "contracts/experimental_qc.schema.json",
        "contracts/outcome_contract.schema.json", "prespecified_experiment_catalog.csv",
        "prespecified_non_ai_strategy.csv", "all_preregistered_predictions.csv",
        "all_experimental_qc.csv", "posterior_history.csv", "experiment_cost_ledger.csv",
        "cycle_calibration.csv", "strategy_comparison.csv", "prospective_exit_gate.json",
        "prospective_run_manifest.json", "report/index.html",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise ValueError(f"missing prospective-loop artifacts: {missing}")
    gate = json.loads((root / "prospective_exit_gate.json").read_text(encoding="utf-8"))
    posterior = pd.read_csv(root / "posterior_history.csv")
    if int(posterior["cycle"].max()) < 3:
        raise ValueError("required three-cycle prospective sequence is incomplete")
    probs = posterior[list(HYPOTHESIS_IDS)].to_numpy(dtype=float)
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("posterior probabilities do not sum to one")
    for cycle in (1, 2, 3):
        cdir = root / f"cycle_{cycle}"
        lock_path = cdir / "preregistration" / "prediction_lock.json"
        freeze_path = cdir / "model_freeze" / "model_freeze_manifest.json"
        if not lock_path.exists() or not freeze_path.exists():
            raise ValueError(f"cycle {cycle} is missing freeze or preregistration lock")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        pred_path = cdir / "preregistration" / "preregistered_predictions.csv"
        if sha256_file(pred_path) != lock["prediction_export_sha256"]:
            raise ValueError(f"cycle {cycle} preregistered prediction hash mismatch")
        qc = pd.read_csv(cdir / "experiment" / "experimental_qc_ingested.csv")
        if "locked_evaluation_eligible" not in qc:
            raise ValueError(f"cycle {cycle} QC eligibility is missing")
    cost = pd.read_csv(root / "experiment_cost_ledger.csv")
    if (cost["attempted_cost"] < cost["evaluable_cost"]).any():
        raise ValueError("evaluable cost cannot exceed attempted cost")
    if (cost.loc[cost["status"] == "failed_qc", "failed_assay_cost"] <= 0).any():
        raise ValueError("failed assays must remain in cost accounting")
    comparison = pd.read_csv(root / "strategy_comparison.csv")
    if comparison["metric"].nunique() != 4:
        raise ValueError("strategy comparison must contain all four exit metrics")
    if require_gate and gate.get("software_gate") != "PASS":
        raise ValueError(f"prospective software gate did not pass: {gate}")
    return {
        "valid": True,
        "version": PROSPECTIVE_VERSION,
        "completed_cycles": int(posterior["cycle"].max()),
        "model_freezes": 3,
        "prediction_locks_verified": 3,
        "failed_assays_accounted": int((cost["status"] == "failed_qc").sum()),
        "software_gate": gate.get("software_gate"),
        "passing_metrics": gate.get("passing_metrics", []),
        "real_prospective_claim_authorized": bool(gate.get("real_prospective_claim_authorized", False)),
    }


def ingest_external_cycle(
    cycle_dir: str | Path,
    qc_csv: str | Path,
    outcomes_csv: str | Path,
) -> dict[str, Any]:
    """Ingest external QC/outcome files against an already locked cycle contract.

    This helper deliberately does not retrain or re-rank. It validates that the
    experiment IDs belong to the frozen contract and copies immutable source
    inputs into an ``external_ingest`` folder with SHA-256 provenance.
    """
    cycle_dir = Path(cycle_dir)
    contract_path = cycle_dir / "experiment" / "lims_eln_experiment_contract.csv"
    lock_path = cycle_dir / "preregistration" / "prediction_lock.json"
    if not contract_path.exists() or not lock_path.exists():
        raise ValueError("cycle must be preregistered before outcome ingestion")
    contract = pd.read_csv(contract_path)
    qc = ingest_experimental_qc(pd.read_csv(qc_csv), contract)
    outcomes = pd.read_csv(outcomes_csv)
    required = {"cycle_id", "experiment_id", "sample_id", "outcome_name", "observed_value", "standard_error", "endpoint_role"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"outcome table missing required columns: {sorted(missing)}")
    known = set(contract["experiment_id"].astype(str))
    unknown = sorted(set(outcomes["experiment_id"].astype(str)) - known)
    if unknown:
        raise ValueError(f"outcomes reference experiments outside the locked contract: {unknown}")
    out = ensure_dir(cycle_dir / "external_ingest")
    qc.to_csv(out / "experimental_qc_ingested.csv", index=False)
    outcomes.to_csv(out / "locked_outcomes.csv", index=False)
    manifest = {
        "qc_source": str(qc_csv), "qc_sha256": sha256_file(qc_csv),
        "outcome_source": str(outcomes_csv), "outcome_sha256": sha256_file(outcomes_csv),
        "locked_preregistration_sha256": sha256_file(lock_path),
        "ingested_at_utc": _utc_now(), "outcome_access_after_prediction_lock": True,
    }
    json_dump(manifest, out / "ingest_manifest.json")
    return manifest
