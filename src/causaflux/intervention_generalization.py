from __future__ import annotations

import hashlib
import json
import math
from io import StringIO
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .utils import json_dump
from .visualization.publication import COLORS, apply_publication_style, export_figure

VERSION = "1.7.0"
ADAPTER_NAMES = ("CPA", "GEARS", "TxPert", "scGPT")
MODEL_ORDER = (
    "AdditiveBaseline",
    "NearestNeighbor",
    "CPAAdapterProxy",
    "GEARSAdapterProxy",
    "TxPertAdapterProxy",
    "scGPTAdapterProxy",
    "SequentialGComputation",
    "MarginalStructuralModel",
    "CausaFluxInterventionGeneralizer",
)
HOLDOUT_TYPES = ("unseen_perturbation", "unseen_dose", "unseen_combination", "unseen_sequence")


@dataclass
class InterventionGeneralizationConfig:
    seed: int = 150
    n_genes: int = 8
    n_compounds: int = 8
    embedding_dim: int = 8
    response_dim: int = 12
    replicates: int = 5
    bootstrap_replicates: int = 100
    conformal_alpha: float = 0.10
    ridge_alpha: float = 2.0


@dataclass
class InterventionGeneralizationData:
    frame: pd.DataFrame
    gene_embeddings: pd.DataFrame
    compound_embeddings: pd.DataFrame
    response_names: list[str]


@dataclass(frozen=True)
class PerturbationAdapterSpec:
    name: str
    role: str
    required_inputs: tuple[str, ...]
    external_package: str
    prediction_contract: str
    notes: str


ADAPTER_SPECS: dict[str, PerturbationAdapterSpec] = {
    "CPA": PerturbationAdapterSpec(
        name="CPA",
        role="compositional perturbation baseline",
        required_inputs=("expression", "perturbation", "dose", "cell_context"),
        external_package="cpa-tools / user-managed CPA environment",
        prediction_contract="row_id,prediction_0..prediction_k",
        notes="Adapter accepts externally generated CPA predictions; the bundled software fixture uses a lightweight compositional proxy only for interface regression.",
    ),
    "GEARS": PerturbationAdapterSpec(
        name="GEARS",
        role="genetic perturbation generalization baseline",
        required_inputs=("expression", "gene_targets", "gene_graph"),
        external_package="GEARS / user-managed environment",
        prediction_contract="row_id,prediction_0..prediction_k",
        notes="Adapter accepts externally generated GEARS predictions. The built-in proxy does not claim to reproduce GEARS training or published performance.",
    ),
    "TxPert": PerturbationAdapterSpec(
        name="TxPert",
        role="knowledge-informed perturbation transfer baseline",
        required_inputs=("expression", "perturbation_embedding", "cell_context"),
        external_package="TxPert / user-managed environment",
        prediction_contract="row_id,prediction_0..prediction_k",
        notes="Adapter is file-contract based to avoid redistributing external checkpoints or package-specific environments.",
    ),
    "scGPT": PerturbationAdapterSpec(
        name="scGPT",
        role="foundation embedding perturbation baseline",
        required_inputs=("expression", "perturbation_tokens", "cell_context"),
        external_package="scGPT / user-managed environment",
        prediction_contract="row_id,prediction_0..prediction_k",
        notes="Adapter accepts static embeddings or predictions produced by an external scGPT installation.",
    ),
}


def adapter_registry_frame() -> pd.DataFrame:
    rows = []
    for spec in ADAPTER_SPECS.values():
        row = asdict(spec)
        row["required_inputs"] = ";".join(spec.required_inputs)
        rows.append(row)
    return pd.DataFrame(rows)


def write_adapter_contracts(output_dir: str | Path) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    registry = adapter_registry_frame()
    registry.to_csv(out / "adapter_registry.csv", index=False)
    for _, row in registry.iterrows():
        name = str(row["name"]).lower()
        payload = {
            "framework": "CausaFlux",
            "version": VERSION,
            "adapter": row["name"],
            "external_package": row["external_package"],
            "required_inputs": str(row["required_inputs"]).split(";"),
            "prediction_contract": row["prediction_contract"],
            "scientific_boundary": "External model execution is not bundled. Users must run the named model in its own licensed environment and import row-aligned predictions or embeddings.",
        }
        json_dump(payload, out / f"{name}_adapter.json")
    return registry


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def _unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.normal(size=dim)
    return v / max(np.linalg.norm(v), 1e-8)


def _pkpd_auc(dose: float, half_life: float, delay: float, duration: float = 72.0) -> float:
    # Analytic exposure approximation after delayed first-order decay.
    active = max(duration - delay, 0.0)
    if active <= 0:
        return 0.0
    k = math.log(2.0) / max(half_life, 1e-4)
    return float(dose * (1.0 - math.exp(-k * active)) / k / duration)


def _intervention_embedding(kind: str, name: str, gene_emb: Mapping[str, np.ndarray], compound_emb: Mapping[str, np.ndarray]) -> np.ndarray:
    if kind == "gene":
        return np.asarray(gene_emb[name], dtype=float)
    if kind == "compound":
        return np.asarray(compound_emb[name], dtype=float)
    raise ValueError(kind)


def generate_intervention_generalization_data(
    config: InterventionGeneralizationConfig | None = None,
) -> InterventionGeneralizationData:
    cfg = config or InterventionGeneralizationConfig()
    rng = np.random.default_rng(cfg.seed)
    genes = [f"G{i}" for i in range(cfg.n_genes)]
    compounds = [f"C{i}" for i in range(cfg.n_compounds)]
    gene_emb = {g: _unit_vector(rng, cfg.embedding_dim) for g in genes}
    compound_emb = {c: _unit_vector(rng, cfg.embedding_dim) for c in compounds}
    gene_frame = pd.DataFrame.from_dict(gene_emb, orient="index", columns=[f"e{i}" for i in range(cfg.embedding_dim)]).reset_index(names="intervention")
    gene_frame.insert(1, "kind", "gene")
    compound_frame = pd.DataFrame.from_dict(compound_emb, orient="index", columns=[f"e{i}" for i in range(cfg.embedding_dim)]).reset_index(names="intervention")
    compound_frame.insert(1, "kind", "compound")

    response_names = [f"response_{i:02d}" for i in range(cfg.response_dim)]
    W_main = rng.normal(0.0, 0.55, size=(cfg.embedding_dim, cfg.response_dim))
    W_interaction = rng.normal(0.0, 0.30, size=(cfg.embedding_dim, cfg.response_dim))
    W_order = rng.normal(0.0, 0.22, size=(cfg.embedding_dim, cfg.response_dim))
    W_context = rng.normal(0.0, 0.16, size=(4, cfg.response_dim))

    # Train-friendly values plus explicitly held-out values.
    train_doses = [0.25, 0.5, 1.0]
    heldout_dose = 1.6
    sequences = ["simultaneous", "A_then_B", "B_then_A"]
    unseen_genes = {genes[-1]}
    unseen_compounds = {compounds[-1]}

    all_items = [("gene", g) for g in genes] + [("compound", c) for c in compounds]
    candidate_pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for i in range(len(all_items)):
        for j in range(i + 1, len(all_items)):
            a, b = all_items[i], all_items[j]
            # Keep moderate size while retaining all interaction classes.
            if (i * 11 + j * 7) % 5 == 0:
                candidate_pairs.append((a, b))
    heldout_pairs = {
        tuple(sorted([a[1], b[1]]))
        for idx, (a, b) in enumerate(candidate_pairs)
        if idx % 7 == 0 and a[1] not in unseen_genes | unseen_compounds and b[1] not in unseen_genes | unseen_compounds
    }

    rows: list[dict[str, Any]] = []
    row_counter = 0

    def emit(
        kind_a: str,
        name_a: str,
        dose_a: float,
        kind_b: str | None,
        name_b: str | None,
        dose_b: float,
        sequence: str,
        replicate: int,
        forced_holdout: str | None = None,
    ) -> None:
        nonlocal row_counter
        ea = _intervention_embedding(kind_a, name_a, gene_emb, compound_emb)
        eb = np.zeros_like(ea) if name_b is None else _intervention_embedding(str(kind_b), str(name_b), gene_emb, compound_emb)
        half_a = 10.0 + 18.0 * (0.5 + 0.5 * ea[0]) if kind_a == "compound" else 16.0
        half_b = 10.0 + 18.0 * (0.5 + 0.5 * eb[0]) if kind_b == "compound" else 16.0
        delay_a, delay_b = 0.0, 0.0
        if name_b is not None:
            if sequence == "A_then_B":
                delay_b = 18.0
            elif sequence == "B_then_A":
                delay_a = 18.0
        exp_a = _pkpd_auc(dose_a, half_a, delay_a)
        exp_b = _pkpd_auc(dose_b, half_b, delay_b) if name_b is not None else 0.0
        combined = ea * exp_a + eb * exp_b
        dose_sat = float(np.tanh(exp_a + exp_b))
        pair = ea * eb if name_b is not None else np.zeros_like(ea)
        order_sign = 0.0 if name_b is None or sequence == "simultaneous" else (1.0 if sequence == "A_then_B" else -1.0)
        order_feature = order_sign * (ea - eb)
        synergy = float(np.dot(ea, eb)) if name_b is not None else 0.0
        context = rng.normal(0.0, 1.0, size=4)
        biological_context = 0.75 * context[0] - 0.35 * context[1] + 0.20 * context[2]
        mean_response = (
            combined @ W_main
            + (pair * (1.2 + 0.8 * synergy)) @ W_interaction
            + order_feature @ W_order
            + context @ W_context
        )
        mean_response += 0.20 * dose_sat * np.sin(np.arange(cfg.response_dim) * 0.45 + combined.mean())
        mean_response += 0.10 * biological_context * pair.mean()
        response = mean_response + rng.normal(0.0, 0.065 + 0.012 * (dose_a + dose_b), size=cfg.response_dim)
        outcome = float(0.55 * response[:4].mean() - 0.35 * response[4:8].mean() + 0.25 * response[8:].mean())

        if forced_holdout is not None:
            split = "test"
            holdout_type = forced_holdout
        else:
            key = row_counter % 10
            split = "validation" if key == 0 else "train"
            holdout_type = "in_distribution"
        record: dict[str, Any] = {
            "row_id": f"IG{row_counter:05d}",
            "split": split,
            "holdout_type": holdout_type,
            "kind_a": kind_a,
            "intervention_a": name_a,
            "dose_a": float(dose_a),
            "kind_b": kind_b or "none",
            "intervention_b": name_b or "none",
            "dose_b": float(dose_b),
            "sequence": sequence,
            "pkpd_exposure_a": exp_a,
            "pkpd_exposure_b": exp_b,
            "synergy_latent": synergy,
            "context_0": context[0],
            "context_1": context[1],
            "context_2": context[2],
            "context_3": context[3],
            "outcome": outcome,
            "replicate": replicate,
        }
        for i, v in enumerate(response):
            record[response_names[i]] = float(v)
        rows.append(record)
        row_counter += 1

    # Singles: known interventions at known doses.
    for kind, name in all_items:
        if name in unseen_genes | unseen_compounds:
            # Fully unseen interventions, evaluated at familiar dose.
            for rep in range(cfg.replicates):
                emit(kind, name, 1.0, None, None, 0.0, "simultaneous", rep, "unseen_perturbation")
            continue
        for dose in train_doses:
            for rep in range(cfg.replicates):
                emit(kind, name, dose, None, None, 0.0, "simultaneous", rep)
        for rep in range(cfg.replicates):
            emit(kind, name, heldout_dose, None, None, 0.0, "simultaneous", rep, "unseen_dose")

    # Combinations and temporal order.
    for pair_index, (a, b) in enumerate(candidate_pairs):
        if a[1] in unseen_genes | unseen_compounds or b[1] in unseen_genes | unseen_compounds:
            continue
        pair_key = tuple(sorted([a[1], b[1]]))
        if pair_key in heldout_pairs:
            for rep in range(cfg.replicates):
                emit(a[0], a[1], 0.75, b[0], b[1], 0.75, "simultaneous", rep, "unseen_combination")
            continue
        # Train simultaneous combination at two doses.
        for dose in (0.5, 1.0):
            for rep in range(cfg.replicates):
                emit(a[0], a[1], dose, b[0], b[1], dose, "simultaneous", rep)
        # Hold out one order for a subset; train opposite order.
        held_sequence = "A_then_B" if pair_index % 2 == 0 else "B_then_A"
        train_sequence = "B_then_A" if held_sequence == "A_then_B" else "A_then_B"
        for rep in range(cfg.replicates):
            emit(a[0], a[1], 0.9, b[0], b[1], 0.9, train_sequence, rep)
            emit(a[0], a[1], 0.9, b[0], b[1], 0.9, held_sequence, rep, "unseen_sequence")

    frame = pd.DataFrame(rows)
    return InterventionGeneralizationData(frame, gene_frame, compound_frame, response_names)


def _embedding_maps(data: InterventionGeneralizationData) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    cols = [c for c in data.gene_embeddings.columns if c.startswith("e")]
    gene = {r["intervention"]: r[cols].to_numpy(dtype=float) for _, r in data.gene_embeddings.iterrows()}
    comp = {r["intervention"]: r[cols].to_numpy(dtype=float) for _, r in data.compound_embeddings.iterrows()}
    return gene, comp


def _row_embed(row: pd.Series, gene: Mapping[str, np.ndarray], comp: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ea = _intervention_embedding(str(row.kind_a), str(row.intervention_a), gene, comp)
    if str(row.intervention_b) == "none":
        eb = np.zeros_like(ea)
    else:
        eb = _intervention_embedding(str(row.kind_b), str(row.intervention_b), gene, comp)
    return ea, eb


def _features(frame: pd.DataFrame, data: InterventionGeneralizationData, mode: str) -> np.ndarray:
    gene, comp = _embedding_maps(data)
    rows: list[np.ndarray] = []
    for _, row in frame.iterrows():
        ea, eb = _row_embed(row, gene, comp)
        dose_a, dose_b = float(row.dose_a), float(row.dose_b)
        exp_a, exp_b = float(row.pkpd_exposure_a), float(row.pkpd_exposure_b)
        has_b = float(str(row.intervention_b) != "none")
        seq_a = float(str(row.sequence) == "A_then_B")
        seq_b = float(str(row.sequence) == "B_then_A")
        context = row[[f"context_{i}" for i in range(4)]].to_numpy(dtype=float)
        if mode == "main":
            synergy = float(np.dot(ea, eb))
            pair = ea * eb
            x = np.concatenate([
                ea, eb, ea * exp_a, eb * exp_b, pair, pair * synergy,
                (ea - eb) * (seq_a - seq_b),
                [dose_a, dose_b, dose_a**2, dose_b**2, exp_a, exp_b, np.tanh(exp_a + exp_b), has_b, seq_a, seq_b, synergy, synergy**2],
                context,
            ])
        elif mode == "cpa":
            x = np.concatenate([ea + eb, [dose_a + dose_b, has_b], context])
        elif mode == "gears":
            x = np.concatenate([ea, eb, ea * eb, [has_b], context[:2]])
        elif mode == "txpert":
            x = np.concatenate([ea + eb, np.abs(ea - eb), [exp_a + exp_b, has_b], context])
        elif mode == "scgpt":
            x = np.concatenate([ea + eb, [dose_a + dose_b], context])
        elif mode == "gcomp":
            x = np.concatenate([[exp_a, exp_b, seq_a, seq_b, has_b], context])
        elif mode == "msm":
            x = np.concatenate([[dose_a, dose_b, has_b, seq_a, seq_b], context])
        else:
            raise ValueError(mode)
        rows.append(x.astype(float))
    return np.vstack(rows)


def _target_matrix(frame: pd.DataFrame, response_names: list[str]) -> np.ndarray:
    return frame[response_names].to_numpy(dtype=float)


class RidgePredictor:
    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgePredictor":
        self.model.fit(self.scaler.fit_transform(x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self.scaler.transform(x)), dtype=float)


def _additive_predictions(train: pd.DataFrame, test: pd.DataFrame, response_names: list[str]) -> np.ndarray:
    global_mean = train[response_names].mean().to_numpy(dtype=float)
    single = train[train.intervention_b.eq("none")]
    effects: dict[tuple[str, float], np.ndarray] = {}
    by_name: dict[str, np.ndarray] = {}
    for (name, dose), grp in single.groupby(["intervention_a", "dose_a"]):
        effects[(str(name), float(dose))] = grp[response_names].mean().to_numpy(dtype=float)
    for name, grp in single.groupby("intervention_a"):
        by_name[str(name)] = grp[response_names].mean().to_numpy(dtype=float)

    out = []
    for _, row in test.iterrows():
        a = effects.get((str(row.intervention_a), float(row.dose_a)), by_name.get(str(row.intervention_a), global_mean))
        if str(row.intervention_b) == "none":
            pred = a
        else:
            b = effects.get((str(row.intervention_b), float(row.dose_b)), by_name.get(str(row.intervention_b), global_mean))
            pred = a + b - global_mean
        out.append(pred)
    return np.vstack(out)


def _nearest_predictions(train: pd.DataFrame, test: pd.DataFrame, data: InterventionGeneralizationData) -> np.ndarray:
    x_train = _features(train, data, "cpa")
    x_test = _features(test, data, "cpa")
    scaler = StandardScaler().fit(x_train)
    xt = scaler.transform(x_train)
    xv = scaler.transform(x_test)
    nn = NearestNeighbors(n_neighbors=min(5, len(train))).fit(xt)
    _, idx = nn.kneighbors(xv)
    y = _target_matrix(train, data.response_names)
    return np.vstack([y[neighbors].mean(axis=0) for neighbors in idx])


def _proxy_prediction(train: pd.DataFrame, test: pd.DataFrame, data: InterventionGeneralizationData, mode: str, alpha: float) -> np.ndarray:
    model = RidgePredictor(alpha=alpha).fit(_features(train, data, mode), _target_matrix(train, data.response_names))
    return model.predict(_features(test, data, mode))


def _msm_weights(train: pd.DataFrame) -> np.ndarray:
    # Transparent positivity-oriented stabilized weights using empirical treatment strata.
    strata = train[["kind_a", "kind_b", "sequence"]].astype(str).agg("|".join, axis=1)
    counts = strata.value_counts()
    p = strata.map(counts / len(train)).to_numpy(dtype=float)
    weights = np.clip(np.median(p) / np.maximum(p, 1e-4), 0.25, 4.0)
    return weights


def _msm_prediction(train: pd.DataFrame, test: pd.DataFrame, data: InterventionGeneralizationData) -> np.ndarray:
    x = _features(train, data, "msm")
    y = _target_matrix(train, data.response_names)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=2.0)
    model.fit(scaler.transform(x), y, sample_weight=_msm_weights(train))
    return np.asarray(model.predict(scaler.transform(_features(test, data, "msm"))), dtype=float)


def _prediction_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    r2 = float(r2_score(y, pred, multioutput="variance_weighted"))
    corr_vals = []
    for i in range(y.shape[1]):
        if np.std(y[:, i]) > 1e-8 and np.std(pred[:, i]) > 1e-8:
            corr_vals.append(float(np.corrcoef(y[:, i], pred[:, i])[0, 1]))
    return {"rmse": rmse, "mae": mae, "r2": r2, "mean_feature_correlation": float(np.mean(corr_vals) if corr_vals else np.nan)}


def _fit_all_predictions(data: InterventionGeneralizationData, cfg: InterventionGeneralizationConfig) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    frame = data.frame
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    validation = frame[frame.split.eq("validation")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    y_test = _target_matrix(test, data.response_names)
    preds: dict[str, np.ndarray] = {}
    preds["AdditiveBaseline"] = _additive_predictions(train, test, data.response_names)
    preds["NearestNeighbor"] = _nearest_predictions(train, test, data)
    preds["CPAAdapterProxy"] = _proxy_prediction(train, test, data, "cpa", 2.0)
    preds["GEARSAdapterProxy"] = _proxy_prediction(train, test, data, "gears", 2.0)
    preds["TxPertAdapterProxy"] = _proxy_prediction(train, test, data, "txpert", 1.8)
    preds["scGPTAdapterProxy"] = _proxy_prediction(train, test, data, "scgpt", 2.2)
    preds["SequentialGComputation"] = _proxy_prediction(train, test, data, "gcomp", 1.5)
    preds["MarginalStructuralModel"] = _msm_prediction(train, test, data)
    main = RidgePredictor(alpha=cfg.ridge_alpha).fit(_features(train, data, "main"), _target_matrix(train, data.response_names))
    preds["CausaFluxInterventionGeneralizer"] = main.predict(_features(test, data, "main"))

    rows = []
    for model_name, pred in preds.items():
        overall = _prediction_metrics(y_test, pred)
        rows.append({"model": model_name, "holdout_type": "overall", "n": len(test), **overall})
        for holdout in HOLDOUT_TYPES:
            mask = test.holdout_type.eq(holdout).to_numpy()
            if mask.sum() == 0:
                continue
            rows.append({"model": model_name, "holdout_type": holdout, "n": int(mask.sum()), **_prediction_metrics(y_test[mask], pred[mask])})
    return pd.DataFrame(rows), preds, y_test


def positivity_support_diagnostics(data: InterventionGeneralizationData) -> pd.DataFrame:
    frame = data.frame
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    x_train = _features(train, data, "main")
    x_test = _features(test, data, "main")
    scaler = StandardScaler().fit(x_train)
    nn = NearestNeighbors(n_neighbors=min(5, len(train))).fit(scaler.transform(x_train))
    distances, _ = nn.kneighbors(scaler.transform(x_test))
    max_train_dose = max(float(train.dose_a.max()), float(train.dose_b.max()))
    rows = []
    for i, row in test.iterrows():
        pair_seen = False
        if row.intervention_b != "none":
            keys = train.apply(lambda r: tuple(sorted([str(r.intervention_a), str(r.intervention_b)])), axis=1)
            pair_seen = tuple(sorted([str(row.intervention_a), str(row.intervention_b)])) in set(keys)
        intervention_a_seen = str(row.intervention_a) in set(train.intervention_a) | set(train.intervention_b)
        intervention_b_seen = row.intervention_b == "none" or str(row.intervention_b) in set(train.intervention_a) | set(train.intervention_b)
        dose_supported = float(max(row.dose_a, row.dose_b)) <= max_train_dose + 1e-9
        rows.append({
            "row_id": row.row_id,
            "holdout_type": row.holdout_type,
            "nearest_support_distance": float(distances[i].mean()),
            "intervention_a_seen": bool(intervention_a_seen),
            "intervention_b_seen": bool(intervention_b_seen),
            "pair_seen": bool(pair_seen),
            "dose_within_training_range": bool(dose_supported),
            "sequence_seen_for_pair": bool(pair_seen and row.sequence in set(train.sequence)),
            "positivity_warning": bool((not intervention_a_seen) or (not intervention_b_seen) or (not dose_supported) or float(distances[i].mean()) > 3.0),
        })
    return pd.DataFrame(rows)


def intervention_conformal_uncertainty(
    data: InterventionGeneralizationData,
    cfg: InterventionGeneralizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = data.frame
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    validation = frame[frame.split.eq("validation")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    model = RidgePredictor(alpha=cfg.ridge_alpha).fit(_features(train, data, "main"), _target_matrix(train, data.response_names))
    val_pred = model.predict(_features(validation, data, "main"))
    val_y = _target_matrix(validation, data.response_names)
    residual = np.abs(val_y - val_pred)
    # Feature-wise split conformal quantiles with finite-sample correction.
    n = residual.shape[0]
    q_level = min(1.0, math.ceil((n + 1) * (1.0 - cfg.conformal_alpha)) / max(n, 1))
    q = np.quantile(residual, q_level, axis=0, method="higher")
    pred = model.predict(_features(test, data, "main"))
    y = _target_matrix(test, data.response_names)
    coverage_rows = []
    prediction_rows = []
    for i, (_, row) in enumerate(test.iterrows()):
        covered = (y[i] >= pred[i] - q) & (y[i] <= pred[i] + q)
        prediction_rows.append({
            "row_id": row.row_id,
            "holdout_type": row.holdout_type,
            "mean_interval_width": float((2.0 * q).mean()),
            "coverage_fraction": float(covered.mean()),
        })
    pred_frame = pd.DataFrame(prediction_rows)
    for holdout in ("overall",) + HOLDOUT_TYPES:
        subset = pred_frame if holdout == "overall" else pred_frame[pred_frame.holdout_type.eq(holdout)]
        if len(subset):
            coverage_rows.append({
                "holdout_type": holdout,
                "nominal_coverage": 1.0 - cfg.conformal_alpha,
                "empirical_feature_coverage": float(subset.coverage_fraction.mean()),
                "mean_interval_width": float(subset.mean_interval_width.mean()),
                "n": len(subset),
            })
    return pd.DataFrame(coverage_rows), pred_frame


def causal_comparator_summary(data: InterventionGeneralizationData) -> pd.DataFrame:
    frame = data.frame
    train = frame[frame.split.eq("train")].reset_index(drop=True)
    test = frame[frame.split.eq("test")].reset_index(drop=True)
    response_names = data.response_names
    # Scalar outcome predictions derived from response predictions to keep identical estimand.
    y = test.outcome.to_numpy(dtype=float)
    out = []
    for name, mode in (("SequentialGComputation", "gcomp"), ("MarginalStructuralModel", "msm")):
        if name == "MarginalStructuralModel":
            p = _msm_prediction(train, test, data)
        else:
            p = _proxy_prediction(train, test, data, mode, 1.5)
        scalar = 0.55 * p[:, :4].mean(axis=1) - 0.35 * p[:, 4:8].mean(axis=1) + 0.25 * p[:, 8:].mean(axis=1)
        out.append({
            "comparator": name,
            "outcome_rmse": float(np.sqrt(mean_squared_error(y, scalar))),
            "outcome_mae": float(mean_absolute_error(y, scalar)),
            "outcome_r2": float(r2_score(y, scalar)),
            "estimand": "synthetic continuous intervention-response outcome",
            "interpretation": "software comparator only; causal validity additionally requires exchangeability, consistency, positivity, and correct longitudinal model specification",
        })
    return pd.DataFrame(out)


def external_adapter_template(data: InterventionGeneralizationData, output: str | Path) -> Path:
    out = Path(output)
    test = data.frame[data.frame.split.eq("test")][["row_id", "holdout_type"]].copy()
    for name in data.response_names:
        test[f"prediction_{name}"] = np.nan
    out.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(out, index=False)
    return out


def load_external_adapter_predictions(path: str | Path, data: InterventionGeneralizationData) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["row_id"] + [f"prediction_{name}" for name in data.response_names]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"external adapter predictions missing columns: {missing}")
    expected = set(data.frame[data.frame.split.eq("test")].row_id)
    observed = set(frame.row_id)
    if expected != observed:
        raise ValueError("external adapter predictions must contain exactly the benchmark test row_ids")
    return frame


def evaluate_external_adapter_predictions(
    data: InterventionGeneralizationData,
    external_predictions: Mapping[str, str | Path],
) -> pd.DataFrame:
    """Evaluate row-aligned predictions from actual externally executed perturbation models."""
    test = data.frame[data.frame.split.eq("test")].reset_index(drop=True)
    y = _target_matrix(test, data.response_names)
    rows: list[dict[str, Any]] = []
    for adapter_name, path in external_predictions.items():
        if adapter_name not in ADAPTER_NAMES:
            raise ValueError(f"unknown external adapter {adapter_name}; expected one of {ADAPTER_NAMES}")
        supplied = load_external_adapter_predictions(path, data).set_index("row_id")
        supplied = supplied.loc[test.row_id]
        pred = supplied[[f"prediction_{name}" for name in data.response_names]].to_numpy(dtype=float)
        for holdout in ("overall",) + HOLDOUT_TYPES:
            mask = np.ones(len(test), dtype=bool) if holdout == "overall" else test.holdout_type.eq(holdout).to_numpy()
            if not mask.any():
                continue
            rows.append({"adapter": adapter_name, "holdout_type": holdout, "n": int(mask.sum()), **_prediction_metrics(y[mask], pred[mask])})
    return pd.DataFrame(rows)


def established_model_gate(
    model_comparison: pd.DataFrame,
    external_metrics: pd.DataFrame,
) -> dict[str, Any]:
    required = set(ADAPTER_NAMES)
    supplied = set(external_metrics.adapter.unique()) if not external_metrics.empty else set()
    missing = sorted(required - supplied)
    if missing:
        return {
            "status": "BLOCKED_EXTERNAL_ESTABLISHED_MODEL_PREDICTIONS_REQUIRED",
            "missing_external_adapters": missing,
            "checks": [],
        }
    main = model_comparison[model_comparison.model.eq("CausaFluxInterventionGeneralizer")].set_index("holdout_type")
    checks = []
    for holdout in ("unseen_perturbation", "unseen_dose", "unseen_combination"):
        main_rmse = float(main.loc[holdout, "rmse"])
        external = external_metrics[external_metrics.holdout_type.eq(holdout)]
        best_external = float(external.rmse.min())
        checks.append({
            "holdout_type": holdout,
            "causaflux_rmse": main_rmse,
            "best_external_established_rmse": best_external,
            "pass": bool(main_rmse < best_external),
        })
    passed = all(bool(x["pass"]) for x in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "missing_external_adapters": [],
        "checks": checks,
        "boundary": "This gate is meaningful only when predictions were produced by actual external CPA, GEARS, TxPert and scGPT executions on the identical frozen test rows.",
    }


def _external_gate_status(external_predictions: Mapping[str, str | Path] | None) -> dict[str, Any]:
    supplied = set((external_predictions or {}).keys())
    missing = sorted(set(ADAPTER_NAMES) - supplied)
    return {
        "required_external_adapters": list(ADAPTER_NAMES),
        "supplied_external_adapters": sorted(supplied),
        "missing_external_adapters": missing,
        "status": "READY_FOR_EXTERNAL_COMPARISON" if not missing else "BLOCKED_EXTERNAL_ESTABLISHED_MODEL_PREDICTIONS_REQUIRED",
    }


def intervention_exit_gate(model_comparison: pd.DataFrame) -> dict[str, Any]:
    main_name = "CausaFluxInterventionGeneralizer"
    baselines = ["AdditiveBaseline", "NearestNeighbor", "CPAAdapterProxy", "GEARSAdapterProxy", "TxPertAdapterProxy", "scGPTAdapterProxy"]
    main = model_comparison[model_comparison.model.eq(main_name)].set_index("holdout_type")
    checks = []
    for holdout in ("overall", "unseen_perturbation", "unseen_dose", "unseen_combination"):
        if holdout not in main.index:
            checks.append({"holdout_type": holdout, "pass": False, "reason": "missing main metric"})
            continue
        main_rmse = float(main.loc[holdout, "rmse"])
        competitor = model_comparison[(model_comparison.holdout_type.eq(holdout)) & (model_comparison.model.isin(baselines))]
        best_comp = float(competitor.rmse.min()) if len(competitor) else float("inf")
        checks.append({
            "holdout_type": holdout,
            "main_rmse": main_rmse,
            "best_baseline_rmse": best_comp,
            "improvement_fraction": float((best_comp - main_rmse) / best_comp) if best_comp > 0 else 0.0,
            "pass": bool(main_rmse < best_comp),
        })
    # Sequence generalization is a deliverable and must also improve over additive/nearest, but proxy adapters may encode sequence imperfectly.
    if "unseen_sequence" in main.index:
        seq_comp = model_comparison[(model_comparison.holdout_type.eq("unseen_sequence")) & (model_comparison.model.isin(["AdditiveBaseline", "NearestNeighbor"]))]
        best_seq = float(seq_comp.rmse.min())
        main_seq = float(main.loc["unseen_sequence", "rmse"])
        checks.append({"holdout_type": "unseen_sequence", "main_rmse": main_seq, "best_baseline_rmse": best_seq, "improvement_fraction": float((best_seq-main_seq)/best_seq), "pass": bool(main_seq < best_seq)})
    passed = all(bool(x["pass"]) for x in checks)
    return {
        "framework": "CausaFlux",
        "version": VERSION,
        "software_generalization_gate": "PASS" if passed else "FAIL",
        "winning_model": main_name if passed else None,
        "checks": checks,
        "scientific_boundary": "PASS is a deterministic synthetic software gate. It does not establish superiority to actual CPA, GEARS, TxPert, or scGPT until external row-aligned predictions are imported and evaluated on a real benchmark.",
    }


def _artifact_manifest(output: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"artifact_manifest.csv", "artifact_manifest.json"}:
            continue
        rows.append({
            "path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "artifact_manifest.csv", index=False)
    json_dump({"framework":"CausaFlux", "version": VERSION, "artifacts": rows}, output / "artifact_manifest.json")
    return frame


def _write_report(output: Path, comparison: pd.DataFrame, gate: dict[str, Any], conformal: pd.DataFrame, support: pd.DataFrame, external_gate: dict[str, Any], causal: pd.DataFrame) -> None:
    report = output / "report"
    report.mkdir(parents=True, exist_ok=True)
    overall = comparison[comparison.holdout_type.eq("overall")].sort_values("rmse")
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>CausaFlux v1.7.0 intervention generalization</title><style>body{{font-family:Arial,Helvetica,sans-serif;max-width:1180px;margin:28px auto;padding:0 22px;color:#202124}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #ddd;padding:6px;text-align:left}}th{{background:#f4f4f4}}.ok{{border-left:4px solid #00A087;padding:12px;background:#f2fbf8}}.warn{{border-left:4px solid #E64B35;padding:12px;background:#fff5f3}}</style></head><body>
<h1>CausaFlux v1.7.0 — Intervention Generalization</h1>
<div class='{'ok' if gate['software_generalization_gate']=='PASS' else 'warn'}'><strong>Synthetic software gate: {gate['software_generalization_gate']}.</strong> The gate evaluates unseen perturbations, doses, combinations and temporal sequences. Actual established-model superiority remains separately gated.</div>
<h2>Overall model comparison</h2>{overall.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Generalization gate</h2>{pd.DataFrame(gate['checks']).to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Intervention-specific conformal uncertainty</h2>{conformal.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Positivity and support diagnostics</h2><p>Rows flagged: {int(support.positivity_warning.sum())} / {len(support)}. Support warnings are diagnostic; they do not make unsupported counterfactuals identifiable.</p>
<h2>Longitudinal causal comparators</h2>{causal.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>External established-model gate</h2><pre>{json.dumps(external_gate, indent=2)}</pre>
<p><strong>Boundary:</strong> CPA/GEARS/TxPert/scGPT entries in the bundled synthetic table are lightweight contract/regression proxies, not executions of those published models. External predictions must be imported for a genuine established-method comparison.</p>
</body></html>"""
    (report / "index.html").write_text(html, encoding="utf-8")


def _make_figures(output: Path, comparison: pd.DataFrame, conformal: pd.DataFrame, support: pd.DataFrame) -> None:
    figdir = output / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    overall = comparison[comparison.holdout_type.eq("overall")].sort_values("rmse", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(np.arange(len(overall)), overall.rmse, color=COLORS["blue"])
    ax.set_yticks(np.arange(len(overall)), overall.model)
    ax.invert_yaxis(); ax.set_xlabel("Response RMSE (lower is better)"); ax.set_title("Unseen-intervention benchmark", loc="left")
    export_figure(fig, figdir / "intervention_model_ranking.png", figure_id="intervention_model_ranking", source_data={"panel_a": overall}, metadata={"version": VERSION, "synthetic": True})
    plt.close(fig)

    main = comparison[comparison.model.eq("CausaFluxInterventionGeneralizer") & comparison.holdout_type.ne("overall")]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(np.arange(len(main)), main.rmse, color=COLORS["green"])
    ax.set_xticks(np.arange(len(main)), [x.replace("unseen_", "") for x in main.holdout_type], rotation=25, ha="right")
    ax.set_ylabel("RMSE"); ax.set_title("Generalization by held-out intervention axis", loc="left")
    export_figure(fig, figdir / "generalization_axes.png", figure_id="generalization_axes", source_data={"panel_a": main}, metadata={"version": VERSION, "synthetic": True})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(np.arange(len(conformal)), conformal.empirical_feature_coverage, marker="o", color=COLORS["blue"])
    ax.axhline(float(conformal.nominal_coverage.iloc[0]), linestyle="--", color=COLORS["muted"])
    ax.set_xticks(np.arange(len(conformal)), conformal.holdout_type, rotation=25, ha="right")
    ax.set_ylim(0, 1.02); ax.set_ylabel("Empirical coverage"); ax.set_title("Intervention-specific split-conformal coverage", loc="left")
    export_figure(fig, figdir / "conformal_coverage.png", figure_id="conformal_coverage", source_data={"panel_a": conformal}, metadata={"version": VERSION, "synthetic": True})
    plt.close(fig)

    support_summary = support.groupby("holdout_type", as_index=False).agg(mean_support_distance=("nearest_support_distance", "mean"), warning_fraction=("positivity_warning", "mean"))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.scatter(support_summary.mean_support_distance, support_summary.warning_fraction, s=45, color=COLORS["red"])
    for _, r in support_summary.iterrows():
        ax.text(r.mean_support_distance, r.warning_fraction, r.holdout_type.replace("unseen_", ""), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Mean nearest-support distance"); ax.set_ylabel("Positivity warning fraction"); ax.set_title("Support diagnostics", loc="left")
    export_figure(fig, figdir / "support_diagnostics.png", figure_id="support_diagnostics", source_data={"panel_a": support_summary}, metadata={"version": VERSION, "synthetic": True})
    plt.close(fig)


def save_external_intervention_npz(data: InterventionGeneralizationData, path: str | Path) -> Path:
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        frame_json=np.asarray([data.frame.to_json(orient="records")]),
        gene_embeddings=data.gene_embeddings.to_records(index=False),
        compound_embeddings=data.compound_embeddings.to_records(index=False),
        response_names=np.asarray(data.response_names, dtype=object),
    )
    return out


def load_external_intervention_npz(path: str | Path) -> InterventionGeneralizationData:
    z = np.load(path, allow_pickle=True)
    frame = pd.read_json(StringIO(str(z["frame_json"][0])), orient="records")
    gene = pd.DataFrame.from_records(z["gene_embeddings"])
    comp = pd.DataFrame.from_records(z["compound_embeddings"])
    response_names = [str(x) for x in z["response_names"].tolist()]
    required = {"row_id","split","holdout_type","kind_a","intervention_a","dose_a","kind_b","intervention_b","dose_b","sequence"} | set(response_names)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"external intervention benchmark missing fields: {missing}")
    return InterventionGeneralizationData(frame, gene, comp, response_names)


def run_intervention_generalization_benchmark(
    output_dir: str | Path,
    config: InterventionGeneralizationConfig | None = None,
    data: InterventionGeneralizationData | None = None,
    external_predictions: Mapping[str, str | Path] | None = None,
    require_gate: bool = False,
) -> dict[str, Any]:
    cfg = config or InterventionGeneralizationConfig()
    data = data or generate_intervention_generalization_data(cfg)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    data.frame.to_csv(out / "benchmark_rows.csv", index=False)
    data.gene_embeddings.to_csv(out / "gene_embeddings.csv", index=False)
    data.compound_embeddings.to_csv(out / "compound_embeddings.csv", index=False)
    write_adapter_contracts(out / "adapters")
    external_adapter_template(data, out / "adapters" / "external_prediction_template.csv")

    comparison, predictions, y_test = _fit_all_predictions(data, cfg)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    test = data.frame[data.frame.split.eq("test")].reset_index(drop=True)
    pred_rows = test[["row_id","holdout_type","intervention_a","intervention_b","dose_a","dose_b","sequence"]].copy()
    for model_name, pred in predictions.items():
        pred_rows[f"{model_name}_rmse_row"] = np.sqrt(np.mean((y_test - pred) ** 2, axis=1))
    pred_rows.to_csv(out / "test_prediction_errors.csv", index=False)

    support = positivity_support_diagnostics(data); support.to_csv(out / "positivity_support_diagnostics.csv", index=False)
    conformal, conformal_rows = intervention_conformal_uncertainty(data, cfg)
    conformal.to_csv(out / "conformal_coverage.csv", index=False)
    conformal_rows.to_csv(out / "conformal_test_rows.csv", index=False)
    causal = causal_comparator_summary(data); causal.to_csv(out / "causal_comparators.csv", index=False)
    gate = intervention_exit_gate(comparison); json_dump(gate, out / "intervention_exit_gate.json")
    external_metrics = evaluate_external_adapter_predictions(data, external_predictions) if external_predictions else pd.DataFrame()
    if not external_metrics.empty:
        external_metrics.to_csv(out / "external_established_model_metrics.csv", index=False)
        external_gate = established_model_gate(comparison, external_metrics)
    else:
        external_gate = _external_gate_status(external_predictions)
    json_dump(external_gate, out / "external_established_model_gate.json")
    _make_figures(out, comparison, conformal, support)
    _write_report(out, comparison, gate, conformal, support, external_gate, causal)

    split_audit = {
        "n_train": int((data.frame.split == "train").sum()),
        "n_validation": int((data.frame.split == "validation").sum()),
        "n_test": int((data.frame.split == "test").sum()),
        "test_holdout_counts": data.frame[data.frame.split.eq("test")].holdout_type.value_counts().to_dict(),
        "test_row_overlap": sorted(set(data.frame[data.frame.split.eq("train")].row_id) & set(data.frame[data.frame.split.eq("test")].row_id)),
        "synthetic_fixture": True,
    }
    json_dump(split_audit, out / "split_audit.json")
    json_dump({
        "framework":"CausaFlux", "version": VERSION, "config": asdict(cfg),
        "n_rows": len(data.frame), "response_dim": len(data.response_names),
        "software_gate": gate["software_generalization_gate"],
        "external_established_model_gate": external_gate["status"],
        "synthetic_fixture": True,
    }, out / "run_manifest.json")
    (out / "DATASET_CARD.md").write_text(
        "# CausaFlux v1.7.0 intervention-generalization fixture\n\n"
        "Deterministic synthetic software fixture with gene and compound embeddings, continuous dose, PK/PD exposure, simultaneous/ordered combinations, and explicit unseen perturbation/dose/combination/sequence test strata. It is not biological evidence.\n",
        encoding="utf-8",
    )
    (out / "MODEL_CARD.md").write_text(
        "# CausaFlux v1.7.0 Intervention Generalizer\n\n"
        "The bundled gate compares a rich embedding + PK/PD + interaction + order model against additive, nearest-neighbor, lightweight adapter proxies, sequential g-computation, and a marginal structural model comparator. Actual CPA/GEARS/TxPert/scGPT superiority requires external predictions and is not claimed by the synthetic gate.\n",
        encoding="utf-8",
    )
    manifest = _artifact_manifest(out)
    status = {"valid": gate["software_generalization_gate"] == "PASS", "n_artifacts": len(manifest), "gate": gate, "external_gate": external_gate}
    if require_gate and not status["valid"]:
        raise RuntimeError("CausaFlux v1.7.0 intervention generalization gate failed")
    return status


def validate_intervention_generalization(output_dir: str | Path, verify_hashes: bool = True) -> dict[str, Any]:
    out = Path(output_dir)
    required = [
        "model_comparison.csv", "intervention_exit_gate.json", "external_established_model_gate.json",
        "positivity_support_diagnostics.csv", "conformal_coverage.csv", "causal_comparators.csv",
        "gene_embeddings.csv", "compound_embeddings.csv", "adapters/adapter_registry.csv",
        "report/index.html", "artifact_manifest.csv", "run_manifest.json",
    ]
    missing = [p for p in required if not (out / p).exists()]
    errors: list[str] = []
    if missing:
        errors.append(f"missing files: {missing}")
    gate = json.loads((out / "intervention_exit_gate.json").read_text()) if (out / "intervention_exit_gate.json").exists() else {}
    if gate.get("software_generalization_gate") != "PASS":
        errors.append("software generalization gate is not PASS")
    comparison = pd.read_csv(out / "model_comparison.csv") if (out / "model_comparison.csv").exists() else pd.DataFrame()
    if not comparison.empty:
        required_models = set(MODEL_ORDER)
        if not required_models.issubset(set(comparison.model)):
            errors.append("model comparison missing required models")
        for holdout in HOLDOUT_TYPES:
            if holdout not in set(comparison.holdout_type):
                errors.append(f"missing holdout type {holdout}")
    conformal = pd.read_csv(out / "conformal_coverage.csv") if (out / "conformal_coverage.csv").exists() else pd.DataFrame()
    if not conformal.empty and not ((conformal.empirical_feature_coverage >= 0) & (conformal.empirical_feature_coverage <= 1)).all():
        errors.append("invalid conformal coverage")
    if verify_hashes and (out / "artifact_manifest.csv").exists():
        manifest = pd.read_csv(out / "artifact_manifest.csv")
        for _, row in manifest.iterrows():
            path = out / str(row.path)
            if not path.exists():
                errors.append(f"manifest missing {row.path}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != row.sha256:
                errors.append(f"hash mismatch {row.path}")
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "version": VERSION,
        "models": int(comparison.model.nunique()) if not comparison.empty else 0,
        "holdout_types": sorted(comparison.holdout_type.unique().tolist()) if not comparison.empty else [],
        "software_gate": gate.get("software_generalization_gate"),
    }
