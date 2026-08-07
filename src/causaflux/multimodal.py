from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from anndata import AnnData
    from mudata import MuData, read_h5mu
    USING_SCVERSE_MUDATA = True
except ImportError:  # pragma: no cover - exercised in minimal build environments
    from ._mudata_compat import AnnData, MuData, read_h5mu
    USING_SCVERSE_MUDATA = False
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .causal_data import BIOMARKER_FEATURES, STATE_ORDER
from .uncertainty import expected_calibration_error

MODALITY_ORDER = ("rna", "atac", "protein", "mutation", "drug_response")
OBS_COLUMNS = (
    "donor_id",
    "sample_id",
    "lineage_id",
    "time_hours",
    "cell_type",
    "state",
    "therapy",
    "future_resistant",
)

RNA_FEATURES = (
    "XBP1", "HSPA5", "DNAJB9", "ERN1", "ATF4", "ATF6", "DDIT3", "NCOA3",
    "HLA_A", "B2M", "TAP1", "PSMB8", "NDUFA1", "TFAM", "PPARGC1A", "SOD2",
    "CXCL9", "CXCL10", "IL6", "STAT1", "VIM", "EPCAM", "MKI67", "BCL2",
)
ATAC_FEATURES = (
    "peak_XBP1_enhancer", "peak_HSPA5_promoter", "peak_ATF4_enhancer",
    "peak_NCOA3_switch", "peak_HLA_A_promoter", "peak_B2M_promoter",
    "peak_TAP1_enhancer", "peak_PSMB8_enhancer", "peak_PGC1A_enhancer",
    "peak_TFAM_promoter", "peak_CXCL9_enhancer", "peak_CXCL10_enhancer",
    "peak_IL6_enhancer", "peak_VIM_enhancer", "peak_EPCAM_promoter",
    "peak_MKI67_promoter", "peak_BCL2_enhancer", "peak_stress_super_enhancer",
)
PROTEIN_FEATURES = (
    "XBP1s", "pIRE1a", "BiP", "ATF4", "ATF6N", "SRC3_pS857",
    "MHC_I", "TAP1", "OXPHOS", "pSTAT1", "Vimentin", "Cleaved_Caspase3",
)
MUTATION_FEATURES = (
    "TP53_mut", "KRAS_mut", "PIK3CA_mut", "APC_mut", "BRAF_mut",
    "CNV_gain_8q", "CNV_loss_9p", "mutation_burden",
)
DRUG_FEATURES = (
    "standard_viability", "ire1i_viability", "mitoi_viability", "ifng_viability",
    "standard_resistance", "ire1i_resistance", "mitoi_resistance", "ifng_resistance",
)


@dataclass(frozen=True)
class MultimodalDemoConfig:
    seed: int = 31
    missing_modality_rate: float = 0.03


def _noise(rng: np.random.Generator, n: int, scale: float = 0.05) -> np.ndarray:
    return rng.normal(0.0, scale, size=n)


def _clip(matrix: np.ndarray) -> np.ndarray:
    return np.clip(matrix.astype(np.float32), 0.0, 1.0)


def _matrix(frame: pd.DataFrame, columns: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack(columns)
    if matrix.shape[0] != len(frame):
        raise RuntimeError("Generated modality row count does not match observations")
    return _clip(matrix)


def generate_multimodal_mudata(
    frame: pd.DataFrame,
    config: MultimodalDemoConfig | None = None,
) -> MuData:
    """Create an aligned five-modality MuData object from the synthetic causal frame.

    The generated measurements are deliberately simplified software-test data. They
    encode known cross-modal relationships but are not intended as biological evidence.
    """

    config = config or MultimodalDemoConfig()
    rng = np.random.default_rng(config.seed)
    n = len(frame)
    z = lambda name: frame[name].to_numpy(dtype=float)
    tumor = (frame["cell_type"].astype(str) == "tumor").to_numpy(dtype=float)
    progress = frame["time_hours"].to_numpy(dtype=float)
    progress = progress / max(float(np.nanmax(progress)), 1.0)

    ire1 = z("ire1_xbp1")
    prot = z("proteostasis_capacity")
    enh = z("enhancer_plasticity")
    mito = z("mitochondrial_reserve")
    antigen = z("antigen_presentation")
    exclusion = z("immune_exclusion")
    inflammation = z("inflammatory_signaling")
    viability = z("viability")
    apoptosis = z("apoptosis_signal")
    resistance = z("resistance_score")
    mutation_burden = z("mutation_burden")

    rna = _matrix(
        frame,
        [
            ire1 + _noise(rng, n), prot + _noise(rng, n), prot * 0.88 + _noise(rng, n),
            ire1 * 0.82 + _noise(rng, n), inflammation * 0.72 + _noise(rng, n),
            prot * 0.58 + _noise(rng, n), apoptosis * 0.82 + _noise(rng, n),
            enh * 0.84 + _noise(rng, n), antigen + _noise(rng, n), antigen * 0.92 + _noise(rng, n),
            antigen * 0.90 + _noise(rng, n), antigen * 0.82 + inflammation * 0.12 + _noise(rng, n),
            mito * 0.82 + _noise(rng, n), mito * 0.74 + _noise(rng, n), mito + _noise(rng, n),
            mito * 0.76 + _noise(rng, n), inflammation * antigen + _noise(rng, n),
            inflammation * antigen * 0.90 + _noise(rng, n), inflammation + _noise(rng, n),
            inflammation * 0.78 + antigen * 0.22 + _noise(rng, n),
            enh * 0.62 + resistance * 0.28 + _noise(rng, n),
            tumor * (1.0 - enh * 0.35) + (1.0 - tumor) * 0.12 + _noise(rng, n),
            tumor * np.clip(0.15 + progress * 0.45 + resistance * 0.25, 0, 1) + _noise(rng, n),
            resistance * 0.70 + viability * 0.20 + _noise(rng, n),
        ],
    )

    atac = _matrix(
        frame,
        [
            ire1 + _noise(rng, n, 0.06), prot + _noise(rng, n, 0.06),
            inflammation * 0.68 + _noise(rng, n, 0.06), enh + _noise(rng, n, 0.05),
            antigen + _noise(rng, n, 0.06), antigen * 0.94 + _noise(rng, n, 0.06),
            antigen * 0.87 + _noise(rng, n, 0.06), antigen * 0.79 + _noise(rng, n, 0.06),
            mito + _noise(rng, n, 0.06), mito * 0.84 + _noise(rng, n, 0.06),
            inflammation * antigen + _noise(rng, n, 0.06),
            inflammation * antigen * 0.92 + _noise(rng, n, 0.06),
            inflammation + _noise(rng, n, 0.06), enh * 0.78 + _noise(rng, n, 0.06),
            tumor * (1 - enh * 0.30) + _noise(rng, n, 0.06),
            tumor * np.clip(0.10 + progress * 0.45 + resistance * 0.30, 0, 1) + _noise(rng, n, 0.06),
            resistance * 0.74 + _noise(rng, n, 0.06),
            np.clip(0.28 * ire1 + 0.30 * enh + 0.24 * resistance + 0.18 * progress, 0, 1)
            + _noise(rng, n, 0.05),
        ],
    )

    protein = _matrix(
        frame,
        [
            ire1 + _noise(rng, n, 0.04), ire1 * 0.91 + _noise(rng, n, 0.04),
            prot + _noise(rng, n, 0.04), inflammation * 0.70 + _noise(rng, n, 0.04),
            prot * 0.60 + _noise(rng, n, 0.04), enh + _noise(rng, n, 0.04),
            antigen + _noise(rng, n, 0.04), antigen * 0.88 + _noise(rng, n, 0.04),
            mito + _noise(rng, n, 0.04), inflammation * antigen + _noise(rng, n, 0.04),
            enh * 0.72 + resistance * 0.20 + _noise(rng, n, 0.04),
            apoptosis + _noise(rng, n, 0.04),
        ],
    )

    # Stable lineage-level pseudo-genotypes are generated by thresholding latent burden
    # plus lineage-specific random effects. Non-tumor populations carry zeros.
    lineage_codes = pd.factorize(frame["lineage_id"].astype(str))[0]
    lineage_random = np.random.default_rng(config.seed + 17).random(lineage_codes.max() + 1)
    lr = lineage_random[lineage_codes]
    mutation = _matrix(
        frame,
        [
            tumor * (mutation_burden + lr * 0.35 > 0.62),
            tumor * (mutation_burden * 0.75 + lr * 0.50 > 0.60),
            tumor * (mutation_burden * 0.55 + lr * 0.65 > 0.72),
            tumor * (mutation_burden * 0.62 + lr * 0.42 > 0.70),
            tumor * (mutation_burden * 0.48 + lr * 0.58 > 0.78),
            tumor * (mutation_burden + resistance * 0.25 > 0.68),
            tumor * (mutation_burden + exclusion * 0.22 > 0.72),
            tumor * mutation_burden,
        ],
    )

    standard_v = np.clip(viability - progress * (0.28 - resistance * 0.18), 0, 1)
    ire1_v = np.clip(standard_v - 0.30 * ire1 * progress + 0.04 * (1 - tumor), 0, 1)
    mito_v = np.clip(standard_v - 0.28 * mito * progress + 0.03 * (1 - tumor), 0, 1)
    ifng_v = np.clip(standard_v - 0.22 * exclusion * antigen * progress, 0, 1)
    drug = _matrix(
        frame,
        [
            standard_v + _noise(rng, n, 0.035), ire1_v + _noise(rng, n, 0.035),
            mito_v + _noise(rng, n, 0.035), ifng_v + _noise(rng, n, 0.035),
            resistance + _noise(rng, n, 0.035),
            np.clip(resistance - 0.34 * ire1 * progress, 0, 1) + _noise(rng, n, 0.035),
            np.clip(resistance - 0.31 * mito * progress, 0, 1) + _noise(rng, n, 0.035),
            np.clip(resistance - 0.27 * antigen * (1 - exclusion) * progress, 0, 1)
            + _noise(rng, n, 0.035),
        ],
    )

    matrices: Mapping[str, tuple[np.ndarray, Sequence[str], str]] = {
        "rna": (rna, RNA_FEATURES, "expression"),
        "atac": (atac, ATAC_FEATURES, "accessibility"),
        "protein": (protein, PROTEIN_FEATURES, "abundance"),
        "mutation": (mutation, MUTATION_FEATURES, "genomic"),
        "drug_response": (drug, DRUG_FEATURES, "phenotype"),
    }

    obs = frame.loc[:, ["row_id", *OBS_COLUMNS]].copy().set_index("row_id")
    obs.index = obs.index.astype(str)
    modalities: dict[str, AnnData] = {}
    for modality, (values, feature_names, feature_type) in matrices.items():
        modality_obs = pd.DataFrame(index=obs.index.copy())
        var = pd.DataFrame(index=pd.Index(feature_names, name="feature"))
        var["modality"] = modality
        var["feature_type"] = feature_type
        X: np.ndarray | sparse.csr_matrix = values
        if modality in {"atac", "mutation"}:
            X = sparse.csr_matrix(values)
        modalities[modality] = AnnData(X=X, obs=modality_obs, var=var)

    mdata = MuData(modalities)
    mdata.obs = obs.copy()
    mdata.uns["causaflux_schema"] = {
        "framework": "CausaFlux",
        "version": "1.7.0",
        "schema_version": "1.1",
        "modalities": list(MODALITY_ORDER),
        "observation_key": "row_id",
        "synthetic": True,
        "backend": "scverse" if USING_SCVERSE_MUDATA else "causaflux_compat",
    }
    mdata.uns["provenance"] = {
        "generator": "causaflux.multimodal.generate_multimodal_mudata",
        "seed": int(config.seed),
        "notice": "Synthetic software-validation data; not biological evidence.",
    }

    rate = float(config.missing_modality_rate)
    if rate > 0:
        missing_rng = np.random.default_rng(config.seed + 901)
        for modality in MODALITY_ORDER:
            # Never remove all data from a donor and retain RNA as the most complete assay.
            actual_rate = rate * (0.35 if modality == "rna" else 1.0)
            missing = missing_rng.random(n) < actual_rate
            mdata.obs[f"has_{modality}"] = ~missing
            adata = mdata.mod[modality]
            if sparse.issparse(adata.X):
                dense = adata.X.toarray().astype(np.float32)
                dense[missing] = np.nan
                adata.X = dense
            else:
                values = np.asarray(adata.X, dtype=np.float32).copy()
                values[missing] = np.nan
                adata.X = values
    else:
        for modality in MODALITY_ORDER:
            mdata.obs[f"has_{modality}"] = True
    return mdata


def read_multimodal(path: str | Path) -> MuData:
    return read_h5mu(Path(path))


def write_multimodal(mdata: MuData, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mdata.write_h5mu(path)
    return path


def validate_multimodal(mdata: MuData, required_modalities: Sequence[str] = MODALITY_ORDER) -> dict[str, Any]:
    missing = [name for name in required_modalities if name not in mdata.mod]
    if missing:
        raise ValueError(f"MuData is missing required modalities: {missing}")
    if not mdata.obs_names.is_unique:
        raise ValueError("MuData observation names must be unique")
    alignment: dict[str, bool] = {}
    inventory: dict[str, dict[str, Any]] = {}
    for name in required_modalities:
        adata = mdata.mod[name]
        alignment[name] = bool(np.array_equal(adata.obs_names.to_numpy(), mdata.obs_names.to_numpy()))
        matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        inventory[name] = {
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "sparse": bool(sparse.issparse(adata.X)),
            "missing_fraction": float(np.isnan(matrix).mean()),
        }
    missing_obs = sorted(set(OBS_COLUMNS) - set(mdata.obs.columns))
    valid = not missing and not missing_obs and all(alignment.values())
    report = {
        "valid": valid,
        "n_obs": int(mdata.n_obs),
        "n_modalities": int(len(mdata.mod)),
        "required_modalities": list(required_modalities),
        "missing_modalities": missing,
        "missing_obs_columns": missing_obs,
        "aligned_observations": alignment,
        "inventory": inventory,
    }
    if not valid:
        raise ValueError(f"Multimodal validation failed: {report}")
    return report


def modality_inventory(mdata: MuData) -> pd.DataFrame:
    rows = []
    for modality in MODALITY_ORDER:
        adata = mdata.mod[modality]
        matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        rows.append(
            {
                "modality": modality,
                "n_observations": int(adata.n_obs),
                "n_features": int(adata.n_vars),
                "storage": "sparse" if sparse.issparse(adata.X) else "dense",
                "missing_fraction": float(np.isnan(matrix).mean()),
                "available_observation_fraction": float(mdata.obs[f"has_{modality}"].mean())
                if f"has_{modality}" in mdata.obs
                else 1.0,
            }
        )
    return pd.DataFrame(rows)


def feature_manifest(mdata: MuData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for modality in MODALITY_ORDER:
        for feature in mdata.mod[modality].var_names.astype(str):
            rows.append(
                {
                    "modality": modality,
                    "feature": feature,
                    "fused_column": f"{modality}__{feature}",
                    "feature_type": str(mdata.mod[modality].var.loc[feature].get("feature_type", "")),
                }
            )
    return pd.DataFrame(rows)


def modality_feature_frame(mdata: MuData, modalities: Sequence[str] = MODALITY_ORDER) -> pd.DataFrame:
    output = mdata.obs.copy().reset_index(names="row_id")
    for modality in modalities:
        adata = mdata.mod[modality]
        matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        columns = [f"{modality}__{name}" for name in adata.var_names.astype(str)]
        part = pd.DataFrame(matrix, index=mdata.obs_names, columns=columns)
        output = output.merge(part.reset_index(names="row_id"), on="row_id", how="left", validate="one_to_one")
    return output


def write_csv_bundle(mdata: MuData, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mdata.obs.reset_index(names="row_id").to_csv(directory / "obs.csv", index=False)
    for modality in MODALITY_ORDER:
        adata = mdata.mod[modality]
        matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        frame = pd.DataFrame(matrix, columns=adata.var_names.astype(str))
        frame.insert(0, "row_id", mdata.obs_names.astype(str))
        frame.to_csv(directory / f"{modality}.csv", index=False)
    return directory


def read_csv_bundle(directory: str | Path) -> MuData:
    directory = Path(directory)
    obs = pd.read_csv(directory / "obs.csv").set_index("row_id")
    obs.index = obs.index.astype(str)
    modalities: dict[str, AnnData] = {}
    for modality in MODALITY_ORDER:
        frame = pd.read_csv(directory / f"{modality}.csv")
        if "row_id" not in frame:
            raise ValueError(f"{modality}.csv must include row_id")
        frame["row_id"] = frame["row_id"].astype(str)
        frame = frame.set_index("row_id").reindex(obs.index)
        if frame.index.isna().any():
            raise ValueError(f"{modality}.csv could not be aligned to obs.csv")
        var = pd.DataFrame(index=pd.Index(frame.columns.astype(str), name="feature"))
        var["modality"] = modality
        modalities[modality] = AnnData(X=frame.to_numpy(dtype=np.float32), obs=pd.DataFrame(index=obs.index), var=var)
    mdata = MuData(modalities)
    mdata.obs = obs
    mdata.uns["causaflux_schema"] = {
        "framework": "CausaFlux", "version": "1.7.0", "schema_version": "1.1",
        "modalities": list(MODALITY_ORDER), "observation_key": "row_id", "synthetic": False,
    }
    for modality in MODALITY_ORDER:
        mdata.obs[f"has_{modality}"] = ~np.isnan(np.asarray(mdata.mod[modality].X)).all(axis=1)
    validate_multimodal(mdata)
    return mdata


def _ablation_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _splitter(groups: pd.Series, mode: str, n_splits: int):
    if mode == "leave_one_donor_out":
        return LeaveOneGroupOut()
    if mode == "group_kfold":
        return GroupKFold(n_splits=min(max(2, n_splits), groups.nunique()))
    raise ValueError("split mode must be leave_one_donor_out or group_kfold")


def evaluate_modality_ablation(
    integrated: pd.DataFrame,
    split_mode: str = "leave_one_donor_out",
    n_splits: int = 4,
    seed: int = 31,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tumor = integrated.loc[
        (integrated["cell_type"] == "tumor") & integrated["state"].isin(STATE_ORDER)
    ].reset_index(drop=True)
    classes = list(STATE_ORDER)
    labels = tumor["state"].map({state: i for i, state in enumerate(classes)}).to_numpy(dtype=int)
    groups = tumor["donor_id"].astype(str)
    splits = list(_splitter(groups, split_mode, n_splits).split(tumor, labels, groups))
    by_modality = {
        modality: [column for column in integrated if column.startswith(f"{modality}__")]
        for modality in MODALITY_ORDER
    }
    by_modality["fusion"] = [column for columns in by_modality.values() for column in columns]
    metrics: list[dict[str, Any]] = []
    probability_cache: dict[str, np.ndarray] = {}
    for index, (name, features) in enumerate(by_modality.items()):
        probabilities = np.zeros((len(tumor), len(classes)), dtype=float)
        for train, test in splits:
            model = _ablation_model(seed + index)
            model.fit(tumor.iloc[train][features], tumor.iloc[train]["state"])
            raw = model.predict_proba(tumor.iloc[test][features])
            aligned = np.full((len(test), len(classes)), 1e-8, dtype=float)
            for source, label in enumerate(model.classes_):
                aligned[:, classes.index(str(label))] = raw[:, source]
            probabilities[test] = aligned / aligned.sum(axis=1, keepdims=True)
        probability_cache[name] = probabilities
        prediction = probabilities.argmax(axis=1)
        target = np.eye(len(classes))[labels]
        metrics.append(
            {
                "feature_set": name,
                "n_features": len(features),
                "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
                "log_loss": float(log_loss(labels, probabilities, labels=np.arange(len(classes)))),
                "brier_score": float(np.mean(np.sum((probabilities - target) ** 2, axis=1))),
                "expected_calibration_error": expected_calibration_error(probabilities, labels),
            }
        )
    metric_frame = pd.DataFrame(metrics).sort_values("log_loss").reset_index(drop=True)

    full = probability_cache["fusion"]
    full_log_loss = float(log_loss(labels, full, labels=np.arange(len(classes))))
    contributions: list[dict[str, Any]] = []
    all_features = by_modality["fusion"]
    for index, modality in enumerate(MODALITY_ORDER):
        keep = [column for column in all_features if not column.startswith(f"{modality}__")]
        probabilities = np.zeros_like(full)
        for train, test in splits:
            model = _ablation_model(seed + 100 + index)
            model.fit(tumor.iloc[train][keep], tumor.iloc[train]["state"])
            raw = model.predict_proba(tumor.iloc[test][keep])
            aligned = np.full((len(test), len(classes)), 1e-8, dtype=float)
            for source, label in enumerate(model.classes_):
                aligned[:, classes.index(str(label))] = raw[:, source]
            probabilities[test] = aligned / aligned.sum(axis=1, keepdims=True)
        removed_loss = float(log_loss(labels, probabilities, labels=np.arange(len(classes))))
        contributions.append(
            {
                "removed_modality": modality,
                "fusion_log_loss": full_log_loss,
                "without_modality_log_loss": removed_loss,
                "delta_log_loss_when_removed": removed_loss - full_log_loss,
                "interpretation": "positive values indicate useful incremental information",
            }
        )
    return metric_frame, pd.DataFrame(contributions).sort_values(
        "delta_log_loss_when_removed", ascending=False
    ).reset_index(drop=True)


def modality_summary_correlations(mdata: MuData) -> pd.DataFrame:
    summaries: dict[str, np.ndarray] = {}
    for modality in MODALITY_ORDER:
        matrix = mdata.mod[modality].X
        matrix = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        counts = np.sum(np.isfinite(matrix), axis=1)
        sums = np.nansum(matrix, axis=1)
        summaries[modality] = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
    return pd.DataFrame(summaries, index=mdata.obs_names).corr()


def plot_modality_ablation(metrics: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = metrics.sort_values("log_loss", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(ordered["feature_set"], ordered["log_loss"])
    ax.set_xlabel("Donor-held-out multiclass log loss (lower is better)")
    ax.set_title("Modality and early-fusion benchmark")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_correlation_matrix(correlation: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    image = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(correlation)), correlation.index)
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            ax.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cross-modal mean-signal correlations")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path
