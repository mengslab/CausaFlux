from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .causal_data import BIOMARKER_FEATURES, CELL_TYPES


COMPARTMENT_MAP = {
    "tumor": "tumor",
    "macrophage": "immune",
    "dendritic_cell": "immune",
    "t_cell": "immune",
    "fibroblast": "stromal",
    "vascular": "vascular",
}

NICHE_ORDER = [
    "tumor_core",
    "immune_infiltrated",
    "macrophage_barrier",
    "stromal_perivascular",
    "mixed_interface",
]

CELL_TYPE_COLORS = {
    "tumor": "#9D3E48",
    "macrophage": "#D48732",
    "dendritic_cell": "#7D66B3",
    "t_cell": "#3B7FB6",
    "fibroblast": "#6B8E4E",
    "vascular": "#B6578D",
}


@dataclass(frozen=True)
class SpatialGraphConfig:
    seed: int = 31
    width: float = 1000.0
    height: float = 1000.0
    n_tumor_nests: int = 3
    k_neighbors: int = 8
    max_distance: float = 230.0
    neighborhood_radius: float = 180.0
    communication_radius: float = 190.0
    min_circuit_edges: int = 3
    bootstrap: int = 50
    export_graphml: bool = True


@dataclass(frozen=True)
class SpatialGraphResult:
    frame: pd.DataFrame
    nodes: pd.DataFrame
    spatial_edges: pd.DataFrame
    communication_edges: pd.DataFrame
    circuits: pd.DataFrame
    niche_summary: pd.DataFrame
    contact_enrichment: pd.DataFrame
    ligand_receptor_catalog: pd.DataFrame
    graph: nx.MultiDiGraph
    qc: dict[str, Any]


LIGAND_RECEPTOR_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "sender": "tumor",
        "receiver": "t_cell",
        "ligand": "CD274",
        "receptor": "PDCD1",
        "family": "checkpoint",
        "effect": "suppressive",
        "sender_feature": "immune_exclusion",
        "receiver_feature": "inflammatory_signaling",
    },
    {
        "sender": "tumor",
        "receiver": "macrophage",
        "ligand": "CSF1",
        "receptor": "CSF1R",
        "family": "myeloid_recruitment",
        "effect": "protective_niche",
        "sender_feature": "inflammatory_signaling",
        "receiver_feature": "viability",
    },
    {
        "sender": "macrophage",
        "receiver": "t_cell",
        "ligand": "IL10",
        "receptor": "IL10RA",
        "family": "immune_suppression",
        "effect": "suppressive",
        "sender_feature": "inflammatory_signaling",
        "receiver_feature": "immune_exclusion",
    },
    {
        "sender": "macrophage",
        "receiver": "dendritic_cell",
        "ligand": "IL10",
        "receptor": "IL10RA",
        "family": "antigen_suppression",
        "effect": "suppressive",
        "sender_feature": "inflammatory_signaling",
        "receiver_feature": "immune_exclusion",
    },
    {
        "sender": "macrophage",
        "receiver": "fibroblast",
        "ligand": "TGFB1",
        "receptor": "TGFBR2",
        "family": "stromal_remodeling",
        "effect": "protective_niche",
        "sender_feature": "immune_exclusion",
        "receiver_feature": "enhancer_plasticity",
    },
    {
        "sender": "fibroblast",
        "receiver": "t_cell",
        "ligand": "CXCL12",
        "receptor": "CXCR4",
        "family": "immune_exclusion",
        "effect": "suppressive",
        "sender_feature": "immune_exclusion",
        "receiver_feature": "viability",
    },
    {
        "sender": "fibroblast",
        "receiver": "tumor",
        "ligand": "HGF",
        "receptor": "MET",
        "family": "stromal_survival",
        "effect": "protective_niche",
        "sender_feature": "proteostasis_capacity",
        "receiver_feature": "mitochondrial_reserve",
    },
    {
        "sender": "dendritic_cell",
        "receiver": "t_cell",
        "ligand": "CXCL9",
        "receptor": "CXCR3",
        "family": "immune_recruitment",
        "effect": "activating",
        "sender_feature": "antigen_presentation",
        "receiver_feature": "antigen_presentation",
    },
    {
        "sender": "dendritic_cell",
        "receiver": "t_cell",
        "ligand": "HLA_I",
        "receptor": "TCR",
        "family": "antigen_presentation",
        "effect": "activating",
        "sender_feature": "antigen_presentation",
        "receiver_feature": "viability",
    },
    {
        "sender": "tumor",
        "receiver": "t_cell",
        "ligand": "HLA_I",
        "receptor": "TCR",
        "family": "tumor_recognition",
        "effect": "activating",
        "sender_feature": "antigen_presentation",
        "receiver_feature": "viability",
    },
    {
        "sender": "t_cell",
        "receiver": "tumor",
        "ligand": "IFNG",
        "receptor": "IFNGR1",
        "family": "antigen_rescue",
        "effect": "activating",
        "sender_feature": "inflammatory_signaling",
        "receiver_feature": "antigen_presentation",
    },
    {
        "sender": "tumor",
        "receiver": "vascular",
        "ligand": "VEGFA",
        "receptor": "KDR",
        "family": "angiogenesis",
        "effect": "protective_niche",
        "sender_feature": "mitochondrial_reserve",
        "receiver_feature": "viability",
    },
    {
        "sender": "vascular",
        "receiver": "tumor",
        "ligand": "ANGPT2",
        "receptor": "TEK",
        "family": "vascular_support",
        "effect": "protective_niche",
        "sender_feature": "inflammatory_signaling",
        "receiver_feature": "viability",
    },
)


def ligand_receptor_catalog() -> pd.DataFrame:
    return pd.DataFrame(LIGAND_RECEPTOR_CATALOG).copy()


def _stable_code(value: str) -> int:
    # Python's hash is intentionally process-randomized, so use a small stable hash.
    code = 0
    for char in str(value):
        code = (code * 131 + ord(char)) % 2_147_483_647
    return code


def _clip_xy(x: np.ndarray, y: np.ndarray, config: SpatialGraphConfig) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.clip(x, 5.0, config.width - 5.0),
        np.clip(y, 5.0, config.height - 5.0),
    )


def generate_spatial_coordinates(
    frame: pd.DataFrame,
    config: SpatialGraphConfig | None = None,
) -> pd.DataFrame:
    """Generate deterministic synthetic spatial coordinates for software validation.

    Tumor cells form nests. Myeloid and stromal cells accumulate at nest interfaces,
    T-cell penetration decreases as sample-level immune exclusion rises, dendritic
    cells occupy immune interfaces, and vascular cells follow synthetic vessel axes.
    """

    config = config or SpatialGraphConfig()
    required = {"row_id", "sample_id", "lineage_id", "cell_type", "immune_exclusion", "state"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Spatial coordinate generation requires columns: {missing}")

    output_parts: list[pd.DataFrame] = []
    for sample_index, (sample_id, group) in enumerate(frame.groupby("sample_id", sort=True)):
        group = group.copy()
        rng = np.random.default_rng(config.seed + 1009 * (sample_index + 1) + _stable_code(sample_id) % 997)
        n_nests = max(1, min(config.n_tumor_nests, int(max(1, (group["cell_type"] == "tumor").sum() // 5))))
        margins = 180.0
        centers = np.column_stack(
            [
                rng.uniform(margins, config.width - margins, size=n_nests),
                rng.uniform(margins, config.height - margins, size=n_nests),
            ]
        )
        sample_exclusion = float(group["immune_exclusion"].mean())
        sample_antigen = float(group.get("antigen_presentation", pd.Series([0.5])).mean())
        x = np.zeros(len(group), dtype=float)
        y = np.zeros(len(group), dtype=float)
        generation_zone: list[str] = []

        for local_index, (_, row) in enumerate(group.iterrows()):
            cell_type = str(row["cell_type"])
            nest = _stable_code(str(row["lineage_id"])) % n_nests
            center_x, center_y = centers[nest]
            angle = rng.uniform(0, 2 * math.pi)
            state = str(row.get("state", "context"))
            if cell_type == "tumor":
                state_scale = {
                    "treatment_sensitive": 78.0,
                    "early_stress": 72.0,
                    "reversible_tolerance": 64.0,
                    "stable_resistance": 56.0,
                }.get(state, 70.0)
                radial = abs(rng.normal(0.0, state_scale))
                x[local_index] = center_x + radial * math.cos(angle) + rng.normal(0, 12)
                y[local_index] = center_y + radial * math.sin(angle) + rng.normal(0, 12)
                generation_zone.append("tumor_nest")
            elif cell_type == "macrophage":
                radial = rng.normal(135 + 42 * sample_exclusion, 34)
                x[local_index] = center_x + radial * math.cos(angle)
                y[local_index] = center_y + radial * math.sin(angle)
                generation_zone.append("myeloid_interface")
            elif cell_type == "dendritic_cell":
                radial = rng.normal(128 + 35 * (1 - sample_antigen), 42)
                x[local_index] = center_x + radial * math.cos(angle)
                y[local_index] = center_y + radial * math.sin(angle)
                generation_zone.append("immune_interface")
            elif cell_type == "t_cell":
                # High exclusion pushes T cells toward the external margin.
                radial = rng.normal(98 + 205 * sample_exclusion, 55)
                x[local_index] = center_x + radial * math.cos(angle)
                y[local_index] = center_y + radial * math.sin(angle)
                generation_zone.append("immune_margin" if sample_exclusion > 0.45 else "immune_infiltrate")
            elif cell_type == "fibroblast":
                radial = rng.normal(185 + 55 * sample_exclusion, 48)
                x[local_index] = center_x + radial * math.cos(angle)
                y[local_index] = center_y + radial * math.sin(angle)
                generation_zone.append("stromal_ring")
            elif cell_type == "vascular":
                vessel_axis = rng.choice([0, 1])
                intercept = (centers[:, 1].mean() if vessel_axis == 0 else centers[:, 0].mean()) + rng.normal(0, 85)
                if vessel_axis == 0:
                    x[local_index] = rng.uniform(40, config.width - 40)
                    y[local_index] = intercept + 0.18 * (x[local_index] - config.width / 2) + rng.normal(0, 22)
                else:
                    y[local_index] = rng.uniform(40, config.height - 40)
                    x[local_index] = intercept + 0.18 * (y[local_index] - config.height / 2) + rng.normal(0, 22)
                generation_zone.append("vascular_track")
            else:
                x[local_index] = rng.uniform(0, config.width)
                y[local_index] = rng.uniform(0, config.height)
                generation_zone.append("unassigned")

        x, y = _clip_xy(x, y, config)
        group["spatial_x"] = x
        group["spatial_y"] = y
        group["spatial_generation_zone"] = generation_zone
        group["spatial_sample_index"] = sample_index
        output_parts.append(group)
    return pd.concat(output_parts, ignore_index=True).sort_values("row_id").reset_index(drop=True)


def attach_spatial_to_mudata(mdata: Any, frame: pd.DataFrame) -> Any:
    indexed = frame.set_index("row_id")
    aligned = indexed.reindex(mdata.obs_names.astype(str))
    if aligned[["spatial_x", "spatial_y"]].isna().any().any():
        raise ValueError("Cannot attach spatial coordinates: MuData and spatial table are not aligned")
    for column in ["spatial_x", "spatial_y", "spatial_generation_zone", "spatial_sample_index"]:
        mdata.obs[column] = aligned[column].to_numpy()
    if not hasattr(mdata, "obsm"):
        mdata.obsm = {}
    mdata.obsm["spatial"] = aligned[["spatial_x", "spatial_y"]].to_numpy(dtype=np.float32)
    if not hasattr(mdata, "uns"):
        mdata.uns = {}
    mdata.uns["spatial_graph"] = {
        "coordinate_key": "spatial",
        "coordinate_columns": ["spatial_x", "spatial_y"],
        "units": "synthetic_microns",
        "notice": "Synthetic spatial coordinates for software validation only.",
    }
    return mdata


def _sample_spatial_edges(group: pd.DataFrame, config: SpatialGraphConfig) -> pd.DataFrame:
    if len(group) < 2:
        return pd.DataFrame()
    coordinates = group[["spatial_x", "spatial_y"]].to_numpy(dtype=float)
    n_neighbors = min(config.k_neighbors + 1, len(group))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(coordinates)
    distances, indices = model.kneighbors(coordinates)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    rows = group.reset_index(drop=True)
    row_records = {str(row["row_id"]): row for row in rows.to_dict("records")}
    for source_index in range(len(rows)):
        for distance, target_index in zip(distances[source_index, 1:], indices[source_index, 1:]):
            if distance > config.max_distance:
                continue
            source_id = str(rows.iloc[source_index]["row_id"])
            target_id = str(rows.iloc[target_index]["row_id"])
            left, right = sorted([source_id, target_id])
            key = (left, right)
            if key in records and records[key]["distance"] <= float(distance):
                continue
            source_row = row_records[left]
            target_row = row_records[right]
            records[key] = {
                "source": left,
                "target": right,
                "sample_id": str(source_row["sample_id"]),
                "donor_id": str(source_row["donor_id"]),
                "therapy": str(source_row["therapy"]),
                "time_hours": float(source_row["time_hours"]),
                "source_cell_type": str(source_row["cell_type"]),
                "target_cell_type": str(target_row["cell_type"]),
                "distance": float(distance),
                "spatial_weight": float(math.exp(-float(distance) / max(config.max_distance, 1e-8))),
                "edge_type": "spatial_proximity",
            }
    return pd.DataFrame(records.values())


def build_spatial_edges(frame: pd.DataFrame, config: SpatialGraphConfig) -> pd.DataFrame:
    pieces = [_sample_spatial_edges(group, config) for _, group in frame.groupby("sample_id", sort=True)]
    pieces = [part for part in pieces if not part.empty]
    if not pieces:
        return pd.DataFrame(
            columns=[
                "source", "target", "sample_id", "donor_id", "therapy", "time_hours",
                "source_cell_type", "target_cell_type", "distance", "spatial_weight", "edge_type",
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def _neighbor_composition(frame: pd.DataFrame, spatial_edges: pd.DataFrame, config: SpatialGraphConfig) -> pd.DataFrame:
    node_types = dict(zip(frame["row_id"].astype(str), frame["cell_type"].astype(str)))
    adjacency: dict[str, list[tuple[str, float]]] = {str(row_id): [] for row_id in frame["row_id"]}
    for edge in spatial_edges.itertuples(index=False):
        adjacency[str(edge.source)].append((str(edge.target), float(edge.distance)))
        adjacency[str(edge.target)].append((str(edge.source), float(edge.distance)))
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        node_id = str(row.row_id)
        neighbors = [target for target, distance in adjacency[node_id] if distance <= config.neighborhood_radius]
        types = [node_types[target] for target in neighbors]
        fractions = {
            f"neighbor_fraction_{cell_type}": float(sum(value == cell_type for value in types) / len(types)) if types else 0.0
            for cell_type in CELL_TYPES
        }
        tumor_fraction = fractions["neighbor_fraction_tumor"]
        immune_fraction = sum(fractions[f"neighbor_fraction_{name}"] for name in ["macrophage", "dendritic_cell", "t_cell"])
        macrophage_stroma = fractions["neighbor_fraction_macrophage"] + fractions["neighbor_fraction_fibroblast"]
        perivascular = fractions["neighbor_fraction_fibroblast"] + fractions["neighbor_fraction_vascular"]
        if tumor_fraction >= 0.55:
            niche = "tumor_core"
        elif tumor_fraction >= 0.15 and (fractions["neighbor_fraction_t_cell"] + fractions["neighbor_fraction_dendritic_cell"]) >= 0.24:
            niche = "immune_infiltrated"
        elif tumor_fraction >= 0.12 and macrophage_stroma >= 0.34:
            niche = "macrophage_barrier"
        elif perivascular >= 0.36:
            niche = "stromal_perivascular"
        else:
            niche = "mixed_interface"
        records.append(
            {
                "row_id": node_id,
                "neighbor_count": int(len(neighbors)),
                "neighbor_tumor_fraction": tumor_fraction,
                "neighbor_immune_fraction": immune_fraction,
                "spatial_niche": niche,
                **fractions,
            }
        )
    return pd.DataFrame(records)


def _value(row: Mapping[str, Any], feature: str) -> float:
    value = float(row.get(feature, 0.0))
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def infer_communication_edges(
    nodes: pd.DataFrame,
    spatial_edges: pd.DataFrame,
    config: SpatialGraphConfig,
    catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    catalog = ligand_receptor_catalog() if catalog is None else catalog.copy()
    node_index = nodes.set_index("row_id").to_dict("index")
    rules: dict[tuple[str, str], list[pd.Series]] = {}
    for _, rule in catalog.iterrows():
        rules.setdefault((str(rule["sender"]), str(rule["receiver"])), []).append(rule)
    records: list[dict[str, Any]] = []
    for edge in spatial_edges.itertuples(index=False):
        if float(edge.distance) > config.communication_radius:
            continue
        pair = [(str(edge.source), str(edge.target)), (str(edge.target), str(edge.source))]
        for source_id, target_id in pair:
            source = node_index[source_id]
            target = node_index[target_id]
            key = (str(source["cell_type"]), str(target["cell_type"]))
            for rule in rules.get(key, []):
                sender_signal = _value(source, str(rule["sender_feature"]))
                receiver_signal = _value(target, str(rule["receiver_feature"]))
                proximity = math.exp(-float(edge.distance) / max(config.communication_radius, 1e-8))
                niche_factor = 1.0
                if str(rule["effect"]) in {"suppressive", "protective_niche"} and str(source.get("spatial_niche", "")) == "macrophage_barrier":
                    niche_factor = 1.12
                if str(rule["effect"]) == "activating" and str(target.get("spatial_niche", "")) == "immune_infiltrated":
                    niche_factor = 1.10
                score = float(np.clip(math.sqrt(sender_signal * receiver_signal) * proximity * niche_factor, 0.0, 1.5))
                records.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "sample_id": str(source["sample_id"]),
                        "donor_id": str(source["donor_id"]),
                        "therapy": str(source["therapy"]),
                        "time_hours": float(source["time_hours"]),
                        "source_cell_type": str(source["cell_type"]),
                        "target_cell_type": str(target["cell_type"]),
                        "source_niche": str(source.get("spatial_niche", "")),
                        "target_niche": str(target.get("spatial_niche", "")),
                        "ligand": str(rule["ligand"]),
                        "receptor": str(rule["receptor"]),
                        "family": str(rule["family"]),
                        "effect": str(rule["effect"]),
                        "distance": float(edge.distance),
                        "ligand_activity": sender_signal,
                        "receptor_activity": receiver_signal,
                        "communication_score": score,
                        "edge_type": "ligand_receptor",
                    }
                )
    return pd.DataFrame(records)


def _bootstrap_circuit_interval(values_by_donor: pd.Series, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float]:
    values = values_by_donor.to_numpy(dtype=float)
    mean = float(np.mean(values)) if len(values) else 0.0
    if len(values) <= 1 or n_bootstrap <= 0:
        return mean, mean, mean
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.choice(values, size=len(values), replace=True)
        estimates[index] = float(np.mean(sampled))
    low = float(np.quantile(estimates, 0.025))
    high = float(np.quantile(estimates, 0.975))
    return mean, min(mean, low), max(mean, high)


def summarize_communication_circuits(
    communication_edges: pd.DataFrame,
    config: SpatialGraphConfig,
) -> pd.DataFrame:
    if communication_edges.empty:
        return pd.DataFrame()
    keys = ["source_cell_type", "target_cell_type", "ligand", "receptor", "family", "effect"]
    sample_summary = (
        communication_edges.groupby(["donor_id", "sample_id", *keys], as_index=False)
        .agg(
            sample_mean_score=("communication_score", "mean"),
            sample_total_score=("communication_score", "sum"),
            supporting_edges=("communication_score", "size"),
            mean_distance=("distance", "mean"),
        )
    )
    donor_summary = (
        sample_summary.groupby(["donor_id", *keys], as_index=False)
        .agg(
            donor_mean_score=("sample_mean_score", "mean"),
            donor_total_score=("sample_total_score", "sum"),
            donor_supporting_edges=("supporting_edges", "sum"),
            donor_mean_distance=("mean_distance", "mean"),
        )
    )
    all_donors = max(1, communication_edges["donor_id"].nunique())
    rng = np.random.default_rng(config.seed + 404)
    records: list[dict[str, Any]] = []
    for key_values, group in donor_summary.groupby(keys, sort=False):
        key_dict = dict(zip(keys, key_values))
        total_edges = int(group["donor_supporting_edges"].sum())
        if total_edges < config.min_circuit_edges:
            continue
        mean, ci_low, ci_high = _bootstrap_circuit_interval(group.set_index("donor_id")["donor_mean_score"], config.bootstrap, rng)
        donor_support = int(group["donor_id"].nunique())
        total_score = float(group["donor_total_score"].sum())
        circuit_score = float(mean * math.log1p(total_edges) * (donor_support / all_donors))
        records.append(
            {
                **key_dict,
                "circuit": f"{key_dict['source_cell_type']}:{key_dict['ligand']}→{key_dict['target_cell_type']}:{key_dict['receptor']}",
                "mean_communication_score": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "total_communication_score": total_score,
                "supporting_edges": total_edges,
                "donor_support": donor_support,
                "donor_support_fraction": donor_support / all_donors,
                "mean_distance": float(group["donor_mean_distance"].mean()),
                "circuit_score": circuit_score,
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["circuit_score", "supporting_edges"], ascending=False).reset_index(drop=True)


def summarize_niches(nodes: pd.DataFrame) -> pd.DataFrame:
    summary = (
        nodes.groupby(["spatial_niche", "cell_type"], as_index=False)
        .agg(
            n_cells=("row_id", "size"),
            n_donors=("donor_id", "nunique"),
            mean_resistance=("resistance_score", "mean"),
            mean_immune_exclusion=("immune_exclusion", "mean"),
            mean_antigen_presentation=("antigen_presentation", "mean"),
        )
    )
    totals = summary.groupby("spatial_niche")["n_cells"].transform("sum")
    summary["within_niche_fraction"] = summary["n_cells"] / totals
    summary["spatial_niche"] = pd.Categorical(summary["spatial_niche"], categories=NICHE_ORDER, ordered=True)
    return summary.sort_values(["spatial_niche", "cell_type"]).reset_index(drop=True)


def contact_enrichment(nodes: pd.DataFrame, spatial_edges: pd.DataFrame) -> pd.DataFrame:
    if spatial_edges.empty:
        return pd.DataFrame()
    node_types = nodes.set_index("row_id")["cell_type"].astype(str)
    counts = node_types.value_counts(normalize=True).to_dict()
    records: list[dict[str, Any]] = []
    pairs = spatial_edges[["source_cell_type", "target_cell_type"]].copy()
    pairs["cell_type_a"] = pairs[["source_cell_type", "target_cell_type"]].min(axis=1)
    pairs["cell_type_b"] = pairs[["source_cell_type", "target_cell_type"]].max(axis=1)
    total_edges = max(1, len(pairs))
    for (cell_a, cell_b), group in pairs.groupby(["cell_type_a", "cell_type_b"]):
        observed = len(group) / total_edges
        expected = counts.get(cell_a, 0.0) ** 2 if cell_a == cell_b else 2 * counts.get(cell_a, 0.0) * counts.get(cell_b, 0.0)
        enrichment = float(observed / max(expected, 1e-12))
        records.append(
            {
                "cell_type_a": cell_a,
                "cell_type_b": cell_b,
                "observed_edge_fraction": observed,
                "expected_random_fraction": expected,
                "contact_enrichment": enrichment,
                "n_edges": int(len(group)),
                "mean_distance": float(group.index.to_series().map(lambda _: 0).mean()) if False else float(
                    spatial_edges.loc[group.index, "distance"].mean()
                ),
            }
        )
    return pd.DataFrame(records).sort_values("contact_enrichment", ascending=False).reset_index(drop=True)


def build_networkx_heterograph(
    nodes: pd.DataFrame,
    spatial_edges: pd.DataFrame,
    communication_edges: pd.DataFrame,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(framework="CausaFlux", version="1.7.0", graph_type="multicellular_spatial_heterograph")
    safe_node_columns = [
        "cell_type", "compartment", "state", "therapy", "sample_id", "donor_id", "time_hours",
        "spatial_x", "spatial_y", "spatial_niche", "resistance_score", "immune_exclusion",
        "antigen_presentation", "ire1_xbp1", "mitochondrial_reserve",
    ]
    for row in nodes.itertuples(index=False):
        attrs: dict[str, Any] = {"node_type": "cell"}
        payload = row._asdict()
        for column in safe_node_columns:
            value = payload.get(column, "")
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            if pd.isna(value):
                value = ""
            attrs[column] = value
        graph.add_node(str(row.row_id), **attrs)
    for edge in spatial_edges.itertuples(index=False):
        graph.add_edge(
            str(edge.source), str(edge.target),
            key=f"spatial::{edge.source}::{edge.target}",
            edge_type="spatial_proximity",
            relation="spatial_proximity",
            distance=float(edge.distance),
            weight=float(edge.spatial_weight),
            directed_semantics="undirected_contact",
        )
        graph.add_edge(
            str(edge.target), str(edge.source),
            key=f"spatial::{edge.target}::{edge.source}",
            edge_type="spatial_proximity",
            relation="spatial_proximity",
            distance=float(edge.distance),
            weight=float(edge.spatial_weight),
            directed_semantics="undirected_contact",
        )
    for index, edge in enumerate(communication_edges.itertuples(index=False)):
        graph.add_edge(
            str(edge.source), str(edge.target),
            key=f"lr::{index}",
            edge_type="ligand_receptor",
            relation=f"{edge.ligand}_{edge.receptor}",
            ligand=str(edge.ligand),
            receptor=str(edge.receptor),
            family=str(edge.family),
            effect=str(edge.effect),
            distance=float(edge.distance),
            weight=float(edge.communication_score),
        )
    return graph


def validate_spatial_graph(
    nodes: pd.DataFrame,
    spatial_edges: pd.DataFrame,
    communication_edges: pd.DataFrame,
    config: SpatialGraphConfig,
) -> dict[str, Any]:
    required_node_columns = {"row_id", "cell_type", "spatial_x", "spatial_y", "spatial_niche"}
    missing_nodes = sorted(required_node_columns - set(nodes.columns))
    node_ids = set(nodes["row_id"].astype(str))
    edge_ids = set(spatial_edges.get("source", pd.Series(dtype=str)).astype(str)) | set(spatial_edges.get("target", pd.Series(dtype=str)).astype(str))
    comm_ids = set(communication_edges.get("source", pd.Series(dtype=str)).astype(str)) | set(communication_edges.get("target", pd.Series(dtype=str)).astype(str))
    unknown_edge_nodes = sorted((edge_ids | comm_ids) - node_ids)[:10]
    invalid_distances = int((spatial_edges.get("distance", pd.Series(dtype=float)) < 0).sum())
    invalid_scores = int((communication_edges.get("communication_score", pd.Series(dtype=float)) < 0).sum())
    valid_niches = set(nodes.get("spatial_niche", pd.Series(dtype=str)).astype(str)).issubset(set(NICHE_ORDER))
    report = {
        "valid": not missing_nodes and not unknown_edge_nodes and invalid_distances == 0 and invalid_scores == 0 and valid_niches,
        "n_nodes": int(len(nodes)),
        "n_spatial_edges": int(len(spatial_edges)),
        "n_communication_edges": int(len(communication_edges)),
        "n_samples": int(nodes["sample_id"].nunique()),
        "n_donors": int(nodes["donor_id"].nunique()),
        "node_types": sorted(nodes["cell_type"].astype(str).unique().tolist()),
        "n_niches": int(nodes["spatial_niche"].nunique()),
        "n_ligand_receptor_pairs": int(communication_edges[["ligand", "receptor"]].drop_duplicates().shape[0]) if not communication_edges.empty else 0,
        "missing_node_columns": missing_nodes,
        "unknown_edge_nodes": unknown_edge_nodes,
        "invalid_distances": invalid_distances,
        "invalid_communication_scores": invalid_scores,
        "valid_niches": valid_niches,
        "k_neighbors": int(config.k_neighbors),
        "max_distance": float(config.max_distance),
        "communication_radius": float(config.communication_radius),
    }
    if not report["valid"]:
        raise ValueError(f"Spatial graph validation failed: {report}")
    return report


def build_spatial_heterograph(
    frame: pd.DataFrame,
    config: SpatialGraphConfig | None = None,
) -> SpatialGraphResult:
    config = config or SpatialGraphConfig()
    if not {"spatial_x", "spatial_y"}.issubset(frame.columns):
        frame = generate_spatial_coordinates(frame, config)
    else:
        frame = frame.copy()
    spatial_edges = build_spatial_edges(frame, config)
    neighborhoods = _neighbor_composition(frame, spatial_edges, config)
    nodes = frame.merge(neighborhoods, on="row_id", how="left", validate="one_to_one")
    nodes["compartment"] = nodes["cell_type"].map(COMPARTMENT_MAP).fillna("other")
    communication_edges = infer_communication_edges(nodes, spatial_edges, config)
    circuits = summarize_communication_circuits(communication_edges, config)
    niche_summary = summarize_niches(nodes)
    contacts = contact_enrichment(nodes, spatial_edges)
    graph = build_networkx_heterograph(nodes, spatial_edges, communication_edges)
    qc = validate_spatial_graph(nodes, spatial_edges, communication_edges, config)
    return SpatialGraphResult(
        frame=frame,
        nodes=nodes,
        spatial_edges=spatial_edges,
        communication_edges=communication_edges,
        circuits=circuits,
        niche_summary=niche_summary,
        contact_enrichment=contacts,
        ligand_receptor_catalog=ligand_receptor_catalog(),
        graph=graph,
        qc=qc,
    )


def pyg_metadata(nodes: pd.DataFrame, communication_edges: pd.DataFrame) -> dict[str, Any]:
    node_types = sorted(nodes["cell_type"].astype(str).unique().tolist())
    spatial_relations = [[source, "spatial_proximity", target] for source in node_types for target in node_types]
    communication_relations = (
        communication_edges[["source_cell_type", "ligand", "receptor", "target_cell_type"]]
        .drop_duplicates()
        .apply(lambda row: [row["source_cell_type"], f"{row['ligand']}_{row['receptor']}", row["target_cell_type"]], axis=1)
        .tolist()
        if not communication_edges.empty
        else []
    )
    return {
        "framework": "CausaFlux",
        "version": "1.7.0",
        "format": "PyTorch Geometric metadata-compatible",
        "node_types": node_types,
        "edge_types": spatial_relations + communication_relations,
        "node_table": "graph_nodes.csv",
        "spatial_edge_table": "spatial_edges.csv",
        "communication_edge_table": "communication_edges.csv",
        "note": "Tables are exported independently of PyTorch Geometric; adapters can construct HeteroData without requiring torch-geometric in the core installation.",
    }


def write_spatial_graph_outputs(result: SpatialGraphResult, directory: str | Path, export_graphml: bool = True) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "nodes": directory / "graph_nodes.csv",
        "spatial_edges": directory / "spatial_edges.csv",
        "communication_edges": directory / "communication_edges.csv",
        "circuits": directory / "communication_circuit_summary.csv",
        "niches": directory / "spatial_niche_summary.csv",
        "contacts": directory / "contact_enrichment.csv",
        "catalog": directory / "ligand_receptor_catalog.csv",
        "qc": directory / "spatial_graph_qc.json",
        "pyg_metadata": directory / "pyg_metadata.json",
    }
    result.nodes.to_csv(paths["nodes"], index=False)
    result.spatial_edges.to_csv(paths["spatial_edges"], index=False)
    result.communication_edges.to_csv(paths["communication_edges"], index=False)
    result.circuits.to_csv(paths["circuits"], index=False)
    result.niche_summary.to_csv(paths["niches"], index=False)
    result.contact_enrichment.to_csv(paths["contacts"], index=False)
    result.ligand_receptor_catalog.to_csv(paths["catalog"], index=False)
    paths["qc"].write_text(json.dumps(result.qc, indent=2, sort_keys=True), encoding="utf-8")
    paths["pyg_metadata"].write_text(json.dumps(pyg_metadata(result.nodes, result.communication_edges), indent=2), encoding="utf-8")
    if export_graphml:
        graphml = directory / "spatial_heterograph.graphml"
        nx.write_graphml(result.graph, graphml)
        paths["graphml"] = graphml
    return paths


def _representative_sample(nodes: pd.DataFrame, sample_id: str | None = None) -> str:
    if sample_id and sample_id in set(nodes["sample_id"].astype(str)):
        return sample_id
    sample_stats = (
        nodes.groupby("sample_id", as_index=False)
        .agg(n_cells=("row_id", "size"), n_types=("cell_type", "nunique"), mean_exclusion=("immune_exclusion", "mean"))
    )
    sample_stats["score"] = sample_stats["n_cells"] + 8 * sample_stats["n_types"] + 5 * sample_stats["mean_exclusion"]
    return str(sample_stats.sort_values("score", ascending=False).iloc[0]["sample_id"])


def plot_spatial_atlas(nodes: pd.DataFrame, path: str | Path, sample_id: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = _representative_sample(nodes, sample_id)
    sample = nodes.loc[nodes["sample_id"].astype(str) == selected]
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    for cell_type in CELL_TYPES:
        part = sample.loc[sample["cell_type"] == cell_type]
        if part.empty:
            continue
        ax.scatter(part["spatial_x"], part["spatial_y"], s=42 if cell_type == "tumor" else 50, alpha=0.82, label=cell_type.replace("_", " "), color=CELL_TYPE_COLORS.get(cell_type))
    ax.set_title(f"Representative multicellular spatial atlas\n{selected}")
    ax.set_xlabel("Spatial x")
    ax.set_ylabel("Spatial y")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_contact_heatmap(contacts: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_types = list(CELL_TYPES)
    matrix = pd.DataFrame(np.nan, index=cell_types, columns=cell_types)
    for row in contacts.itertuples(index=False):
        matrix.loc[row.cell_type_a, row.cell_type_b] = row.contact_enrichment
        matrix.loc[row.cell_type_b, row.cell_type_a] = row.contact_enrichment
    values = matrix.fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(7.7, 6.5))
    image = ax.imshow(values, cmap="coolwarm", vmin=0, vmax=max(2.0, float(np.nanquantile(values, 0.95))))
    ax.set_xticks(range(len(cell_types)), [name.replace("_", " ") for name in cell_types], rotation=45, ha="right")
    ax.set_yticks(range(len(cell_types)), [name.replace("_", " ") for name in cell_types])
    for i in range(len(cell_types)):
        for j in range(len(cell_types)):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Spatial contact enrichment over random mixing")
    fig.colorbar(image, ax=ax, label="Enrichment")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_communication_circuits(circuits: pd.DataFrame, path: str | Path, top_n: int = 12) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = circuits.head(top_n).sort_values("circuit_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, max(5.0, 0.48 * max(1, len(top)) + 1.6)))
    if top.empty:
        ax.text(0.5, 0.5, "No communication circuits passed the configured threshold", ha="center", va="center")
        ax.axis("off")
    else:
        colors = ["#A05252" if effect in {"suppressive", "protective_niche"} else "#3B7F6B" for effect in top["effect"]]
        ax.barh(top["circuit"], top["circuit_score"], color=colors)
        ax.set_xlabel("Circuit score")
        ax.set_title("Top spatial communication circuits")
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heterograph_summary(nodes: pd.DataFrame, communication_edges: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.DiGraph()
    counts = nodes["cell_type"].value_counts()
    for cell_type, count in counts.items():
        graph.add_node(cell_type, count=int(count))
    if not communication_edges.empty:
        aggregated = (
            communication_edges.groupby(["source_cell_type", "target_cell_type"], as_index=False)["communication_score"]
            .sum()
        )
        for row in aggregated.itertuples(index=False):
            graph.add_edge(row.source_cell_type, row.target_cell_type, weight=float(row.communication_score))
    positions = nx.circular_layout(graph)
    fig, ax = plt.subplots(figsize=(8.3, 7.2))
    node_sizes = [650 + 1600 * counts.get(node, 1) / max(counts.max(), 1) for node in graph.nodes]
    node_colors = [CELL_TYPE_COLORS.get(node, "#7B8794") for node in graph.nodes]
    nx.draw_networkx_nodes(graph, positions, node_size=node_sizes, node_color=node_colors, alpha=0.92, ax=ax)
    weights = [graph.edges[edge].get("weight", 1.0) for edge in graph.edges]
    max_weight = max(weights, default=1.0)
    widths = [0.7 + 4.5 * weight / max_weight for weight in weights]
    nx.draw_networkx_edges(graph, positions, width=widths, alpha=0.45, arrows=True, arrowsize=14, connectionstyle="arc3,rad=0.12", ax=ax)
    nx.draw_networkx_labels(graph, positions, labels={node: node.replace("_", " ") for node in graph.nodes}, font_size=9, ax=ax)
    ax.set_title("Tumor–immune–stromal heterogeneous communication graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_niche_composition(niche_summary: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot = niche_summary.pivot_table(index="spatial_niche", columns="cell_type", values="within_niche_fraction", fill_value=0, observed=False).reindex(NICHE_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    bottom = np.zeros(len(pivot))
    for cell_type in CELL_TYPES:
        values = pivot.get(cell_type, pd.Series(0, index=pivot.index)).to_numpy()
        ax.bar([name.replace("_", " ") for name in pivot.index], values, bottom=bottom, label=cell_type.replace("_", " "), color=CELL_TYPE_COLORS.get(cell_type))
        bottom += values
    ax.set_ylabel("Cell fraction")
    ax.set_title("Cellular composition of inferred spatial niches")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.22), loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path
