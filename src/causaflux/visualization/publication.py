from __future__ import annotations

import hashlib
import json
import math
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

MM_TO_IN = 1.0 / 25.4

EXPORT_PROFILES: dict[str, dict[str, Any]] = {
    "nature_single": {"width_mm": 89.0, "height_mm": 68.0, "font_pt": 7.0, "label_pt": 8.0, "dpi": 600},
    "nature_double": {"width_mm": 183.0, "height_mm": 120.0, "font_pt": 7.5, "label_pt": 9.0, "dpi": 600},
    "cell_single": {"width_mm": 85.0, "height_mm": 72.0, "font_pt": 7.0, "label_pt": 8.0, "dpi": 600},
    "cell_double": {"width_mm": 178.0, "height_mm": 125.0, "font_pt": 7.5, "label_pt": 9.0, "dpi": 600},
    "cell_square": {"width_mm": 150.0, "height_mm": 150.0, "font_pt": 8.0, "label_pt": 10.0, "dpi": 600},
    "nature_portrait": {"width_mm": 150.0, "height_mm": 200.0, "font_pt": 7.5, "label_pt": 9.0, "dpi": 600},
}

COLORS = {
    "ink": "#202124",
    "muted": "#687078",
    "grid": "#D9DDE1",
    "blue": "#3F6C8E",
    "teal": "#2D7F78",
    "gold": "#C78B2C",
    "red": "#B64C4C",
    "purple": "#7868A6",
    "green": "#5E8C61",
    "sky": "#7FA7C3",
    "orange": "#D0784A",
    "light": "#F4F6F7",
}

CELL_TYPE_COLORS = {
    "tumor": "#B64C4C",
    "macrophage": "#C78B2C",
    "dendritic_cell": "#7B6DA8",
    "t_cell": "#3F6C8E",
    "fibroblast": "#5E8C61",
    "vascular": "#4E9B9A",
    "excitatory_neuron": "#3F6C8E",
    "inhibitory_neuron": "#7B6DA8",
    "astrocyte": "#5E8C61",
    "microglia": "#C78B2C",
    "oligodendrocyte": "#4E9B9A",
}

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("causaflux_seq", ["#F6F8F9", "#9DB8C9", "#3F6C8E", "#22384B"])
DIVERGING_CMAP = LinearSegmentedColormap.from_list("causaflux_div", ["#3F6C8E", "#F6F6F4", "#B64C4C"])


@dataclass(frozen=True)
class FigureExport:
    figure_id: str
    profile: str
    png: str
    svg: str
    pdf: str
    tiff: str
    source_data: list[str]
    manifest: str
    width_mm: float
    height_mm: float
    dpi: int
    synthetic_only: bool = True


def _font_family() -> list[str]:
    return ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"]


def apply_publication_style(profile: str = "nature_double") -> dict[str, Any]:
    if profile not in EXPORT_PROFILES:
        raise ValueError(f"Unknown export profile: {profile}")
    spec = EXPORT_PROFILES[profile]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _font_family(),
            "font.size": spec["font_pt"],
            "axes.titlesize": spec["label_pt"],
            "axes.labelsize": spec["font_pt"],
            "axes.titleweight": "semibold",
            "axes.titlepad": 6.0,
            "axes.linewidth": 0.65,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.labelsize": spec["font_pt"] - 0.5,
            "ytick.labelsize": spec["font_pt"] - 0.5,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": spec["font_pt"] - 0.4,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
            "patch.linewidth": 0.6,
        }
    )
    return spec


def _new_figure(profile: str, *, height_mm: float | None = None) -> tuple[plt.Figure, plt.Axes]:
    spec = apply_publication_style(profile)
    height = float(height_mm or spec["height_mm"])
    fig, ax = plt.subplots(figsize=(spec["width_mm"] * MM_TO_IN, height * MM_TO_IN))
    return fig, ax


def _clean_axis(ax: plt.Axes, *, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.55, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)


def _human(value: Any) -> str:
    return str(value).replace("_", " ").replace("→", " → ")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perceptual_hash(path: str | Path, hash_size: int = 16) -> str:
    image = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.int16)
    difference = pixels[:, 1:] > pixels[:, :-1]
    bits = "".join("1" if flag else "0" for flag in difference.ravel())
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def _hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def compare_visual_baseline(current_png: str | Path, expected_hash: str, tolerance: int = 8) -> dict[str, Any]:
    current_hash = perceptual_hash(current_png)
    distance = _hamming_hex(current_hash, expected_hash)
    return {"valid": distance <= tolerance, "distance": distance, "tolerance": tolerance, "current_hash": current_hash, "expected_hash": expected_hash}


def _write_source_data(source_dir: Path, figure_id: str, source_data: Any) -> list[str]:
    source_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    if source_data is None:
        return outputs
    if isinstance(source_data, pd.DataFrame):
        mapping: Mapping[str, pd.DataFrame] = {"panel_a": source_data}
    elif isinstance(source_data, Mapping):
        mapping = {str(key): value for key, value in source_data.items() if isinstance(value, pd.DataFrame)}
    else:
        raise TypeError("source_data must be a DataFrame, mapping of DataFrames, or None")
    for panel, frame in mapping.items():
        safe_panel = panel.lower().replace(" ", "_")
        path = source_dir / f"{figure_id}__{safe_panel}.csv"
        frame.to_csv(path, index=True)
        outputs.append(path.as_posix())
    return outputs


def export_figure(
    fig: plt.Figure,
    output_path: str | Path,
    *,
    figure_id: str | None = None,
    profile: str = "nature_double",
    source_data: Any = None,
    metadata: Mapping[str, Any] | None = None,
    synthetic_only: bool = True,
) -> FigureExport:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_id = figure_id or output_path.stem
    spec = EXPORT_PROFILES[profile]
    base = output_path.with_suffix("")
    paths = {
        "png": base.with_suffix(".png"),
        "svg": base.with_suffix(".svg"),
        "pdf": base.with_suffix(".pdf"),
        "tiff": base.with_suffix(".tiff"),
    }
    fig.set_size_inches(spec["width_mm"] * MM_TO_IN, spec["height_mm"] * MM_TO_IN, forward=True)
    fig.savefig(paths["png"], dpi=spec["dpi"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["tiff"], dpi=spec["dpi"], bbox_inches="tight", pad_inches=0.03, pil_kwargs={"compression": "tiff_lzw"})
    source_paths = _write_source_data(output_path.parent / "source_data", figure_id, source_data)
    manifest_dir = output_path.parent / "figure_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{figure_id}.json"
    payload = {
        "framework": "CausaFlux",
        "version": "1.9.0",
        "figure_id": figure_id,
        "profile": profile,
        "width_mm": spec["width_mm"],
        "height_mm": spec["height_mm"],
        "dpi": spec["dpi"],
        "formats": {name: path.name for name, path in paths.items()},
        "source_data": [Path(path).name for path in source_paths],
        "synthetic_only": synthetic_only,
        "visual_hash": perceptual_hash(paths["png"]),
        "sha256": {name: _hash_file(path) for name, path in paths.items()},
        "metadata": dict(metadata or {}),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return FigureExport(
        figure_id=figure_id,
        profile=profile,
        png=paths["png"].as_posix(),
        svg=paths["svg"].as_posix(),
        pdf=paths["pdf"].as_posix(),
        tiff=paths["tiff"].as_posix(),
        source_data=source_paths,
        manifest=manifest_path.as_posix(),
        width_mm=spec["width_mm"],
        height_mm=spec["height_mm"],
        dpi=spec["dpi"],
        synthetic_only=synthetic_only,
    )


def _finish(fig: plt.Figure, output_path: Path, *, source_data: Any, profile: str = "nature_double", metadata: Mapping[str, Any] | None = None) -> FigureExport:
    fig.tight_layout(pad=0.5)
    result = export_figure(fig, output_path, profile=profile, source_data=source_data, metadata=metadata)
    plt.close(fig)
    return result


def plot_ranked_bar(frame: pd.DataFrame, label: str, value: str, output: Path, title: str, xlabel: str, *, top_n: int = 12, lower_better: bool = False, color: str = COLORS["blue"], profile: str = "nature_double") -> FigureExport:
    data = frame.dropna(subset=[label, value]).copy()
    data = data.nsmallest(top_n, value) if lower_better else data.nlargest(top_n, value)
    data = data.sort_values(value, ascending=not lower_better)
    fig, ax = _new_figure(profile)
    y = np.arange(len(data))
    ax.barh(y, data[value], color=color, edgecolor="none", height=0.66, zorder=3)
    ax.set_yticks(y, [_human(x) for x in data[label]])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    _clean_axis(ax, grid="x")
    if len(data):
        span = float(data[value].max() - data[value].min()) or max(abs(float(data[value].max())), 1.0)
        for index, value_item in enumerate(data[value]):
            ax.text(float(value_item) + 0.018 * span, index, f"{float(value_item):.3f}", va="center", fontsize=6.4, color=COLORS["muted"])
    return _finish(fig, output, source_data={"panel_a": data}, profile=profile)


def plot_heatmap(matrix: pd.DataFrame, output: Path, title: str, colorbar_label: str, *, diverging: bool = False, annotate: bool = True, profile: str = "nature_double", vmin: float | None = None, vmax: float | None = None) -> FigureExport:
    values = matrix.to_numpy(dtype=float)
    fig, ax = _new_figure(profile)
    cmap = DIVERGING_CMAP if diverging else SEQUENTIAL_CMAP
    if diverging:
        bound = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 1e-8)
        image = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-bound if vmin is None else vmin, vcenter=0, vmax=bound if vmax is None else vmax), aspect="auto")
    else:
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [_human(x) for x in matrix.columns], rotation=38, ha="right")
    ax.set_yticks(range(len(matrix.index)), [_human(x) for x in matrix.index])
    ax.set_title(title, loc="left")
    if annotate and values.size <= 100:
        threshold = np.nanmin(values) + 0.60 * (np.nanmax(values) - np.nanmin(values) + 1e-9)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value_item = values[row, column]
                ax.text(column, row, f"{value_item:.2f}", ha="center", va="center", fontsize=6.2, color="white" if value_item > threshold else COLORS["ink"])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label(colorbar_label)
    return _finish(fig, output, source_data={"panel_a": matrix.reset_index()}, profile=profile)


def plot_reliability(raw: pd.DataFrame, calibrated: pd.DataFrame, output: Path) -> FigureExport:
    fig, ax = _new_figure("nature_single")
    ax.plot([0, 1], [0, 1], color=COLORS["muted"], linestyle="--", linewidth=0.9, label="Ideal")
    ax.plot(raw.iloc[:, 0], raw.iloc[:, 1], marker="o", color=COLORS["red"], label="Selected baseline")
    ax.plot(calibrated.iloc[:, 0], calibrated.iloc[:, 1], marker="o", color=COLORS["blue"], label="Calibrated ensemble")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence", ylabel="Observed accuracy")
    ax.set_title("Donor-cross-fitted reliability", loc="left")
    ax.legend(loc="lower right")
    _clean_axis(ax, grid="both")
    return _finish(fig, output, source_data={"panel_a_raw": raw, "panel_b_calibrated": calibrated}, profile="nature_single")


def _topological_layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    generations = list(nx.topological_generations(graph))
    positions: dict[str, tuple[float, float]] = {}
    for x_index, generation in enumerate(generations):
        nodes = sorted(generation)
        offset = (len(nodes) - 1) / 2
        for y_index, node in enumerate(nodes):
            positions[node] = (float(x_index), float(offset - y_index))
    return positions


def plot_causal_dag(nodes: pd.DataFrame, edges: pd.DataFrame, output: Path) -> FigureExport:
    graph = nx.DiGraph()
    for row in nodes.itertuples(index=False):
        graph.add_node(row.name, type=getattr(row, "type", "mechanism"))
    for row in edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, sign=getattr(row, "sign", "positive"), evidence=getattr(row, "evidence", ""))
    raw_positions = _topological_layout(graph)
    positions = {node: (1.55 * xy[0], 0.92 * xy[1]) for node, xy in raw_positions.items()}
    fig, ax = _new_figure("nature_double", height_mm=118)
    type_colors = {"intervention": COLORS["purple"], "mechanism": COLORS["blue"], "state": COLORS["teal"], "outcome": COLORS["red"], "biomarker": COLORS["gold"]}
    box_width, box_height = 1.20, 0.46
    for source, target, attrs in graph.edges(data=True):
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        sign = attrs.get("sign", "positive")
        color = COLORS["red"] if sign == "negative" else COLORS["ink"]
        arrow = FancyArrowPatch(
            (x1 + box_width / 2, y1),
            (x2 - box_width / 2, y2),
            arrowstyle="-|>", mutation_scale=7.5, linewidth=0.72, color=color,
            alpha=0.72, connectionstyle="arc3,rad=0.0", zorder=2,
        )
        ax.add_patch(arrow)
    for node, attrs in graph.nodes(data=True):
        x, y = positions[node]
        color = type_colors.get(attrs.get("type", "mechanism"), COLORS["blue"])
        patch = FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2), box_width, box_height,
            boxstyle="round,pad=0.025,rounding_size=0.055", facecolor=color,
            edgecolor="white", linewidth=0.7, zorder=3,
        )
        ax.add_patch(patch)
        label = textwrap.fill(_human(node), width=19, break_long_words=False)
        ax.text(x, y, label, ha="center", va="center", fontsize=5.7, color="white", fontweight="semibold", zorder=4, linespacing=0.96)
    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    ax.set_xlim(min(xs) - 0.72, max(xs) + 0.72)
    ax.set_ylim(min(ys) - 0.48, max(ys) + 0.48)
    ax.set_title("Editable causal model with signed intervention paths", loc="left")
    ax.axis("off")
    layout = pd.DataFrame([{"name": node, "x": xy[0], "y": xy[1]} for node, xy in positions.items()])
    return _finish(fig, output, source_data={"panel_a_nodes": nodes, "panel_b_edges": edges, "panel_c_layout": layout}, profile="nature_double")


def plot_spatial_atlas(nodes: pd.DataFrame, output: Path) -> FigureExport:
    stats = nodes.groupby("sample_id").agg(n_cells=("row_id", "size"), n_types=("cell_type", "nunique"), mean_exclusion=("immune_exclusion", "mean"))
    sample_id = stats.assign(score=lambda x: x.n_cells + 8 * x.n_types + 5 * x.mean_exclusion).sort_values("score", ascending=False).index[0]
    data = nodes[nodes["sample_id"] == sample_id].copy()
    fig, ax = _new_figure("nature_double", height_mm=112)
    for cell_type, group in data.groupby("cell_type", sort=False):
        ax.scatter(group["spatial_x"], group["spatial_y"], s=13 if cell_type == "tumor" else 18, c=CELL_TYPE_COLORS.get(cell_type, COLORS["muted"]), edgecolors="white", linewidths=0.25, alpha=0.88, label=_human(cell_type), rasterized=False)
    ax.set(xlabel="Spatial x", ylabel="Spatial y")
    ax.set_title(f"Representative multicellular spatial atlas · {sample_id}", loc="left")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Cell population")
    _clean_axis(ax)
    return _finish(fig, output, source_data={"panel_a": data}, profile="nature_double", metadata={"representative_sample": sample_id})


def plot_communication_circuits(circuits: pd.DataFrame, output: Path) -> FigureExport:
    data = circuits.nlargest(12, "circuit_score").sort_values("circuit_score")
    fig, ax = _new_figure("nature_double")
    palette = [COLORS["red"] if effect in {"suppressive", "protective_niche"} else COLORS["teal"] for effect in data["effect"]]
    y = np.arange(len(data))
    ax.barh(y, data["mean_communication_score"], color=palette, height=0.65)
    lower = np.maximum(0.0, data["mean_communication_score"].to_numpy() - data["ci_low"].to_numpy())
    upper = np.maximum(0.0, data["ci_high"].to_numpy() - data["mean_communication_score"].to_numpy())
    ax.errorbar(data["mean_communication_score"], y, xerr=np.vstack([lower, upper]), fmt="none", ecolor=COLORS["ink"], elinewidth=0.7, capsize=1.5)
    ax.set_yticks(y, [_human(x) for x in data["circuit"]])
    ax.set_xlabel("Mean communication score with donor-bootstrap interval")
    ax.set_title("Spatial communication circuits", loc="left")
    _clean_axis(ax, grid="x")
    return _finish(fig, output, source_data={"panel_a": data}, profile="nature_double")


def plot_heterograph(nodes: pd.DataFrame, edges: pd.DataFrame, output: Path) -> FigureExport:
    counts = nodes["cell_type"].value_counts().sort_index()
    aggregate = edges.groupby(["source_cell_type", "target_cell_type"], as_index=False)["communication_score"].sum()
    graph = nx.DiGraph()
    for node, count in counts.items():
        graph.add_node(node, count=int(count))
    for row in aggregate.itertuples(index=False):
        graph.add_edge(row.source_cell_type, row.target_cell_type, weight=float(row.communication_score))
    order = [item for item in ["tumor", "macrophage", "dendritic_cell", "t_cell", "fibroblast", "vascular"] if item in graph]
    positions = {node: (math.cos(2 * math.pi * index / max(len(order), 1)), math.sin(2 * math.pi * index / max(len(order), 1))) for index, node in enumerate(order)}
    fig, ax = _new_figure("nature_double", height_mm=112)
    max_count = max(counts.max(), 1)
    max_weight = max([attrs["weight"] for _, _, attrs in graph.edges(data=True)] or [1.0])
    for source, target, attrs in graph.edges(data=True):
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        width = 0.35 + 2.7 * attrs["weight"] / max_weight
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, linewidth=width, color=COLORS["muted"], alpha=0.34, connectionstyle="arc3,rad=0.16")
        ax.add_patch(arrow)
    for node in order:
        x, y = positions[node]
        size = 0.11 + 0.16 * counts[node] / max_count
        circle = plt.Circle((x, y), size, color=CELL_TYPE_COLORS.get(node, COLORS["muted"]), ec="white", lw=1.0, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, _human(node), ha="center", va="center", color="white", fontsize=6.4, fontweight="semibold", zorder=4)
    ax.set(xlim=(-1.35, 1.35), ylim=(-1.3, 1.3))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Tumor–immune–stromal communication overview", loc="left")
    layout = pd.DataFrame([{"cell_type": node, "x": positions[node][0], "y": positions[node][1], "n_cells": int(counts[node])} for node in order])
    return _finish(fig, output, source_data={"panel_a_nodes": layout, "panel_b_edges": aggregate}, profile="nature_double")


def plot_stacked_composition(summary: pd.DataFrame, output: Path) -> FigureExport:
    pivot = summary.pivot_table(index="spatial_niche", columns="cell_type", values="within_niche_fraction", fill_value=0, observed=False)
    fig, ax = _new_figure("nature_double")
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for cell_type in pivot.columns:
        values = pivot[cell_type].to_numpy()
        ax.bar(x, values, bottom=bottom, color=CELL_TYPE_COLORS.get(cell_type, COLORS["muted"]), width=0.72, label=_human(cell_type))
        bottom += values
    ax.set_xticks(x, [_human(v) for v in pivot.index], rotation=28, ha="right")
    ax.set_ylabel("Within-niche cell fraction")
    ax.set_title("Cellular composition of inferred spatial niches", loc="left")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _clean_axis(ax, grid="y")
    return _finish(fig, output, source_data={"panel_a": summary, "panel_b_matrix": pivot.reset_index()}, profile="nature_double")


def plot_pareto(predictions: pd.DataFrame, output: Path) -> FigureExport:
    data = predictions.copy()
    fig, ax = _new_figure("nature_double")
    categories = list(data["regimen_category"].dropna().unique())
    palette = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["purple"], COLORS["red"]]
    for color, category in zip(palette, categories):
        group = data[data["regimen_category"] == category]
        ax.scatter(group["normal_cell_toxicity"], group["resistance_risk_reduction"], s=14 + 25 * group["pareto_optimal"].astype(float), color=color, alpha=0.64, edgecolor="white", linewidth=0.35, label=_human(category))
    pareto = data[data["pareto_optimal"].astype(bool)].sort_values("normal_cell_toxicity")
    if not pareto.empty:
        ax.plot(pareto["normal_cell_toxicity"], pareto["resistance_risk_reduction"], color=COLORS["ink"], linewidth=0.85, linestyle="--", label="Pareto frontier")
    ax.set(xlabel="Normal-cell toxicity penalty", ylabel="Predicted resistance-risk reduction")
    ax.set_title("Benefit–toxicity trade-off", loc="left")
    ax.legend(ncol=2)
    _clean_axis(ax, grid="both")
    return _finish(fig, output, source_data={"panel_a": data, "panel_b_pareto": pareto}, profile="nature_double")


def plot_timing(predictions: pd.DataFrame, output: Path) -> FigureExport:
    data = predictions.copy()
    matrix = data.pivot_table(index="regimen_name", columns="first_start_hour", values="uncertainty_adjusted_utility", aggfunc="mean")
    matrix = matrix.loc[matrix.max(axis=1).sort_values(ascending=False).head(12).index]
    return plot_heatmap(matrix, output, "Treatment timing landscape", "Uncertainty-adjusted utility", annotate=False, profile="nature_double")


def plot_sequence(predictions: pd.DataFrame, output: Path) -> FigureExport:
    data = predictions.nlargest(16, "uncertainty_adjusted_utility").sort_values("uncertainty_adjusted_utility")
    return plot_ranked_bar(data, "regimen_name", "uncertainty_adjusted_utility", output, "Directional treatment-sequence comparison", "Uncertainty-adjusted utility", top_n=16, color=COLORS["purple"])


def plot_waterfall(predictions: pd.DataFrame, output: Path) -> FigureExport:
    data = predictions.nsmallest(12, "rank").sort_values("resistance_risk_reduction_bootstrap_mean")
    fig, ax = _new_figure("nature_double")
    y = np.arange(len(data))
    mean = data["resistance_risk_reduction_bootstrap_mean"]
    ax.errorbar(mean, y, xerr=[mean - data["resistance_risk_reduction_ci_low"], data["resistance_risk_reduction_ci_high"] - mean], fmt="o", color=COLORS["blue"], ecolor=COLORS["ink"], elinewidth=0.8, capsize=2)
    ax.axvline(0, color=COLORS["muted"], linewidth=0.7)
    ax.set_yticks(y, [_human(x) for x in data["regimen_name"]])
    ax.set_xlabel("Predicted resistance-risk reduction")
    ax.set_title("Counterfactual effect intervals", loc="left")
    _clean_axis(ax, grid="x")
    return _finish(fig, output, source_data={"panel_a": data}, profile="nature_double")


def plot_biomarker_heatmap(timecourse: pd.DataFrame, ranking: pd.DataFrame, output: Path) -> FigureExport:
    top = ranking.nsmallest(10, "rank")["biomarker"]
    data = timecourse[timecourse["biomarker"].isin(top)]
    matrix = data.pivot(index="biomarker", columns="time_hours", values="association_auc").reindex(top)
    return plot_heatmap(matrix, output, "Early-warning association across time", "Association AUC", annotate=True, profile="nature_double", vmin=0.5, vmax=1.0)


def plot_causal_lead(ranking: pd.DataFrame, output: Path) -> FigureExport:
    data = ranking.copy()
    fig, ax = _new_figure("nature_double")
    size = 18 + 70 * data.get("assayability", pd.Series(0.5, index=data.index)).fillna(0.5)
    scatter = ax.scatter(data["causal_proximity"], data["early_warning_lead_hours"], s=size, c=data["uncertainty_adjusted_score"], cmap=SEQUENTIAL_CMAP, edgecolors="white", linewidths=0.5)
    for row in data.nsmallest(8, "rank").itertuples(index=False):
        ax.annotate(_human(row.biomarker), (row.causal_proximity, row.early_warning_lead_hours), xytext=(3, 3), textcoords="offset points", fontsize=6.1)
    ax.set(xlabel="Causal-proximity score", ylabel="Warning lead time (hours)")
    ax.set_title("Lead time and causal proximity remain separate", loc="left")
    fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.025, label="Uncertainty-adjusted score")
    _clean_axis(ax, grid="both")
    return _finish(fig, output, source_data={"panel_a": data}, profile="nature_double")


def plot_panel_performance(panels: pd.DataFrame, output: Path) -> FigureExport:
    data = panels.copy()
    fig, ax = _new_figure("nature_single")
    ax.plot(data["panel_size"], data["donor_held_out_auc"], marker="o", color=COLORS["blue"], label="Pooled donor-held-out AUC")
    ax.plot(data["panel_size"], data["mean_donor_auc"], marker="s", color=COLORS["teal"], label="Mean donor AUC")
    ax.set_xticks(data["panel_size"])
    ax.set(xlabel="Panel size", ylabel="AUC", ylim=(0.45, 1.0))
    ax.set_title("Compact biomarker panels", loc="left")
    ax.legend()
    _clean_axis(ax, grid="both")
    return _finish(fig, output, source_data={"panel_a": data}, profile="nature_single")


def plot_info_gain_by_type(ranking: pd.DataFrame, output: Path) -> FigureExport:
    summary = ranking.groupby("experiment_type", as_index=False).agg(mean_eig=("expected_information_gain_nats", "mean"), max_eig=("expected_information_gain_nats", "max"), n=("experiment_id", "size"))
    fig, ax = _new_figure("nature_single")
    x = np.arange(len(summary))
    ax.bar(x, summary["mean_eig"], color=[COLORS["purple"], COLORS["teal"], COLORS["gold"], COLORS["blue"]][: len(summary)], width=0.65)
    ax.scatter(x, summary["max_eig"], color=COLORS["ink"], s=14, zorder=4, label="Maximum")
    ax.set_xticks(x, [_human(x) for x in summary["experiment_type"]], rotation=25, ha="right")
    ax.set_ylabel("Expected information gain (nats)")
    ax.set_title("Information value by experiment class", loc="left")
    ax.legend()
    _clean_axis(ax, grid="y")
    return _finish(fig, output, source_data={"panel_a": summary}, profile="nature_single")


def plot_posterior(history: pd.DataFrame, output: Path) -> FigureExport:
    hypothesis_cols = [column for column in history.columns if column.startswith("H")]
    fig, ax = _new_figure("nature_double")
    colors = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["purple"]]
    for color, column in zip(colors, hypothesis_cols):
        ax.plot(history["update_step"], history[column], marker="o", color=color, label=_human(column))
    ax.set(xlabel="Sequential update", ylabel="Posterior probability", ylim=(0, 1))
    ax.set_title("Mechanism posterior update", loc="left")
    ax.legend(ncol=2)
    _clean_axis(ax, grid="both")
    return _finish(fig, output, source_data={"panel_a": history}, profile="nature_double")


def plot_batch(batch: pd.DataFrame, output: Path) -> FigureExport:
    data = batch.sort_values("batch_position")
    fig, ax = _new_figure("nature_double")
    x = np.arange(len(data))
    colors = {"crispr": COLORS["purple"], "drug": COLORS["teal"], "imaging": COLORS["gold"], "sampling_time": COLORS["blue"]}
    bars = ax.bar(x, data["relative_cost"], color=[colors.get(v, COLORS["muted"]) for v in data["experiment_type"]], width=0.62)
    for bar, score in zip(bars, data["priority_score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"priority {score:.2f}", ha="center", fontsize=6.1)
    ax.set_xticks(x, [_human(x) for x in data["experiment_name"]], rotation=24, ha="right")
    ax.set_ylabel("Relative cost")
    ax.set_title("Budget-constrained experiment portfolio", loc="left")
    _clean_axis(ax, grid="y")
    return _finish(fig, output, source_data={"panel_a": data}, profile="nature_double")


def plot_sampling_times(ranking: pd.DataFrame, output: Path) -> FigureExport:
    data = ranking[ranking["experiment_type"].astype(str).str.contains("sampling", case=False)].sort_values("sample_time_hours")
    fig, ax = _new_figure("nature_single")
    ax.plot(data["sample_time_hours"], data["expected_information_gain_nats"], marker="o", color=COLORS["blue"])
    if "eig_ci_low" in data and "eig_ci_high" in data:
        ax.fill_between(data["sample_time_hours"], data["eig_ci_low"], data["eig_ci_high"], color=COLORS["blue"], alpha=0.18, linewidth=0)
    ax.set(xlabel="Sampling time (hours)", ylabel="Expected information gain (nats)")
    ax.set_title("Sampling-time recommendations", loc="left")
    _clean_axis(ax, grid="both")
    return _finish(fig, output, source_data={"panel_a": data}, profile="nature_single")


def plot_trajectories(summary: pd.DataFrame, output: Path) -> FigureExport:
    fig, ax = _new_figure("nature_double")
    for cell_type, group in summary.groupby("cell_type"):
        ax.plot(group["time_days"], group["probability_irreversible_degeneration"], marker="o", color=CELL_TYPE_COLORS.get(cell_type, COLORS["muted"]), label=_human(cell_type))
    ax.set(xlabel="Time (days)", ylabel="Irreversible-degeneration probability", ylim=(0, 1))
    ax.set_title("Neural–glial degeneration trajectories", loc="left")
    ax.legend(ncol=2)
    _clean_axis(ax, grid="both")
    return _finish(fig, output, source_data={"panel_a": summary}, profile="nature_double")


def plot_alignment(alignment: pd.DataFrame, output: Path) -> FigureExport:
    data = alignment.nlargest(14, "absolute_correlation").sort_values("spearman_correlation")
    data = data.assign(pair=data["imaging_feature"].map(_human) + " ↔ " + data["electrophysiology_feature"].map(_human))
    fig, ax = _new_figure("nature_double")
    colors = [COLORS["red"] if value < 0 else COLORS["blue"] for value in data["spearman_correlation"]]
    y = np.arange(len(data))
    ax.barh(y, data["spearman_correlation"], color=colors, height=0.65)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_yticks(y, data["pair"])
    ax.set_xlabel("Spearman correlation")
    ax.set_title("Live-imaging and electrophysiology alignment", loc="left")
    _clean_axis(ax, grid="x")
    return _finish(fig, output, source_data={"panel_a": data}, profile="nature_double")


def plot_apoe(apoe: pd.DataFrame, output: Path) -> FigureExport:
    data = apoe.groupby(["apoe_genotype", "time_days"], as_index=False)["predicted_degeneration_probability"].mean()
    fig, ax = _new_figure("nature_single")
    colors = {"APOE3": COLORS["blue"], "APOE4": COLORS["red"]}
    for genotype, group in data.groupby("apoe_genotype"):
        ax.plot(group["time_days"], group["predicted_degeneration_probability"], marker="o", color=colors.get(genotype, COLORS["muted"]), label=genotype)
    ax.set(xlabel="Time (days)", ylabel="Donor-held-out degeneration risk", ylim=(0, 1))
    ax.set_title("APOE-stratified neural risk", loc="left")
    ax.legend()
    _clean_axis(ax, grid="both")
    return _finish(fig, output, source_data={"panel_a": data}, profile="nature_single")


def _read_csv(path: Path, *, index_col: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col)


def rebuild_reference_figures(reference_dir: str | Path, profile: str = "nature_double") -> pd.DataFrame:
    root = Path(reference_dir)
    exports: list[FigureExport] = []
    # Multimodal and uncertainty
    modality = _read_csv(root / "multimodal/modality_ablation_metrics.csv")
    exports.append(plot_ranked_bar(modality, "feature_set", "log_loss", root / "multimodal/modality_ablation.png", "Modality and early-fusion benchmark", "Donor-held-out log loss (lower is better)", lower_better=True, top_n=10))
    corr = _read_csv(root / "multimodal/cross_modal_summary_correlations.csv", index_col=0)
    exports.append(plot_heatmap(corr, root / "multimodal/cross_modal_correlation.png", "Cross-modal mean-signal correlations", "Pearson correlation", diverging=True, profile="nature_single", vmin=-1, vmax=1))
    baseline = _read_csv(root / "baselines/linear_baseline_metrics.csv")
    baseline = baseline.assign(display_name=baseline["model"].map(_human) + " · " + baseline["variant"].map(_human))
    exports.append(plot_ranked_bar(baseline, "display_name", "log_loss", root / "baselines/linear_baseline_benchmark.png", "Linear baseline and calibration benchmark", "Donor-held-out log loss (lower is better)", lower_better=True, top_n=12))
    exports.append(plot_reliability(_read_csv(root / "calibration/selected_model_reliability.csv"), _read_csv(root / "calibration/ensemble_reliability.csv"), root / "calibration/reliability_diagram.png"))
    transition = _read_csv(root / "transitions/transition_matrix.csv", index_col=0)
    exports.append(plot_heatmap(transition, root / "transitions/transition_heatmap.png", "Tumor-state transition probabilities", "Probability", profile="nature_single", vmin=0, vmax=1))
    exports.append(plot_causal_dag(_read_csv(root / "graph/causal_nodes.csv"), _read_csv(root / "graph/causal_edges.csv"), root / "graph/causal_graph.png"))
    # Spatial
    nodes = _read_csv(root / "spatial_graph/graph_nodes.csv")
    contacts = _read_csv(root / "spatial_graph/contact_enrichment.csv")
    cell_types = sorted(set(contacts["cell_type_a"]) | set(contacts["cell_type_b"]))
    contact_matrix = pd.DataFrame(0.0, index=cell_types, columns=cell_types)
    for row in contacts.itertuples(index=False):
        contact_matrix.loc[row.cell_type_a, row.cell_type_b] = row.contact_enrichment
        contact_matrix.loc[row.cell_type_b, row.cell_type_a] = row.contact_enrichment
    exports.append(plot_spatial_atlas(nodes, root / "spatial_graph/spatial_atlas.png"))
    exports.append(plot_heatmap(contact_matrix, root / "spatial_graph/contact_enrichment_heatmap.png", "Spatial contact enrichment over random mixing", "Enrichment", profile="nature_single", vmin=0))
    circuits = _read_csv(root / "spatial_graph/communication_circuit_summary.csv")
    exports.append(plot_communication_circuits(circuits, root / "spatial_graph/communication_circuits.png"))
    comm_edges = _read_csv(root / "spatial_graph/communication_edges.csv")
    exports.append(plot_heterograph(nodes, comm_edges, root / "spatial_graph/heterograph_summary.png"))
    niche = _read_csv(root / "spatial_graph/spatial_niche_summary.csv")
    exports.append(plot_stacked_composition(niche, root / "spatial_graph/spatial_niche_composition.png"))
    # Therapeutics
    therapy = _read_csv(root / "therapeutics/all_regimen_predictions.csv")
    therapy = therapy.assign(display_name=therapy["regimen_category"].map(_human) + " · " + therapy["regimen_name"].map(_human))
    exports.append(plot_ranked_bar(therapy.nsmallest(15, "rank"), "display_name", "uncertainty_adjusted_utility", root / "therapeutics/therapeutic_ranking.png", "Counterfactual therapeutic ranking", "Uncertainty-adjusted utility", top_n=15, color=COLORS["teal"]))
    exports.append(plot_timing(_read_csv(root / "therapeutics/timing_predictions.csv"), root / "therapeutics/timing_heatmap.png"))
    exports.append(plot_sequence(_read_csv(root / "therapeutics/sequence_predictions.csv"), root / "therapeutics/sequence_comparison.png"))
    exports.append(plot_pareto(therapy, root / "therapeutics/benefit_toxicity_pareto.png"))
    exports.append(plot_waterfall(therapy, root / "therapeutics/counterfactual_waterfall.png"))
    # Biomarkers
    biomarker = _read_csv(root / "biomarkers/causal_biomarker_ranking.csv")
    exports.append(plot_ranked_bar(biomarker.nsmallest(12, "rank"), "biomarker", "uncertainty_adjusted_score", root / "biomarkers/biomarker_ranking.png", "Early-warning and causal-proximity ranking", "Uncertainty-adjusted biomarker score", top_n=12, color=COLORS["gold"]))
    timecourse = _read_csv(root / "biomarkers/early_warning_timecourse.csv")
    exports.append(plot_biomarker_heatmap(timecourse, biomarker, root / "biomarkers/early_warning_heatmap.png"))
    exports.append(plot_causal_lead(biomarker, root / "biomarkers/causal_lead_map.png"))
    panels = _read_csv(root / "biomarkers/biomarker_panel_metrics.csv")
    exports.append(plot_panel_performance(panels, root / "biomarkers/biomarker_panel_performance.png"))
    # Closed-loop
    experiments = _read_csv(root / "active_learning/round1_experiment_recommendations.csv")
    exports.append(plot_ranked_bar(experiments.nsmallest(14, "rank"), "experiment_name", "priority_score", root / "active_learning/experiment_priority_ranking.png", "Closed-loop experiment priority", "Priority score", top_n=14, color=COLORS["purple"]))
    exports.append(plot_info_gain_by_type(experiments, root / "active_learning/information_gain_by_type.png"))
    posterior = _read_csv(root / "active_learning/hypothesis_posterior_history.csv")
    exports.append(plot_posterior(posterior, root / "active_learning/hypothesis_posterior_update.png"))
    batch = _read_csv(root / "active_learning/round1_selected_batch.csv")
    exports.append(plot_batch(batch, root / "active_learning/batch_portfolio.png"))
    exports.append(plot_sampling_times(experiments, root / "active_learning/sampling_time_recommendations.png"))
    # Neurobiology
    trajectory = _read_csv(root / "neurobiology/neural_glial_trajectory_summary.csv")
    exports.append(plot_trajectories(trajectory, root / "neurobiology/neural_glial_trajectories.png"))
    alignment = _read_csv(root / "neurobiology/imaging_ephys_alignment.csv")
    exports.append(plot_alignment(alignment, root / "neurobiology/imaging_ephys_alignment.png"))
    apoe = _read_csv(root / "neurobiology/apoe_stratified_risk.csv")
    exports.append(plot_apoe(apoe, root / "neurobiology/apoe_neural_risk.png"))
    drivers = _read_csv(root / "neurobiology/cell_type_driver_scores.csv")
    exports.append(plot_ranked_bar(drivers, "cell_type", "driver_score", root / "neurobiology/cell_type_drivers.png", "Cell types associated with future neuronal degeneration", "Cross-modal driver score", top_n=10, color=COLORS["orange"], profile="nature_single"))
    neuro_transition = _read_csv(root / "neurobiology/neural_glial_transition_matrix.csv", index_col=0)
    exports.append(plot_heatmap(neuro_transition, root / "neurobiology/neural_glial_transition_matrix.png", "Neural–glial state transitions", "Probability", profile="nature_double", vmin=0, vmax=1))
    # Older dynamic-model plots, if present, are also upgraded without changing data.
    dynamic_map = [
        (root / "training/training_history.csv", root / "training/training_history.png"),
    ]
    inventory = pd.DataFrame([asdict(item) for item in exports])
    publication_dir = root / "publication_graphics"
    publication_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(publication_dir / "figure_inventory.csv", index=False)
    baselines = []
    for item in exports:
        png = Path(item.png)
        with Image.open(png) as image:
            width_px, height_px = image.size
        baselines.append({"figure_id": item.figure_id, "png": png.relative_to(root).as_posix(), "perceptual_hash": perceptual_hash(png), "sha256": _hash_file(png), "width_px": width_px, "height_px": height_px, "profile": item.profile, "tolerance": 8})
    pd.DataFrame(baselines).to_csv(publication_dir / "visual_regression_baselines.csv", index=False)
    summary = {
        "framework": "CausaFlux",
        "version": "1.7.0",
        "n_figures": len(exports),
        "profiles": EXPORT_PROFILES,
        "formats_per_figure": ["png", "svg", "pdf", "tiff"],
        "panel_source_data": True,
        "visual_regression": True,
        "synthetic_only": True,
    }
    (publication_dir / "publication_graphics_qc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return inventory


def validate_publication_bundle(reference_dir: str | Path, *, check_hashes: bool = True) -> dict[str, Any]:
    root = Path(reference_dir)
    inventory_path = root / "publication_graphics/figure_inventory.csv"
    baseline_path = root / "publication_graphics/visual_regression_baselines.csv"
    errors: list[str] = []
    if not inventory_path.exists():
        return {"valid": False, "errors": ["Missing figure inventory"], "n_figures": 0}
    inventory = pd.read_csv(inventory_path)
    if len(inventory) < 30:
        errors.append(f"Expected at least 30 figures, found {len(inventory)}")
    for row in inventory.itertuples(index=False):
        for field in ["png", "svg", "pdf", "tiff", "manifest"]:
            path = Path(getattr(row, field))
            if not path.exists():
                errors.append(f"Missing {field}: {path}")
        if not row.source_data or row.source_data == "[]":
            errors.append(f"No panel source data for {row.figure_id}")
    if not baseline_path.exists():
        errors.append("Missing visual-regression baselines")
    elif check_hashes:
        baselines = pd.read_csv(baseline_path)
        for row in baselines.itertuples(index=False):
            image = root / row.png
            if not image.exists():
                errors.append(f"Missing regression image: {image}")
                continue
            result = compare_visual_baseline(image, row.perceptual_hash, int(row.tolerance))
            if not result["valid"]:
                errors.append(f"Visual regression failed for {row.figure_id}: distance {result['distance']}")
    return {"valid": not errors, "errors": errors, "n_figures": int(len(inventory)), "inventory": inventory_path.as_posix(), "baselines": baseline_path.as_posix()}

PUBLICATION_GROUPS = ("core", "spatial", "therapeutics", "biomarkers", "active_learning", "neurobiology")


def rebuild_reference_figure_group(reference_dir: str | Path, group: str) -> pd.DataFrame:
    """Render one bounded publication-graphics group.

    Groups are intentionally process-isolatable because scientific Python stacks can
    retain large raster buffers after repeated 600-dpi TIFF exports.
    """
    if group not in PUBLICATION_GROUPS:
        raise ValueError(f"Unknown publication group {group!r}; choose from {PUBLICATION_GROUPS}")
    root = Path(reference_dir)
    exports: list[FigureExport] = []
    if group == "core":
        modality = _read_csv(root / "multimodal/modality_ablation_metrics.csv")
        exports.append(plot_ranked_bar(modality, "feature_set", "log_loss", root / "multimodal/modality_ablation.png", "Modality and early-fusion benchmark", "Donor-held-out log loss (lower is better)", lower_better=True, top_n=10))
        corr = _read_csv(root / "multimodal/cross_modal_summary_correlations.csv", index_col=0)
        exports.append(plot_heatmap(corr, root / "multimodal/cross_modal_correlation.png", "Cross-modal mean-signal correlations", "Pearson correlation", diverging=True, profile="nature_single", vmin=-1, vmax=1))
        baseline = _read_csv(root / "baselines/linear_baseline_metrics.csv")
        baseline = baseline.assign(display_name=baseline["model"].map(_human) + " · " + baseline["variant"].map(_human))
        exports.append(plot_ranked_bar(baseline, "display_name", "log_loss", root / "baselines/linear_baseline_benchmark.png", "Linear baseline and calibration benchmark", "Donor-held-out log loss (lower is better)", lower_better=True, top_n=12))
        exports.append(plot_reliability(_read_csv(root / "calibration/selected_model_reliability.csv"), _read_csv(root / "calibration/ensemble_reliability.csv"), root / "calibration/reliability_diagram.png"))
        transition = _read_csv(root / "transitions/transition_matrix.csv", index_col=0)
        exports.append(plot_heatmap(transition, root / "transitions/transition_heatmap.png", "Tumor-state transition probabilities", "Probability", profile="nature_single", vmin=0, vmax=1))
        exports.append(plot_causal_dag(_read_csv(root / "graph/causal_nodes.csv"), _read_csv(root / "graph/causal_edges.csv"), root / "graph/causal_graph.png"))
    elif group == "spatial":
        nodes = _read_csv(root / "spatial_graph/graph_nodes.csv")
        contacts = _read_csv(root / "spatial_graph/contact_enrichment.csv")
        cell_types = sorted(set(contacts["cell_type_a"]) | set(contacts["cell_type_b"]))
        contact_matrix = pd.DataFrame(0.0, index=cell_types, columns=cell_types)
        for row in contacts.itertuples(index=False):
            contact_matrix.loc[row.cell_type_a, row.cell_type_b] = row.contact_enrichment
            contact_matrix.loc[row.cell_type_b, row.cell_type_a] = row.contact_enrichment
        exports.append(plot_spatial_atlas(nodes, root / "spatial_graph/spatial_atlas.png"))
        exports.append(plot_heatmap(contact_matrix, root / "spatial_graph/contact_enrichment_heatmap.png", "Spatial contact enrichment over random mixing", "Enrichment", profile="nature_single", vmin=0))
        circuits = _read_csv(root / "spatial_graph/communication_circuit_summary.csv")
        exports.append(plot_communication_circuits(circuits, root / "spatial_graph/communication_circuits.png"))
        comm_edges = _read_csv(root / "spatial_graph/communication_edges.csv")
        exports.append(plot_heterograph(nodes, comm_edges, root / "spatial_graph/heterograph_summary.png"))
        niche = _read_csv(root / "spatial_graph/spatial_niche_summary.csv")
        exports.append(plot_stacked_composition(niche, root / "spatial_graph/spatial_niche_composition.png"))
    elif group == "therapeutics":
        therapy = _read_csv(root / "therapeutics/all_regimen_predictions.csv")
        therapy = therapy.assign(display_name=therapy["regimen_category"].map(_human) + " · " + therapy["regimen_name"].map(_human))
        exports.append(plot_ranked_bar(therapy.nsmallest(15, "rank"), "display_name", "uncertainty_adjusted_utility", root / "therapeutics/therapeutic_ranking.png", "Counterfactual therapeutic ranking", "Uncertainty-adjusted utility", top_n=15, color=COLORS["teal"]))
        exports.append(plot_timing(_read_csv(root / "therapeutics/timing_predictions.csv"), root / "therapeutics/timing_heatmap.png"))
        exports.append(plot_sequence(_read_csv(root / "therapeutics/sequence_predictions.csv"), root / "therapeutics/sequence_comparison.png"))
        exports.append(plot_pareto(therapy, root / "therapeutics/benefit_toxicity_pareto.png"))
        exports.append(plot_waterfall(therapy, root / "therapeutics/counterfactual_waterfall.png"))
    elif group == "biomarkers":
        biomarker = _read_csv(root / "biomarkers/causal_biomarker_ranking.csv")
        exports.append(plot_ranked_bar(biomarker.nsmallest(12, "rank"), "biomarker", "uncertainty_adjusted_score", root / "biomarkers/biomarker_ranking.png", "Early-warning and causal-proximity ranking", "Uncertainty-adjusted biomarker score", top_n=12, color=COLORS["gold"]))
        timecourse = _read_csv(root / "biomarkers/early_warning_timecourse.csv")
        exports.append(plot_biomarker_heatmap(timecourse, biomarker, root / "biomarkers/early_warning_heatmap.png"))
        exports.append(plot_causal_lead(biomarker, root / "biomarkers/causal_lead_map.png"))
        panels = _read_csv(root / "biomarkers/biomarker_panel_metrics.csv")
        exports.append(plot_panel_performance(panels, root / "biomarkers/biomarker_panel_performance.png"))
    elif group == "active_learning":
        experiments = _read_csv(root / "active_learning/round1_experiment_recommendations.csv")
        exports.append(plot_ranked_bar(experiments.nsmallest(14, "rank"), "experiment_name", "priority_score", root / "active_learning/experiment_priority_ranking.png", "Closed-loop experiment priority", "Priority score", top_n=14, color=COLORS["purple"]))
        exports.append(plot_info_gain_by_type(experiments, root / "active_learning/information_gain_by_type.png"))
        exports.append(plot_posterior(_read_csv(root / "active_learning/hypothesis_posterior_history.csv"), root / "active_learning/hypothesis_posterior_update.png"))
        exports.append(plot_batch(_read_csv(root / "active_learning/round1_selected_batch.csv"), root / "active_learning/batch_portfolio.png"))
        exports.append(plot_sampling_times(experiments, root / "active_learning/sampling_time_recommendations.png"))
    elif group == "neurobiology":
        exports.append(plot_trajectories(_read_csv(root / "neurobiology/neural_glial_trajectory_summary.csv"), root / "neurobiology/neural_glial_trajectories.png"))
        exports.append(plot_alignment(_read_csv(root / "neurobiology/imaging_ephys_alignment.csv"), root / "neurobiology/imaging_ephys_alignment.png"))
        exports.append(plot_apoe(_read_csv(root / "neurobiology/apoe_stratified_risk.csv"), root / "neurobiology/apoe_neural_risk.png"))
        exports.append(plot_ranked_bar(_read_csv(root / "neurobiology/cell_type_driver_scores.csv"), "cell_type", "driver_score", root / "neurobiology/cell_type_drivers.png", "Cell types associated with future neuronal degeneration", "Cross-modal driver score", top_n=10, color=COLORS["orange"], profile="nature_single"))
        neuro_transition = _read_csv(root / "neurobiology/neural_glial_transition_matrix.csv", index_col=0)
        exports.append(plot_heatmap(neuro_transition, root / "neurobiology/neural_glial_transition_matrix.png", "Neural–glial state transitions", "Probability", profile="nature_double", vmin=0, vmax=1))
    return pd.DataFrame([asdict(item) for item in exports])


def finalize_publication_inventory(reference_dir: str | Path) -> pd.DataFrame:
    root = Path(reference_dir)
    records: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("**/figure_manifests/*.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        folder = manifest_path.parent.parent
        figure_id = payload["figure_id"]
        record = {
            "figure_id": figure_id,
            "profile": payload["profile"],
            "png": (folder / payload["formats"]["png"]).as_posix(),
            "svg": (folder / payload["formats"]["svg"]).as_posix(),
            "pdf": (folder / payload["formats"]["pdf"]).as_posix(),
            "tiff": (folder / payload["formats"]["tiff"]).as_posix(),
            "source_data": json.dumps([(folder / "source_data" / name).as_posix() for name in payload.get("source_data", [])]),
            "manifest": manifest_path.as_posix(),
            "width_mm": payload["width_mm"],
            "height_mm": payload["height_mm"],
            "dpi": payload["dpi"],
            "synthetic_only": payload.get("synthetic_only", True),
        }
        records.append(record)
    inventory = pd.DataFrame(records).drop_duplicates("figure_id").sort_values("figure_id").reset_index(drop=True)
    publication_dir = root / "publication_graphics"
    publication_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(publication_dir / "figure_inventory.csv", index=False)
    baselines = []
    for row in inventory.itertuples(index=False):
        png = Path(row.png)
        with Image.open(png) as image:
            width_px, height_px = image.size
        baselines.append({"figure_id": row.figure_id, "png": png.relative_to(root).as_posix(), "perceptual_hash": perceptual_hash(png), "sha256": _hash_file(png), "width_px": width_px, "height_px": height_px, "profile": row.profile, "tolerance": 8})
    pd.DataFrame(baselines).to_csv(publication_dir / "visual_regression_baselines.csv", index=False)
    summary = {"framework": "CausaFlux", "version": "1.7.0", "n_figures": int(len(inventory)), "profiles": EXPORT_PROFILES, "formats_per_figure": ["png", "svg", "pdf", "tiff"], "panel_source_data": True, "visual_regression": True, "synthetic_only": True}
    (publication_dir / "publication_graphics_qc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return inventory
