from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from causaflux.visualization.publication import (
    EXPORT_PROFILES,
    compare_visual_baseline,
    export_figure,
    perceptual_hash,
    validate_publication_bundle,
)


def test_export_profiles_are_publication_sized() -> None:
    assert EXPORT_PROFILES["nature_single"]["width_mm"] == 89.0
    assert EXPORT_PROFILES["nature_double"]["width_mm"] == 183.0
    assert EXPORT_PROFILES["cell_double"]["dpi"] == 600


def test_export_figure_writes_all_formats_and_source_data(tmp_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3, 2))
    frame = pd.DataFrame({"x": [0, 1, 2], "y": [0.1, 0.4, 0.8]})
    ax.plot(frame["x"], frame["y"])
    result = export_figure(
        fig,
        tmp_path / "test_panel.png",
        profile="nature_single",
        source_data={"panel_a": frame},
        metadata={"purpose": "visual regression fixture"},
    )
    plt.close(fig)
    for path in [result.png, result.svg, result.pdf, result.tiff, result.manifest]:
        assert Path(path).exists()
    assert result.source_data and Path(result.source_data[0]).exists()
    payload = json.loads(Path(result.manifest).read_text())
    assert payload["dpi"] == 600
    assert payload["synthetic_only"] is True


def test_visual_regression_detects_identical_image(tmp_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    result = export_figure(fig, tmp_path / "regression.png", profile="nature_single", source_data=pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
    plt.close(fig)
    expected = perceptual_hash(result.png)
    comparison = compare_visual_baseline(result.png, expected, tolerance=0)
    assert comparison["valid"]
    assert comparison["distance"] == 0


def test_reference_publication_bundle_is_complete() -> None:
    root = Path(__file__).resolve().parents[1] / "reference_demo"
    if not (root / "publication_graphics/figure_inventory.csv").exists():
        return
    report = validate_publication_bundle(root, check_hashes=True)
    assert report["valid"], report["errors"]
    assert report["n_figures"] >= 30
