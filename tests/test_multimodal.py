from pathlib import Path

import numpy as np

from causaflux.causal_data import CancerDemoConfig, generate_cancer_demo
from causaflux.multimodal import (
    MODALITY_ORDER,
    MultimodalDemoConfig,
    evaluate_modality_ablation,
    generate_multimodal_mudata,
    modality_feature_frame,
    read_csv_bundle,
    read_multimodal,
    validate_multimodal,
    write_csv_bundle,
    write_multimodal,
)


def _small_mdata():
    frame = generate_cancer_demo(
        CancerDemoConfig(n_donors=4, clones_per_donor=6, non_tumor_cells_per_type=1, seed=43)
    )
    return frame, generate_multimodal_mudata(
        frame, MultimodalDemoConfig(seed=43, missing_modality_rate=0.02)
    )


def test_multimodal_schema_and_alignment():
    frame, mdata = _small_mdata()
    report = validate_multimodal(mdata)
    assert report["valid"]
    assert list(mdata.mod) == list(MODALITY_ORDER)
    assert mdata.n_obs == len(frame)
    assert set(mdata.obs_names.astype(str)) == set(frame["row_id"].astype(str))
    assert all(name in mdata.obs for name in [f"has_{m}" for m in MODALITY_ORDER])


def test_h5mu_roundtrip(tmp_path: Path):
    _, mdata = _small_mdata()
    path = write_multimodal(mdata, tmp_path / "demo.h5mu")
    restored = read_multimodal(path)
    assert validate_multimodal(restored)["valid"]
    assert restored.n_obs == mdata.n_obs
    assert set(restored.mod) == set(mdata.mod)


def test_csv_bundle_roundtrip(tmp_path: Path):
    _, mdata = _small_mdata()
    directory = write_csv_bundle(mdata, tmp_path / "bundle")
    restored = read_csv_bundle(directory)
    assert validate_multimodal(restored)["valid"]
    assert restored.n_obs == mdata.n_obs


def test_fusion_and_ablation_outputs():
    _, mdata = _small_mdata()
    integrated = modality_feature_frame(mdata)
    assert any(column.startswith("rna__") for column in integrated)
    assert any(column.startswith("atac__") for column in integrated)
    metrics, contributions = evaluate_modality_ablation(integrated, seed=43)
    assert set(metrics["feature_set"]) == set(MODALITY_ORDER) | {"fusion"}
    assert np.isfinite(metrics["log_loss"]).all()
    assert set(contributions["removed_modality"]) == set(MODALITY_ORDER)
