from pathlib import Path

import numpy as np
import torch

from causaflux.multimodal_dynamic import (
    MODALITY_ORDER,
    MultimodalDynamicConfig,
    ProductOfExpertsFusion,
    generate_multimodal_dynamic_data,
    _trapezoid_integral,
    history_split,
    load_external_multimodal_npz,
    run_multimodal_dynamic_benchmark,
    save_external_multimodal_npz,
    split_audit,
    validate_multimodal_dynamic_benchmark,
)



def test_numpy_126_trapezoid_compatibility(monkeypatch):
    # NumPy 1.26 does not expose np.trapezoid; the supported Intel-macOS
    # environment must fall back to the numerically equivalent np.trapz path.
    monkeypatch.setattr(np, "trapezoid", None, raising=False)
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([0.0, 1.0, 2.0], dtype=float)
    assert np.isclose(_trapezoid_integral(y, x), 2.0)


def test_multimodal_fixture_has_all_required_modalities():
    cfg = MultimodalDynamicConfig(replicates_per_history=2, epochs=2, bootstrap_replicates=2)
    data = generate_multimodal_dynamic_data(cfg)
    assert set(data.modalities) == set(MODALITY_ORDER)
    assert set(data.observed_masks) == set(MODALITY_ORDER)
    assert data.modalities["rna"].shape[:2] == (72, cfg.steps)
    assert 0.15 < float(data.destructive_label.mean()) < 0.85
    assert data.baseline_covariates.shape[0] == len(data)


def test_history_split_reports_intentional_donor_overlap_not_donor_holdout():
    data = generate_multimodal_dynamic_data(MultimodalDynamicConfig(replicates_per_history=2))
    audit = split_audit(data, history_split(data, 140))
    assert audit["history_split_valid"] is True
    assert audit["history_overlap_train_test"] == []
    assert audit["donor_holdout_enforced"] is False
    assert audit["donor_overlap_expected"] is True
    assert len(audit["donor_overlap_train_test"]) > 0


def test_product_of_experts_ignores_missing_expert():
    fusion = ProductOfExpertsFusion()
    mu1 = torch.ones(2, 3, 4)
    lv1 = torch.zeros_like(mu1)
    mu2 = torch.full_like(mu1, 8.0)
    lv2 = torch.zeros_like(mu2)
    mask1 = torch.ones(2, 3)
    mask2 = torch.zeros(2, 3)
    fused, logvar = fusion([(mu1, lv1), (mu2, lv2)], [mask1, mask2])
    # Unit Gaussian prior plus expert 1 => mean 0.5; expert 2 cannot contribute.
    assert torch.allclose(fused, torch.full_like(fused, 0.5), atol=1e-6)
    assert torch.isfinite(logvar).all()


def test_external_multimodal_contract_roundtrip(tmp_path: Path):
    cfg = MultimodalDynamicConfig(replicates_per_history=2)
    data = generate_multimodal_dynamic_data(cfg)
    path = save_external_multimodal_npz(data, tmp_path / "fixture.npz")
    loaded = load_external_multimodal_npz(path)
    assert len(loaded) == len(data)
    assert np.array_equal(loaded.history_ids, data.history_ids)
    for modality in MODALITY_ORDER:
        assert loaded.modalities[modality].shape == data.modalities[modality].shape
        assert loaded.observed_masks[modality].shape == data.observed_masks[modality].shape


def test_multimodal_benchmark_exit_gate_and_mnar_outputs(tmp_path: Path):
    cfg = MultimodalDynamicConfig(
        replicates_per_history=3,
        epochs=10,
        patience=3,
        bootstrap_replicates=4,
        hidden_dim=32,
        latent_dim=16,
        batch_size=24,
    )
    result = run_multimodal_dynamic_benchmark(tmp_path / "benchmark", cfg)
    assert result["gate"]["software_exit_gate_passed"] is True
    assert result["gate"]["foundation_pretraining_authorization"].startswith("BLOCKED")
    assert (tmp_path / "benchmark" / "missingness_sensitivity.csv").exists()
    assert (tmp_path / "benchmark" / "cross_modal_forecasting.csv").exists()
    qc = validate_multimodal_dynamic_benchmark(tmp_path / "benchmark")
    assert qc["valid"] is True
