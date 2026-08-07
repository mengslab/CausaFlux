from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from causaflux.dynamic_benchmark import (
    DynamicBenchmarkConfig,
    MODEL_ORDER,
    audit_split,
    build_model,
    external_benchmark_contract,
    generate_dynamic_benchmark_data,
    make_split,
    run_dynamic_benchmark,
    validate_dynamic_benchmark,
    save_external_benchmark_npz,
    load_external_benchmark_npz,
    attach_precomputed_embeddings,
)


def test_dynamic_fixture_has_complete_histories() -> None:
    cfg = DynamicBenchmarkConfig(replicates_per_history=2)
    data = generate_dynamic_benchmark_data(cfg)
    assert data.observations.shape == (90, cfg.steps, cfg.observation_dim)
    assert data.interventions.shape == (90, cfg.steps, cfg.intervention_dim)
    assert len(set(data.history_ids)) == 45
    assert set(np.unique(data.fates)).issubset({0, 1, 2})


def test_split_policies_prevent_required_leakage() -> None:
    data = generate_dynamic_benchmark_data(DynamicBenchmarkConfig(replicates_per_history=2))
    history = make_split(data, "perturbation_history")
    audit = audit_split(data, history, "perturbation_history")
    assert not audit["history_leakage"]
    donor = make_split(data, "donor_holdout")
    donor_audit = audit_split(data, donor, "donor_holdout")
    assert not donor_audit["donor_leakage"]
    for mode in ["dose_holdout", "sequence_holdout", "temporal_extrapolation"]:
        split = make_split(data, mode)
        assert all(len(split[name]) > 0 for name in ["train", "validation", "test"])


def test_all_model_registry_members_forecast_expected_shapes() -> None:
    cfg = DynamicBenchmarkConfig(replicates_per_history=1, hidden_dim=32)
    batch = 3
    context_obs = torch.randn(batch, cfg.context_steps, cfg.observation_dim)
    context_int = torch.randn(batch, cfg.context_steps, cfg.intervention_dim)
    context_times = torch.sort(torch.rand(batch, cfg.context_steps) * 48, dim=1).values
    future_int = torch.randn(batch, cfg.horizon, cfg.intervention_dim)
    future_times = context_times[:, -1:] + torch.cumsum(torch.rand(batch, cfg.horizon) * 12 + 4, dim=1)
    for name in MODEL_ORDER:
        model = build_model(name, cfg)
        out = model(context_obs, context_int, context_times, future_int, future_times)
        assert out["mean"].shape == (batch, cfg.horizon, cfg.observation_dim)
        assert out["logvar"].shape == out["mean"].shape
        assert out["fate_logits"].shape == (batch, 3)


def test_external_contract_requires_history_and_donor_identity() -> None:
    contract = external_benchmark_contract()
    assert contract["schema_version"] == "1.7.0"
    assert "history_ids" in contract["required_arrays"]
    assert "donor_ids" in contract["required_arrays"]
    assert "perturbation_history" in contract["supported_split_modes"]


def test_mini_dynamic_benchmark_passes_exit_gate(tmp_path: Path) -> None:
    cfg = DynamicBenchmarkConfig(
        replicates_per_history=3,
        epochs=8,
        patience=3,
        hidden_dim=32,
        batch_size=32,
        bootstrap_replicates=20,
    )
    status = run_dynamic_benchmark(tmp_path, cfg)
    assert status["gate"]["status"] == "PASS"
    assert status["gate"]["performance_gate_passed"]
    assert not status["gate"]["foundation_pretraining_allowed"]
    assert status["gate"]["evaluation_scope"] == "synthetic_software_fixture"
    result = validate_dynamic_benchmark(tmp_path)
    assert result["valid"]
    gate = json.loads((tmp_path / "foundation_pretraining_gate.json").read_text())
    assert gate["winning_dynamic_model"] in status["gate"]["passing_dynamic_models"]
    assert (tmp_path / "figures" / "trajectory_forecast_benchmark.svg").exists()


def test_external_npz_roundtrip_and_embedding_adapter(tmp_path: Path) -> None:
    data = generate_dynamic_benchmark_data(DynamicBenchmarkConfig(replicates_per_history=1))
    path = save_external_benchmark_npz(data, tmp_path / "fixture.npz")
    loaded = load_external_benchmark_npz(path)
    assert loaded.observations.shape == data.observations.shape
    embeddings = np.zeros((len(loaded), 5), dtype=np.float32)
    augmented = attach_precomputed_embeddings(loaded, embeddings, prefix="scvi", mode="append")
    assert augmented.observations.shape[-1] == loaded.observations.shape[-1] + 5
    assert augmented.feature_names[-1] == "scvi_004"
