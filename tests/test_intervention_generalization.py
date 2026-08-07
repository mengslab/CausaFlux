from pathlib import Path
import json

import numpy as np
import pandas as pd

from causaflux.intervention_generalization import (
    ADAPTER_NAMES,
    HOLDOUT_TYPES,
    InterventionGeneralizationConfig,
    adapter_registry_frame,
    generate_intervention_generalization_data,
    load_external_intervention_npz,
    positivity_support_diagnostics,
    run_intervention_generalization_benchmark,
    save_external_intervention_npz,
    validate_intervention_generalization,
)


def small_cfg():
    return InterventionGeneralizationConfig(seed=150, replicates=2, bootstrap_replicates=10)


def test_fixture_contains_all_generalization_axes_and_embeddings():
    data = generate_intervention_generalization_data(small_cfg())
    test = data.frame[data.frame.split.eq("test")]
    assert set(HOLDOUT_TYPES).issubset(set(test.holdout_type))
    assert data.gene_embeddings.shape[0] == small_cfg().n_genes
    assert data.compound_embeddings.shape[0] == small_cfg().n_compounds
    assert len(data.response_names) == small_cfg().response_dim


def test_support_diagnostics_flag_out_of_support_rows():
    data = generate_intervention_generalization_data(small_cfg())
    support = positivity_support_diagnostics(data)
    assert support.positivity_warning.any()
    dose = support[support.holdout_type.eq("unseen_dose")]
    assert (~dose.dose_within_training_range).all()


def test_external_contract_roundtrip(tmp_path: Path):
    data = generate_intervention_generalization_data(small_cfg())
    path = save_external_intervention_npz(data, tmp_path / "fixture.npz")
    restored = load_external_intervention_npz(path)
    assert len(restored.frame) == len(data.frame)
    assert restored.response_names == data.response_names
    assert set(restored.frame.holdout_type) == set(data.frame.holdout_type)


def test_adapter_registry_is_explicitly_external():
    registry = adapter_registry_frame()
    assert set(ADAPTER_NAMES) == set(registry.name)
    assert registry.notes.str.contains("extern", case=False).all()


def test_intervention_benchmark_gate_and_validator(tmp_path: Path):
    out = tmp_path / "benchmark"
    status = run_intervention_generalization_benchmark(out, small_cfg(), require_gate=True)
    assert status["valid"]
    gate = json.loads((out / "intervention_exit_gate.json").read_text())
    assert gate["software_generalization_gate"] == "PASS"
    assert json.loads((out / "external_established_model_gate.json").read_text())["status"].startswith("BLOCKED_")
    comparison = pd.read_csv(out / "model_comparison.csv")
    main = comparison[comparison.model.eq("CausaFluxInterventionGeneralizer")].set_index("holdout_type")
    for axis in ("unseen_perturbation", "unseen_dose", "unseen_combination", "unseen_sequence"):
        assert np.isfinite(main.loc[axis, "rmse"])
    validation = validate_intervention_generalization(out)
    assert validation["valid"], validation
