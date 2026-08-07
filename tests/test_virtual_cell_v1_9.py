from pathlib import Path
import json

import numpy as np
import pandas as pd

from causaflux.virtual_cell import DEFAULT_SCENARIOS, load_module_evidence, simulate_scenario
from causaflux.virtual_cell_release import run_virtual_cell_release
from causaflux.virtual_cell_validation import validate_virtual_cell_release
from causaflux.real_world_hub import UserDatasetContract, register_user_dataset, preview_tabular_dataset
from causaflux.ui_app import self_test


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_model_router_is_normalized_and_validated():
    frame = load_module_evidence(project_root())
    assert len(frame) == 6
    assert set(frame["gate"]) == {"PASS"}
    assert np.isclose(frame["normalized_weight"].sum(), 1.0)
    assert frame["reliability"].between(0.35, 0.98).all()


def test_virtual_cell_scenario_has_bounded_states():
    evidence = load_module_evidence(project_root())
    trajectory, components = simulate_scenario(DEFAULT_SCENARIOS[1], evidence)
    mean_cols = [c for c in trajectory if c.endswith("_mean")]
    assert len(trajectory) == 37
    assert trajectory[mean_cols].to_numpy().min() >= 0.0
    assert trajectory[mean_cols].to_numpy().max() <= 1.0
    assert not components.empty


def test_real_world_registration_hashes_input(tmp_path):
    source = tmp_path / "real.csv"
    pd.DataFrame({"donor": ["D1", "D2"], "time": [0, 24], "outcome": [0.2, 0.8]}).to_csv(source, index=False)
    contract = UserDatasetContract(
        dataset_id="test_real", path=str(source), data_class="experimental", modalities=("rna",),
        longitudinal=True, perturbational=False, spatial=False, prospective=True, outcome_available=True,
        donor_column="donor", time_column="time", outcome_column="outcome",
    )
    target = register_user_dataset(contract, tmp_path / "registry")
    payload = json.loads(target.read_text())
    assert len(payload["sha256"]) == 64
    assert payload["prospective_evidence_candidate"] is True
    preview = preview_tabular_dataset(source)
    assert preview["rows_previewed"] == 2


def test_full_v1_9_release(tmp_path):
    out = tmp_path / "v190"
    run_virtual_cell_release(project_root(), out)
    validation = validate_virtual_cell_release(out)
    assert validation["valid"] is True
    assert validation["software_gate"] == "PASS"
    assert validation["real_prospective_gate"] == "PENDING"
    inventory = pd.read_csv(out / "figures" / "figure_inventory.csv")
    assert len(inventory) >= 6
    assert inventory["validated"].all()


def test_real_prospective_requirement_is_not_falsely_satisfied(tmp_path):
    out = tmp_path / "v190"
    run_virtual_cell_release(project_root(), out)
    validation = validate_virtual_cell_release(out, require_real_prospective=True)
    assert validation["valid"] is False
    assert validation["real_prospective_gate"] == "PENDING"


def test_ui_self_test():
    assert self_test()["status"] == "PASS"
