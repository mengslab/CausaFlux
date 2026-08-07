import json
from pathlib import Path

import pandas as pd
import pytest

from causaflux.prospective_loop import (
    CONTRACT_VERSION,
    ProspectiveLoopConfig,
    default_experiment_catalog,
    experiment_contract_schema,
    ingest_experimental_qc,
    run_prospective_loop,
    validate_prospective_loop,
    write_contract_bundle,
)


def test_contract_bundle_has_lims_eln_qc_and_outcome_schemas(tmp_path):
    write_contract_bundle(tmp_path)
    assert (tmp_path / "experiment_contract.schema.json").exists()
    assert (tmp_path / "experimental_qc.schema.json").exists()
    assert (tmp_path / "outcome_contract.schema.json").exists()
    assert (tmp_path / "ELN_TEMPLATE.md").exists()
    schema = experiment_contract_schema()
    assert schema["properties"]["contract_version"]["const"] == CONTRACT_VERSION
    for field in ["model_freeze_id", "preregistration_id", "expected_cost", "randomization_block"]:
        assert field in schema["required"]


def test_qc_ingestion_rejects_unknown_experiment():
    contract = pd.DataFrame({"experiment_id": ["KNOWN"]})
    qc = pd.DataFrame([
        {"cycle_id": 1, "experiment_id": "UNKNOWN", "sample_id": "S1", "assay_status": "pass", "qc_pass": True, "usable_for_primary_endpoint": True}
    ])
    with pytest.raises(ValueError, match="outside the locked contract"):
        ingest_experimental_qc(qc, contract)


def test_reference_loop_completes_three_locked_cycles_and_gate(tmp_path):
    config = ProspectiveLoopConfig(seed=180, max_cycles=3, min_cycles=3)
    result = run_prospective_loop(tmp_path, config=config)
    report = validate_prospective_loop(tmp_path, require_gate=True)
    assert report["valid"] is True
    assert report["completed_cycles"] == 3
    assert report["software_gate"] == "PASS"
    assert result.gate["cycle3_independent_confirmation_or_falsification"] is True
    assert result.gate["real_prospective_claim_authorized"] is False
    assert len(result.gate["passing_metrics"]) >= 1
    assert (tmp_path / "report" / "index.html").exists()


def test_failed_assay_is_costed_but_not_evaluated(tmp_path):
    run_prospective_loop(tmp_path, ProspectiveLoopConfig(seed=180, synthetic_failure_experiment="IMG_MITO_24H"))
    ledger = pd.read_csv(tmp_path / "experiment_cost_ledger.csv")
    failed = ledger.loc[ledger["status"] == "failed_qc"]
    assert len(failed) >= 1
    assert (failed["failed_assay_cost"] > 0).all()
    assert (failed["evaluable_cost"] == 0).all()


def test_preregistered_prediction_tampering_is_detected(tmp_path):
    run_prospective_loop(tmp_path, ProspectiveLoopConfig(seed=180))
    path = tmp_path / "cycle_1" / "preregistration" / "preregistered_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "predicted_mean"] = float(frame.loc[0, "predicted_mean"]) + 5.0
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_prospective_loop(tmp_path)


def test_catalog_contains_prespecified_non_ai_order_and_orthogonal_assays():
    catalog = default_experiment_catalog()
    assert catalog["baseline_order"].is_unique
    assert catalog["baseline_order"].min() == 1
    assert catalog["assay_family"].nunique() >= 5
    assert catalog["confirmation_group"].nunique() >= 4
