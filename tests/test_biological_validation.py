from pathlib import Path
import json
import pandas as pd

from causaflux.biological_validation import (
    load_hypothesis_registry,
    run_biological_validation,
    write_biological_validation,
    validate_biological_validation,
)

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "benchmarks" / "snapshots" / "sea_ad"


def test_hypothesis_registry_is_frozen_and_mixed_status():
    specs = load_hypothesis_registry()
    assert len(specs) == 6
    assert sum(s.status == "executed" for s in specs) == 3
    assert all(len(s.preregistration_sha256) == 64 for s in specs)


def test_seaad_primary_hypotheses_replicate_without_causal_claims():
    run = run_biological_validation(SNAP, n_boot=30, seed=9)
    assert len(run.primary_results) == 6
    assert run.primary_results.groupby("hypothesis_id")["supported"].all().all()
    assert run.qc["causal_claims"] == 0
    assert run.qc["clinical_claims"] == 0
    assert not run.evidence_ledger["causal_claim_permitted"].any()


def test_endpoint_replication_and_established_methods_present():
    run = run_biological_validation(SNAP, n_boot=20, seed=10)
    assert set(run.endpoint_replication["endpoint"]) == {"Gabitto 2024", "Travaglini 2026", "Kana 2026"}
    assert {"Mann-Whitney U", "Welch t-test", "Spearman", "Pearson", "Age/sex-adjusted OLS"}.issubset(set(run.method_comparison["method"]))


def test_validation_output_and_manuscript_package(tmp_path):
    run = run_biological_validation(SNAP, n_boot=20, seed=11)
    out = write_biological_validation(run, tmp_path / "validation")
    report = validate_biological_validation(out)
    assert report["valid"], report
    inventory = pd.read_csv(out / "manuscript_package" / "figure_inventory.csv")
    assert len(inventory) == 4
    status = json.loads((out / "biological_validation_status.json").read_text())
    assert status["external_dataset_replication_established"] == 0
    assert status["perturbational_validation_established"] == 0
