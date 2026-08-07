from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

import causaflux
from causaflux.cli import build_parser
from causaflux.platform import (
    PLATFORM_VERSION,
    build_artifact_manifest,
    environment_snapshot,
    get_demo_registry,
    platform_doctor,
    sha256_file,
    validate_research_platform,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_release_version_and_retained_platform_schema() -> None:
    assert causaflux.__version__ == "2.0.0"
    # Retained integrated-platform reference artifacts preserve their v1.7 schema.
    assert PLATFORM_VERSION == "1.7.0"


def test_demo_registry_has_ten_supported_profiles() -> None:
    demos = get_demo_registry(project_root())
    assert {demo.demo_id for demo in demos} == {
        "foundation_pretraining",
        "spatiotemporal_digital_tissue",
        "intervention_generalization",
        "multimodal_dynamic_state",
        "dynamic_model_benchmark",
        "cancer_quickstart",
        "neurobiology_quickstart",
        "integrated_reference",
        "realdata_registry",
        "biological_validation",
    }
    assert {demo.domain for demo in demos} == {"foundation-pretraining", "spatiotemporal-tissue", "intervention-generalization", "multimodal-dynamic", "dynamic-modeling", "cancer", "neurobiology", "cross-domain", "cross-domain-real-data", "neurobiology-real-data"}
    assert all(Path(demo.config).exists() for demo in demos)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_bytes(b"causaflux-v1")
    assert sha256_file(path) == hashlib.sha256(b"causaflux-v1").hexdigest()


def test_artifact_manifest_is_sorted_and_excludes_provenance(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "provenance").mkdir()
    (tmp_path / "data" / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "provenance" / "self.csv").write_text("self")
    frame = build_artifact_manifest(tmp_path)
    assert frame["relative_path"].tolist() == ["a.txt", "data/b.txt"]
    assert frame["sha256"].str.len().eq(64).all()


def test_environment_snapshot_contains_reproducibility_fields() -> None:
    snapshot = environment_snapshot()
    assert snapshot["framework"] == "CausaFlux"
    assert snapshot["version"] == "1.7.0"
    assert snapshot["python"]
    assert "packages" in snapshot
    assert "numpy" in snapshot["packages"]


def test_platform_doctor_reports_packaged_project_readiness() -> None:
    report = platform_doctor(project_root())
    assert report["required_files_present"]
    assert report["demo_count"] == 10
    assert report["ready"] == report["python_supported"]


def test_incomplete_platform_output_fails_validation(tmp_path: Path) -> None:
    report = validate_research_platform(tmp_path, verify_hashes=False)
    assert not report.valid
    assert any(check.status == "fail" for check in report.checks)


def test_reference_demo_passes_platform_validation() -> None:
    report = validate_research_platform(project_root() / "reference_demo", verify_hashes=True)
    assert report.valid
    assert not [check for check in report.checks if check.status == "fail"]


def test_cli_exposes_v1_platform_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["version"]).command == "version"
    assert parser.parse_args(["doctor"]).command == "doctor"
    assert parser.parse_args(["demo-list"]).command == "demo-list"
    args = parser.parse_args(["platform-validate", "--input", "reference_demo"])
    assert args.command == "platform-validate"
