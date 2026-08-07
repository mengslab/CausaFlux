"""Unified real-world data and evidence hub for CausaFlux v1.9.0."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

import pandas as pd

REAL_WORLD_HUB_VERSION = "1.9.0"


@dataclass(frozen=True)
class UserDatasetContract:
    dataset_id: str
    path: str
    data_class: str
    modalities: tuple[str, ...]
    longitudinal: bool
    perturbational: bool
    spatial: bool
    prospective: bool
    outcome_available: bool
    donor_column: str | None = None
    time_column: str | None = None
    intervention_column: str | None = None
    outcome_column: str | None = None
    notes: str = ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_real_world_evidence_matrix(project_root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(root / "realdata_reference" / "benchmark_registry.csv")
    accession = pd.read_csv(root / "realdata_reference" / "accession_manifest.csv")
    biological = pd.read_csv(root / "biological_validation_reference" / "results" / "evidence_ledger.csv")
    primary = pd.read_csv(root / "biological_validation_reference" / "results" / "primary_validation_results.csv")

    registry_out = registry.copy()
    registry_out["integration_role"] = registry_out.domain.map(
        lambda x: "spatial_context" if "spatial" in str(x) else ("intervention" if "perturb" in str(x) else ("multimodal_state" if "molecular" in str(x) or "multiomics" in str(x) else "dynamic_state"))
    )
    registry_out["virtual_cell_readiness"] = registry_out.status.map(lambda x: "metadata_integrated" if "snapshot" in str(x) else "accession_ready")
    registry_path = out / "real_world_benchmark_matrix.csv"
    registry_out.to_csv(registry_path, index=False)

    evidence = biological.copy()
    evidence["evidence_tier"] = "registered_only"
    evidence.loc[evidence.source_cohort_replication_supported.astype(bool), "evidence_tier"] = "real_observational_replication"
    evidence.loc[evidence.perturbational_status.astype(str).str.contains("executed|validated", case=False, regex=True), "evidence_tier"] = "real_perturbational"
    evidence_path = out / "real_world_evidence_ledger.csv"
    evidence.to_csv(evidence_path, index=False)

    primary_path = out / "real_world_primary_validation.csv"
    primary.to_csv(primary_path, index=False)
    accession_path = out / "real_world_accession_manifest.csv"
    accession.to_csv(accession_path, index=False)

    summary = {
        "framework": "CausaFlux",
        "version": REAL_WORLD_HUB_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_families": int(len(registry)),
        "registered_sources": int(len(accession)),
        "real_observational_hypotheses_supported": int((evidence.evidence_tier == "real_observational_replication").sum()),
        "real_perturbational_hypotheses_supported": int((evidence.evidence_tier == "real_perturbational").sum()),
        "real_prospective_virtual_cell_cycles": 0,
        "large_or_controlled_data_redistributed": False,
        "status": "REAL_WORLD_EVIDENCE_INTEGRATED__REAL_PROSPECTIVE_VIRTUAL_CELL_PENDING",
    }
    summary_path = out / "real_world_hub_status.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"registry": registry_path, "evidence": evidence_path, "primary": primary_path, "accession": accession_path, "status": summary_path}


def register_user_dataset(contract: UserDatasetContract, output_dir: str | Path) -> Path:
    path = Path(contract.path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = asdict(contract)
    payload["modalities"] = list(contract.modalities)
    payload["size_bytes"] = path.stat().st_size
    payload["sha256"] = _sha256(path)
    payload["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["source_file_modified_at_utc"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    payload["real_world"] = True
    payload["prospective_evidence_candidate"] = bool(contract.prospective and contract.outcome_available)
    target = out / f"{contract.dataset_id}_dataset_contract.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def preview_tabular_dataset(path: str | Path, max_rows: int = 200) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, nrows=max_rows)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(path, sep="\t", nrows=max_rows)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, nrows=max_rows)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path).head(max_rows)
    else:
        raise ValueError("Preview supports CSV, TSV, XLSX and Parquet. H5MU/NPZ are registered by contract and handled by their native CausaFlux loaders.")
    return {
        "rows_previewed": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "numeric_columns": [str(c) for c in frame.select_dtypes(include="number").columns],
        "missing_fraction": {str(c): float(frame[c].isna().mean()) for c in frame.columns},
    }
