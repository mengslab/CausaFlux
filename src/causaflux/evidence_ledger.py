"""Claim-linked evidence ledger for CausaFlux v2.0.0.

The ledger deliberately separates software evidence from real biological evidence.
A v2 prospective-validation claim cannot be satisfied by synthetic fixtures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
from typing import Any, Iterable

import pandas as pd

EVIDENCE_LEDGER_VERSION = "2.0.0"
REAL_EVIDENCE_KINDS = {
    "real_longitudinal_perturbation",
    "real_multimodal_perturbation",
    "real_spatial_perturbation",
    "prospective_cycle",
    "external_lab_replication",
    "independent_cohort_replication",
    "distribution_shift_calibration",
    "real_negative_result",
    "real_failed_assay",
}

REQUIRED_CLAIMS: tuple[tuple[str, str], ...] = (
    ("CF2_PHASE1_DYNAMIC_SUPERIORITY", "Phase 1 dynamic superiority established on real held-out longitudinal perturbation data"),
    ("CF2_MULTIMODAL_FORECASTING", "Multimodal forecasting validated on real data"),
    ("CF2_UNSEEN_INTERVENTION_GENERALIZATION", "Unseen intervention generalization validated on real held-out interventions"),
    ("CF2_SPATIAL_CONTEXT_BENEFIT", "Spatial context adds predictive value on independent real tissue data"),
    ("CF2_PROSPECTIVE_CYCLE_1", "Prospective experimental Cycle 1 completed with preregistered locked predictions"),
    ("CF2_PROSPECTIVE_CYCLE_2", "Prospective experimental Cycle 2 completed after model update"),
    ("CF2_EXTERNAL_REPLICATION", "Independent cohort or external laboratory replication completed"),
    ("CF2_SHIFT_CALIBRATION", "Calibrated uncertainty maintained under prespecified distribution shift"),
    ("CF2_REAL_LONGITUDINAL_CONNECTED", "Actual longitudinal perturbation dataset connected to model training and prospective loop"),
    ("CF2_NEGATIVE_FAILURE_REPORTING", "Failures and negative results are explicitly retained and reported"),
)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    claim_id: str
    status: str
    evidence_kind: str
    source: str
    independent: bool = False
    prospective: bool = False
    synthetic: bool = False
    negative_or_failure: bool = False
    cycle: int | None = None
    metric: str = ""
    value: float | None = None
    threshold: str = ""
    notes: str = ""
    sha256: str = ""
    recorded_at_utc: str = ""

    def normalized(self) -> dict[str, Any]:
        payload = asdict(self)
        if not payload["recorded_at_utc"]:
            payload["recorded_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return payload


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def claim_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(REQUIRED_CLAIMS, columns=["claim_id", "claim_text"])


def write_ledger(records: Iterable[EvidenceRecord | dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.normalized() if isinstance(r, EvidenceRecord) else dict(r) for r in records]
    frame = pd.DataFrame(rows)
    expected = list(EvidenceRecord.__dataclass_fields__)
    for col in expected:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[expected]
    frame.to_csv(path, index=False)
    return path


def load_ledger(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for col in ("independent", "prospective", "synthetic", "negative_or_failure"):
        if col in frame.columns:
            frame[col] = frame[col].astype(str).str.lower().isin({"true", "1", "yes"})
    if "cycle" in frame.columns:
        frame["cycle"] = pd.to_numeric(frame["cycle"], errors="coerce")
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _record_from_json_status(evidence_id: str, claim_id: str, source: Path, passed: bool, notes: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        claim_id=claim_id,
        status="SOFTWARE_PASS" if passed else "SOFTWARE_FAIL",
        evidence_kind="software_fixture",
        source=str(source),
        synthetic=True,
        sha256=sha256_file(source),
        notes=notes,
    )


def build_reference_ledger(project_root: str | Path, output_dir: str | Path) -> Path:
    """Build a transparent baseline ledger from retained references.

    These rows document software readiness but intentionally do not satisfy the
    real v2 prospective-validation claims.
    """
    root = Path(project_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[EvidenceRecord] = []

    refs = [
        ("SW_DYNAMIC", "CF2_PHASE1_DYNAMIC_SUPERIORITY", root / "dynamic_benchmark_reference/dynamic_benchmark_status.json", lambda d: d.get("gate", {}).get("status") == "PASS", "Synthetic dynamic superiority software gate only."),
        ("SW_MULTIMODAL", "CF2_MULTIMODAL_FORECASTING", root / "multimodal_dynamic_reference/multimodal_exit_gate.json", lambda d: bool(d.get("software_exit_gate_passed")), "Synthetic multimodal forecasting gate only."),
        ("SW_INTERVENTION", "CF2_UNSEEN_INTERVENTION_GENERALIZATION", root / "intervention_generalization_reference/intervention_exit_gate.json", lambda d: d.get("software_generalization_gate") == "PASS", "Synthetic unseen-intervention software gate only."),
        ("SW_SPATIAL", "CF2_SPATIAL_CONTEXT_BENEFIT", root / "spatiotemporal_tissue_reference/spatiotemporal_exit_gate.json", lambda d: d.get("software_spatiotemporal_gate") == "PASS", "Synthetic spatial-context software gate only."),
    ]
    for evid, claim, source, predicate, notes in refs:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows.append(_record_from_json_status(evid, claim, source, bool(predicate(payload)), notes))

    prospective_path = root / "prospective_loop_reference/prospective_exit_gate.json"
    p = json.loads(prospective_path.read_text(encoding="utf-8"))
    for cycle in (1, 2):
        rows.append(EvidenceRecord(
            evidence_id=f"SW_PROSPECTIVE_CYCLE_{cycle}", claim_id=f"CF2_PROSPECTIVE_CYCLE_{cycle}",
            status="SOFTWARE_PASS" if p.get("software_gate") == "PASS" else "SOFTWARE_FAIL",
            evidence_kind="software_fixture", source=str(prospective_path), prospective=True,
            synthetic=True, cycle=cycle, sha256=sha256_file(prospective_path),
            notes="Prospectively locked synthetic workflow; not a real experimental cycle.",
        ))

    biological_path = root / "biological_validation_reference/biological_validation_status.json"
    b = json.loads(biological_path.read_text(encoding="utf-8"))
    rows.append(EvidenceRecord(
        evidence_id="REAL_OBSERVATIONAL_REPLICATION", claim_id="CF2_EXTERNAL_REPLICATION",
        status="PARTIAL", evidence_kind="independent_cohort_replication", source=str(biological_path),
        independent=True, synthetic=False, sha256=sha256_file(biological_path),
        notes=f"Retained observational source-cohort replication={b.get('independent_source_cohort_replication_established', 0)}; does not satisfy perturbational external replication.",
    ))

    # Explicitly record unmet claims rather than silently omitting them.
    pending = {
        "CF2_SHIFT_CALIBRATION": "No real distribution-shift calibration evidence ingested.",
        "CF2_REAL_LONGITUDINAL_CONNECTED": "Public dataset adapters are available, but no local real longitudinal perturbation training run is bundled as biological evidence.",
        "CF2_NEGATIVE_FAILURE_REPORTING": "Reporting machinery is present; real-study completeness must be attested from the actual prospective study ledger.",
    }
    for idx, (claim, note) in enumerate(pending.items(), start=1):
        rows.append(EvidenceRecord(
            evidence_id=f"PENDING_{idx:02d}", claim_id=claim, status="PENDING",
            evidence_kind="release_boundary", source="CausaFlux v2.0.0 release gate", notes=note,
        ))

    path = write_ledger(rows, out / "evidence_ledger.csv")
    claim_registry_frame().to_csv(out / "claim_registry.csv", index=False)
    return path


def merge_external_evidence(base_ledger: str | Path, evidence_files: Iterable[str | Path], output: str | Path) -> Path:
    frames = [load_ledger(base_ledger)]
    for source in evidence_files:
        frame = load_ledger(source)
        frame["source"] = frame["source"].fillna(str(source))
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    # External evidence with the same evidence_id supersedes the bundled placeholder.
    merged = merged.drop_duplicates(subset=["evidence_id"], keep="last")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return Path(output)


def validate_ledger(path: str | Path) -> dict[str, Any]:
    frame = load_ledger(path)
    errors: list[str] = []
    required_columns = set(EvidenceRecord.__dataclass_fields__)
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        errors.append(f"missing columns: {missing}")
    if frame.get("evidence_id", pd.Series(dtype=str)).duplicated().any():
        errors.append("evidence_id values must be unique")
    known_claims = {claim for claim, _ in REQUIRED_CLAIMS}
    unknown = sorted(set(frame.get("claim_id", pd.Series(dtype=str)).dropna()) - known_claims)
    if unknown:
        errors.append(f"unknown claim ids: {unknown}")
    claim_counts = frame.groupby("claim_id").size().to_dict() if len(frame) else {}
    unlinked = [claim for claim in known_claims if claim_counts.get(claim, 0) == 0]
    if unlinked:
        errors.append(f"claims without ledger rows: {unlinked}")
    return {
        "valid": not errors,
        "errors": errors,
        "n_records": int(len(frame)),
        "n_claims_linked": int(len(set(frame.get("claim_id", [])) & known_claims)),
        "n_required_claims": len(known_claims),
        "negative_or_failure_records": int(frame.get("negative_or_failure", pd.Series(dtype=bool)).fillna(False).sum()) if len(frame) else 0,
    }
