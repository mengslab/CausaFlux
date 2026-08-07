"""Hard release gate for the CausaFlux v2 prospectively validated claim."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any
import pandas as pd
from .evidence_ledger import REQUIRED_CLAIMS, REAL_EVIDENCE_KINDS, load_ledger, validate_ledger, sha256_file

V2_GATE_VERSION="2.0.0"
PASS_STATUSES={"PASS","VALIDATED","CONFIRMED","SUPPORTED"}


def _locked_source(row: pd.Series, ledger_dir: Path) -> bool:
    source = str(row.get("source", "") or "").strip()
    expected = str(row.get("sha256", "") or "").strip().lower()
    if not source or not expected:
        return False
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = (ledger_dir / path).resolve()
    if not path.is_file():
        return False
    try:
        return sha256_file(path).lower() == expected
    except OSError:
        return False


def _real_pass(frame:pd.DataFrame,claim_id:str,ledger_dir:Path)->pd.DataFrame:
    sub=frame[(frame.claim_id==claim_id)&frame.status.astype(str).str.upper().isin(PASS_STATUSES)].copy()
    sub=sub[(~sub.synthetic.fillna(False)) & sub.evidence_kind.astype(str).isin(REAL_EVIDENCE_KINDS)]
    if len(sub):
        sub["provenance_locked"] = sub.apply(lambda row: _locked_source(row, ledger_dir), axis=1)
        sub = sub[sub.provenance_locked]
    return sub


def evaluate_v2_release_gate(ledger_path:str|Path, *, software_checks:dict[str,bool]|None=None)->tuple[pd.DataFrame,dict[str,Any]]:
    ledger_path=Path(ledger_path); ledger=load_ledger(ledger_path); ledger_check=validate_ledger(ledger_path); ledger_dir=ledger_path.parent
    checks=[]
    for claim_id,claim_text in REQUIRED_CLAIMS:
        matched=_real_pass(ledger,claim_id,ledger_dir)
        passed=len(matched)>0
        detail=f"{len(matched)} qualifying real evidence record(s)" if passed else "no qualifying real PASS evidence"
        checks.append({"criterion":claim_id,"description":claim_text,"passed":passed,"detail":detail})
    # Strengthen the prospective and replication semantics beyond a single status label.
    cycles=ledger[(ledger.evidence_kind=="prospective_cycle")&(~ledger.synthetic.fillna(False))&ledger.status.astype(str).str.upper().isin(PASS_STATUSES)].copy()
    if len(cycles): cycles=cycles[cycles.apply(lambda row:_locked_source(row,ledger_dir),axis=1)]
    distinct_cycles=sorted(set(pd.to_numeric(cycles.cycle,errors="coerce").dropna().astype(int)))
    cycle_ok=len([c for c in distinct_cycles if c>=1])>=2
    for row in checks:
        if row["criterion"] in {"CF2_PROSPECTIVE_CYCLE_1","CF2_PROSPECTIVE_CYCLE_2"}: row["passed"]=row["passed"] and cycle_ok
    rep=ledger[(ledger.claim_id=="CF2_EXTERNAL_REPLICATION")&ledger.independent.fillna(False)&(~ledger.synthetic.fillna(False))&ledger.status.astype(str).str.upper().isin(PASS_STATUSES)].copy()
    if len(rep): rep=rep[rep.apply(lambda row:_locked_source(row,ledger_dir),axis=1)]
    for row in checks:
        if row["criterion"]=="CF2_EXTERNAL_REPLICATION": row["passed"]=len(rep)>0; row["detail"]=f"independent qualifying replications={len(rep)}"
    neg=ledger[(ledger.claim_id=="CF2_NEGATIVE_FAILURE_REPORTING")&ledger.status.astype(str).str.upper().isin(PASS_STATUSES)&(~ledger.synthetic.fillna(False))].copy()
    if len(neg): neg=neg[neg.apply(lambda row:_locked_source(row,ledger_dir),axis=1)]
    for row in checks:
        if row["criterion"]=="CF2_NEGATIVE_FAILURE_REPORTING": row["passed"]=len(neg)>0; row["detail"]="real-study negative/failure reporting completeness attested" if len(neg) else "real-study reporting completeness not attested"
    matrix=pd.DataFrame(checks)
    software_checks=software_checks or {}
    software_ready=all(software_checks.values()) if software_checks else True
    all_real=bool(matrix.passed.all()) and ledger_check["valid"]
    status={
        "framework":"CausaFlux","version":V2_GATE_VERSION,"generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "software_release_ready":bool(software_ready),"evidence_ledger_valid":bool(ledger_check["valid"]),
        "real_required_criteria":int(len(matrix)),"real_required_criteria_passed":int(matrix.passed.sum()),
        "real_prospective_cycles_completed":distinct_cycles,"independent_replication_records":int(len(rep)),
        "prospectively_validated_virtual_cell":bool(software_ready and all_real),
        "release_claim_status":"PROSPECTIVELY_VALIDATED" if software_ready and all_real else "NOT_YET_ELIGIBLE",
        "authorization_boundary":"The v2 prospectively validated claim is emitted only when every criterion is supported by qualifying non-synthetic evidence in the locked evidence ledger.",
    }
    return matrix,status


def validate_v2_output(output_dir:str|Path, *, require_prospectively_validated:bool=False)->dict[str,Any]:
    out=Path(output_dir)
    required=["evidence/evidence_ledger.csv","validation/v2_release_gate.json","validation/v2_release_matrix.csv","real_longitudinal/longitudinal_perturbation_registry.csv","report/index.html","figures/figure_inventory.csv","release_artifact_manifest.csv"]
    missing=[x for x in required if not (out/x).exists()]
    status=json.loads((out/"validation/v2_release_gate.json").read_text()) if not missing else {}
    valid=not missing and status.get("software_release_ready") is True and (status.get("prospectively_validated_virtual_cell") is True if require_prospectively_validated else True)
    return {"valid":bool(valid),"missing":missing,"software_release_ready":status.get("software_release_ready"),"prospectively_validated_virtual_cell":status.get("prospectively_validated_virtual_cell"),"release_claim_status":status.get("release_claim_status","UNKNOWN"),"require_prospectively_validated":require_prospectively_validated}
