"""CausaFlux v2.0 stable software release and real-evidence gate workflow."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, shutil
import pandas as pd

from .evidence_ledger import build_reference_ledger, merge_external_evidence, validate_ledger
from .longitudinal_realdata import write_public_dataset_bundle, gse8057_sample_metadata
from .v2_release_gate import evaluate_v2_release_gate, validate_v2_output
from .v2_report import generate_v2_figures, generate_v2_report

V2_RELEASE_VERSION="2.0.0"


def _load(path:Path): return json.loads(path.read_text(encoding="utf-8"))
def _sha(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def _software_checks(root:Path)->dict[str,bool]:
    return {
        "dynamic_software_gate":_load(root/"dynamic_benchmark_reference/dynamic_benchmark_status.json").get("gate",{}).get("status")=="PASS",
        "multimodal_software_gate":bool(_load(root/"multimodal_dynamic_reference/multimodal_exit_gate.json").get("software_exit_gate_passed")),
        "intervention_software_gate":_load(root/"intervention_generalization_reference/intervention_exit_gate.json").get("software_generalization_gate")=="PASS",
        "spatial_software_gate":_load(root/"spatiotemporal_tissue_reference/spatiotemporal_exit_gate.json").get("software_spatiotemporal_gate")=="PASS",
        "prospective_locking_software_gate":_load(root/"prospective_loop_reference/prospective_exit_gate.json").get("software_gate")=="PASS",
        "v1_9_integrated_virtual_cell_gate":_load(root/"virtual_cell_reference/validation/prospective_virtual_cell_status.json").get("software_integrated_virtual_cell_gate")=="PASS",
        "realdata_registry_gate":bool(_load(root/"realdata_reference/realdata_status.json").get("registry_valid")),
    }


def _external_ledger_files(directory:Path)->list[Path]:
    if not directory.exists(): return []
    exact=directory/"evidence_ledger.csv"
    files=[exact] if exact.exists() else []
    files += [p for p in sorted(directory.glob("*.csv")) if "evidence" in p.name.lower() and p != exact]
    return files


def _artifact_manifest(out:Path)->Path:
    rows=[]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name!="release_artifact_manifest.csv": rows.append({"relative_path":p.relative_to(out).as_posix(),"size_bytes":p.stat().st_size,"sha256":_sha(p)})
    path=out/"release_artifact_manifest.csv"; pd.DataFrame(rows).to_csv(path,index=False); return path


def run_v2_release(project_root:str|Path, output_dir:str|Path, *, external_evidence_dir:str|Path|None=None)->dict:
    root=Path(project_root).resolve(); out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    for sub in ["evidence","validation","real_longitudinal","report","figures","ai","real_world","prospective"]: (out/sub).mkdir(exist_ok=True)
    # Preserve the v1.9 user-facing virtual-cell outputs as the AI baseline inside v2.
    baseline = root / "virtual_cell_reference"
    for sub in ("ai", "real_world", "prospective"):
        src = baseline / sub
        if src.exists():
            shutil.copytree(src, out / sub, dirs_exist_ok=True)
    base=build_reference_ledger(root,out/"evidence")
    files=_external_ledger_files(Path(external_evidence_dir).resolve()) if external_evidence_dir else []
    ledger=base
    if files: ledger=merge_external_evidence(base,files,out/"evidence/evidence_ledger.csv")
    bundle=write_public_dataset_bundle(out/"real_longitudinal")
    gse8057_sample_metadata().to_csv(out/"real_longitudinal/gse8057_curated_design_metadata.csv",index=False)
    checks=_software_checks(root); (out/"validation/software_readiness.json").write_text(json.dumps(checks,indent=2,sort_keys=True),encoding="utf-8")
    matrix,status=evaluate_v2_release_gate(ledger,software_checks=checks); matrix.to_csv(out/"validation/v2_release_matrix.csv",index=False); (out/"validation/v2_release_gate.json").write_text(json.dumps(status,indent=2,sort_keys=True),encoding="utf-8")
    generate_v2_figures(out); report=generate_v2_report(out); manifest=_artifact_manifest(out)
    validation=validate_v2_output(out)
    run={"framework":"CausaFlux","version":V2_RELEASE_VERSION,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"software_checks":checks,"external_evidence_files":[str(x) for x in files],"release_gate":status,"validation":validation,"report":str(report.relative_to(out)),"artifact_manifest":str(manifest.relative_to(out))}
    (out/"run_manifest.json").write_text(json.dumps(run,indent=2,sort_keys=True),encoding="utf-8")
    return run
