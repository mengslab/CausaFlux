"""Integrated v1.9.0 virtual-cell release workflow."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
from typing import Any

import pandas as pd

from .real_world_hub import build_real_world_evidence_matrix
from .virtual_cell import run_virtual_cell_ensemble
from .virtual_cell_report import generate_virtual_cell_figures, generate_virtual_cell_report
from .virtual_cell_validation import build_validation_matrix, validate_virtual_cell_release

RELEASE_VERSION = "1.9.0"


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def _copy_prospective_reference(root: Path, out: Path) -> None:
    src=root/'prospective_loop_reference'; dst=out/'prospective'; dst.mkdir(parents=True,exist_ok=True)
    names=['cycle_calibration.csv','strategy_comparison.csv','prospective_exit_gate.json','all_preregistered_predictions.csv','all_locked_outcomes.csv','all_experimental_qc.csv','posterior_history.csv','experiment_cost_ledger.csv']
    for name in names:
        shutil.copy2(src/name,dst/name)
    contracts=dst/'contracts'; contracts.mkdir(exist_ok=True)
    for path in (src/'contracts').iterdir():
        if path.is_file(): shutil.copy2(path,contracts/path.name)


def _write_manifest(out: Path) -> Path:
    files=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and '.DS_Store' not in path.name and path.name != 'release_artifact_manifest.csv':
            files.append({'relative_path':path.relative_to(out).as_posix(),'size_bytes':path.stat().st_size,'sha256':_sha(path)})
    frame=pd.DataFrame(files)
    path=out/'release_artifact_manifest.csv'; frame.to_csv(path,index=False); return path


def run_virtual_cell_release(project_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root=Path(project_root).resolve(); out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    (out/'ai').mkdir(exist_ok=True); (out/'real_world').mkdir(exist_ok=True); (out/'validation').mkdir(exist_ok=True)
    _copy_prospective_reference(root,out)
    ai=run_virtual_cell_ensemble(root,out/'ai')
    rw=build_real_world_evidence_matrix(root,out/'real_world')
    matrix,status=build_validation_matrix(root,out/'validation')
    figures=generate_virtual_cell_figures(out)
    report=generate_virtual_cell_report(out)
    manifest=_write_manifest(out)
    validation=validate_virtual_cell_release(out)
    run_manifest={
        'framework':'CausaFlux','version':RELEASE_VERSION,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'project_root':str(root),'output_dir':str(out),'software_validation':validation,
        'virtual_cell_status':status,
        'ai_outputs':{k:str(v.relative_to(out)) for k,v in ai.items()},
        'real_world_outputs':{k:str(v.relative_to(out)) for k,v in rw.items()},
        'figure_count':int(len(figures)),'report':str(report.relative_to(out)),'artifact_manifest':str(manifest.relative_to(out)),
    }
    (out/'run_manifest.json').write_text(json.dumps(run_manifest,indent=2,sort_keys=True),encoding='utf-8')
    return run_manifest
