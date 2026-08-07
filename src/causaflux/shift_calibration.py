"""Distribution-shift uncertainty calibration checks for CausaFlux v2.0.0."""
from __future__ import annotations
from pathlib import Path
import json
from typing import Any
import numpy as np
import pandas as pd

SHIFT_CALIBRATION_VERSION="2.0.0"


def evaluate_shift_calibration(frame: pd.DataFrame, *, z:float=1.6448536269514722, target_coverage:float=0.90, tolerance:float=0.08, min_group_coverage:float=0.75) -> tuple[pd.DataFrame,dict[str,Any]]:
    required={"observed","predicted_mean","predicted_sd","shift_group"}
    missing=sorted(required-set(frame.columns))
    if missing: raise ValueError(f"missing columns: {missing}")
    x=frame.copy(); x["predicted_sd"]=pd.to_numeric(x["predicted_sd"],errors="coerce").clip(lower=1e-8)
    x["observed"]=pd.to_numeric(x["observed"],errors="coerce"); x["predicted_mean"]=pd.to_numeric(x["predicted_mean"],errors="coerce")
    x=x.dropna(subset=["observed","predicted_mean","predicted_sd","shift_group"])
    x["lower"]=x.predicted_mean-z*x.predicted_sd; x["upper"]=x.predicted_mean+z*x.predicted_sd
    x["covered"]=(x.observed>=x.lower)&(x.observed<=x.upper)
    x["standardized_error"]=(x.observed-x.predicted_mean)/x.predicted_sd
    rows=[]
    for group,g in x.groupby("shift_group"):
        rows.append({"shift_group":group,"n":len(g),"coverage_90":float(g.covered.mean()),"rmse":float(np.sqrt(np.mean((g.observed-g.predicted_mean)**2))),"mean_abs_standardized_error":float(np.mean(np.abs(g.standardized_error)))})
    metrics=pd.DataFrame(rows)
    overall=float(x.covered.mean()) if len(x) else float("nan")
    coverage_gap=abs(overall-target_coverage) if np.isfinite(overall) else float("inf")
    group_min=float(metrics.coverage_90.min()) if len(metrics) else float("nan")
    gate=bool(len(x)>0 and coverage_gap<=tolerance and group_min>=min_group_coverage)
    status={"framework":"CausaFlux","version":SHIFT_CALIBRATION_VERSION,"real_distribution_shift_gate":"PASS" if gate else "FAIL","n":int(len(x)),"overall_coverage_90":overall,"target_coverage":target_coverage,"coverage_tolerance":tolerance,"minimum_group_coverage":group_min,"required_min_group_coverage":min_group_coverage}
    return metrics,status


def evaluate_shift_calibration_file(input_path:str|Path, output_dir:str|Path)->dict[str,Any]:
    input_path=Path(input_path); suffix=input_path.suffix.lower()
    frame=pd.read_csv(input_path) if suffix==".csv" else pd.read_csv(input_path,sep="\t") if suffix in {".tsv",".txt"} else pd.read_excel(input_path)
    metrics,status=evaluate_shift_calibration(frame); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(out/"distribution_shift_calibration.csv",index=False); (out/"distribution_shift_calibration_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    return status
