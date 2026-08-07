"""Real longitudinal perturbation adapters for CausaFlux v2.0.0."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .dynamic_benchmark import DynamicBenchmarkData, save_external_benchmark_npz, run_dynamic_benchmark, DynamicBenchmarkConfig

LONGITUDINAL_REALDATA_VERSION = "2.0.0"


@dataclass(frozen=True)
class PublicLongitudinalDataset:
    dataset_id: str
    repository: str
    accession: str
    title: str
    organism: str
    modality: str
    perturbations: str
    temporal_design: str
    download_page: str
    integration_role: str
    bundled_data: bool = False


PUBLIC_DATASETS = (
    PublicLongitudinalDataset(
        dataset_id="geo_gse8057_platinum_timecourse",
        repository="NCBI GEO", accession="GSE8057",
        title="Expression data from ovarian cancer cells with time-course and concentration-profiles",
        organism="Homo sapiens", modality="transcriptomic microarray",
        perturbations="cisplatin; oxaliplatin; vehicle",
        temporal_design="pre, 0h, 2h, 6h, 16h, 24h after 2h drug exposure; dose-response at 16h",
        download_page="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8057",
        integration_role="real longitudinal perturbation benchmark",
    ),
    PublicLongitudinalDataset(
        dataset_id="lincs_gse70138_l1000",
        repository="NIH LINCS / NCBI GEO", accession="GSE70138",
        title="L1000 Connectivity Map perturbational profiles, LINCS Phase II",
        organism="Homo sapiens", modality="L1000 transcriptomic signatures",
        perturbations="small molecules; genetic loss/gain of function",
        temporal_design="condition metadata include perturbagen, cell line, dose and time point",
        download_page="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138",
        integration_role="large-scale unseen-intervention/time/dose benchmark",
    ),
    PublicLongitudinalDataset(
        dataset_id="lincs_gse101406_multimodal",
        repository="NIH LINCS / NCBI GEO", accession="GSE101406",
        title="Perturbational proteomic and transcriptional profiles of 90 small molecules",
        organism="Homo sapiens", modality="P100 phosphoproteomics; GCP chromatin PTM; L1000 transcriptomics",
        perturbations="90 small molecules in six cell lines",
        temporal_design="P100 3h; L1000 6h; GCP 24h",
        download_page="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE101406",
        integration_role="real multimodal perturbation validation",
    ),
)


def public_dataset_registry() -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in PUBLIC_DATASETS])


def write_public_dataset_bundle(output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    registry = public_dataset_registry()
    registry_path = out / "longitudinal_perturbation_registry.csv"
    registry.to_csv(registry_path, index=False)
    plan = registry[["dataset_id","repository","accession","download_page","integration_role"]].copy()
    plan["action"] = "Download from authoritative repository; retain repository checksum/metadata; transform with CausaFlux longitudinal contract."
    plan["redistributed"] = False
    plan_path = out / "download_and_ingestion_plan.csv"; plan.to_csv(plan_path, index=False)
    contract = longitudinal_table_contract()
    contract_path = out / "longitudinal_table_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    template = pd.DataFrame({
        "trajectory_id":["traj_001","traj_001","traj_001","traj_001"],
        "donor_id":["donor_001"]*4,"time":[0.0,2.0,6.0,24.0],"history_id":["drugA_1uM"]*4,
        "target":["drugA"]*4,"dose":[1.0]*4,"sequence":["continuous"]*4,"fate":["recovery"]*4,
        "int__drugA":[0.0,1.0,1.0,1.0],"feature__state_1":[0.1,0.4,0.6,0.2],"feature__state_2":[0.8,0.6,0.3,0.7],
    })
    template_path = out / "longitudinal_table_template.csv"; template.to_csv(template_path, index=False)
    return {"registry": registry_path, "plan": plan_path, "contract": contract_path, "template": template_path}


def longitudinal_table_contract() -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "required_columns": ["trajectory_id","donor_id","time","history_id","target","dose","sequence","fate"],
        "feature_columns": "one or more columns prefixed feature__",
        "intervention_columns": "one or more columns prefixed int__",
        "requirements": [
            "each trajectory contains at least four ordered observations",
            "all trajectories used together have the same number of time points for the current benchmark adapter",
            "time is numeric and strictly increasing within each trajectory",
            "history_id encodes the complete intervention history and must be held out as a group for unseen-history evaluation",
            "donor_id is the biological or independent experimental unit",
            "fate is constant within trajectory",
            "future intervention schedule is known before outcome measurement",
        ],
    }


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def read_longitudinal_table(path: str | Path) -> pd.DataFrame:
    path=Path(path); suffix=path.suffix.lower()
    if suffix==".csv": return pd.read_csv(path)
    if suffix in {".tsv",".txt"}: return pd.read_csv(path,sep="\t")
    if suffix in {".xlsx",".xls"}: return pd.read_excel(path)
    if suffix==".parquet": return pd.read_parquet(path)
    raise ValueError("Supported real longitudinal tables: CSV, TSV, XLSX, Parquet")


def validate_longitudinal_table(frame: pd.DataFrame) -> dict[str, Any]:
    required=set(longitudinal_table_contract()["required_columns"])
    errors=[]
    missing=sorted(required-set(frame.columns))
    if missing: errors.append(f"missing required columns: {missing}")
    features=[c for c in frame.columns if str(c).startswith("feature__")]
    interventions=[c for c in frame.columns if str(c).startswith("int__")]
    if not features: errors.append("at least one feature__ column is required")
    if not interventions: errors.append("at least one int__ column is required")
    if errors: return {"valid":False,"errors":errors,"feature_columns":features,"intervention_columns":interventions}
    local=frame.copy(); local["time"]=pd.to_numeric(local["time"],errors="coerce")
    if local.time.isna().any(): errors.append("time contains non-numeric values")
    lengths=[]
    for tid,g in local.groupby("trajectory_id",sort=False):
        g=g.sort_values("time"); lengths.append(len(g))
        if len(g)<4: errors.append(f"trajectory {tid} has fewer than four observations")
        if len(g)>1 and not np.all(np.diff(g.time.to_numpy(float))>0): errors.append(f"trajectory {tid} time is not strictly increasing")
        for col in ["donor_id","history_id","target","dose","sequence","fate"]:
            if g[col].nunique(dropna=False)!=1: errors.append(f"trajectory {tid} has non-constant {col}")
    if len(set(lengths))>1: errors.append("trajectories have unequal observation counts; resample/alignment is required before this adapter")
    return {"valid":not errors,"errors":errors,"n_rows":int(len(local)),"n_trajectories":int(local.trajectory_id.nunique()),"steps":int(lengths[0]) if lengths else 0,"feature_columns":features,"intervention_columns":interventions}


def table_to_dynamic_benchmark(frame: pd.DataFrame) -> DynamicBenchmarkData:
    check=validate_longitudinal_table(frame)
    if not check["valid"]: raise ValueError("; ".join(check["errors"]))
    features=check["feature_columns"]; intervention_cols=check["intervention_columns"]
    fate_names=sorted(frame["fate"].astype(str).unique().tolist())
    fate_index={x:i for i,x in enumerate(fate_names)}
    groups=[]
    for tid,g in frame.groupby("trajectory_id",sort=True): groups.append((str(tid),g.sort_values("time")))
    observations=np.stack([g[features].to_numpy(np.float32) for _,g in groups])
    interventions=np.stack([g[intervention_cols].to_numpy(np.float32) for _,g in groups])
    times=np.stack([g["time"].to_numpy(np.float32) for _,g in groups])
    first=lambda g,c:g[c].iloc[0]
    return DynamicBenchmarkData(
        observations=observations, interventions=interventions, times=times,
        fates=np.asarray([fate_index[str(first(g,"fate"))] for _,g in groups],dtype=np.int64),
        trajectory_ids=np.asarray([tid for tid,_ in groups],dtype=object),
        donor_ids=np.asarray([str(first(g,"donor_id")) for _,g in groups],dtype=object),
        history_ids=np.asarray([str(first(g,"history_id")) for _,g in groups],dtype=object),
        targets=np.asarray([str(first(g,"target")) for _,g in groups],dtype=object),
        doses=np.asarray([float(first(g,"dose")) for _,g in groups],dtype=np.float32),
        sequences=np.asarray([str(first(g,"sequence")) for _,g in groups],dtype=object),
        feature_names=[str(c).removeprefix("feature__") for c in features],
        intervention_names=[str(c).removeprefix("int__") for c in intervention_cols], fate_names=fate_names,
    )


def convert_longitudinal_table(input_path: str | Path, output_npz: str | Path, metadata_output: str | Path | None=None) -> dict[str, Any]:
    frame=read_longitudinal_table(input_path); data=table_to_dynamic_benchmark(frame); save_external_benchmark_npz(data,output_npz)
    manifest={"framework":"CausaFlux","version":LONGITUDINAL_REALDATA_VERSION,"source":str(Path(input_path).resolve()),"source_sha256":sha256_file(input_path),"output_npz":str(Path(output_npz).resolve()),"n_trajectories":len(data),"steps":int(data.observations.shape[1]),"features":data.feature_names,"interventions":data.intervention_names,"real_data":True}
    if metadata_output:
        Path(metadata_output).write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
    return manifest


def run_real_longitudinal_benchmark(input_path: str | Path, output_dir: str | Path, *, model_names: Sequence[str] | None=None, seed:int=200) -> dict[str, Any]:
    frame=read_longitudinal_table(input_path); data=table_to_dynamic_benchmark(frame)
    steps=int(data.observations.shape[1]); context=max(2,min(steps-2,steps//2))
    cfg=DynamicBenchmarkConfig(seed=seed,steps=steps,context_steps=context,observation_dim=int(data.observations.shape[-1]),intervention_dim=int(data.interventions.shape[-1]),epochs=18,patience=4,bootstrap_replicates=50)
    result=run_dynamic_benchmark(output_dir,cfg,model_names=model_names,data=data)
    manifest={"source":str(Path(input_path).resolve()),"source_sha256":sha256_file(input_path),"real_longitudinal_perturbation":True,"dynamic_result":result}
    Path(output_dir,"real_longitudinal_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    return manifest


def gse8057_sample_metadata() -> pd.DataFrame:
    """Curated public sample-design metadata from the GEO series description.

    Expression values are intentionally not redistributed by CausaFlux.
    """
    rows=[]
    # Time-course sample identifiers listed on GEO. 16 h treated samples are also dose-response replicates.
    names={
        "cisplatin": {"pre":["GSM198888","GSM198889"],"0":["GSM198893","GSM198894"],"2":["GSM198895","GSM198896"],"6":["GSM198897","GSM198898"],"24":["GSM198899"]},
        "oxaliplatin": {"pre":["GSM198891","GSM198892"],"0":["GSM198900","GSM198901"],"2":["GSM198902","GSM198903"],"6":["GSM198904","GSM198905"],"24":["GSM198906"]},
        "vehicle": {"pre":["GSM198890"],"0":["GSM198907"],"2":["GSM198908"],"6":["GSM198909"],"16":["GSM198910","GSM198911","GSM198912","GSM198913"]},
    }
    dose={"cisplatin":25.0,"oxaliplatin":32.0,"vehicle":0.0}
    for drug,times in names.items():
        for time_label,accessions in times.items():
            time=-2.0 if time_label=="pre" else float(time_label)
            for rep,acc in enumerate(accessions,1): rows.append({"sample_accession":acc,"perturbation":drug,"dose_uM":dose[drug],"time_hours_after_exposure":time,"replicate":rep,"design_role":"time_course"})
    return pd.DataFrame(rows)
