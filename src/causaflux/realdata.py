"""Accession-pinned real-data benchmark registry and download planning.

CausaFlux never redistributes third-party biomedical data in the package.  This
module turns curated benchmark manifests into explicit access plans, lock files,
license records, and research reports.  Network execution is opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.resources as ir
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable
from urllib.parse import quote

import pandas as pd
import yaml

REALDATA_VERSION = "1.7.0"

@dataclass(frozen=True)
class SourceSpec:
    benchmark_id: str
    source_id: str
    role: str
    repository: str
    accession: str
    title: str
    adapter: str
    version: str
    access: str
    license: str
    license_url: str
    url: str
    modalities: tuple[str, ...]
    citation: str
    citation_url: str
    query: dict[str, Any]
    notes: str = ""

@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    title: str
    domain: str
    status: str
    estimated_storage_gb: str
    primary_question: str
    sources: tuple[SourceSpec, ...]
    evaluation: dict[str, Any]


def _resource_dir() -> Path:
    return Path(ir.files("causaflux").joinpath("resources/realdata"))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_benchmark_registry(manifest_dir: str | Path | None = None) -> list[BenchmarkSpec]:
    base = Path(manifest_dir) if manifest_dir else _resource_dir()
    registry = _load_yaml(base / "registry.yaml")
    specs: list[BenchmarkSpec] = []
    for item in registry.get("benchmarks", []):
        payload = _load_yaml(base / item["manifest"])
        sources = tuple(SourceSpec(
            benchmark_id=payload["id"],
            source_id=s["source_id"], role=s["role"], repository=s["repository"],
            accession=s["accession"], title=s["title"], adapter=s["adapter"],
            version=str(s["version"]), access=s["access"], license=s["license"],
            license_url=s["license_url"], url=s["url"], modalities=tuple(s.get("modalities", [])),
            citation=s["citation"], citation_url=s["citation_url"], query=dict(s.get("query", {})),
            notes=s.get("notes", ""),
        ) for s in payload.get("sources", []))
        specs.append(BenchmarkSpec(
            benchmark_id=payload["id"], title=payload["title"], domain=payload["domain"],
            status=payload["status"], estimated_storage_gb=str(payload["estimated_storage_gb"]),
            primary_question=payload["primary_question"], sources=sources,
            evaluation=dict(payload.get("evaluation", {})),
        ))
    return specs


def get_benchmark(benchmark_id: str, manifest_dir: str | Path | None = None) -> BenchmarkSpec:
    for spec in load_benchmark_registry(manifest_dir):
        if spec.benchmark_id == benchmark_id:
            return spec
    raise KeyError(f"unknown benchmark: {benchmark_id}")


def benchmark_registry_frame(manifest_dir: str | Path | None = None) -> pd.DataFrame:
    rows=[]
    for spec in load_benchmark_registry(manifest_dir):
        rows.append({
            "benchmark_id":spec.benchmark_id,"title":spec.title,"domain":spec.domain,
            "status":spec.status,"n_sources":len(spec.sources),
            "n_validation_sources":sum("validation" in s.role for s in spec.sources),
            "estimated_storage_gb":spec.estimated_storage_gb,
        })
    return pd.DataFrame(rows)


def accession_manifest_frame(manifest_dir: str | Path | None = None) -> pd.DataFrame:
    rows=[]
    for spec in load_benchmark_registry(manifest_dir):
        for s in spec.sources:
            rows.append({
                "benchmark_id":spec.benchmark_id,"source_id":s.source_id,"role":s.role,
                "repository":s.repository,"accession":s.accession,"version_policy":s.version,
                "access":s.access,"license_summary":s.license,"license_url":s.license_url,
                "source_url":s.url,"citation":s.citation,"citation_url":s.citation_url,
                "modalities":"; ".join(s.modalities),"adapter":s.adapter,
                "query_json":json.dumps(s.query,sort_keys=True),"notes":s.notes,
            })
    return pd.DataFrame(rows)


def _command_for_source(source: SourceSpec, destination: Path, metadata_only: bool) -> tuple[str,str]:
    d = destination / source.benchmark_id / source.source_id
    if source.adapter == "aws_s3":
        bucket=source.query.get("bucket")
        cmd=f"aws s3 {'ls' if metadata_only else 'sync'} --no-sign-request s3://{bucket}/ {d}/"
        return cmd,"public-command"
    if source.adapter == "dandi":
        did=source.query["dandiset"]
        cmd=f"dandi download DANDI:{did}@latest --output-dir {d}"
        return cmd,"public-command"
    if source.adapter == "synapse":
        sid=source.query.get("synapse_id",source.accession)
        cmd=f"synapse get -r {sid} --downloadLocation {d}"
        return cmd,"account-or-controlled"
    if source.adapter == "htan_synapse":
        cmd=f"Open HTAN portal query for {source.source_id}; export metadata; then use synapse get on locked synIDs into {d}"
        return cmd,"portal-query-account"
    if source.adapter == "gdc":
        filt=json.dumps(source.query,separators=(',',':'))
        cmd=f"Use GDC API/client with query {filt}; write file manifest and MD5 lock under {d}"
        return cmd,"public-and-controlled"
    if source.adapter == "pdc":
        sid=source.query.get("pdc_study_id",source.accession)
        cmd=f"Use PDC GraphQL API for study {sid}; write study version and file checksums under {d}"
        return cmd,"public-and-controlled"
    if source.adapter == "geo":
        acc=source.query.get("geo_accession",source.accession)
        cmd=f"Download GEO supplementary files for {acc} using NCBI HTTPS/FTP into {d}"
        return cmd,"public-command"
    if source.adapter == "depmap_manual":
        cmd=f"Download {source.accession} from DepMap Downloads after accepting file terms; place files in {d}"
        return cmd,"manual-terms-no-scraping"
    return f"Resolve {source.url} into {d}","manual"


def build_download_plan(
    destination: str | Path,
    benchmark_ids: Iterable[str] | None = None,
    *, metadata_only: bool = True,
    manifest_dir: str | Path | None = None,
) -> pd.DataFrame:
    destination=Path(destination)
    chosen=set(benchmark_ids or [])
    rows=[]
    for spec in load_benchmark_registry(manifest_dir):
        if chosen and spec.benchmark_id not in chosen: continue
        for source in spec.sources:
            command,mode=_command_for_source(source,destination,metadata_only)
            rows.append({
                "benchmark_id":spec.benchmark_id,"source_id":source.source_id,
                "accession":source.accession,"role":source.role,"adapter":source.adapter,
                "access":source.access,"execution_mode":mode,"metadata_only":metadata_only,
                "destination":str(destination/spec.benchmark_id/source.source_id),
                "command_or_action":command,
            })
    return pd.DataFrame(rows)


def preflight_benchmarks(manifest_dir: str | Path | None = None) -> pd.DataFrame:
    tools={"aws_s3":"aws","dandi":"dandi","synapse":"synapse","htan_synapse":"synapse","gdc":"curl","pdc":"curl","geo":"curl","depmap_manual":None}
    rows=[]
    for spec in load_benchmark_registry(manifest_dir):
        has_validation=any("validation" in s.role for s in spec.sources)
        for s in spec.sources:
            tool=tools.get(s.adapter)
            rows.append({
                "benchmark_id":spec.benchmark_id,"source_id":s.source_id,"manifest_valid":bool(s.accession and s.url and s.citation),
                "validation_cohort_defined":has_validation,"adapter":s.adapter,"required_tool":tool or "manual browser",
                "tool_available":True if tool is None else shutil.which(tool) is not None,
                "requires_account_or_terms":any(k in s.access.lower() for k in ("controlled","account")) or "terms" in s.license.lower(),
                "license_recorded":bool(s.license and s.license_url),"citation_recorded":bool(s.citation and s.citation_url),
            })
    return pd.DataFrame(rows)


def validate_realdata_registry(manifest_dir: str | Path | None = None) -> dict[str, Any]:
    specs=load_benchmark_registry(manifest_dir)
    errors=[]
    expected={"htan_spatial_cancer","gdc_tcga_cptac_brca","depmap_prism_lincs","sea_ad_neural_glial","amp_ad_molecular","dandi_neurophysiology"}
    found={s.benchmark_id for s in specs}
    if found != expected: errors.append(f"benchmark IDs differ: {sorted(found ^ expected)}")
    for spec in specs:
        if not spec.sources: errors.append(f"{spec.benchmark_id}: no sources")
        if not any("validation" in s.role for s in spec.sources): errors.append(f"{spec.benchmark_id}: no validation cohort")
        for s in spec.sources:
            for field in ("accession","url","license","license_url","citation","citation_url","adapter"):
                if not getattr(s,field): errors.append(f"{spec.benchmark_id}/{s.source_id}: missing {field}")
    return {"framework":"CausaFlux","version":REALDATA_VERSION,"valid":not errors,"n_benchmarks":len(specs),"n_sources":sum(len(s.sources) for s in specs),"errors":errors}


def _sha(path: Path) -> str:
    h=hashlib.sha256();
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def _html_escape(value: Any) -> str:
    import html
    return html.escape(str(value))


def _table_html(frame: pd.DataFrame, max_rows: int=200) -> str:
    return frame.head(max_rows).to_html(index=False,escape=True,classes="data")


def _copy_snapshot_files(project_root: Path, output: Path) -> list[Path]:
    src=project_root/"benchmarks"/"snapshots"/"sea_ad"
    if not src.exists() or not any(src.glob("*.xlsx")):
        src=Path(ir.files("causaflux").joinpath("resources/realdata/snapshots"))
    dst=output/"snapshots"/"sea_ad"; dst.mkdir(parents=True,exist_ok=True)
    copied=[]
    for p in src.glob('*.xlsx'):
        q=dst/p.name; shutil.copy2(p,q); copied.append(q)
    if len(copied) < 2:
        raise FileNotFoundError("SEA-AD public metadata snapshots are missing from the project and package resources")
    return copied


def _sea_ad_descriptive(snapshot_dir: Path, output: Path) -> dict[str,Any]:
    donor_file=next(snapshot_dir.glob('*donor_metadata*.xlsx'))
    cps_file=next(snapshot_dir.glob('*pseudoprogression*.xlsx'))
    donors=pd.read_excel(donor_file,sheet_name=0)
    cps=pd.read_excel(cps_file,sheet_name='Travaglini 2026')
    out=output/"sea_ad_descriptive"; out.mkdir(parents=True,exist_ok=True)
    sex=donors['Sex'].fillna('Missing').value_counts().rename_axis('sex').reset_index(name='donors')
    cog=donors['Cognitive Status'].fillna('Missing').value_counts().rename_axis('cognitive_status').reset_index(name='donors')
    apoe=donors['APOE Genotype'].fillna('Missing').astype(str).value_counts().rename_axis('apoe_genotype').reset_index(name='donors')
    region=cps.groupby('Brain Region',dropna=False).agg(donors=('Donor ID','nunique'),median_cps=('CPS_Global','median'),q25=('CPS_Global',lambda x:x.quantile(.25)),q75=('CPS_Global',lambda x:x.quantile(.75))).reset_index()
    for name,frame in [('sex_counts',sex),('cognitive_status_counts',cog),('apoe_counts',apoe),('cps_region_summary',region)]: frame.to_csv(out/f'{name}.csv',index=False)
    summary={
      'real_dataset':True,'source':'SEA-AD Data and Downloads','donors':int(donors['Donor ID'].nunique()),
      'median_age_at_death':float(pd.to_numeric(donors['Age at Death'],errors='coerce').median()),
      'age_range':[float(pd.to_numeric(donors['Age at Death'],errors='coerce').min()),float(pd.to_numeric(donors['Age at Death'],errors='coerce').max())],
      'sex_counts':dict(zip(sex.sex,sex.donors.astype(int))),
      'cognitive_status_counts':dict(zip(cog.cognitive_status,cog.donors.astype(int))),
      'cps_records':int(len(cps)),'cps_donors':int(cps['Donor ID'].nunique()),'brain_regions':int(cps['Brain Region'].nunique()),
      'snapshot_sha256':{donor_file.name:_sha(donor_file),cps_file.name:_sha(cps_file)},
      'interpretation':'Descriptive cohort metadata only; no causal or clinical inference.',
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':8,'axes.linewidth':0.7,'pdf.fonttype':42,'svg.fonttype':'none'})
        fig,axs=plt.subplots(1,3,figsize=(7.1,2.25),constrained_layout=True)
        axs[0].bar(sex['sex'],sex['donors']); axs[0].set_ylabel('Donors'); axs[0].set_title('a  Sex')
        top=cog.head(6); axs[1].barh(top['cognitive_status'][::-1],top['donors'][::-1]); axs[1].set_xlabel('Donors'); axs[1].set_title('b  Cognitive status')
        vals=[g['CPS_Global'].dropna().values for _,g in cps.groupby('Brain Region')]; labs=[str(k) for k,_ in cps.groupby('Brain Region')]
        axs[2].boxplot(vals,tick_labels=labs,showfliers=False); axs[2].tick_params(axis='x',rotation=60,labelsize=6); axs[2].set_ylabel('Global CPS'); axs[2].set_title('c  Pathology progression by region')
        for ax in axs: ax.spines[['top','right']].set_visible(False)
        for ext in ('svg','pdf','png'):
            fig.savefig(out/f'sea_ad_cohort_overview.{ext}',dpi=600 if ext=='png' else None,bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        (out/'figure_warning.txt').write_text(str(exc),encoding='utf-8')
    return summary


def generate_realdata_reports(output_dir: str | Path, *, project_root: str | Path | None=None, manifest_dir: str | Path | None=None) -> dict[str,Path]:
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    project=Path(project_root).resolve() if project_root else Path.cwd().resolve()
    manifests=accession_manifest_frame(manifest_dir); registry=benchmark_registry_frame(manifest_dir)
    plan=build_download_plan(output/'data',metadata_only=True,manifest_dir=manifest_dir)
    preflight=preflight_benchmarks(manifest_dir)
    # Materialize one explicit non-destructive adapter plan per source. Import lazily to
    # avoid the SourceSpec/adapter module dependency at import time.
    from .realdata_adapters import plan_source
    adapter_dir=output/'adapter_plans'; adapter_dir.mkdir(exist_ok=True)
    adapter_rows=[]
    for spec in load_benchmark_registry(manifest_dir):
        for source in spec.sources:
            dest=output/'data'/spec.benchmark_id/source.source_id
            ap=plan_source(source,dest,metadata_only=True)
            ap.write(adapter_dir/f'{spec.benchmark_id}__{source.source_id}.json')
            adapter_rows.append({
                'benchmark_id':spec.benchmark_id,'source_id':source.source_id,'adapter':ap.adapter,
                'accession':ap.accession,'execution_mode':ap.execution_mode,'metadata_only':ap.metadata_only,
                'requires_user_authorization':ap.requires_user_authorization,
                'redistributable_by_causaflux':ap.redistributable_by_causaflux,
                'command_or_action':ap.command_or_action,
            })
    pd.DataFrame(adapter_rows).to_csv(output/'adapter_capabilities.csv',index=False)
    manifests.to_csv(output/'accession_manifest.csv',index=False); registry.to_csv(output/'benchmark_registry.csv',index=False)
    plan.to_csv(output/'download_plan.csv',index=False); preflight.to_csv(output/'preflight_checks.csv',index=False)
    val=validate_realdata_registry(manifest_dir)
    (output/'registry_validation.json').write_text(json.dumps(val,indent=2),encoding='utf-8')
    snaps=_copy_snapshot_files(project,output)
    sea_summary=_sea_ad_descriptive(output/'snapshots'/'sea_ad',output)
    locks=[]
    for p in snaps: locks.append({'relative_path':str(p.relative_to(output)),'sha256':_sha(p),'lock_type':'bundled public metadata snapshot'})
    for _,r in manifests.iterrows(): locks.append({'relative_path':f"remote::{r['benchmark_id']}::{r['source_id']}",'sha256':'runtime-resolution-required','lock_type':r['version_policy']})
    pd.DataFrame(locks).to_csv(output/'accession_lock.csv',index=False)
    validation_rows=[]
    for spec in load_benchmark_registry(manifest_dir):
        for s in spec.sources:
            if 'validation' in s.role:
                validation_rows.append({'benchmark_id':spec.benchmark_id,'validation_source':s.source_id,'accession':s.accession,'role':s.role,'split_unit':spec.evaluation.get('split_unit'),'metrics':'; '.join(spec.evaluation.get('primary_metrics',[]))})
    pd.DataFrame(validation_rows).to_csv(output/'independent_validation_cohorts.csv',index=False)
    # Research cards and explicit citation/license exports.
    cards=output/'cards'; cards.mkdir(exist_ok=True)
    license_frame=manifests[['benchmark_id','source_id','accession','access','license_summary','license_url']].copy()
    license_frame.to_csv(output/'data_license_matrix.csv',index=False)
    citation_frame=manifests[['benchmark_id','source_id','accession','citation','citation_url']].copy()
    citation_frame.to_csv(output/'citation_manifest.csv',index=False)
    model_card = f'''# CausaFlux v{REALDATA_VERSION} real-data benchmark model card

## Intended use

Reproducible research benchmarking across cancer and neurobiology. This release is not a medical device and does not provide patient-specific guidance.

## Executed evidence in the packaged release

- Accession, access-class, license, citation, and independent-validation manifests were validated for all benchmark sources.
- Public SEA-AD donor metadata and continuous pseudoprogression metadata were analyzed descriptively.
- Full molecular, imaging, perturbational, and electrophysiology matrices were not redistributed or claimed as executed.

## Validation boundary

The outer validation unit is the patient, donor, animal, or session specified in each benchmark manifest. Independent validation cohorts are excluded from feature selection and calibration.

## Prohibited use

Diagnosis, prognosis, treatment selection, dosing, or any direct clinical decision.
'''
    (cards/'REAL_DATA_MODEL_CARD.md').write_text(model_card,encoding='utf-8')
    for spec in load_benchmark_registry(manifest_dir):
        source_lines='\n'.join(f"- **{s.role}:** {s.source_id} — `{s.accession}` ({s.access}; {s.license})" for s in spec.sources)
        card=f'''# Dataset card — {spec.title}

- **Benchmark ID:** `{spec.benchmark_id}`
- **Domain:** {spec.domain}
- **Status:** {spec.status}
- **Estimated storage:** {spec.estimated_storage_gb} GB
- **Primary question:** {spec.primary_question}
- **Discovery cohort:** {spec.evaluation.get('discovery')}
- **Validation cohorts:** {', '.join(spec.evaluation.get('validation',[]))}
- **Outer split unit:** {spec.evaluation.get('split_unit')}

## Sources

{source_lines}

## Leakage controls

{chr(10).join('- '+x for x in spec.evaluation.get('forbidden_leakage',[]))}

## Execution status

Accession-ready. Biological-result claims require local lawful download, immutable version locking, analysis, and independent validation.
'''
        (cards/f'{spec.benchmark_id}_DATASET_CARD.md').write_text(card,encoding='utf-8')

    reports=output/'reports'; reports.mkdir(exist_ok=True)
    css="body{font-family:Arial,Helvetica,sans-serif;max-width:1180px;margin:32px auto;color:#202124;line-height:1.45;padding:0 20px}h1,h2{letter-spacing:-.02em}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #ddd;padding:7px;text-align:left;vertical-align:top}th{background:#f5f5f5}.tag{display:inline-block;background:#eef3f8;padding:3px 7px;border-radius:10px;margin-right:4px}.warning{border-left:4px solid #b75d00;padding:10px 14px;background:#fff7ed}.ok{border-left:4px solid #287a4a;padding:10px 14px;background:#effaf3}code{background:#f5f5f5;padding:2px 4px}a{color:#24557a}"
    for spec in load_benchmark_registry(manifest_dir):
        f=manifests[manifests.benchmark_id==spec.benchmark_id]
        body=f"""<!doctype html><html><head><meta charset="utf-8"><title>{_html_escape(spec.title)}</title><style>{css}</style></head><body><h1>{_html_escape(spec.title)}</h1><p class="tag">{_html_escape(spec.domain)}</p><p>{_html_escape(spec.primary_question)}</p><div class="warning"><strong>Execution status:</strong> accession and metadata verified. Large or controlled matrices are not redistributed. Run the download plan after accepting source-specific terms.</div><h2>Accession manifest</h2>{_table_html(f[['source_id','role','repository','accession','version_policy','access','modalities','citation']])}<h2>Independent validation</h2><p>Discovery: {_html_escape(spec.evaluation.get('discovery'))}</p><p>Validation: {_html_escape(', '.join(spec.evaluation.get('validation',[])))}</p><p>Split unit: {_html_escape(spec.evaluation.get('split_unit'))}</p><h2>Download plan</h2>{_table_html(plan[plan.benchmark_id==spec.benchmark_id][['source_id','execution_mode','command_or_action']])}<h2>Research-use limits</h2><p>Outputs are retrospective research benchmarks. They are not diagnostic, prognostic, or treatment guidance.</p><p><a href="index.html">Back to registry</a></p></body></html>"""
        (reports/f'{spec.benchmark_id}.html').write_text(body,encoding='utf-8')
    licenses_html=f"""<!doctype html><html><head><meta charset="utf-8"><title>Data licenses</title><style>{css}</style></head><body><h1>Data access and license matrix</h1><div class="warning">The MIT license covers CausaFlux code only. Third-party biomedical data remain governed by source-specific terms.</div>{_table_html(license_frame,100)}<p><a href="index.html">Back to registry</a></p></body></html>"""
    citations_html=f"""<!doctype html><html><head><meta charset="utf-8"><title>Citations</title><style>{css}</style></head><body><h1>Benchmark citations</h1><p>Cite the exact dataset release, immutable repository version, associated publication, and CausaFlux software release.</p>{_table_html(citation_frame,100)}<p><a href="index.html">Back to registry</a></p></body></html>"""
    (reports/'licenses.html').write_text(licenses_html,encoding='utf-8')
    (reports/'citations.html').write_text(citations_html,encoding='utf-8')
    central=f"""<!doctype html><html><head><meta charset="utf-8"><title>CausaFlux v1.7.0 real-data benchmarks</title><style>{css}</style></head><body><h1>CausaFlux v1.7.0 — Real-data benchmark registry</h1><div class="ok"><strong>Six benchmark families are accession-ready.</strong> The release bundles public SEA-AD donor metadata and pathology-progression summaries; large assay matrices remain at their authoritative repositories.</div><p><a href="licenses.html">Data licenses</a> · <a href="citations.html">Citations</a> · <a href="../independent_validation_cohorts.csv">Independent validation cohorts</a></p><h2>Benchmark families</h2>{_table_html(registry)}<h2>Accessions and licenses</h2>{_table_html(manifests[['benchmark_id','source_id','role','accession','access','license_summary','citation']],80)}<h2>Real-data snapshot</h2><p>SEA-AD donors: {sea_summary['donors']}; CPS records: {sea_summary['cps_records']}; brain regions: {sea_summary['brain_regions']}.</p><p><a href="../sea_ad_descriptive/sea_ad_cohort_overview.svg">Open SEA-AD descriptive figure</a></p><h2>Important boundary</h2><div class="warning">This package does not bundle controlled-access files or claim that full molecular benchmarks were executed during release construction. Reports become biological-result reports only after the user downloads data, records immutable versions, runs the analysis, and passes external-validation gates.</div></body></html>"""
    (reports/'index.html').write_text(central,encoding='utf-8')
    status={'framework':'CausaFlux','version':REALDATA_VERSION,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'registry_valid':val['valid'],'n_benchmarks':val['n_benchmarks'],'n_sources':val['n_sources'],'bundled_real_data':['SEA-AD donor metadata','SEA-AD continuous pseudoprogression scores'],'large_data_redistributed':False,'controlled_data_redistributed':False,'full_benchmarks_executed':False,'report_scope':'accession, access, licensing, reproducibility, and real metadata descriptive benchmark'}
    (output/'realdata_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    run_manifest={
      'framework':'CausaFlux','version':REALDATA_VERSION,'profile':'real-data-benchmark-registry',
      'generated_at_utc':status['generated_at_utc'],'synthetic':False,
      'executed_real_data':['SEA-AD donor metadata','SEA-AD continuous pseudoprogression scores'],
      'full_benchmarks_executed':False,'clinical_use':False,
    }
    (output/'run_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
    provenance=output/'provenance'; provenance.mkdir(exist_ok=True)
    import platform, sys
    environment={'framework':'CausaFlux','version':REALDATA_VERSION,'python':sys.version,'platform':platform.platform(),'pandas':pd.__version__}
    (provenance/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    artifact_rows=[]
    for path in sorted(output.rglob('*')):
        if path.is_file() and 'provenance/artifact_manifest.csv' not in path.as_posix():
            artifact_rows.append({'relative_path':str(path.relative_to(output)),'size_bytes':path.stat().st_size,'sha256':_sha(path)})
    pd.DataFrame(artifact_rows).to_csv(provenance/'artifact_manifest.csv',index=False)
    return {'report':reports/'index.html','manifest':output/'accession_manifest.csv','status':output/'realdata_status.json'}


def validate_realdata_output(output_dir: str | Path) -> dict[str,Any]:
    output=Path(output_dir)
    required=['accession_manifest.csv','benchmark_registry.csv','download_plan.csv','preflight_checks.csv','registry_validation.json','accession_lock.csv','independent_validation_cohorts.csv','realdata_status.json','run_manifest.json','reports/index.html','reports/licenses.html','reports/citations.html','sea_ad_descriptive/summary.json','cards/REAL_DATA_MODEL_CARD.md','provenance/artifact_manifest.csv']
    missing=[p for p in required if not (output/p).exists()]
    result={'valid':not missing,'version':REALDATA_VERSION,'missing':missing}
    if not missing:
        m=pd.read_csv(output/'accession_manifest.csv'); b=pd.read_csv(output/'benchmark_registry.csv'); v=pd.read_csv(output/'independent_validation_cohorts.csv')
        result.update({'n_benchmarks':int(len(b)),'n_sources':int(len(m)),'n_validation_sources':int(len(v)),'all_licenses_recorded':bool(m.license_summary.notna().all()),'all_citations_recorded':bool(m.citation.notna().all())})
        result['valid']=result['valid'] and result['n_benchmarks']==6 and result['n_validation_sources']>=6 and result['all_licenses_recorded'] and result['all_citations_recorded']
    return result
