#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import pandas as pd


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    raise SystemExit(f'REAL-DATA VERIFICATION FAILED: {msg}')


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else root/'realdata_reference'
    required=[
      'run_manifest.json','realdata_status.json','accession_manifest.csv','benchmark_registry.csv',
      'download_plan.csv','preflight_checks.csv','accession_lock.csv','adapter_capabilities.csv',
      'independent_validation_cohorts.csv','data_license_matrix.csv','citation_manifest.csv',
      'reports/index.html','reports/licenses.html','reports/citations.html',
      'cards/REAL_DATA_MODEL_CARD.md','sea_ad_descriptive/summary.json',
      'sea_ad_descriptive/sea_ad_cohort_overview.svg','provenance/environment.json',
      'provenance/artifact_manifest.csv',
    ]
    for rel in required:
        p=out/rel
        if not p.exists() or p.stat().st_size==0: fail(f'missing or empty {rel}')
    status=json.loads((out/'realdata_status.json').read_text())
    if status.get('version')!='1.7.0' or status.get('n_benchmarks')!=6 or status.get('n_sources')!=24:
        fail('status counts/version mismatch')
    if status.get('full_benchmarks_executed') or status.get('controlled_data_redistributed'):
        fail('execution or controlled-data boundary is incorrectly declared')
    manifest=pd.read_csv(out/'accession_manifest.csv')
    if manifest['benchmark_id'].nunique()!=6 or len(manifest)!=24: fail('accession manifest incomplete')
    if manifest[['accession','license_summary','license_url','citation','citation_url']].isna().any().any():
        fail('license or citation fields missing')
    validation=pd.read_csv(out/'independent_validation_cohorts.csv')
    if len(validation)<12: fail('independent validation matrix incomplete')
    adapters=pd.read_csv(out/'adapter_capabilities.csv')
    if len(adapters)!=24 or adapters['redistributable_by_causaflux'].astype(str).str.lower().isin(['true','1']).any():
        fail('adapter capability matrix invalid')
    if len(list((out/'adapter_plans').glob('*.json')))!=24: fail('adapter plans incomplete')
    sea=json.loads((out/'sea_ad_descriptive/summary.json').read_text())
    if sea.get('donors')!=84 or sea.get('cps_records')!=632 or sea.get('brain_regions')<9:
        fail('SEA-AD metadata snapshot counts unexpected')
    for name,digest in sea.get('snapshot_sha256',{}).items():
        p=out/'snapshots/sea_ad'/name
        if not p.exists() or sha(p)!=digest: fail(f'SEA-AD snapshot hash mismatch: {name}')
    cards=list((out/'cards').glob('*_DATASET_CARD.md'))
    if len(cards)!=6: fail('dataset-card count is not six')
    artifacts=pd.read_csv(out/'provenance/artifact_manifest.csv')
    bad=[]
    for row in artifacts.itertuples(index=False):
        p=out/row.relative_path
        if not p.exists() or p.stat().st_size!=row.size_bytes or sha(p)!=row.sha256: bad.append(row.relative_path)
    if bad: fail(f'artifact hash mismatch for {len(bad)} files')
    print(json.dumps({'valid':True,'framework':'CausaFlux','version':'1.7.0','benchmarks':6,
                      'sources':24,'validation_sources':len(validation),'adapter_plans':len(adapters),
                      'sea_ad_donors':84,'sea_ad_cps_records':632,'hashed_artifacts':len(artifacts)},indent=2))

if __name__=='__main__': main()
