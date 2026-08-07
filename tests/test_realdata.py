from pathlib import Path
import pandas as pd
from causaflux.realdata import (
    load_benchmark_registry, benchmark_registry_frame, accession_manifest_frame,
    build_download_plan, validate_realdata_registry, generate_realdata_reports,
    validate_realdata_output,
)


def test_registry_has_six_benchmarks():
    specs=load_benchmark_registry()
    assert len(specs)==6
    assert {s.benchmark_id for s in specs}=={
      'htan_spatial_cancer','gdc_tcga_cptac_brca','depmap_prism_lincs',
      'sea_ad_neural_glial','amp_ad_molecular','dandi_neurophysiology'}


def test_every_benchmark_has_validation_and_licenses():
    for spec in load_benchmark_registry():
        assert any('validation' in s.role for s in spec.sources)
        for source in spec.sources:
            assert source.accession and source.license and source.citation
            assert source.license_url.startswith('http') and source.citation_url.startswith('http')


def test_key_accessions_are_pinned():
    frame=accession_manifest_frame()
    text=' '.join(frame.accession.astype(str))
    for accession in ['TCGA-BRCA','PDC000120','GSE92742','syn21241740','syn3219045','DANDI:000048']:
        assert accession in text


def test_depmap_plan_is_manual_not_scraped(tmp_path):
    plan=build_download_plan(tmp_path,['depmap_prism_lincs'])
    dep=plan[plan.adapter=='depmap_manual']
    assert not dep.empty
    assert dep.execution_mode.eq('manual-terms-no-scraping').all()


def test_report_generation_uses_real_seaad_snapshots(tmp_path):
    project=Path(__file__).resolve().parents[1]
    paths=generate_realdata_reports(tmp_path,project_root=project)
    assert paths['report'].exists()
    result=validate_realdata_output(tmp_path)
    assert result['valid']
    summary=pd.read_json(tmp_path/'sea_ad_descriptive/summary.json',typ='series')
    assert int(summary['donors'])==84
    assert (tmp_path/'sea_ad_descriptive/sea_ad_cohort_overview.svg').exists()


def test_registry_validation():
    result=validate_realdata_registry()
    assert result['valid']
    assert result['n_sources']>=20
