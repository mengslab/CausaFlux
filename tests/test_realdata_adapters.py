from pathlib import Path
import json

from causaflux.realdata import load_benchmark_registry
from causaflux.realdata_adapters import adapter_names, plan_source, write_accession_lock


def test_every_manifest_adapter_is_implemented(tmp_path: Path):
    implemented=set(adapter_names())
    for benchmark in load_benchmark_registry():
        for source in benchmark.sources:
            assert source.adapter in implemented
            plan=plan_source(source,tmp_path/source.source_id)
            assert plan.source_id==source.source_id
            assert plan.command_or_action
            assert not plan.redistributable_by_causaflux


def test_depmap_adapter_forbids_scraping(tmp_path: Path):
    source=next(s for b in load_benchmark_registry() for s in b.sources if s.adapter=='depmap_manual')
    plan=plan_source(source,tmp_path)
    assert plan.execution_mode=='manual-terms-no-scraping'
    assert 'scraping is intentionally disabled' in plan.command_or_action


def test_accession_lock_records_version_and_license(tmp_path: Path):
    source=load_benchmark_registry()[0].sources[0]
    path=write_accession_lock(source,tmp_path,resolved_version='syn123.4',files=[{'id':'syn123','version':4,'sha256':'a'*64}])
    payload=json.loads(path.read_text())
    assert payload['resolved_version']=='syn123.4'
    assert payload['license_url'].startswith('http')
