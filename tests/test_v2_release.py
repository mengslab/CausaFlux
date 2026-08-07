from pathlib import Path
import json
import numpy as np
import pandas as pd

from causaflux.evidence_ledger import EvidenceRecord, REQUIRED_CLAIMS, build_reference_ledger, write_ledger, validate_ledger, sha256_file
from causaflux.longitudinal_realdata import public_dataset_registry, validate_longitudinal_table, table_to_dynamic_benchmark, gse8057_sample_metadata
from causaflux.shift_calibration import evaluate_shift_calibration
from causaflux.v2_release_gate import evaluate_v2_release_gate
from causaflux.v2_release import run_v2_release


def _real_evidence_records(tmp_path):
    kind_by_claim={
        'CF2_PHASE1_DYNAMIC_SUPERIORITY':'real_longitudinal_perturbation',
        'CF2_MULTIMODAL_FORECASTING':'real_multimodal_perturbation',
        'CF2_UNSEEN_INTERVENTION_GENERALIZATION':'real_longitudinal_perturbation',
        'CF2_SPATIAL_CONTEXT_BENEFIT':'real_spatial_perturbation',
        'CF2_PROSPECTIVE_CYCLE_1':'prospective_cycle',
        'CF2_PROSPECTIVE_CYCLE_2':'prospective_cycle',
        'CF2_EXTERNAL_REPLICATION':'external_lab_replication',
        'CF2_SHIFT_CALIBRATION':'distribution_shift_calibration',
        'CF2_REAL_LONGITUDINAL_CONNECTED':'real_longitudinal_perturbation',
        'CF2_NEGATIVE_FAILURE_REPORTING':'real_negative_result',
    }
    rows=[]
    for i,(claim,_) in enumerate(REQUIRED_CLAIMS,1):
        source=tmp_path/f'evidence_{i}.json'; source.write_text('{"locked": true}')
        rows.append(EvidenceRecord(f'E{i}',claim,'PASS',kind_by_claim[claim],str(source),independent=claim=='CF2_EXTERNAL_REPLICATION',prospective='PROSPECTIVE_CYCLE' in claim,cycle=1 if claim.endswith('_1') else 2 if claim.endswith('_2') else None,negative_or_failure=claim=='CF2_NEGATIVE_FAILURE_REPORTING',sha256=sha256_file(source)))
    return rows


def test_public_longitudinal_registry_contains_real_accessions():
    frame=public_dataset_registry()
    assert {'GSE8057','GSE70138','GSE101406'}.issubset(set(frame.accession))
    assert not frame.bundled_data.any()


def test_gse8057_design_metadata_has_timecourse():
    frame=gse8057_sample_metadata()
    assert {'cisplatin','oxaliplatin','vehicle'}.issubset(set(frame.perturbation))
    assert frame.time_hours_after_exposure.nunique() >= 5
    assert frame.sample_accession.str.startswith('GSM').all()


def test_longitudinal_table_adapter():
    rows=[]
    for traj,history,drug in [('T1','H1','drugA'),('T2','H2','drugB')]:
        for t in [0,2,6,24]:
            rows.append({'trajectory_id':traj,'donor_id':traj,'time':t,'history_id':history,'target':drug,'dose':1.0,'sequence':'continuous','fate':'recovery','int__drug':float(t>0),'feature__a':t/24,'feature__b':1-t/48})
    frame=pd.DataFrame(rows)
    assert validate_longitudinal_table(frame)['valid']
    data=table_to_dynamic_benchmark(frame)
    assert data.observations.shape==(2,4,2)
    assert data.interventions.shape==(2,4,1)


def test_shift_calibration_gate_passes_calibrated_predictions():
    rng=np.random.default_rng(2); n=400; mean=rng.normal(size=n); sd=np.ones(n); observed=mean+rng.normal(size=n)
    frame=pd.DataFrame({'observed':observed,'predicted_mean':mean,'predicted_sd':sd,'shift_group':np.where(np.arange(n)%2,'external_donor','external_tissue')})
    metrics,status=evaluate_shift_calibration(frame)
    assert len(metrics)==2
    assert status['real_distribution_shift_gate']=='PASS'


def test_synthetic_reference_does_not_unlock_v2(tmp_path):
    root=Path(__file__).resolve().parents[1]
    ledger=build_reference_ledger(root,tmp_path/'evidence')
    matrix,status=evaluate_v2_release_gate(ledger,software_checks={'all':True})
    assert not status['prospectively_validated_virtual_cell']
    assert status['release_claim_status']=='NOT_YET_ELIGIBLE'
    assert matrix.passed.sum() < len(matrix)


def test_real_evidence_can_unlock_when_all_claims_pass(tmp_path):
    ledger=write_ledger(_real_evidence_records(tmp_path),tmp_path/'ledger.csv')
    assert validate_ledger(ledger)['valid']
    matrix,status=evaluate_v2_release_gate(ledger,software_checks={'all':True})
    assert matrix.passed.all()
    assert status['prospectively_validated_virtual_cell']


def test_reference_release_bundle_builds_without_claim(tmp_path):
    root=Path(__file__).resolve().parents[1]
    result=run_v2_release(root,tmp_path/'release')
    assert result['validation']['valid']
    assert result['release_gate']['software_release_ready']
    assert not result['release_gate']['prospectively_validated_virtual_cell']
    assert (tmp_path/'release/report/index.html').exists()
    inventory=pd.read_csv(tmp_path/'release/figures/figure_inventory.csv')
    assert len(inventory)>=4 and inventory.validated.all()
