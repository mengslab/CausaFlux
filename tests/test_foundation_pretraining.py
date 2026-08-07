import json
from pathlib import Path
import numpy as np

from causaflux.foundation_pretraining import (
    ADAPTERS, OBJECTIVES, EVAL_MODES, SPLITS,
    FoundationPretrainingConfig, adapter_registry, objective_registry_frame,
    generate_foundation_data, save_external_foundation_npz, load_external_foundation_npz,
    validate_foundation_pretraining,
)

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'foundation_pretraining_reference'

def test_adapter_and_objective_registry_complete():
    assert {x['name'] for x in adapter_registry()} == set(ADAPTERS)
    assert set(objective_registry_frame()['objective']) == set(OBJECTIVES)

def test_foundation_reference_valid_and_gate_passes():
    result=validate_foundation_pretraining(REF)
    assert result['valid']
    assert result['software_gate']=='PASS'
    assert result['real_authorization'] is False

def test_required_evaluation_matrix_present():
    import pandas as pd
    m=pd.read_csv(REF/'foundation_evaluation_matrix.csv')
    assert set(EVAL_MODES).issubset(set(m['evaluation']))
    assert set(SPLITS).issubset(set(m['split']))
    assert {'future_state_rmse','intervention_effect_rmse','cell_type_accuracy'}.issubset(m.columns)

def test_external_npz_roundtrip(tmp_path):
    cfg=FoundationPretrainingConfig(n_samples=60,seed=171)
    data=generate_foundation_data(cfg)
    p=save_external_foundation_npz(data,tmp_path/'f.npz')
    loaded=load_external_foundation_npz(p)
    assert set(loaded)==set(data)
    assert loaded['X'].shape==data['X'].shape
    assert np.allclose(loaded['future'],data['future'])

def test_gate_is_not_real_pretraining_authorization():
    gate=json.loads((REF/'foundation_pretraining_gate.json').read_text())
    assert gate['software_pretraining_gate']=='PASS'
    assert gate['foundation_pretraining_authorized'] is False
    assert gate['synthetic_fixture'] is True
