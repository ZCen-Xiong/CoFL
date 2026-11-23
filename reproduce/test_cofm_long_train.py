"""Budget calculation and atomic serialization used by unattended continuation."""
import json
import pytest
import torch
from cofl.experiments.cofm_long_train import atomic_json,atomic_torch,planned_epochs


def test_epoch_budget_accounts_for_validation():
    assert planned_epochs(1,2.,10.,5)==900
    assert planned_epochs(8,.63,.11,95)>45000
    with pytest.raises(ValueError):planned_epochs(8,0,1,1)


def test_atomic_status_and_resume_payload(tmp_path):
    p=tmp_path/'status.json'
    atomic_json(p,dict(epoch=1,status='running'))
    atomic_json(p,dict(epoch=2,status='completed'))
    assert json.loads(p.read_text())['epoch']==2
    assert not list(tmp_path.glob('*.tmp'))
    state=dict(weights=torch.arange(3),optimizer={'step':3},cursor=2,
               numpy_rng={'bit_generator':'PCG64','state':{'state':2**120,'inc':2**117}})
    atomic_torch(tmp_path/'resume.pt',state)
    restored=torch.load(tmp_path/'resume.pt',weights_only=True)
    torch.testing.assert_close(restored['weights'],state['weights'])
    assert restored['numpy_rng']==state['numpy_rng']
