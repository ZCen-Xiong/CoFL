"""Manual point tasks and the Studio contract; no body/q filtering."""
import json
import numpy as np
import pytest
from cofl.app.cofm import CofMService, MTaskRequest
from cofl.app.config import SceneResource, StudioConfig
from cofl.generation.obstacle_scene import SceneConfig, generate_scene


def make_service(tmp_path):
    folder=tmp_path/'scene'
    generate_scene(folder,SceneConfig(length=5,width=4,poles=0))
    config=StudioConfig(device='cpu',cofm_output=tmp_path/'tasks',
        scenes={'example':SceneResource(path=folder/'obstacles.glb')})
    return CofMService(config)


def test_point_near_wall_is_valid_without_body_envelope(tmp_path):
    s=make_service(tmp_path)
    assert s.validate('example',(.025,0)).valid
    assert not s.validate('example',(0,0)).valid
    assert not s.validate('example',(.013,0)).valid
    assert s.scene('example').docking['proxy']=='point'


def test_clearance_is_configurable(tmp_path):
    s=make_service(tmp_path);s.point_clearance_m=.05
    assert not s.validate('example',(.025,0)).valid
    assert s.validate('example',(.075,0)).valid


def test_manual_point_field_export(tmp_path):
    s=make_service(tmp_path)
    result=s.teacher(MTaskRequest(scene_id='example',start=(.025,0),goal=(4.,0)))
    assert result.reachable
    assert len(result.path[0])==2 and len(result.arrows[0])==4
    assert not hasattr(result,'slices')
    assert result.path[0]==pytest.approx((.025,0))
    assert result.path[-1]==pytest.approx((4.,0))
    with np.load(s.output/result.artifact_id/'teacher.npz') as f:
        assert 'q' not in f and f['field'].ndim==3 and f['field'].shape[-1]==2
        assert f['terminal'].sum()==1
        assert (f['field'][f['terminal']]==0).all()
    task=json.loads((s.output/result.artifact_id/'task.json').read_text())
    assert task['clearance_m']==.02 and 'robot' not in task
    with pytest.raises(ValueError,match='手动'):
        s.random_pair('example',0)
