"""Temporal teacher equivariance, frame loss, memory validity and A reference joins."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from cofl.runtime.temporal_field import rotation, ego_points, world_vectors, WorldFieldMemory
from cofl.planning.endpoint_flow import EndpointFlowPlanner, EndpointFlowConfig
from cofl.training.cofm_temporal import aligned_pair_loss
from cofl.planning.point_field import PointTeacher
from cofl.planning.configuration_field import PlanarWorld,Box2D
from cofl.experiments.matterport_routes import local_labels


def test_teacher_same_world_points_under_translation_yaw_and_visibility(tmp_path):
    t=PointTeacher(PlanarWorld('p',Box2D(0,4,-2,2),(),(.5,0),(3.8,0)),resolution=.05).solve()
    t.save(tmp_path/'teacher.npz')
    with np.load(tmp_path/'teacher.npz') as f:
        a=local_labels(f,[.5,0],heading=0,depth=np.full((240,320),15.))
        b=local_labels(f,[.55,.01],heading=.15,depth=np.full((240,320),15.))
        c=local_labels(f,[.5,0],heading=0,depth=np.full((240,320),.6))
    _,ia,ib=np.intersect1d(a['world_index'],b['world_index'],return_indices=True)
    assert len(ia)>100
    np.testing.assert_allclose(world_vectors(a['field'][ia],0),world_vectors(b['field'][ib],.15),atol=1e-7)
    _,ia,ic=np.intersect1d(a['world_index'],c['world_index'],return_indices=True)
    assert 0<len(ic)<len(a['query'])
    np.testing.assert_array_equal(a['field'][ia],c['field'][ic])


def test_pair_loss_rotates_vectors_and_backpropagates():
    p=torch.tensor([[[1.,0.]],[[0.,-1.]]],requires_grad=True)
    yaw=torch.tensor([0.,np.pi/2])
    assert aligned_pair_loss(p,yaw).item()<1e-12
    bad=p.detach().clone();bad[1,0,0]=.3;bad.requires_grad_()
    loss=aligned_pair_loss(bad,yaw);loss.backward()
    assert loss.item()>.01 and bad.grad.abs().sum()>0


def test_memory_translation_rotation_staleness_and_conflict():
    p=np.array([[1.,0.],[1.,.025]])
    np.testing.assert_allclose(ego_points(p,[.2,0],np.pi/2),[[0,-.8],[.025,-.8]],atol=1e-8)
    m=WorldFieldMemory(max_age_s=.4)
    m.update(p,np.tile([1.,0.],(2,1)),0.)
    m.update(p[:1],np.array([[-1.,0.]]),.2)
    np.testing.assert_allclose(m.sample(p[:1]),[[-1,0]],atol=1e-6)
    assert np.isnan(m.sample([[2,2]])).all()
    m.update(np.empty((0,2)),np.empty((0,2)),.7)
    assert np.isnan(m.sample(p)).all()
    with pytest.raises(ValueError):m.update(p,p,.1)


def constant_field(p):
    return np.broadcast_to([1.,0.],np.asarray(p).shape).copy()


def free(p):return np.ones(np.asarray(p).shape[:-1],bool)


def test_A_same_width_perpendicular_and_pointwise_time_join():
    planner=EndpointFlowPlanner()
    a=planner.plan(np.array([0.,0.]),0.,.82,0.,constant_field,free)
    assert a['status']=='ok' and a['perpendicular_deg']==pytest.approx(90)
    assert a['initial_width_m']==pytest.approx(.82)
    b=planner.plan(np.array([.04,0.]),.03,.82,.2,constant_field,free)
    assert b['status']=='ok'
    assert b['join_position_error_m']<1e-8 and b['join_yaw_error_rad']<1e-8
    np.testing.assert_allclose(np.array(a['endpoints_world'])[10],b['endpoints_world'][0],atol=1e-8)
    assert np.max(np.diff(b['width_m']))<1e-8


def test_yaw_allowed_and_no_invalid_teleport_or_unsupported_query():
    field=lambda p:np.broadcast_to([1.,1.],np.asarray(p).shape).copy()
    planner=EndpointFlowPlanner()
    result=planner.plan(np.array([0.,0.]),np.pi/4,.82,0.,field,free)
    assert result['status']=='ok' and result['yaw_rad'][0]==pytest.approx(np.pi/4)
    expired=planner.plan(np.array([0.,0.]),np.pi/4,.82,1.,field,free)
    assert expired['status']=='reference_expired'
    bad=EndpointFlowPlanner().plan(np.array([0.,0.]),0.,.82,0.,
        lambda p:np.full(np.asarray(p).shape,np.nan),free)
    assert bad['status']=='no_feasible_pair'
    with pytest.raises(ValueError):EndpointFlowPlanner().plan(np.zeros(2),0.,.1,0.,field,free)


def test_no_false_success_if_chord_crosses_obstacle():
    blocked=lambda p:np.asarray(p)[...,1]>.05
    result=EndpointFlowPlanner(replace(EndpointFlowConfig(),max_lookahead_m=.6)).plan(
        np.zeros(2),0.,.82,0.,constant_field,blocked)
    assert result['status']=='no_feasible_pair'


def test_converging_streams_shrink_and_remain_continuous_across_02s_frames():
    planner=EndpointFlowPlanner()
    field=lambda p:np.stack((np.ones(np.asarray(p).shape[:-1]),-.5*np.asarray(p)[...,1]),-1)
    width=.82
    for i in range(12):
        result=planner.plan(np.array([i*.04,0.]),0.,width,i*.2,field,free)
        assert result['status']=='ok'
        assert result['initial_width_m']==pytest.approx(width)
        assert max(np.diff(result['width_m']))<=1e-6
        if i:
            assert result['join_position_error_m']<1e-8
            assert result['join_endpoint_velocity_error_m_s']<1e-8
        width=result['width_m'][10]
    assert .60<=width<.82


def test_point_model_has_two_outputs_and_strict_checkpoint_version(tmp_path):
    from cofl.models.cofm import CofMConfig,CofMPolicy
    torch.set_num_threads(2)
    model=CofMPolicy(CofMConfig(image_size=64,d_model=32)).eval()
    args=dict(query=torch.tensor([[[0.,.1],[.3,.15]]]),rgb=torch.rand(1,3,64,64),
              depth=torch.ones(1,1,64,64),goal_ego=torch.tensor([[1.,0.]]))
    with torch.no_grad():expected=model(**args)
    assert expected.shape==(1,2,2) and torch.isfinite(expected).all()
    model.save(tmp_path/'policy.pt')
    restored=CofMPolicy.load(tmp_path/'policy.pt').eval()
    with torch.no_grad():torch.testing.assert_close(restored(**args),expected)
    torch.save(dict(format='cofm_rgbd_v1'),tmp_path/'old.pt')
    with pytest.raises(ValueError,match='retraining'):CofMPolicy.load(tmp_path/'old.pt')
