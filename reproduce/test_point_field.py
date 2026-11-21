"""Verify reused CoFL-S field math and point-only geometry."""
import numpy as np
import pytest
from dataclasses import replace
from cofl.planning.configuration_field import PlanarWorld, Box2D
from cofl.planning.point_field import PointTeacher
from cofl.generation.sector.fields.sector_gt import (
    _compose_potential,_compute_obstacle_distance,_compute_bev_escape_direction,_compute_bev_velocity_field)


def world():
    return PlanarWorld('point',Box2D(0,2,-1,1),(),(.05,0),(1.8,0))


def test_same_cofls_kernels_and_magnitude():
    t=PointTeacher(world(),resolution=.025).solve()
    p=_compose_potential(t.reachable,t.value,_compute_obstacle_distance(~t.reachable))
    expected=_compute_bev_velocity_field(p.T,t.reachable.T,t.distance.T,
        _compute_bev_escape_direction(t.reachable.T),v_norm=5.).transpose(2,1,0).astype(np.float32)
    expected[t.terminal]=0
    np.testing.assert_array_equal(t.vectors,expected)
    assert t.vectors[t.locate((.5,0))][0]>0
    assert np.linalg.norm(t.vectors[t.locate((.5,0))])==pytest.approx(t.distance[t.locate((.5,0))]/5,rel=.01)


def test_two_cm_band_not_body_size():
    t=PointTeacher(world(),resolution=.01)
    assert t.free[t.locate((.02,0))]
    assert not t.free[t.locate((.01,0))]


def test_unreachable_not_terminal():
    w=replace(world(),obstacles=(Box2D(.9,1.1,-1,1),))
    t=PointTeacher(w,resolution=.025).solve()
    assert not t.reachable[t.locate(w.start_xy)]
    assert len(t.rollout())==0 and t.terminal.sum()==1
