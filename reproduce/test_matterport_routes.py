"""Point-route raster mapping, narrow filtering, and local RGB-D field labels."""
import numpy as np
import pytest
from cofl.generation.matterport_worker import planning_raster
from cofl.planning.narrow import GroundRaster,narrow_profile
from cofl.planning.point_field import PointTeacher
from cofl.planning.configuration_field import PlanarWorld,Box2D
from cofl.experiments.matterport_routes import local_labels,path_heading


def test_mapping():
    free=np.array([[True,False,True],[False,True,True]])
    mapped,ox,oy=planning_raster(free,[[10,0,20],[11,3,21]],.1)
    assert np.array_equal(mapped,free[::-1,::-1].T)
    assert ox==pytest.approx(-20.1) and -(-.1+oy)==pytest.approx(10.2)


def test_narrow_and_one_wall():
    yy,xx=np.meshgrid(np.arange(241)*.025-3,np.arange(321)*.025,indexing='ij')
    path=np.c_[np.arange(1,7.01,.1),np.zeros(61)]
    for width,accept in [(.8,True),(2.,False)]:
        r=GroundRaster((abs(yy)<=width/2)|(xx<2)|(xx>6),.025,-3)
        assert narrow_profile(path,r)['accepted']==accept
    assert not narrow_profile(path,GroundRaster(yy>-.2,.025,-3))['accepted']


def test_point_queries_no_q_and_depth_visibility(tmp_path):
    t=PointTeacher(PlanarWorld('point',Box2D(0,4,-2,2),(),(1,0),(3,0)),resolution=.05).solve()
    t.save(tmp_path/'field.npz')
    with np.load(tmp_path/'field.npz') as f:
        a=local_labels(f,[1,0],depth=np.full((240,320),15.),height=.2)
        b=local_labels(f,[1,0],depth=np.full((240,320),.1),height=.2)
    assert len(a['query'])>len(b['query'])
    assert a['field'].shape[-1]==2 and np.isfinite(a['field']).all()
    assert path_heading([[0,0],[0,1]],0)==pytest.approx(np.pi/2)
