"""Point geodesics and fields using the original CoFL-S numerical kernels."""
import numpy as np

from cofl.generation.sector.fields.global_geodesic import (
    _build_grid_graph_uniform, _build_world_safety_cost_map, _dijkstra_from_cell,
)
from cofl.generation.sector.fields.sector_gt import (
    _compose_potential, _compute_obstacle_distance,
    _compute_bev_escape_direction, _compute_bev_velocity_field,
)
from .configuration_field import Disc2D


class PointTeacher:
    def __init__(self, world, *, resolution=.025, clearance_m=.02,
                 safety_band_m=.02, safety_weight=5., v_norm=5.,
                 smoothing_iters=2):
        self.world, self.resolution, self.clearance_m = world, resolution, clearance_m
        b = world.bounds
        self.x = np.arange(np.ceil(b.xmin/resolution), np.floor(b.xmax/resolution)+1)*resolution
        self.y = np.arange(np.ceil(b.ymin/resolution), np.floor(b.ymax/resolution)+1)*resolution
        if len(self.x)*len(self.y)>1_000_000:
            raise ValueError('质点场网格过大，请裁剪场景或增大场分辨率')
        x, y = np.meshgrid(self.x, self.y)
        self.states = np.stack((x,y), -1)
        self.clearance = np.minimum.reduce([x-b.xmin,b.xmax-x,y-b.ymin,b.ymax-y])
        if world.raster is not None:
            self.clearance = np.minimum(self.clearance, world.raster.point_clearance(x,y))
        for o in world.obstacles:
            if isinstance(o, Disc2D):
                distance = np.hypot(x-o.x,y-o.y)-o.radius
            else:
                distance = np.hypot(np.maximum(np.maximum(o.xmin-x,x-o.xmax),0),
                                    np.maximum(np.maximum(o.ymin-y,y-o.ymax),0))
                distance = np.where((x>=o.xmin)&(x<=o.xmax)&(y>=o.ymin)&(y<=o.ymax),0,distance)
            self.clearance = np.minimum(self.clearance,distance)
        self.free = self.clearance >= clearance_m-1e-9
        self.shape = self.free.shape
        cost = _build_world_safety_cost_map(self.free,safety_band_m/resolution,safety_weight)
        self.graph = _build_grid_graph_uniform(self.free,resolution,cost_map=cost)
        self.v_norm,self.smoothing_iters = v_norm,smoothing_iters

    def locate(self, point):
        x,y=point
        ix=int(np.argmin(abs(self.x-x)));iy=int(np.argmin(abs(self.y-y)))
        if abs(self.x[ix]-x)>1e-7 or abs(self.y[iy]-y)>1e-7:
            raise ValueError(f'点需对齐 {self.resolution}m 网格，不自动挪动端点')
        return iy,ix

    def solve(self):
        self.value,self.distance,self.predecessor = _dijkstra_from_cell(
            self.graph,self.free,self.locate(self.world.goal_xy),self.resolution)
        self.reachable = self.free & np.isfinite(self.value)
        self.terminal = np.zeros(self.shape,bool)
        goal=self.locate(self.world.goal_xy)
        self.terminal[goal]=self.reachable[goal]
        self.potential = _compose_potential(self.reachable,self.value,
            _compute_obstacle_distance(~self.reachable))
        # CoFL-S rows are forward/X and columns left/Y; stored map is [Y,X].
        raw = _compute_bev_velocity_field(self.potential.T,self.reachable.T,
            self.distance.T,_compute_bev_escape_direction(self.reachable.T),
            v_norm=self.v_norm,smoothing_iters=self.smoothing_iters)
        self.vectors=raw.transpose(2,1,0).astype(np.float32)
        self.vectors[self.terminal]=0
        return self

    def field(self):
        return self.vectors,self.reachable,self.terminal

    def rollout(self):
        start=self.locate(self.world.start_xy)
        if not self.reachable[start]:return np.empty((0,2))
        current=np.ravel_multi_index(start,self.shape);result=[]
        for _ in range(self.free.size):
            result.append(self.states.reshape(-1,2)[current])
            nxt=int(self.predecessor[current])
            if nxt<0:return np.asarray(result)
            current=nxt
        raise RuntimeError('Predecessor cycle')

    def save(self,path):
        np.savez_compressed(path,x=self.x,y=self.y,field=self.vectors,value=self.value,
            geodesic_m=self.distance,potential=self.potential,collision_free=self.free,
            reachable=self.reachable,terminal=self.terminal,
            predecessor=self.predecessor.reshape(self.shape),clearance_m=self.clearance_m,
            format='cofm_point_field_v1')
