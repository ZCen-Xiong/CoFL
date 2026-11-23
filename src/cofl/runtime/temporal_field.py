"""SE(2) field alignment and finite-support, short-lived world-frame memory."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def ego_points(world, position, yaw):
    return (np.asarray(world) - position) @ rotation(yaw)


def world_vectors(ego, yaw):
    return np.asarray(ego) @ rotation(yaw).T


def polar_queries(world, position, yaw, radius=5., half_fov=np.pi/4):
    p = ego_points(world, position, yaw)
    return np.stack((np.arctan2(p[..., 1], p[..., 0])/half_fov,
                     np.linalg.norm(p, axis=-1)/radius), -1)


@dataclass
class WorldFieldMemory:
    resolution: float = .025
    max_age_s: float = 1.0
    agreement_deg: float = 30.

    def __post_init__(self):
        if self.resolution <= 0 or self.max_age_s < 0:
            raise ValueError('Positive resolution and nonnegative memory age required')
        self.cells = {}
        self.tree = None
        self.time = -np.inf

    def update(self, world_xy, vectors_world, timestamp):
        world_xy=np.asarray(world_xy);vectors_world=np.asarray(vectors_world)
        if world_xy.ndim!=2 or world_xy.shape[-1]!=2 or vectors_world.shape!=world_xy.shape:
            raise ValueError('Matching [N,2] world positions and world vectors required')
        if not np.isfinite(timestamp) or timestamp < self.time:
            raise ValueError('Reset memory after a clock/pose discontinuity')
        self.time = float(timestamp)
        self.cells = {k: v for k, v in self.cells.items() if timestamp-v[2] <= self.max_age_s+1e-8}
        for p, v in zip(world_xy, vectors_world):
            if not np.isfinite(p).all() or not np.isfinite(v).all():
                continue
            key = tuple(np.rint(p/self.resolution).astype(int))
            if key in self.cells:
                old = self.cells[key][1]
                norm = np.linalg.norm(v)*np.linalg.norm(old)
                # Never average contradictory route directions into a wall/zero field.
                if norm > 1e-10 and np.dot(v, old)/norm >= np.cos(np.deg2rad(self.agreement_deg)):
                    v = .5*(v+old)
            self.cells[key] = (np.array(p), np.array(v), float(timestamp))
        if self.cells:
            entries = list(self.cells.values())
            self.points = np.array([v[0] for v in entries])
            self.vectors = np.array([v[1] for v in entries])
            self.tree = cKDTree(self.points)
        else:
            self.tree = None

    def sample(self, points):
        p = np.asarray(points, dtype=float)
        shape = p.shape
        p = p.reshape(-1, 2)
        out = np.full_like(p, np.nan)
        if self.tree is not None:
            d, i = self.tree.query(p, k=min(4, len(self.points)))
            d, i = d.reshape(len(p), -1), i.reshape(len(p), -1)
            supported = d[:, 0] <= self.resolution*.8
            weight = np.where(d <= self.resolution*1.5, 1/np.maximum(d, 1e-9), 0)
            values = (self.vectors[i]*weight[..., None]).sum(1)/np.maximum(weight.sum(1)[:, None], 1e-9)
            out[supported] = values[supported]
        return out.reshape(shape)
