"""Experimental (x,y,q) graph teacher for a fixed-altitude, fixed-yaw robot.

Not an aircraft dynamics model. The footprint is an explicit capsule proxy,
and graph edges conservatively check the entire interpolated swept footprint.
The teacher tangent is per unit configuration arc length, not metres/second.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Box2D:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.xmin, self.xmax, self.ymin, self.ymax)):
            raise ValueError("Box coordinates must be finite")
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("Box must have positive area")


@dataclass(frozen=True)
class Disc2D:
    x: float
    y: float
    radius: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.radius)) or self.radius <= 0:
            raise ValueError("Disc coordinates must be finite and radius positive")


@dataclass(frozen=True)
class PlanarWorld:
    name: str
    bounds: Box2D
    obstacles: tuple[Box2D | Disc2D, ...]
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]


@dataclass(frozen=True)
class CoupledCrossProxy:
    # Outer spans are 0.80 +/- 2q metres, with constant 0.20m width.
    # Capsule centerline lengths exclude the two 0.10m end caps.
    e0: float = 0.30
    horizontal_radius: float = 0.10
    vertical_radius: float = 0.10
    body_radius: float = 0.10
    q_limit: float = 0.10
    margin: float = 0.01

    def __post_init__(self):
        values = (self.e0, self.horizontal_radius, self.vertical_radius,
                  self.body_radius, self.q_limit, self.margin)
        if not all(math.isfinite(x) and x >= 0 for x in values) or self.q_limit > self.e0:
            raise ValueError("Proxy dimensions must be finite, nonnegative and q_limit <= e0")

    def half_extents(self, q, yaw=0.0):
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        horizontal, vertical = self.e0 + q, self.e0 - q
        return np.maximum.reduce([
            np.asarray([horizontal * c + self.horizontal_radius,
                        horizontal * s + self.horizontal_radius]),
            np.asarray([vertical * s + self.vertical_radius,
                        vertical * c + self.vertical_radius]),
            np.full(2, self.body_radius),
        ])

    def swept_clearance(self, world: PlanarWorld, start, end):
        """Lower bound on clearance of the full linear transition, at yaw=0.

        Capsule centreline endpoint extrema are linear in (x,y,q). Their
        AABB contains every intermediate centreline. Distance of that rectangle
        to an obstacle, minus capsule radius, is thus a safe clearance bound.
        This is exact for individual state checks and axis/q-only graph edges;
        diagonal translations can be conservatively rejected. Bounds are walls.
        """
        start, end = np.broadcast_arrays(np.asarray(start, float), np.asarray(end, float))
        if start.shape[-1] != 3 or not np.isfinite(start).all() or not np.isfinite(end).all():
            raise ValueError("States must be finite arrays [...,3] in (x,y,q)")
        x0, y0, q0 = np.moveaxis(start, -1, 0)
        x1, y1, q1 = np.moveaxis(end, -1, 0)
        xmin, xmax = np.minimum(x0, x1), np.maximum(x0, x1)
        ymin, ymax = np.minimum(y0, y1), np.maximum(y0, y1)
        parts = [
            (np.minimum(x0 - self.e0 - q0, x1 - self.e0 - q1),
             np.maximum(x0 + self.e0 + q0, x1 + self.e0 + q1),
             ymin, ymax, self.horizontal_radius),
            (xmin, xmax, np.minimum(y0 - self.e0 + q0, y1 - self.e0 + q1),
             np.maximum(y0 + self.e0 - q0, y1 + self.e0 - q1), self.vertical_radius),
            (xmin, xmax, ymin, ymax, self.body_radius),
        ]
        clearance = np.full(start.shape[:-1], np.inf)
        bounds = world.bounds
        for left, right, bottom, top, radius in parts:
            for distance in (left - bounds.xmin - radius, bounds.xmax - right - radius,
                             bottom - bounds.ymin - radius, bounds.ymax - top - radius):
                clearance = np.minimum(clearance, distance)
            for obstacle in world.obstacles:
                if isinstance(obstacle, Disc2D):
                    dx = np.maximum.reduce([left - obstacle.x, obstacle.x - right,
                                            np.zeros_like(left)])
                    dy = np.maximum.reduce([bottom - obstacle.y, obstacle.y - top,
                                            np.zeros_like(bottom)])
                    clearance = np.minimum(clearance, np.hypot(dx, dy) - obstacle.radius - radius)
                    continue
                dx = np.maximum.reduce([obstacle.xmin - right, left - obstacle.xmax,
                                        np.zeros_like(left)])
                dy = np.maximum.reduce([obstacle.ymin - top, bottom - obstacle.ymax,
                                        np.zeros_like(bottom)])
                clearance = np.minimum(clearance, np.hypot(dx, dy) - radius)
        in_limits = (np.abs(q0) <= self.q_limit + 1e-10) & (np.abs(q1) <= self.q_limit + 1e-10)
        return np.where(in_limits, clearance, -np.inf)

    def clearance(self, world, states):
        return self.swept_clearance(world, states, states)


class ConfigurationTeacher:
    """Positive-cost lattice graph, with goal-rooted Dijkstra successor labels."""

    def __init__(self, world, robot=None, *, resolution=0.05, q_values=None,
                 morph_weight=2.0, safety_weight=0.2, safety_band=0.25):
        from scipy import sparse
        self.world, self.robot = world, robot or CoupledCrossProxy()
        self.resolution, self.morph_weight = resolution, morph_weight
        if (not math.isfinite(resolution) or resolution <= 0 or
                not math.isfinite(morph_weight) or morph_weight <= 0 or
                not math.isfinite(safety_weight) or safety_weight < 0 or
                not math.isfinite(safety_band) or safety_band <= 0):
            raise ValueError("Invalid resolution or cost settings")
        self.q = np.asarray(np.linspace(-0.1, 0.1, 9) if q_values is None else q_values, float)
        if (self.q.ndim != 1 or not len(self.q) or not np.isfinite(self.q).all()
                or np.any(np.diff(self.q) <= 0) or np.any(abs(self.q) > self.robot.q_limit + 1e-10)):
            raise ValueError("q_values must be sorted, unique and inside the robot limits")
        b = world.bounds
        self.x = np.arange(round((b.xmax - b.xmin) / resolution) + 1) * resolution + b.xmin
        self.y = np.arange(round((b.ymax - b.ymin) / resolution) + 1) * resolution + b.ymin
        q, y, x = np.meshgrid(self.q, self.y, self.x, indexing="ij")
        self.states = np.stack((x, y, q), axis=-1)
        self.shape = q.shape
        self.clearance = self.robot.clearance(world, self.states)
        self.free = self.clearance > self.robot.margin + 1e-10
        indices = np.arange(self.free.size).reshape(self.shape)
        rows, columns, weights = [], [], []
        # Undirected 8-neighbour translation and adjacent shape changes.
        for offset in ((0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, -1), (1, 0, 0)):
            source_slices, target_slices = [], []
            for length, shift in zip(self.shape, offset):
                source_slices.append(slice(max(0, -shift), min(length, length - shift)))
                target_slices.append(slice(max(0, shift), min(length, length + shift)))
            source_slices, target_slices = tuple(source_slices), tuple(target_slices)
            valid = self.free[source_slices] & self.free[target_slices]
            a, b = self.states[source_slices][valid], self.states[target_slices][valid]
            clearance = self.robot.swept_clearance(world, a, b)
            valid_edge = clearance > self.robot.margin + 1e-10
            a, b, clearance = a[valid_edge], b[valid_edge], clearance[valid_edge]
            source, target = indices[source_slices][valid][valid_edge], indices[target_slices][valid][valid_edge]
            length = np.linalg.norm((b - a)[:, :2], axis=1) + morph_weight * abs(b[:, 2] - a[:, 2])
            cost = length * (1 + safety_weight * np.maximum(0, 1 - clearance / safety_band))
            rows.extend((source, target))
            columns.extend((target, source))
            weights.extend((cost, cost))
        self.graph = sparse.coo_matrix(
            (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
            shape=(self.free.size, self.free.size),
        ).tocsr()
        self.value = np.full(self.shape, np.inf)
        self.predecessor = np.full(self.free.size, -9999, dtype=np.int64)

    def locate(self, state):
        state = np.asarray(state, float)
        if state.shape != (3,) or not np.isfinite(state).all():
            raise ValueError("Requested state must be finite (x,y,q)")
        axes = (self.x, self.y, self.q)
        nearest = [int(np.argmin(abs(axis - coordinate))) for axis, coordinate in zip(axes, state)]
        if any(abs(axes[i][nearest[i]] - state[i]) > 1e-8 for i in range(3)):
            raise ValueError("Requested state must lie exactly on the lattice; no obstacle snapping")
        return nearest[2], nearest[1], nearest[0]

    def solve(self, *, goal_q=0.0):
        from scipy.sparse.csgraph import dijkstra
        # Explicit target configuration. An invalid goal is not snapped to free space.
        self.value.fill(np.inf)
        self.predecessor.fill(-9999)
        goal = self.locate((*self.world.goal_xy, goal_q))
        if self.free[goal]:
            values, pred = dijkstra(self.graph, directed=True,
                                    indices=np.ravel_multi_index(goal, self.shape),
                                    return_predecessors=True)
            self.value, self.predecessor = values.reshape(self.shape), pred
        return self

    def field(self):
        """Return tangent, reachable mask, terminal mask; unreachable isn't STOP."""
        values = self.value.ravel()
        valid = np.isfinite(values)
        advancing = valid & (self.predecessor >= 0)
        field = np.zeros((len(values), 3))
        states = self.states.reshape(-1, 3)
        delta = states[self.predecessor[advancing]] - states[advancing]
        norm = np.sqrt(np.square(delta[:, :2]).sum(1) + np.square(self.morph_weight * delta[:, 2]))
        field[advancing] = delta / norm[:, None]
        terminal = valid & ~advancing
        return field.reshape((*self.shape, 3)), valid.reshape(self.shape), terminal.reshape(self.shape)

    def rollout(self, *, start_q=0.0):
        start = self.locate((*self.world.start_xy, start_q))
        if not np.isfinite(self.value[start]):
            return np.empty((0, 3))
        current = np.ravel_multi_index(start, self.shape)
        path, values = [], self.value.ravel()
        states = self.states.reshape(-1, 3)
        for _ in range(self.free.size):
            path.append(states[current])
            successor = self.predecessor[current]
            if successor < 0:
                return np.asarray(path)
            if not values[successor] < values[current]:
                raise RuntimeError("Successor failed to strictly descend the value function")
            current = successor
        raise RuntimeError("Successor cycle")

    def save(self, path):
        field, reachable, terminal = self.field()
        np.savez_compressed(path, x=self.x, y=self.y, q=self.q, value=self.value,
                            field=field.astype(np.float32), collision_free=self.free,
                            reachable=reachable, terminal=terminal,
                            predecessor=self.predecessor.reshape(self.shape))
