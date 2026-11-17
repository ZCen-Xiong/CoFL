"""Checks of geometry semantics, edge safety and mask handling, independent of Habitat."""

import numpy as np
import pytest

from cofl.planning.configuration_field import (
    Box2D, ConfigurationTeacher, CoupledCrossProxy, PlanarWorld,
)



def make_worlds():
    bounds = Box2D(0, 10, -3, 3)

    def gap(name, width):
        return PlanarWorld(name, bounds, (
            Box2D(4.9, 5.1, -3, -width / 2), Box2D(4.9, 5.1, width / 2, 3),
        ), (1, 0), (9, 0))

    return dict(
        empty=PlanarWorld("empty", bounds, (), (1, 0), (9, 0)),
        narrow_065=gap("narrow_065", 0.65),
        narrow_059=gap("narrow_059", 0.59),
        u_pocket=PlanarWorld("u_pocket", bounds, (
            Box2D(5.8, 6.1, -1.6, 1.6), Box2D(2.8, 6.1, 1.4, 1.6),
            Box2D(2.8, 6.1, -1.6, -1.4),
        ), (4, 0), (8, 0)),
        two_slots=PlanarWorld("two_slots", bounds, (
            Box2D(3.9, 4.1, -3, -1.525), Box2D(3.9, 4.1, -0.875, 3),
            Box2D(0, 6.675, 1.1, 1.3), Box2D(7.325, 10, 1.1, 1.3),
        ), (1, -1.2), (7, 2.4)),
        upper_blocked=PlanarWorld("upper_blocked", bounds, (
            Box2D(3, 4, -2, 2), Box2D(4, 6, 0, 3),
        ), (1, 0), (9, 0)),
        lower_blocked=PlanarWorld("lower_blocked", bounds, (
            Box2D(3, 4, -2, 2), Box2D(4, 6, -3, 0),
        ), (1, 0), (9, 0)),
    )


def optical_depth_scan(world, angles):
    """Analytic horizontal pinhole scan, Z depth (not ray range)."""
    ox, oy = world.start_xy
    slope = np.tan(angles)
    depth = np.full_like(slope, np.inf)
    for box in world.obstacles:
        with np.errstate(divide="ignore", invalid="ignore"):
            low = (box.ymin - oy) / slope
            high = (box.ymax - oy) / slope
        tmin = np.maximum(box.xmin - ox, np.minimum(low, high))
        tmax = np.minimum(box.xmax - ox, np.maximum(low, high))
        parallel = abs(slope) < 1e-12
        y_inside = box.ymin <= oy <= box.ymax
        tmin = np.where(parallel, box.xmin - ox if y_inside else np.inf, tmin)
        tmax = np.where(parallel, box.xmax - ox if y_inside else -np.inf, tmax)
        hit = (tmin <= tmax) & (tmax > 0)
        depth = np.minimum(depth, np.where(hit, np.maximum(tmin, 0), np.inf))
    return depth


def test_shape_envelope_and_coupling():
    robot = CoupledCrossProxy()
    np.testing.assert_allclose(2*robot.half_extents(0), [0.80, 0.80])
    np.testing.assert_allclose(2*robot.half_extents(0.1), [1.00, 0.60])
    np.testing.assert_allclose(2*robot.half_extents(-0.1), [0.60, 1.00])
    assert np.all(robot.half_extents(0, np.pi/4) + robot.margin < 0.325)


def test_free_endpoints_do_not_imply_safe_swept_motion():
    world = PlanarWorld("thin_wall", Box2D(0, 2, -1, 1),
                        (Box2D(0.99, 1.01, -1, 1),), (0.9, 0), (1.1, 0))
    robot = CoupledCrossProxy(e0=0, horizontal_radius=0, vertical_radius=0,
                              body_radius=0, q_limit=0)
    a, b = np.array([0.9, 0, 0]), np.array([1.1, 0, 0])
    assert np.all(robot.clearance(world, [a, b]) > robot.margin)
    assert robot.swept_clearance(world, a, b) <= robot.margin


def test_swept_bound_covers_interpolated_diagonal_and_shape_motion():
    robot, world = CoupledCrossProxy(), make_worlds()["two_slots"]
    rng = np.random.default_rng(123)
    a = rng.uniform([1, -2, -0.1], [9, 2, 0.1], size=(200, 3))
    b = a + rng.uniform([-0.05, -0.05, -0.025], [0.05, 0.05, 0.025], size=a.shape)
    b[:, 2] = np.clip(b[:, 2], -0.1, 0.1)
    bound = robot.swept_clearance(world, a, b)
    for t in np.linspace(0, 1, 21):
        assert np.all(robot.clearance(world, a*(1-t)+b*t) >= bound - 1e-10)


def test_gap_requires_shape_and_unreachable_is_not_terminal():
    world = make_worlds()["narrow_065"]
    fixed = ConfigurationTeacher(world, q_values=[0]).solve()
    field, reachable, terminal = fixed.field()
    start = fixed.locate((1, 0, 0))
    assert not reachable[start] and not terminal[start]
    assert not len(fixed.rollout())
    variable = ConfigurationTeacher(world).solve()
    path = variable.rollout()
    assert len(path) and path[:, 2].max() == pytest.approx(0.1)
    assert path[0, 2] == pytest.approx(0) and path[-1, 2] == pytest.approx(0)
    assert np.all(variable.robot.swept_clearance(world, path[:-1], path[1:]) > variable.robot.margin)


def test_orthogonal_slits_need_opposite_deformations():
    teacher = ConfigurationTeacher(make_worlds()["two_slots"]).solve()
    path = teacher.rollout()
    assert len(path) and path[:, 2].min() < 0 < path[:, 2].max()
    field, reachable, terminal = teacher.field()
    index = np.flatnonzero(reachable.ravel() & ~terminal.ravel())
    assert np.all(teacher.value.ravel()[teacher.predecessor[index]] < teacher.value.ravel()[index])
    # The scaled metric gives a unit tangent, including pure shape actions.
    tangent = field.reshape(-1, 3)[index]
    np.testing.assert_allclose(np.linalg.norm(tangent*[1, 1, 2], axis=-1), 1)


def test_no_silent_goal_snapping_and_repeated_solve_resets():
    teacher = ConfigurationTeacher(make_worlds()["narrow_065"]).solve()
    assert len(teacher.rollout())
    # Move the goal into solid wall; no stale field or nearest-free replacement.
    teacher.world = PlanarWorld("blocked_goal", teacher.world.bounds, teacher.world.obstacles,
                                teacher.world.start_xy, (5, 1))
    teacher.solve()
    assert not np.isfinite(teacher.value).any()
    assert not teacher.field()[2].any()
    with pytest.raises(ValueError, match="exactly"):
        teacher.locate((1.013, 0, 0))


def test_identical_scan_with_opposite_optimal_labels():
    worlds = make_worlds()
    angles = np.linspace(-np.pi/4, np.pi/4, 321)
    a, b = worlds["upper_blocked"], worlds["lower_blocked"]
    np.testing.assert_array_equal(optical_depth_scan(a, angles), optical_depth_scan(b, angles))
    ta, tb = [ConfigurationTeacher(world, q_values=[0]).solve() for world in (a, b)]
    fa, fb = [teacher.field()[0][teacher.locate((1, 0, 0))] for teacher in (ta, tb)]
    assert fa[1] < 0 < fb[1]
