"""Checks of geometry semantics, edge safety and mask handling, independent of Habitat."""

import numpy as np
import pytest

from cofl.experiments.supervision_geometry import make_worlds, optical_depth_scan
from cofl.generation.configuration_field import (
    Box2D, ConfigurationTeacher, CoupledCrossProxy, PlanarWorld,
)


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
