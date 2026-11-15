"""Validate generated surface gaps and height-invariant obstacle geometry."""

from dataclasses import replace
import math

import numpy as np
import pytest

from cofl.generation.obstacle_scene import (
    SceneConfig, audit_obstacles, build_obstacles, footprint_gap, generate_scene, obstacle_mesh,
)


@pytest.mark.parametrize("seed", range(12))
def test_default_forest_surface_spacing_and_full_height(seed):
    cfg = SceneConfig(seed=seed)
    obstacles = build_obstacles(cfg)
    poles = [o for o in obstacles if o["kind"] == "cylinder"]
    assert len(poles) == 45
    assert len(obstacles) == 46  # 45 poles and a floor, no rings or default gap wall.
    for index, a in enumerate(poles):
        for b in poles[:index]:
            # Independent pair check against the circular outer envelope.
            actual = math.hypot(a["center"][0]-b["center"][0],
                                a["center"][1]-b["center"][1]) - a["radius"] - b["radius"]
            assert actual > 0.65
        vertices, _ = obstacle_mesh(a)
        np.testing.assert_allclose(np.unique(vertices[:, 2]), [0, cfg.height])


def test_mixed_walls_and_columns_and_optional_gap():
    cfg = SceneConfig(poles=25, walls=8, gap_wall=1)
    obstacles = build_obstacles(cfg)
    audit = audit_obstacles(cfg, obstacles)
    assert audit["min_surface_gap_m"] > 0.65
    assert len([o for o in obstacles if o["id"].startswith("wall_")]) == 8
    for obstacle in obstacles[1:]:
        vertices, _ = obstacle_mesh(obstacle)
        np.testing.assert_allclose(np.unique(vertices[:, 2]), [0, cfg.height])
    assert build_obstacles(cfg) == obstacles


def test_wall_only_mode_and_manifest(tmp_path):
    cfg = SceneConfig(poles=0, walls=12)
    metadata = generate_scene(tmp_path / "scene", cfg)
    assert metadata["gap"] is None
    assert metadata["planning"]["plane"] == "XOY"
    assert metadata["geometry_profile"] == "vertical_extrusion_xy_v1"
    assert metadata["geometry_audit"]["min_surface_gap_m"] > 0.65
    assert all(o["kind"] == "box" for o in metadata["obstacles"])


def test_surface_distance_is_not_center_distance():
    circle = dict(kind="cylinder", center=[0, 0, 1.5], radius=0.3)
    other = dict(kind="cylinder", center=[0.9, 0, 1.5], radius=0.3)
    wall = dict(kind="box", center=[1, 0, 1.5], size=[0.2, 2, 3])
    assert footprint_gap(circle, other) == pytest.approx(0.3)
    assert footprint_gap(circle, wall) == pytest.approx(0.6)
    assert footprint_gap(wall, circle) == pytest.approx(0.6)
    wall2 = dict(kind="box", center=[2, 0, 1.5], size=[0.2, 2, 3])
    assert footprint_gap(wall, wall2) == pytest.approx(0.8)


@pytest.mark.parametrize("change", [dict(rings=1), dict(min_clearance=0.64),
                                   dict(gap_wall=1, gap_width=0.65)])
def test_reject_disallowed_geometry(change):
    with pytest.raises(ValueError):
        replace(SceneConfig(), **change).validate()


def test_crowded_scene_fails_instead_of_relaxing_spacing():
    with pytest.raises(ValueError, match="Cannot place"):
        build_obstacles(SceneConfig(length=4, width=2, gap_x=2, camera_end=0, poles=20))
