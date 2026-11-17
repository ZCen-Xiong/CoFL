"""Docking semantics and real-scene teacher checks for the new Studio mode."""

import json
from pathlib import Path

import numpy as np
import pytest

from cofl.app.cofm import CofMService, MTaskRequest
from cofl.app.config import SceneResource, StudioConfig
from cofl.app.server import create_app
from cofl.generation.obstacle_scene import SceneConfig, generate_scene, save_json, write_glb


def make_service(tmp_path, obstacles=()):
    folder = tmp_path / "scene"
    # These collision fixtures use x=5..6m, independent of the scene-generator defaults.
    metadata = generate_scene(folder, SceneConfig(length=16, width=10, poles=0))
    metadata["obstacles"].extend(obstacles)
    save_json(folder / "scene.json", metadata)
    write_glb(folder / "obstacles.glb", metadata["obstacles"])
    config = StudioConfig(device="cpu", scenes={"example": SceneResource(path=folder/"obstacles.glb")},
                          cofm_output=tmp_path/"teachers")
    return CofMService(config), config


def test_new_entry_and_separate_api(tmp_path):
    service, config = make_service(tmp_path)
    app = create_app(config)
    assert "cofm" in app.state.domains
    assert app.state.domains["cofm"].resource_kinds == []
    paths = app.openapi()["paths"]
    assert "/api/v1/cofm/teacher" in paths
    assert "/api/v1/cofm/observation" in paths
    assert "/api/v1/predict" in paths
    assert "/api/v1/domains/ego2d/sessions" in paths
    dock = service.scene("example").docking
    assert sum(dock["centerline_lengths_m"]) > 0.82
    assert min(dock["horizontal_width_m"], dock["vertical_width_m"]) > 0.15


def test_whole_cross_catches_obstacle_between_center_and_tip(tmp_path):
    obstacle = dict(id="small_pole", kind="cylinder", center=[5.2, 0, 1.5], radius=0.05, height=3)
    service, _ = make_service(tmp_path, [obstacle])
    # Centre and outer centreline tips are clear, but the arm interior intersects.
    assert abs(5.2-5) > 0.05 and abs(5.4-5.2) > 0.05
    assert not service.validate("example", (5, 0)).valid


def test_cross_not_its_bounding_square_and_all_shapes_fit(tmp_path):
    obstacle = dict(id="corner_pole", kind="cylinder", center=[5.35, 0.35, 1.5], radius=0.05, height=3)
    service, _ = make_service(tmp_path, [obstacle])
    assert service.validate("example", (5, 0)).valid
    world = service._load("example")[3]
    for q in np.linspace(-0.1, 0.1, 41):
        assert service.robot.clearance(world, [5, 0, q]) > service.robot.margin


def test_reject_boundary_and_off_grid_without_silent_snap(tmp_path):
    service, _ = make_service(tmp_path)
    assert not service.validate("example", (0.1, 0)).valid
    assert not service.validate("example", (5.013, 0)).valid
    with pytest.raises(ValueError):
        service.teacher(MTaskRequest(scene_id="example", start=(5.013, 0), goal=(8, 0)))
    with pytest.raises(FileNotFoundError):
        service.scene("../../elsewhere")


def test_seeded_docking_pair_and_saved_teacher(tmp_path):
    service, _ = make_service(tmp_path)
    pair = service.random_pair("example", 42)
    assert pair == service.random_pair("example", 42)
    assert pair.start.valid and pair.goal.valid
    assert pair.start.point[0] <= pair.goal.point[0]
    assert np.linalg.norm(np.asarray(pair.start.point)-pair.goal.point) >= 3
    result = service.teacher(MTaskRequest(scene_id="example", start=pair.start.point, goal=pair.goal.point))
    assert result.reachable and result.min_path_clearance_m > service.robot.margin
    assert len(result.slices) == 9
    root = service.output / result.artifact_id
    with np.load(root/"teacher.npz") as labels:
        assert labels["field"].shape[-1] == 3
        assert labels["terminal"].sum() == 1
        assert not (labels["reachable"] & ~labels["collision_free"]).any()
    task = json.loads((root/"task.json").read_text())
    assert task["height_m"] == 1.2 and task["source"] == "geometry_teacher"


def test_disk_swept_clearance_handles_thin_intersection(tmp_path):
    obstacle = dict(id="pole", kind="cylinder", center=[5, 0, 1.5], radius=0.05, height=3)
    service, _ = make_service(tmp_path, [obstacle])
    world = service._load("example")[3]
    a, b = [4, 0, 0], [6, 0, 0]
    assert np.all(service.robot.clearance(world, [a, b]) > service.robot.margin)
    assert service.robot.swept_clearance(world, a, b) < service.robot.margin
