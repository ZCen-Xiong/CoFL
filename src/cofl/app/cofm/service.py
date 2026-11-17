"""Studio orchestration: scene resources, teacher previews, exports and depth capture.

Planning algorithms live in cofl.planning; neural networks in cofl.models.
This service adapts those capabilities to the Studio request/response contracts.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import uuid

import numpy as np

from cofl.planning.configuration_field import (
    Box2D, ConfigurationTeacher, CoupledCrossProxy, Disc2D, PlanarWorld,
)
from cofl.generation import obstacle_scene
from .contracts import MDock, MFieldSlice, MObservation, MObstacle, MPair, MScene, MSceneSummary, MTeacher


class CofMService:
    resolution = 0.1
    body_clearance_m = 0.01  # Extra clearance from the whole body to an obstacle.
    teacher_safety_weight = 0.0  # No additional preference for larger clearance.
    teacher_safety_band_m = body_clearance_m
    arrow_stride = 2  # Display spacing = resolution * stride = 0.2m.
    max_display_arrows = 4000

    def __init__(self, config):
        self.config = config
        self.robot = CoupledCrossProxy(margin=self.body_clearance_m)
        root = Path(__file__).resolve().parents[4]
        self.habitat_python = config.cofm_habitat_python or root / ".habitat/cofm/bin/python"
        self.output = config.cofm_output or root / "artifacts/cof-m/studio-tasks"
        self.lock, self.render_lock = threading.RLock(), threading.Lock()
        self.cache = {}  # At most two CPU graphs; no model or GPU state is retained.

    def _load(self, scene_id):
        resource = self.config.scenes.get(scene_id)
        if resource is None:
            raise FileNotFoundError("Unknown scene")
        manifest = resource.path.parent / "scene.json"
        metadata = json.loads(manifest.read_text())
        if metadata.get("geometry_profile") != "vertical_extrusion_xy_v1":
            raise ValueError("Select a generated vertical-extrusion cof-m scene")
        cfg = obstacle_scene.SceneConfig(**metadata["config"])
        cfg.validate()
        obstacle_scene.audit_obstacles(cfg, metadata["obstacles"])
        obstacles, display = [], []
        for o in metadata["obstacles"]:
            if o["id"] == "floor":
                continue
            x, y = o["center"][:2]
            if o["kind"] == "cylinder":
                obstacles.append(Disc2D(x, y, o["radius"]))
                display.append(MObstacle(kind="circle", center=(x, y), radius=o["radius"]))
            else:
                sx, sy = o["size"][:2]
                obstacles.append(Box2D(x-sx/2, x+sx/2, y-sy/2, y+sy/2))
                display.append(MObstacle(kind="box", center=(x, y), size=(sx, sy)))
        world = PlanarWorld(scene_id, Box2D(0, cfg.length, -cfg.width/2, cfg.width/2),
                            tuple(obstacles), (1, 0), (cfg.length-1, 0))
        return resource, metadata, cfg, world, display

    def scenes(self):
        result = []
        for scene_id, resource in self.config.scenes.items():
            try:
                self._load(scene_id)
            except (ValueError, OSError, KeyError, TypeError):
                continue
            result.append(MSceneSummary(id=scene_id, label=resource.label or scene_id))
        return result

    def scene(self, scene_id):
        resource, _, cfg, _, display = self._load(scene_id)
        r = self.robot
        return MScene(id=scene_id, label=resource.label or scene_id, length_m=cfg.length,
                      width_m=cfg.width, height_m=cfg.camera_height, resolution_m=self.resolution,
                      obstacles=display, habitat_available=self.habitat_python.is_file(), docking=dict(
                          proxy="coupled-cross union over all q, fixed yaw=0",
                          q_range_m=[-r.q_limit, r.q_limit],
                          centerline_lengths_m=[2*(r.e0+r.q_limit)]*2,
                          horizontal_width_m=2*r.horizontal_radius,
                          vertical_width_m=2*r.vertical_radius,
                          horizontal_span_m=2*(r.e0+r.q_limit+r.horizontal_radius),
                          vertical_span_m=2*(r.e0+r.q_limit+r.vertical_radius),
                          margin_m=r.margin,
                      ))

    def validate(self, scene_id, point):
        _, _, _, world, _ = self._load(scene_id)
        x, y = point
        clearance = float(self.robot.swept_clearance(world, (x, y, -0.1), (x, y, 0.1)))
        on_grid = np.allclose(np.asarray(point)/self.resolution,
                             np.round(np.asarray(point)/self.resolution), atol=1e-8, rtol=0)
        valid = clearance > self.robot.margin + 1e-10 and on_grid
        reason = ("停泊净空通过：全部 q 形态均可容纳（固定朝向）" if valid else
                  "请对齐 0.1m 网格" if not on_grid else
                  "停泊净空不足：十字包络与障碍或规划边界冲突")
        return MDock(point=point, valid=bool(valid), clearance_m=clearance, reason=reason)

    def _graph(self, scene_id):
        resource, _, _, world, _ = self._load(scene_id)
        path = resource.path.parent / "scene.json"
        key = (scene_id, path.stat().st_mtime_ns, path.stat().st_size,
               self.resolution, self.robot, self.teacher_safety_weight, self.teacher_safety_band_m)
        if key not in self.cache:
            if len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = ConfigurationTeacher(
                world, self.robot, resolution=self.resolution,
                safety_weight=self.teacher_safety_weight, safety_band=self.teacher_safety_band_m,
            )
        return self.cache[key]

    def random_pair(self, scene_id, seed):
        from scipy.sparse.csgraph import connected_components
        with self.lock:
            teacher = self._graph(scene_id)
            xy = teacher.states[len(teacher.q)//2].copy()
            low, high = xy.copy(), xy.copy()
            low[..., 2], high[..., 2] = -0.1, 0.1
            docking = self.robot.swept_clearance(teacher.world, low, high) > self.robot.margin + 1e-10
            ids = np.arange(teacher.free.size).reshape(teacher.shape)[len(teacher.q)//2][docking]
            candidates = xy[docking, :2]
            if len(candidates) < 2:
                raise ValueError("This scene has fewer than two valid docking points")
            _, component = connected_components(teacher.graph, directed=False)
            labels = component[ids]
            rng = np.random.default_rng(seed)
            for _ in range(200):
                i = int(rng.integers(len(candidates)))
                possible = np.flatnonzero((labels == labels[i]) &
                                          (np.linalg.norm(candidates-candidates[i], axis=1) >= 3))
                if len(possible):
                    j = int(rng.choice(possible))
                    if candidates[i, 0] > candidates[j, 0]:
                        i, j = j, i  # Face the sampled task in the default +X camera direction.
                    return MPair(start=self.validate(scene_id, tuple(candidates[i])),
                                 goal=self.validate(scene_id, tuple(candidates[j])))
            raise ValueError("No connected docking pair at least 3m apart was found")

    def teacher(self, body):
        with self.lock:
            start, goal = self.validate(body.scene_id, body.start), self.validate(body.scene_id, body.goal)
            if not start.valid or not goal.valid:
                raise ValueError(f"起点：{start.reason}；终点：{goal.reason}")
            if np.linalg.norm(np.asarray(body.start)-body.goal) < 0.2:
                raise ValueError("起终点距离应至少为 0.2m")
            teacher = self._graph(body.scene_id)
            teacher.world = replace(teacher.world, start_xy=body.start, goal_xy=body.goal)
            teacher.solve()
            path = teacher.rollout()
            field, reachable, _ = teacher.field()
            slices = []
            for i, q in enumerate(teacher.q):
                selected = np.zeros(teacher.shape[1:], bool)
                selected[::self.arrow_stride, ::self.arrow_stride] = True
                selected &= reachable[i]
                values = np.concatenate((teacher.states[i][selected, :2], field[i][selected]), axis=1)
                if len(values) > self.max_display_arrows:
                    # Keep coverage across the whole map, rather than cutting
                    # off one side when the scene exceeds the display budget.
                    indices = np.linspace(0, len(values)-1, self.max_display_arrows, dtype=int)
                    values = values[indices]
                slices.append(MFieldSlice(q_m=float(q), arrows=values.tolist()))
            clearance = (float(self.robot.swept_clearance(teacher.world, path[:-1], path[1:]).min())
                         if len(path) > 1 else None)
            artifact_id = uuid.uuid4().hex
            output = self.output / artifact_id
            output.mkdir(parents=True)
            teacher.save(output / "teacher.npz")
            np.save(output / "path.npy", path, allow_pickle=False)
            obstacle_scene.save_json(output / "task.json", dict(
                format="cof_m_studio_teacher", version=1, source="geometry_teacher",
                scene_id=body.scene_id, scene_manifest=str(self.config.scenes[body.scene_id].path.parent / "scene.json"),
                start=body.start, goal=body.goal, height_m=self.scene(body.scene_id).height_m,
                robot=asdict(self.robot), yaw_rad=0, resolution_m=self.resolution,
                reachable=bool(len(path)), endpoint_q=0, morph_weight=teacher.morph_weight,
                safety_weight=self.teacher_safety_weight, safety_band_m=self.teacher_safety_band_m,
                scene_config=self._load(body.scene_id)[1]["config"],
                scene_obstacles=self._load(body.scene_id)[1]["obstacles"],
            ))
            return MTeacher(reachable=bool(len(path)), resolution_m=self.resolution,
                            path=path.tolist(), slices=slices, min_path_clearance_m=clearance,
                            artifact_id=artifact_id, message=(
                                "离散 teacher 路径可达；已检查整机扫掠碰撞" if len(path) else
                                "停泊点有效，但此 0.1m 构型图中不可达；不将其标作 STOP"))

    def observation(self, body):
        from PIL import Image
        resource, metadata, cfg, world, _ = self._load(body.scene_id)
        state = [*body.point, body.q_m]
        if float(self.robot.clearance(world, state)) <= self.robot.margin:
            raise ValueError("当前构型与障碍或规划边界冲突")
        if not self.habitat_python.is_file():
            raise RuntimeError("Habitat interpreter unavailable; configure cofm_habitat_python")
        with self.render_lock, tempfile.TemporaryDirectory(prefix="cofm-observation-") as directory:
            output = Path(directory)
            (output / "obstacles.glb").symlink_to(resource.path)
            obstacle_scene.save_json(output / "obstacles.stage_config.json", dict(
                render_asset="obstacles.glb", collision_asset="obstacles.glb",
                up=[0, 1, 0], front=[0, 0, -1], units_to_meters=1.0,
            ))
            position = np.asarray([*body.point, cfg.camera_height])
            metadata["frames"] = [dict(index=0, position_planning=position.tolist(),
                position_habitat=(obstacle_scene.PLANNING_TO_HABITAT @ position).tolist(),
                rotation_xyzw_habitat=[0, 0, 0, 1])]
            metadata["depth_rendered"] = False
            obstacle_scene.save_json(output / "scene.json", metadata)
            try:
                result = subprocess.run([str(self.habitat_python.absolute()),
                    str(Path(obstacle_scene.__file__).resolve()), "--output", str(output),
                    "--render-existing"], capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("Habitat depth rendering timed out") from error
            if result.returncode:
                raise RuntimeError("Habitat depth rendering failed; verify EGL/GPU access. " + result.stderr[-600:])
            depth = np.load(output / "depth/000000.npy", allow_pickle=False)
            valid = np.load(output / "valid/000000.npy", allow_pickle=False)
            # A viewing transform only; no RGB sensor or learned RGB appearance.
            image = np.where(valid, 30 + 225*(1-np.clip(depth/cfg.far, 0, 1)), 0).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            return MObservation(image="data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode(),
                                position=tuple(position), q_m=body.q_m,
                                valid_fraction=float(valid.mean()), far_m=cfg.far)


