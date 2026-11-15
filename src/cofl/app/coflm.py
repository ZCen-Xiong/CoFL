"""Depth-only CoFL-M tasks: docking, exact-scene teacher, and Habitat observations."""

from __future__ import annotations

import base64
from dataclasses import asdict, replace
from functools import partial
import io
import json
import re
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Literal
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
import numpy as np
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from cofl.generation.configuration_field import (
    Box2D, ConfigurationTeacher, CoupledCrossProxy, Disc2D, PlanarWorld,
)
from cofl.generation import obstacle_scene
from .contracts import Contract, Finite, Point2, ResourceId
from .resources import require_same_origin


class MObstacle(Contract):
    kind: Literal["circle", "box"]
    center: Point2
    radius: float = 0
    size: Point2 = (0, 0)


class MSceneSummary(Contract):
    id: str
    label: str


class MScene(Contract):
    id: str
    label: str
    length_m: float
    width_m: float
    height_m: float
    resolution_m: float
    obstacles: list[MObstacle]
    docking: dict[str, float | list[float] | str]
    habitat_available: bool


class MPointRequest(Contract):
    scene_id: ResourceId
    point: Point2


class MDock(Contract):
    point: Point2
    valid: bool
    clearance_m: float
    reason: str


class MRandomRequest(Contract):
    scene_id: ResourceId
    seed: int = Field(default=0, ge=0, le=2**32-1)


class MPair(Contract):
    start: MDock
    goal: MDock


class MTaskRequest(Contract):
    scene_id: ResourceId
    start: Point2
    goal: Point2


class MFieldSlice(Contract):
    q_m: float
    # x,y,Fx,Fy,Fq; these are configuration tangents, not physical velocities.
    arrows: list[tuple[float, float, float, float, float]]


class MTeacher(Contract):
    source: Literal["geometry_teacher"] = "geometry_teacher"
    reachable: bool
    resolution_m: float
    path: list[tuple[float, float, float]]
    slices: list[MFieldSlice]
    min_path_clearance_m: float | None
    artifact_id: str
    message: str


class MObservationRequest(Contract):
    scene_id: ResourceId
    point: Point2
    q_m: Finite = Field(default=0, ge=-0.1, le=0.1)


class MObservation(Contract):
    image: str
    position: tuple[float, float, float]
    q_m: float
    heading_deg: float = 0
    valid_fraction: float
    far_m: float
    source: Literal["habitat_depth"] = "habitat_depth"


class CoFLMService:
    resolution = 0.1

    def __init__(self, config):
        self.config = config
        self.robot = CoupledCrossProxy()
        root = Path(__file__).resolve().parents[3]
        self.habitat_python = config.coflm_habitat_python or root / ".habitat/coflm/bin/python"
        self.output = config.coflm_output or root / "artifacts/cofl-m/studio-tasks"
        self.lock, self.render_lock = threading.RLock(), threading.Lock()
        self.cache = {}  # At most two CPU graphs; no model or GPU state is retained.

    def _load(self, scene_id):
        resource = self.config.scenes.get(scene_id)
        if resource is None:
            raise FileNotFoundError("Unknown scene")
        manifest = resource.path.parent / "scene.json"
        metadata = json.loads(manifest.read_text())
        if metadata.get("geometry_profile") != "vertical_extrusion_xy_v1":
            raise ValueError("Select a generated vertical-extrusion CoFL-M scene")
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
        key = (scene_id, path.stat().st_mtime_ns, path.stat().st_size)
        if key not in self.cache:
            if len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = ConfigurationTeacher(world, self.robot, resolution=self.resolution)
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
                selected[::4, ::4] = True
                selected &= reachable[i]
                values = np.concatenate((teacher.states[i][selected, :2], field[i][selected]), axis=1)
                slices.append(MFieldSlice(q_m=float(q), arrows=values[:1100].tolist()))
            clearance = (float(self.robot.swept_clearance(teacher.world, path[:-1], path[1:]).min())
                         if len(path) > 1 else None)
            artifact_id = uuid.uuid4().hex
            output = self.output / artifact_id
            output.mkdir(parents=True)
            teacher.save(output / "teacher.npz")
            np.save(output / "path.npy", path, allow_pickle=False)
            obstacle_scene.save_json(output / "task.json", dict(
                format="cofl_m_studio_teacher", version=1, source="geometry_teacher",
                scene_id=body.scene_id, scene_manifest=str(self.config.scenes[body.scene_id].path.parent / "scene.json"),
                start=body.start, goal=body.goal, height_m=self.scene(body.scene_id).height_m,
                robot=asdict(self.robot), yaw_rad=0, resolution_m=self.resolution,
                reachable=bool(len(path)), endpoint_q=0, morph_weight=teacher.morph_weight,
                safety_weight=0.2, safety_band_m=0.25,
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
            raise RuntimeError("Habitat interpreter unavailable; configure coflm_habitat_python")
        with self.render_lock, tempfile.TemporaryDirectory(prefix="coflm-observation-") as directory:
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


def create_router(service, require_open, online_running):
    router = APIRouter(prefix="/api/v1/coflm", tags=["CoFL-M"])

    async def execute(request, method, *args):
        require_same_origin(request)
        require_open()
        try:
            return await run_in_threadpool(partial(method, *args))
        except (FileNotFoundError, KeyError) as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        except (RuntimeError, OSError) as error:
            raise HTTPException(503, str(error)) from error
        except ImportError as error:
            raise HTTPException(503, "CoFL-M teacher requires the generation extra (SciPy): " + str(error)) from error

    @router.get("/scenes", response_model=list[MSceneSummary])
    async def scenes(request: Request):
        return await execute(request, service.scenes)

    @router.get("/scenes/{scene_id}", response_model=MScene)
    async def scene(scene_id: str, request: Request):
        return await execute(request, service.scene, scene_id)

    @router.post("/validate", response_model=MDock)
    async def validate(body: MPointRequest, request: Request):
        return await execute(request, service.validate, body.scene_id, body.point)

    @router.post("/sample", response_model=MPair)
    async def sample(body: MRandomRequest, request: Request):
        return await execute(request, service.random_pair, body.scene_id, body.seed)

    @router.post("/teacher", response_model=MTeacher)
    async def teacher(body: MTaskRequest, request: Request):
        return await execute(request, service.teacher, body)

    @router.post("/observation", response_model=MObservation)
    async def observation(body: MObservationRequest, request: Request):
        if online_running():
            raise HTTPException(409, "End the active CoFL-S session before opening CoFL-M Habitat")
        return await execute(request, service.observation, body)

    @router.get("/tasks/{artifact_id}/teacher")
    async def download(artifact_id: str, request: Request):
        require_same_origin(request)
        require_open()
        if not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
            raise HTTPException(404, "Unknown teacher artifact")
        path = service.output / artifact_id / "teacher.npz"
        if not path.is_file():
            raise HTTPException(404, "Unknown teacher artifact")
        return FileResponse(path, media_type="application/octet-stream", filename="teacher.npz")

    return router
