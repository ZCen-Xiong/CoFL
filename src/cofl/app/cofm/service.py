"""Studio orchestration: scene resources, teacher previews, exports and depth capture.

Planning algorithms live in cofl.planning; neural networks in cofl.models.
This service adapts those capabilities to the Studio request/response contracts.
"""

from __future__ import annotations

import base64
from dataclasses import replace
from types import SimpleNamespace
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import uuid

import numpy as np

from cofl.planning.configuration_field import (
    Box2D, Disc2D, PlanarWorld,
)
from cofl.generation import obstacle_scene, matterport_worker
from cofl.planning.narrow import GroundRaster, narrow_profile
from cofl.planning.point_field import PointTeacher
from .contracts import MDock, MObservation, MObstacle, MPair, MScene, MSceneSummary, MTeacher


class CofMService:
    resolution = 0.025
    arrow_stride = 4  # Default display spacing: 0.025m * 4 = 0.1m.
    max_display_arrows = 4000

    def __init__(self, config):
        self.config = config
        self.resolution = config.cofm_field_resolution_m
        self.point_clearance_m = config.cofm_point_clearance_m
        root = Path(__file__).resolve().parents[4]
        self.habitat_python = config.cofm_habitat_python or root / ".habitat/cofm/bin/python"
        self.output = config.cofm_output or root / "artifacts/cof-m/studio-tasks"
        self.lock, self.render_lock = threading.RLock(), threading.Lock()
        self.rasters = {}
        self.cache = {}  # At most two CPU graphs; no model or GPU state is retained.

    def _load(self, scene_id):
        resource = self.config.scenes.get(scene_id)
        if resource is None:
            raise FileNotFoundError("Unknown scene")
        manifest = resource.path.parent / "scene.json"
        metadata = json.loads(manifest.read_text())
        if metadata.get("geometry_profile") == "ground_navmesh_v1":
            m=metadata["raster"]; cfg=SimpleNamespace(**metadata["config"])
            key=(str(manifest), manifest.stat().st_mtime_ns)
            if key not in self.rasters:
                if len(self.rasters)>=2:self.rasters.pop(next(iter(self.rasters)))
                self.rasters[key]=GroundRaster(np.load(manifest.parent/'free.npy',allow_pickle=False),m['resolution_m'],-cfg.width/2)
            raster=self.rasters[key]
            world=PlanarWorld(scene_id,Box2D(0,cfg.length,-cfg.width/2,cfg.width/2),(),(1,0),(cfg.length-1,0),raster)
            return resource,metadata,cfg,world,[]
        if metadata.get("geometry_profile") != "vertical_extrusion_xy_v1":
            raise ValueError("Select an imported Matterport floor or generated CoF-M scene")
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
        resource, metadata, cfg, world, display = self._load(scene_id)
        map_image=None
        if world.raster is not None:
            from PIL import Image
            image=np.where(world.raster.free[::-1,:,None],np.array([241,244,239]),np.array([112,135,125])).astype(np.uint8)
            buffer=io.BytesIO();Image.fromarray(image).save(buffer,format="PNG")
            map_image="data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()
        return MScene(id=scene_id,label=resource.label or scene_id,length_m=cfg.length,
            width_m=cfg.width,height_m=cfg.camera_height,resolution_m=self.resolution,
            obstacles=display,map_image=map_image,source_kind='matterport' if world.raster is not None else 'procedural',
            habitat_available=self.habitat_python.is_file(),docking={'proxy':'point', 'margin_m':self.point_clearance_m})

    def validate(self, scene_id, point):
        teacher=self._graph(scene_id)
        try:
            index=teacher.locate(point)
        except ValueError as error:
            return MDock(point=point,valid=False,clearance_m=0,reason=str(error))
        clearance=float(teacher.clearance[index])
        valid=bool(teacher.free[index])
        return MDock(point=point,valid=valid,clearance_m=clearance,
            reason='质点净空通过' if valid else '质点位于障碍或净空带内')

    def _graph(self, scene_id):
        resource,metadata,_,world,_=self._load(scene_id)
        path=resource.path.parent/'scene.json'
        if world.raster is not None and metadata['raster'].get('point_agent_radius_m',0)>0:
            raise ValueError('旧场景 navmesh 已预先侵蚀，需用更新后的导入命令重新生成零半径质点场景')
        key=(scene_id,path.stat().st_mtime_ns,self.resolution,self.point_clearance_m,
             self.config.cofm_safety_band_m,self.config.cofm_safety_cost_weight)
        if key not in self.cache:
            if len(self.cache)>=2:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=PointTeacher(world,resolution=self.resolution,clearance_m=self.point_clearance_m,
                safety_band_m=self.config.cofm_safety_band_m,
                safety_weight=self.config.cofm_safety_cost_weight)
        return self.cache[key]

    def random_pair(self, scene_id, seed):
        raise ValueError('当前任务使用地图手动选择起点和终点')

    def teacher(self, body):
        with self.lock:
            start,goal=self.validate(body.scene_id,body.start),self.validate(body.scene_id,body.goal)
            if not start.valid or not goal.valid:raise ValueError(f'起点：{start.reason}；终点：{goal.reason}')
            if np.linalg.norm(np.asarray(body.start)-body.goal)<self.resolution:
                raise ValueError('起终点必须不同')
            t=self._graph(body.scene_id)
            t.world=replace(t.world,start_xy=body.start,goal_xy=body.goal);t.solve()
            path=t.rollout();narrow=None
            if t.world.raster is not None:
                narrow=narrow_profile(path,t.world.raster,max_width=body.max_width_m,min_length=body.min_narrow_length_m)
                if body.narrow_only is True and not narrow['accepted']:raise ValueError('路径未经过指定窄段')
            field,reachable,_=t.field()
            selected=np.zeros(t.shape,bool);selected[::self.arrow_stride,::self.arrow_stride]=True
            selected &= reachable
            arrows=np.concatenate((t.states[selected],field[selected]),axis=1)
            if len(arrows)>self.max_display_arrows:arrows=arrows[np.linspace(0,len(arrows)-1,self.max_display_arrows,dtype=int)]
            artifact_id=uuid.uuid4().hex;output=self.output/artifact_id;output.mkdir(parents=True)
            t.save(output/'teacher.npz');np.save(output/'path.npy',path,allow_pickle=False)
            obstacle_scene.save_json(output/'task.json',dict(format='cofm_point_task_v1',scene_id=body.scene_id,
                scene_manifest=str(self.config.scenes[body.scene_id].path.parent/'scene.json'),
                start=body.start,goal=body.goal,height_m=self.scene(body.scene_id).height_m,
                clearance_m=self.point_clearance_m,resolution_m=self.resolution,narrow=narrow,
                field_source='cofl_s_geodesic_potential_kernels'))
            return MTeacher(reachable=bool(len(path)),resolution_m=self.resolution,path=path.tolist(),
                arrows=arrows.tolist(),min_path_clearance_m=float(t.clearance[reachable].min()) if reachable.any() else None,
                artifact_id=artifact_id,narrow=narrow,message='CoFL-S 质点测地势场；未执行整机或 q 约束')

    def observation(self, body):
        from PIL import Image
        resource, metadata, cfg, world, _ = self._load(body.scene_id)
        if not self.validate(body.scene_id,body.point).valid:
            raise ValueError('观测位置不满足质点净空')
        if not self.habitat_python.is_file():
            raise RuntimeError("Habitat interpreter unavailable; configure cofm_habitat_python")
        with self.render_lock, tempfile.TemporaryDirectory(prefix="cofm-observation-") as directory:
            output=Path(directory)
            if world.raster is not None:
                m=metadata['raster']
                position=np.asarray([*body.point,cfg.camera_height])
                poses=[dict(position_habitat=[-(body.point[1]+m['origin_y']),m['floor_y']+cfg.camera_height,-(body.point[0]+m['origin_x'])])]
                (output/'poses.json').write_text(json.dumps(poses))
                cmd=[str(self.habitat_python.absolute()),str(Path(matterport_worker.__file__).resolve()),'--mode','render',
                     '--scene',metadata['source_scene'],'--poses',str(output/'poses.json'),'--output',str(output)]
                if metadata.get('scene_dataset'):cmd.extend(['--scene-dataset',metadata['scene_dataset']])
                try:
                    result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError('Habitat RGB-D rendering timed out') from error
                if result.returncode:raise RuntimeError('Habitat RGB-D failed: '+result.stderr[-600:])
                (output/'depth').mkdir();(output/'valid').mkdir();(output/'rgb').mkdir()
                (output/'000000-depth.npy').rename(output/'depth/000000.npy')
                (output/'000000-valid.npy').rename(output/'valid/000000.npy')
                (output/'000000-rgb.png').rename(output/'rgb/000000.png')
            else:
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
            # Metric depth remains separate from the grayscale viewing transform.
            image = np.where(valid, 30 + 225*(1-np.clip(depth/cfg.far, 0, 1)), 0).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            rgb_image=None
            if (output/'rgb/000000.png').is_file():
                rgb_image='data:image/png;base64,'+base64.b64encode((output/'rgb/000000.png').read_bytes()).decode()
            return MObservation(rgb_image=rgb_image,image="data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode(),
                                position=tuple(position),
                                valid_fraction=float(valid.mean()), far_m=cfg.far)


