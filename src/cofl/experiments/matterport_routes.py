"""Collect RGB-D only near narrow sections of user-specified planar routes."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from cofl.app.config import load_config
from cofl.app.cofm import CofMService, MTaskRequest
from cofl.generation import matterport_worker
from cofl.generation.sector.fields.sector_gt import _build_visible_bev_mask_3d


def local_labels(labels, state, *, heading=0., radius=5., half_fov=np.pi/4,
                 depth=None, height=.2):
    """Sample one fixed world teacher; visibility changes supervision, never geometry."""
    x, y = np.meshgrid(labels['x'], labels['y'])
    wx, wy = x-state[0], y-state[1]
    c, s = np.cos(heading), np.sin(heading)
    dx, dy = c*wx+s*wy, -s*wx+c*wy
    distance = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    if 'q' in labels:
        raise ValueError('旧构型标签不能作为质点场使用，请重新生成')
    reachable=labels['reachable'].copy()
    if depth is not None:
        reachable &= _build_visible_bev_mask_3d(dx,dy,np.zeros(dx.shape),
            np.array([0,height,0]),np.array([1,0,0]),np.array([0,0,1]),
            2*half_fov,radius,depth,depth_visibility_tol_m=.1)
    mask = (distance <= radius) & (abs(angle) <= half_fov) & reachable
    field = labels['field'][mask].copy()
    fx, fy = field[:, 0].copy(), field[:, 1].copy()
    field[:, 0], field[:, 1] = c*fx+s*fy, -s*fx+c*fy
    return dict(query=np.stack((angle[mask]/half_fov, distance[mask]/radius), -1).astype(np.float32),
                field=field,
                terminal=labels['terminal'][mask],
                world_xy=np.stack((x[mask],y[mask]),-1).astype(np.float32),
                world_index=np.flatnonzero(mask).astype(np.int64))


def path_heading(path, index):
    """Use the next spatial transition, skipping shape-only states."""
    path = np.asarray(path)
    for stop in range(index+1, len(path)):
        delta = path[stop, :2]-path[index, :2]
        if np.linalg.norm(delta) > 1e-8:
            return float(np.arctan2(delta[1], delta[0]))
    for start in range(index-1, -1, -1):
        delta = path[index, :2]-path[start, :2]
        if np.linalg.norm(delta) > 1e-8:
            return float(np.arctan2(delta[1], delta[0]))
    return 0.


def collect(config_path, task_path, output, *, max_width=1.1, min_length=.3, max_frames=200, camera_heading='path'):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty collection output directory')
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(config_path))
    config.cofm_output = output.resolve() / 'teachers'
    service = CofMService(config)
    tasks = json.loads(Path(task_path).read_text())
    if isinstance(tasks, dict):
        tasks = tasks.get('tasks', [tasks])
    manifest = dict(format='cofm_point_rgbd_routes_v2', teacher_visibility='supervision_mask_only', clearance_m=service.point_clearance_m, camera=dict(width=320, height=240,
                    hfov_deg=90, near_m=.05, far_m=15, heading=camera_heading),
                    query='[theta/(pi/4), range_m/5]', field='[Fx,Fy] CoFL-S canonical field',
                    accepted=[], rejected=[])
    for number, task in enumerate(tasks):
        scene_id = task['scene_id']
        _, metadata, cfg, world, _ = service._load(scene_id)
        if world.raster is None:
            raise ValueError('This collector requires an imported Matterport floor')
        request = MTaskRequest(scene_id=scene_id, start=task['start'], goal=task['goal'],
                               narrow_only=True, max_width_m=max_width, min_narrow_length_m=min_length)
        try:
            teacher = service.teacher(request)
        except ValueError as error:
            manifest['rejected'].append(dict(task=number, scene_id=scene_id, reason=str(error)))
            continue
        indices = teacher.narrow['selected_indices']
        if len(indices) > max_frames:
            indices = [indices[i] for i in np.linspace(0, len(indices)-1, max_frames, dtype=int)]
        root = output / f'route-{number:04d}'
        root.mkdir()
        raster = metadata['raster']
        poses = []
        for i in indices:
            x, y = teacher.path[i]
            heading = path_heading(teacher.path, i) if camera_heading == 'path' else 0.
            c, s = np.cos(heading), np.sin(heading)
            dx, dy = task['goal'][0]-x, task['goal'][1]-y
            poses.append(dict(path_index=i, state_xy=[x, y],
                goal_ego=[c*dx+s*dy, -s*dx+c*dy], heading_rad=heading,
                position_habitat=[-(y+raster['origin_y']), raster['floor_y']+cfg.camera_height,
                                  -(x+raster['origin_x'])],
                rotation_xyzw_habitat=[0, float(np.sin(heading/2)), 0, float(np.cos(heading/2))]))
        (root/'poses.json').write_text(json.dumps(poses, indent=2))
        command = [str(service.habitat_python.absolute()), str(Path(matterport_worker.__file__).resolve()),
                   '--mode', 'render', '--scene', metadata['source_scene'],
                   '--poses', str((root/'poses.json').resolve()), '--output', str(root.resolve())]
        if metadata.get('scene_dataset'):
            command.extend(['--scene-dataset', metadata['scene_dataset']])
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (root/'render.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'RGB-D rendering failed: {root / "render.log"}')
        with np.load(service.output/teacher.artifact_id/'teacher.npz', allow_pickle=False) as labels:
            for frame, pose in enumerate(poses):
                depth=np.load(root/f'{frame:06d}-depth.npy',allow_pickle=False)
                arrays = local_labels(labels, pose['state_xy'], heading=pose['heading_rad'],
                                      depth=depth,height=cfg.camera_height)
                np.savez_compressed(root/f'{frame:06d}-field.npz', **arrays)
        manifest['accepted'].append(dict(task=number, scene_id=scene_id, directory=root.name,
            teacher=f'teachers/{teacher.artifact_id}', frames=len(poses),
            start=task['start'], goal=task['goal'], narrow=teacher.narrow))
        # Persist after each route so a long collection remains inspectable.
        (output/'dataset.json').write_text(json.dumps(manifest, indent=2))
    (output/'dataset.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True, help='Task JSON or list: scene_id, start, goal')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-width', type=float, default=1.1)
    p.add_argument('--min-length', type=float, default=.3)
    p.add_argument('--max-frames', type=int, default=200, help='Per accepted route')
    p.add_argument('--camera-heading', choices=['path', 'fixed'], default='path',
                   help='Camera direction only; teacher body yaw remains fixed at zero')
    a = p.parse_args(argv)
    if a.max_frames <= 0:
        p.error('--max-frames must be positive')
    report = collect(a.config, a.tasks, a.output, max_width=a.max_width,
                     min_length=a.min_length, max_frames=a.max_frames, camera_heading=a.camera_heading)
    print(f"Accepted {len(report['accepted'])}, rejected {len(report['rejected'])}; {a.output/'dataset.json'}")


if __name__ == '__main__':
    main()
