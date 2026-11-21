"""Standalone Habitat interpreter worker: floor raster import and aligned RGB-D.

Run this file with the separate Habitat Python, not the model environment.
"""
import argparse
import json
from pathlib import Path
import sys

# Direct worker execution must not shadow stdlib random with generation/random.py.
if not __package__:
    sys.path = [p for p in sys.path if Path(p or '.').resolve() != Path(__file__).resolve().parent]

import numpy as np


def planning_raster(free, bounds, resolution, crop=None):
    """Reindex Habitat's sampled [Z,X] map; keep its exact sampling origin."""
    z = float(bounds[0][2]) + np.arange(free.shape[0]) * resolution
    x = float(bounds[0][0]) + np.arange(free.shape[1]) * resolution
    if crop is not None:
        xmin, xmax, zmin, zmax = crop
        if not xmin < xmax or not zmin < zmax:
            raise ValueError('Crop bounds must increase')
        ix = np.flatnonzero((x >= xmin) & (x <= xmax))
        iz = np.flatnonzero((z >= zmin) & (z <= zmax))
        if not len(ix) or not len(iz):
            raise ValueError('Crop outside scene bounds')
        free = free[np.ix_(iz, ix)]
        x, z = x[ix], z[iz]
    free = free[::-1, ::-1].T.copy()
    if not free.any():
        raise ValueError('No free floor cells at --floor-y; check floor elevation and crop')
    width = (free.shape[0] - 1) * resolution
    return free, -float(z[-1]), -float(x[-1]) + width / 2


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['import','render'],required=True)
    p.add_argument('--scene',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--floor',type=float,default=0)
    p.add_argument('--resolution',type=float,default=.025)
    p.add_argument('--height',type=float,default=1.2)
    p.add_argument('--floor-tolerance',type=float,default=.1)
    p.add_argument('--poses',type=Path)
    p.add_argument('--scene-dataset')
    p.add_argument('--crop-habitat', type=float, nargs=4, metavar=('XMIN','XMAX','ZMIN','ZMAX'))
    args=p.parse_args()
    if args.resolution<=0 or args.height<=0:raise ValueError('Positive height/resolution required')
    import habitat_sim
    import magnum as mn
    from habitat_sim.utils.common import quat_from_coeffs
    b=habitat_sim.SimulatorConfiguration();b.scene_id=args.scene;b.enable_physics=False
    if args.scene_dataset:b.scene_dataset_config_file=args.scene_dataset
    specs=[]
    if args.mode=='render':
        for name,kind in [('rgb',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH)]:
            spec=habitat_sim.CameraSensorSpec();spec.uuid=name;spec.sensor_type=kind
            spec.resolution=[240,320];spec.position=mn.Vector3(0);spec.hfov=mn.Deg(90)
            spec.near=.05;spec.far=15;specs.append(spec)
    agent=habitat_sim.agent.AgentConfiguration();agent.sensor_specifications=specs
    sim=habitat_sim.Simulator(habitat_sim.Configuration(b,[agent]))
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        if args.mode=='import':
            settings=habitat_sim.NavMeshSettings();settings.set_defaults()
            settings.agent_radius=0.;settings.agent_height=args.height
            settings.cell_size=args.resolution;settings.cell_height=.02
            # Tolerate scan surface noise; the floor slice still excludes other floors.
            settings.agent_max_climb=.1;settings.agent_max_slope=30
            if not sim.recompute_navmesh(sim.pathfinder,settings):raise RuntimeError('Navmesh build failed')
            free=np.asarray(sim.pathfinder.get_topdown_view(args.resolution,args.floor,eps=args.floor_tolerance),bool)
            bounds=sim.pathfinder.get_bounds()
            free, origin_x, origin_y = planning_raster(free, bounds, args.resolution, args.crop_habitat)
            if free.size>1500000:raise ValueError('Floor raster too large; use a room-level asset or larger resolution')
            width=(free.shape[0]-1)*args.resolution
            np.save(args.output/'free.npy',free,allow_pickle=False)
            meta=dict(resolution_m=args.resolution,length_m=(free.shape[1]-1)*args.resolution,
                      width_m=width,origin_x=origin_x,origin_y=origin_y,floor_y=args.floor,
                      flight_height_m=args.height,point_agent_radius_m=0.,
                      crop_habitat=args.crop_habitat,
                      habitat_bounds=[list(map(float, p)) for p in bounds],
                      navmesh_height_m=args.height,flat_floor_tolerance_m=args.floor_tolerance,
                      max_climb_m=.1,max_slope_deg=30)
            (args.output/'raster.json').write_text(json.dumps(meta,indent=2))
        else:
            from PIL import Image
            poses=json.loads(args.poses.read_text())
            for index,pose in enumerate(poses):
                state=habitat_sim.AgentState();state.position=np.asarray(pose['position_habitat'],np.float32)
                state.rotation=quat_from_coeffs(pose.get('rotation_xyzw_habitat',[0,0,0,1]))
                sim.get_agent(0).set_state(state,reset_sensors=True)
                obs=sim.get_sensor_observations();raw=np.asarray(obs['depth'],np.float32)
                valid=np.isfinite(raw)&(raw>=.05)&(raw<15*(1-1e-6))
                depth=np.where(valid,raw,0).astype(np.float32);stem=f'{index:06d}'
                np.save(args.output/(stem+'-depth.npy'),depth,allow_pickle=False)
                np.save(args.output/(stem+'-valid.npy'),valid,allow_pickle=False)
                Image.fromarray(np.asarray(obs['rgb'])[:,:,:3]).save(args.output/(stem+'-rgb.png'))
            (args.output/'poses.json').write_text(json.dumps(poses,indent=2))
    finally:sim.close()


if __name__=='__main__':main()
