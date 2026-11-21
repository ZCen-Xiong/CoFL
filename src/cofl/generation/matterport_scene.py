"""Prepare a Matterport/Habitat floor for CoF-M without language episodes."""
import argparse
import json
from pathlib import Path
import subprocess
import yaml
from . import matterport_worker


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True,help='GLB or Habitat stage config')
    p.add_argument('--scene-dataset',type=Path)
    p.add_argument('--floor-y',type=float,required=True,help='Habitat Y-up floor elevation in metres')
    p.add_argument('--height',type=float,default=1.2)
    p.add_argument('--floor-tolerance',type=float,default=.1)
    p.add_argument('--resolution',type=float,default=.025)
    p.add_argument('--crop-habitat', type=float, nargs=4, metavar=('XMIN','XMAX','ZMIN','ZMAX'),
                   help='Optional room bounds in Habitat horizontal X/Z coordinates')
    p.add_argument('--habitat-python',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if not a.scene.is_file():raise FileNotFoundError(a.scene)
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Use an empty output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    cmd=[str(a.habitat_python.absolute()),str(Path(matterport_worker.__file__).resolve()),
         '--mode','import','--scene',str(a.scene.resolve()),'--output',str(a.output.resolve()),
         '--floor',str(a.floor_y),'--height',str(a.height),'--resolution',str(a.resolution),
         '--floor-tolerance',str(a.floor_tolerance)]
    if a.scene_dataset:cmd.extend(['--scene-dataset',str(a.scene_dataset.resolve())])
    if a.crop_habitat:cmd.extend(['--crop-habitat',*map(str,a.crop_habitat)])
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    (a.output/'import.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(f'Import failed: {a.output / "import.log"}')
    raster=json.loads((a.output/'raster.json').read_text())
    meta=dict(format='cof_m_ground_scene',version=1,geometry_profile='ground_navmesh_v1',
              source_scene=str(a.scene.resolve()),scene_dataset=None if not a.scene_dataset else str(a.scene_dataset.resolve()),
              raster=raster,obstacles=[],config=dict(length=raster['length_m'],width=raster['width_m'],
              height=3.,camera_height=a.height,image_width=320,image_height=240,near=.05,far=15.))
    (a.output/'scene.json').write_text(json.dumps(meta,indent=2))
    # Studio scene catalog resources point to this manifest; no scene is copied.
    catalog=dict(device='cpu',cofm_habitat_python=str(a.habitat_python.absolute()),
                 scenes={'mp3d-floor':dict(path='scene.json',label='Matterport RGB-D floor',domain_id='cofm')})
    (a.output/'studio.yaml').write_text(yaml.safe_dump(catalog,sort_keys=False))
    print('Imported floor. Start Studio with --config',a.output/'studio.yaml')


if __name__=='__main__':main()
