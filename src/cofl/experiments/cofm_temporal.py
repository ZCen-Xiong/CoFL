"""Collect, train and replay the A1 point-field experiment on real MP3D RGB-D.

This is an offline same-scene feasibility experiment, not a flight dynamics demo.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from cofl.app.config import load_config
from cofl.app.cofm import CofMService, MTaskRequest
from cofl.experiments.matterport_routes import local_labels, path_heading
from cofl.generation import matterport_worker
from cofl.runtime.temporal_field import WorldFieldMemory, rotation, world_vectors
from cofl.planning.endpoint_flow import EndpointFlowConfig, EndpointFlowPlanner, pair_state


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def resample_path(path, spacing):
    path = np.asarray(path)
    distance = np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    sample = np.arange(0,distance[-1]+1e-9,spacing)
    return np.stack([np.interp(sample,distance,path[:,i]) for i in range(2)],-1)


def collect(config_path, tasks_path, output, *, period_s=.2, speed=.25, max_frames=32):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Collection output must be empty')
    output.mkdir(parents=True,exist_ok=True)
    config=load_config(Path(config_path));config.cofm_output=output.resolve()/'teachers'
    service=CofMService(config)
    tasks=json.loads(Path(tasks_path).read_text())
    if isinstance(tasks,dict):tasks=tasks.get('tasks',[tasks])
    manifest=dict(format='cofm_temporal_rgbd_v1', config=str(Path(config_path).resolve()),
        teacher='one fixed world field per task; visibility masks supervision only',
        period_s=period_s, speed_m_s=speed, camera_height_m=None,
        frame_relation='same world_index identifies the SAME stationary world point in all frames',
        simulation='pose playback along point-teacher paths; no AMB dynamics', routes=[])
    for number,task in enumerate(tasks):
        scene=task['scene_id']; teacher=service.teacher(MTaskRequest(scene_id=scene,start=task['start'],goal=task['goal']))
        if not teacher.reachable:raise ValueError(f'Task {number} unreachable')
        _,metadata,cfg,world,_=service._load(scene)
        if world.raster is None:raise ValueError('Real MP3D imported floor required')
        points=resample_path(teacher.path,speed*period_s)
        # Keep a contiguous window around the first detected narrow section, never skip time steps.
        if len(points)>max_frames:
            indices=teacher.narrow['selected_indices']
            anchor=np.asarray(teacher.path)[indices[len(indices)//2]] if indices else points[len(points)//2]
            center=int(np.argmin(np.linalg.norm(points-anchor,axis=1)))
            start=max(0,min(len(points)-max_frames,center-max_frames//2))
            points=points[start:start+max_frames]
        if len(points)<6:raise ValueError('Need at least six consecutive frames')
        headings=np.unwrap([path_heading(points,i) for i in range(len(points))])
        # Bound camera yaw changes; a smooth offset exposes actual coordinate-rotation effects.
        for i in range(1,len(headings)):
            headings[i]=headings[i-1]+np.clip(headings[i]-headings[i-1],-.08,.08)
        headings+=.12*np.sin(np.arange(len(points))*.35)
        directory=output/f'route-{number:04d}';directory.mkdir()
        poses=[];m=metadata['raster']
        for i,((x,y),yaw) in enumerate(zip(points,headings)):
            poses.append(dict(state_xy=[float(x),float(y)],heading_rad=float(yaw),timestamp_s=i*period_s,
                goal_ego=((np.asarray(task['goal'])-[x,y])@rotation(yaw)).tolist(),
                position_habitat=[-(y+m['origin_y']),m['floor_y']+cfg.camera_height,-(x+m['origin_x'])],
                rotation_xyzw_habitat=[0,float(np.sin(yaw/2)),0,float(np.cos(yaw/2))]))
        write_json(directory/'poses.json',poses)
        cmd=[str(service.habitat_python.resolve()),str(Path(matterport_worker.__file__).resolve()),
             '--mode','render','--scene',metadata['source_scene'],'--poses',str((directory/'poses.json').resolve()),
             '--output',str(directory.resolve())]
        if metadata.get('scene_dataset'):cmd.extend(['--scene-dataset',metadata['scene_dataset']])
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        (directory/'render.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(f'Rendering failed: {directory}/render.log')
        with np.load(service.output/teacher.artifact_id/'teacher.npz') as labels:
            for i,pose in enumerate(poses):
                arrays=local_labels(labels,pose['state_xy'],heading=pose['heading_rad'],
                    depth=np.load(directory/f'{i:06d}-depth.npy'),height=cfg.camera_height)
                if len(arrays['query'])<32:raise ValueError(f'Frame {number}/{i} has too few visible labels')
                np.savez_compressed(directory/f'{i:06d}-field.npz',**arrays)
        manifest['routes'].append(dict(directory=directory.name,scene_id=scene,frames=len(poses),
            teacher=f'teachers/{teacher.artifact_id}',start=task['start'],goal=task['goal'],
            label=task.get('label','manual route'),clearance_m=service.point_clearance_m))
        manifest['camera_height_m']=cfg.camera_height
        write_json(output/'dataset.json',manifest)
        print(f'Collected {directory}: {len(poses)} consecutive RGB-D frames',flush=True)
    return manifest


def replay(data, predictions, *, continuous=True, width=.82, min_width=.60, monotonic=True):
    config=load_config(Path(data.manifest['config']))
    service=CofMService(config)
    cfg=EndpointFlowConfig(monotonic_shrink=monotonic,min_width_m=min_width)
    results=[];offset=0
    for route in data.manifest['routes']:
        planner=EndpointFlowPlanner(cfg,continuous=continuous)
        memory=WorldFieldMemory(max_age_s=1. if continuous else 0.)
        _,_,_,world,_=service._load(route['scene_id'])
        raster=world.raster;clearance=route['clearance_m']
        def free(p):
            return raster.point_clearance(p[...,0],p[...,1])>=clearance
        measured_width=width
        for i in range(route['frames']):
            frame=data.frames[offset+i];pose=frame['pose'];labels=frame['labels']
            vectors=world_vectors(predictions[offset+i],pose['heading_rad'])
            memory_start=time.perf_counter()
            memory.update(labels['world_xy'],vectors,pose['timestamp_s'])
            memory_seconds=time.perf_counter()-memory_start
            start_time=time.perf_counter()
            plan=planner.plan(np.array(pose['state_xy']),pose['heading_rad'],measured_width,
                pose['timestamp_s'],memory.sample,free)
            plan['planning_seconds']=time.perf_counter()-start_time
            plan['memory_update_seconds']=memory_seconds
            plan.update(route=route['directory'],frame=i,observed_width_m=float(measured_width),
                        measured_cog_world=pose['state_xy'],measured_camera_yaw_rad=pose['heading_rad'])
            if plan['status']=='ok':
                # Ideal span tracking ONLY; COG/camera poses are recorded playback, not simulated flight.
                measured_width=plan['width_m'][int(round(cfg.period_s/cfg.integration_dt_s))]
            results.append(plan)
        offset+=route['frames']
    good=[r for r in results if r['status']=='ok']
    joins=[r['join_position_error_m'] for r in good if r['join_position_error_m'] is not None]
    yaw_joins=[r['join_yaw_error_rad'] for r in good if r['join_yaw_error_rad'] is not None]
    velocity_joins=[r['join_endpoint_velocity_error_m_s'] for r in good if r['join_endpoint_velocity_error_m_s'] is not None]
    route_summary={}
    for route in data.manifest['routes']:
        subset=[r for r in results if r['route']==route['directory']]
        run=longest=0
        for r in subset:
            run=run+1 if r['status']=='ok' else 0;longest=max(longest,run)
        route_summary[route['directory']]=dict(label=route['label'],frames=len(subset),
            feasible_frames=sum(r['status']=='ok' for r in subset),longest_successive_frames=longest,
            statuses={s:sum(r['status']==s for r in subset) for s in sorted({r['status'] for r in subset})})
    summary=dict(frames=len(results),feasible_frames=len(good),
        join_position_error_max_m=max(joins) if joins else None,
        join_yaw_error_max_deg=float(np.rad2deg(max(yaw_joins))) if yaw_joins else None,
        join_endpoint_velocity_error_max_m_s=max(velocity_joins) if velocity_joins else None,
        planning_median_s=float(np.median([r['planning_seconds'] for r in results])),
        planning_p95_s=float(np.percentile([r['planning_seconds'] for r in results],95)),
        memory_and_planning_p95_s=float(np.percentile([r['planning_seconds']+r['memory_update_seconds'] for r in results],95)),
        routes=route_summary,
        minimum_reference_width_m=min(min(r['width_m']) for r in good) if good else None,
        selection_angle_min_deg=min(r['perpendicular_deg'] for r in good) if good else None,
        selection_angle_max_deg=max(r['perpendicular_deg'] for r in good) if good else None,
        initial_test_width_m=width,config=asdict(cfg),
        validity='offline oracle raster point/chord check; not a sensor-only collision detector',
        execution='recorded COG/camera playback + ideal scalar-span feedback; not flight closed loop',
        limitations=['reference is ahead of COG; actual-state connector and true AMB inverse kinematics are not implemented',
                     'no whole-body or flight dynamics guarantee; failures are retained, not silently reseeded'])
    return summary,results


def visualize(data, predictions, records, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    output=Path(output);frames=[];saved_success=False
    for index in range(len(data.frames)):
        frame=data.frames[index];labels=frame['labels'];pose=frame['pose']
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        axes[0].imshow(frame['rgb'].cpu().permute(1,2,0));axes[0].set_title(f'Actual Habitat RGB / frame {index}');axes[0].axis('off')
        for ax,field,title in [(axes[1],labels['field'],'Canonical teacher'),(axes[2],predictions[index],'Trained field + A reference')]:
            points=labels['world_xy'];ids=np.arange(0,len(points),max(1,len(points)//650))
            vectors=world_vectors(field,pose['heading_rad'])
            ax.quiver(points[ids,0],points[ids,1],vectors[ids,0],vectors[ids,1],color='#428a80',scale=12)
            ax.scatter(*pose['state_xy'],c='k',s=20,label='Recorded COG')
            ax.set(xlim=(pose['state_xy'][0]-1.2,pose['state_xy'][0]+2),
                   ylim=(pose['state_xy'][1]-1.6,pose['state_xy'][1]+1.6),title=title,xlabel='World X (m)',ylabel='World Y (m)')
            ax.set_aspect('equal')
        record=records[index]
        if record['status']=='ok':
            pair=np.array(record['endpoints_world']);center=pair.mean(1)
            for j,color in [(0,'#e6a23c'),(1,'#9c6bd3')]:axes[2].plot(pair[:,j,0],pair[:,j,1],color=color)
            axes[2].plot(center[:,0],center[:,1],'b-',label='Reference COG')
            axes[2].plot(pair[0,:,0],pair[0,:,1],'r-',lw=2)
        else:axes[2].text(.03,.95,'No feasible pair: retain failure',transform=axes[2].transAxes,color='r',va='top')
        fig.tight_layout();fig.canvas.draw()
        rgba=np.asarray(fig.canvas.buffer_rgba());frames.append(Image.fromarray(rgba[...,:3].copy()))
        if index==0:frames[-1].save(output/'preview.png')
        if record['status']=='ok' and not saved_success:
            frames[-1].save(output/'feasible-preview.png');saved_success=True
        plt.close(fig)
    frames[0].save(output/'continuity.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)


def plot_metrics(data,predictions,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,fields in predictions.items():
        x=[];angles=[];errors=[]
        for a,b,ia,ib in data.pairs:
            va=world_vectors(fields[a][ia],data.frames[a]['pose']['heading_rad'])
            vb=world_vectors(fields[b][ib],data.frames[b]['pose']['heading_rad'])
            norm=np.linalg.norm(va,axis=1)*np.linalg.norm(vb,axis=1);valid=norm>1e-8
            angle=np.rad2deg(np.arccos(np.clip(np.sum(va[valid]*vb[valid],axis=1)/norm[valid],-1,1)))
            x.append(b);angles.append(float(np.mean(angle)));errors.append(float(np.sqrt(np.mean((va-vb)**2))))
        axes[0].plot(x,angles,label=name);axes[1].plot(x,errors,label=name)
    axes[0].set(ylabel='Mean aligned direction difference (degrees)')
    axes[1].set(ylabel='Aligned canonical vector RMSE')
    for ax in axes:ax.set_xlabel('Frame index (two routes)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(Path(output)/'continuity-metrics.png',dpi=150);plt.close(fig)


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    dataset=output/'dataset'
    if args.stage in ('all','collect'):
        collect(args.config,args.tasks,dataset,max_frames=args.max_frames)
    if args.stage=='collect':return
    import torch
    from cofl.training.cofm_temporal import train,SequenceData,predict_all,metrics
    from cofl.models.cofm import CofMPolicy
    torch.set_num_threads(4)
    models={};data=None
    for name,weight in [('baseline',0.),('temporal',args.temporal_weight)]:
        folder=output/name
        if args.stage in ('all','train'):
            model,data,_=train(dataset,folder,steps=args.steps,temporal_weight=weight,
                device=args.device,queries=args.queries)
        else:
            model=CofMPolicy.load(folder/'policy.pt').to(args.device).eval()
            data=SequenceData(dataset,args.device)
        models[name]=model
    report=dict(experiment='A1 canonical world teacher, single-frame DFormer, SE(2)-aligned pair loss',
        created_at_utc=datetime.now(timezone.utc).isoformat(), data=str(dataset.resolve()),
        same_scene_smoke_only=True, metrics={}, planning={})
    final_predictions=None;final_records=None;all_predictions={}
    for name,model in models.items():
        predictions=predict_all(model,data,output/name/'predictions')
        all_predictions[name]=predictions
        report['metrics'][name]=metrics(data,predictions)
        report['metrics'][name]['inference']=json.loads((output/name/'predictions/inference-timing.json').read_text())
        if name=='temporal':final_predictions=predictions
    teachers=[f['labels']['field'] for f in data.frames]
    plot_metrics(data,dict(teacher=teachers,**all_predictions),output)
    for name,predictions in [('teacher',teachers),('trained',final_predictions)]:
        for mode in ('A','B'):
            summary,records=replay(data,predictions,continuous=mode=='A',width=args.width,
                                   min_width=args.min_width,monotonic=not args.allow_expansion)
            key=f'{name}_{mode}';report['planning'][key]=summary
            write_json(output/f'{key}-references.json',records)
            print(key,summary['feasible_frames'],'/',summary['frames'],flush=True)
            if key=='trained_A':final_records=records
    write_json(output/'report.json',report)
    visualize(data,final_predictions,final_records,output)
    # Human-facing report stays in the two established Markdown files; artifacts use JSON/PNG/GIF.
    print(json.dumps(report['metrics'],indent=2),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('artifacts/cof-m/matterport-point-020/studio.yaml'))
    p.add_argument('--tasks',type=Path,default=Path('configs/experiments/cofm_temporal_tasks.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['all','collect','train','evaluate'],default='all')
    p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=300)
    p.add_argument('--max-frames',type=int,default=32);p.add_argument('--queries',type=int,default=512)
    p.add_argument('--temporal-weight',type=float,default=2.);p.add_argument('--width',type=float,default=.82)
    p.add_argument('--min-width',type=float,default=.60)
    p.add_argument('--allow-expansion',action='store_true')
    args=p.parse_args(argv)
    if (args.steps<1 or args.queries<32 or args.max_frames<6
        or not np.isfinite([args.width,args.min_width,args.temporal_weight]).all()
        or args.min_width<=0 or args.width<args.min_width or args.temporal_weight<0):
        p.error('Positive steps/width, >=32 queries, >=6 frames and nonnegative temporal weight required')
    run(args)


if __name__=='__main__':main()
