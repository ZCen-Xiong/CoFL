"""Time-budgeted epoch continuation of the A1 RGB-D field experiment."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from cofl.models.cofm import CofMPolicy
from cofl.training.cofm_temporal import SequenceData,training_step,validation_score,predict_all,metrics


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def atomic_torch(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,tmp);tmp.replace(path)


def save_policy(model,path):
    atomic_torch(path,dict(format='cofm_point_rgbd_v2',config=asdict(model.config),state_dict=model.state_dict()))


def planned_epochs(hours,epoch_seconds,validation_seconds,validate_every):
    if min(hours,epoch_seconds,validate_every)<=0 or validation_seconds<0:
        raise ValueError('Positive duration, epoch time and validation interval required')
    return math.ceil(hours*3600/(epoch_seconds+validation_seconds/validate_every))


def evaluate_best(data,folder,device,*,final=False):
    model=CofMPolicy.load(folder/'policy.pt').to(device).eval()
    predictions=predict_all(model,data,folder/'predictions')
    result=metrics(data,predictions)
    result['created_at_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(folder/'metrics-latest.json',result)
    with (folder/'metrics-history.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    if final:
        from cofl.experiments.cofm_temporal import replay,visualize,plot_metrics
        records=None;result['planning']={}
        teachers=[f['labels']['field'] for f in data.frames]
        for name,fields in [('teacher',teachers),('trained',predictions)]:
            for mode in ['A','B']:
                summary,refs=replay(data,fields,continuous=mode=='A')
                result['planning'][name+'_'+mode]=summary
                atomic_json(folder/(name+'_'+mode+'-references.json'),refs)
                if name=='trained' and mode=='A':records=refs
        visualize(data,predictions,records,folder)
        plot_metrics(data,dict(teacher=teachers,long_train=predictions),folder)
        atomic_json(folder/'evaluation-final.json',result)
    del model
    return result


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'status.json').exists() and not args.resume:
        raise FileExistsError('Use an empty output or explicit --resume')
    torch.set_num_threads(4);torch.manual_seed(17)
    initial=CofMPolicy.load(args.initial).to(args.device)
    data=SequenceData(args.dataset,args.device,initial.config.image_size)
    model=initial
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=10,min_lr=1e-6)
    rng=np.random.default_rng(17)
    fingerprint=hashlib.sha256((args.dataset/'dataset.json').read_bytes()).hexdigest()
    config=dict(dataset=str(args.dataset.resolve()),dataset_manifest_sha256=fingerprint,
        initial_weights=str(args.initial.resolve()),initial_optimizer='new AdamW: old smoke checkpoint has no optimizer state',
        train_pairs=len(data.train_pairs),train_frames=len(data.train_ids),validation_frames=len(data.validation_ids),
        epoch_definition='one shuffled traversal of ALL training pairs, no replacement; random queries each visit',
        epochs=args.epochs,hours=args.hours,queries=args.queries,temporal_weight=args.temporal_weight,
        lr=args.lr,validate_every=args.validate_every,checkpoint_seconds=args.checkpoint_seconds,
        device=args.device,seed=17,same_scene_only=True)
    epoch=step=0;order=[];cursor=0;prior_elapsed=0.;best_epoch=0
    if args.resume:
        state=torch.load(output/'resume.pt',map_location='cpu',weights_only=True)
        for key in ['dataset_manifest_sha256','queries','temporal_weight','train_pairs']:
            if state['config'][key]!=config[key]:raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);rng.bit_generator.state=state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
        epoch,step,order,cursor=state['epoch'],state['step'],state['order'],state['cursor']
        prior_elapsed,best_score,best_epoch=state['elapsed_s'],state['best_score'],state['best_epoch']
    else:
        best_score=validation_score(model,data,args.temporal_weight,args.device)
        save_policy(model,output/'policy.pt')
    atomic_json(output/'config.json',config)
    stop=[False]
    def request_stop(signum,frame):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    started=time.monotonic();last_checkpoint=last_metrics=started
    epoch_times=[];last_score=best_score
    def elapsed():return prior_elapsed+time.monotonic()-started
    def status(kind,**extra):
        value=dict(status=kind,pid=os.getpid(),updated_at_utc=datetime.now(timezone.utc).isoformat(),
            epoch=epoch,step=step,epoch_pair_cursor=cursor,elapsed_s=elapsed(),remaining_s=max(0,args.hours*3600-elapsed()),
            target_epochs=args.epochs,best_validation_score=best_score,best_epoch=best_epoch,
            last_validation_score=last_score,learning_rate=optimizer.param_groups[0]['lr'],**extra)
        atomic_json(output/'status.json',value);return value
    def checkpoint():
        atomic_torch(output/'resume.pt',dict(format='cofm_temporal_resume_v1',config=config,
            state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            epoch=epoch,step=step,order=order,cursor=cursor,elapsed_s=elapsed(),best_score=best_score,best_epoch=best_epoch))
        save_policy(model,output/'policy-last.pt')
    status('running');checkpoint()
    try:
        model.train()
        while epoch<args.epochs and elapsed()<args.hours*3600 and not stop[0]:
            epoch_start=time.monotonic()
            if not order:order=rng.permutation(len(data.train_pairs)).tolist();cursor=0
            while cursor<len(order) and elapsed()<args.hours*3600 and not stop[0]:
                values=training_step(model,data,data.train_pairs[order[cursor]],rng,args.queries,
                    args.device,optimizer,args.temporal_weight)
                cursor+=1;step+=1
            if cursor<len(order):break
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            epoch_times.append(time.monotonic()-epoch_start)
            epoch+=1;order=[];cursor=0
            if epoch%args.validate_every==0 or args.benchmark or epoch==args.epochs:
                start=time.monotonic();last_score=validation_score(model,data,args.temporal_weight,args.device)
                validation_seconds=time.monotonic()-start
                scheduler.step(last_score)
                if last_score<best_score:
                    best_score,best_epoch=last_score,epoch;save_policy(model,output/'policy.pt')
                with (output/'validation.jsonl').open('a') as f:
                    f.write(json.dumps(dict(epoch=epoch,step=step,score=last_score,best=best_score,elapsed_s=elapsed()))+'\n')
            value=status('running',last_epoch_seconds=epoch_times[-1],loss=float(values['loss']))
            if epoch%args.validate_every==0 or args.benchmark:print(json.dumps(value),flush=True)
            if time.monotonic()-last_checkpoint>=args.checkpoint_seconds:
                checkpoint();last_checkpoint=time.monotonic()
            if not args.benchmark and time.monotonic()-last_metrics>=args.metrics_seconds:
                evaluate_best(data,output,args.device);last_metrics=time.monotonic();model.train()
        checkpoint()
        if args.benchmark:
            measured=epoch_times[1:] if len(epoch_times)>1 else epoch_times
            result=dict(epoch_seconds_median=float(np.median(measured)),epoch_seconds_all=epoch_times,
                validation_seconds=validation_seconds,train_pairs=len(data.train_pairs),steps_per_epoch=len(data.train_pairs),
                epoch_timing='training only, warmup epoch excluded from median',
                planned_validation_every=max(1,round(60/float(np.median(measured)))))
            result['estimated_8h_epochs']=planned_epochs(8,result['epoch_seconds_median'],result['validation_seconds'],result['planned_validation_every'])
            atomic_json(output/'benchmark.json',result);print(json.dumps(result),flush=True)
        else:
            evaluate_best(data,output,args.device,final=True)
        final=status('stopped' if stop[0] else 'completed',
            stop_reason='signal' if stop[0] else ('wall_time_budget' if elapsed()>=args.hours*3600 else 'epoch_budget'))
        print(json.dumps(final),flush=True)
    except BaseException as exc:
        status('failed',error=repr(exc))
        try:checkpoint()
        except Exception:pass
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('artifacts/cof-m/temporal-a-20261122/dataset'))
    p.add_argument('--initial',type=Path,default=Path('artifacts/cof-m/temporal-a-20261122/temporal/policy.pt'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--hours',type=float,default=8);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--queries',type=int,default=512);p.add_argument('--temporal-weight',type=float,default=2.)
    p.add_argument('--validate-every',type=int,default=30);p.add_argument('--checkpoint-seconds',type=float,default=300)
    p.add_argument('--metrics-seconds',type=float,default=1800)
    p.add_argument('--resume',action='store_true');p.add_argument('--benchmark',action='store_true')
    args=p.parse_args(argv)
    if min(args.epochs,args.hours,args.lr,args.queries,args.validate_every,args.checkpoint_seconds,args.metrics_seconds)<=0:
        p.error('Positive budgets, learning rate, query count and intervals required')
    run(args)


if __name__=='__main__':main()
