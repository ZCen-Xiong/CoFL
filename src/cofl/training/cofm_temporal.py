"""Small native RGB-D point-field trainer with pose-aligned pair supervision.

No language, q labels, hidden-state teacher or pretrained-asset downloads.
"""
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from cofl.models.cofm import CofMConfig, CofMPolicy
from cofl.runtime.temporal_field import rotation


def aligned_world_vectors(prediction, yaw):
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    world = torch.stack((c*prediction[..., 0]-s*prediction[..., 1],
                         s*prediction[..., 0]+c*prediction[..., 1]), -1)
    return world


def aligned_pair_loss(prediction, yaw):
    """Both query lists address the same world points, in the same order."""
    world=aligned_world_vectors(prediction,yaw)
    return F.mse_loss(world[0], world[1])


def temporal_terms(prediction, yaw, target):
    world=aligned_world_vectors(prediction,yaw)
    mse=F.mse_loss(world[0],world[1])
    valid=(torch.linalg.vector_norm(target[0],dim=-1)>.01)&(torch.linalg.vector_norm(target[1],dim=-1)>.01)
    angular=(1-F.cosine_similarity(world[0],world[1],dim=-1))[valid].mean() if valid.any() else mse*0
    return mse,angular


class SequenceData:
    def __init__(self, root, device, image_size=224):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'dataset.json').read_text())
        if self.manifest.get('format') != 'cofm_temporal_rgbd_v1':
            raise ValueError('Regenerate pose-paired data with a canonical world teacher')
        self.frames, self.pairs = [], []
        self.train_ids, self.validation_ids = [], []
        for route in self.manifest['routes']:
            folder = self.root/route['directory']
            poses = json.loads((folder/'poses.json').read_text())
            offset = len(self.frames)
            for i, pose in enumerate(poses):
                stem = folder/f'{i:06d}'
                with np.load(str(stem)+'-field.npz') as f:
                    labels = {k: f[k] for k in f.files}
                rgb = np.array(Image.open(str(stem)+'-rgb.png').convert('RGB'), copy=True)
                depth = np.load(str(stem)+'-depth.npy')
                rgb = torch.from_numpy(rgb).permute(2,0,1).float()[None]/255
                depth = torch.from_numpy(depth)[None,None]
                rgb = F.interpolate(rgb, (image_size, image_size), mode='bilinear', align_corners=False)[0]
                depth = F.interpolate(depth, (image_size, image_size), mode='nearest')[0]
                self.frames.append(dict(labels=labels, pose=pose, rgb=rgb.to(device), depth=depth.to(device),
                    goal=torch.tensor(pose['goal_ego'], device=device, dtype=torch.float32)))
                # Hold out actual frames, not just randomly selected queries in training images.
                (self.validation_ids if i % 5 == 3 else self.train_ids).append(offset+i)
            for i in range(len(poses)-1):
                a, b = offset+i, offset+i+1
                ids, ia, ib = np.intersect1d(self.frames[a]['labels']['world_index'],
                    self.frames[b]['labels']['world_index'], return_indices=True)
                if len(ids) >= 16:
                    self.pairs.append((a,b,ia,ib))
        train = set(self.train_ids)
        self.train_pairs = [p for p in self.pairs if p[0] in train and p[1] in train]
        if not self.train_pairs or not self.validation_ids:
            raise ValueError('Need adjacent training pairs and held-out frames')

    def inputs(self, ids):
        return dict(rgb=torch.stack([self.frames[i]['rgb'] for i in ids]),
                    depth=torch.stack([self.frames[i]['depth'] for i in ids]),
                    goal_ego=torch.stack([self.frames[i]['goal'] for i in ids]))


def training_step(model,data,pair,rng,queries,device,optimizer,temporal_weight):
    """One update shared by the short experiment and timed epoch runner."""
    a, b, ia, ib = pair
    n = queries//2
    choose = rng.choice(len(ia), n, replace=len(ia)<n)
    indices = []
    for frame, shared in [(a,ia), (b,ib)]:
        labels = data.frames[frame]['labels']
        extra = rng.integers(len(labels['query']), size=queries-n)
        indices.append(np.r_[shared[choose], extra])
    q = torch.tensor(np.stack([data.frames[f]['labels']['query'][i] for f,i in zip((a,b),indices)]),device=device)
    target = torch.tensor(np.stack([data.frames[f]['labels']['field'][i] for f,i in zip((a,b),indices)]),device=device)
    prediction = model(q, **data.inputs([a,b]))
    supervised = F.mse_loss(prediction, target)
    valid = torch.linalg.vector_norm(target, dim=-1) > 1e-4
    direction = (1-F.cosine_similarity(prediction, target, dim=-1))[valid].mean() if valid.any() else supervised*0
    yaw = torch.tensor([data.frames[f]['pose']['heading_rad'] for f in (a,b)],device=device)
    temporal,temporal_angle = temporal_terms(prediction[:,:n], yaw,target[:,:n])
    loss = 10*supervised + .1*direction + temporal_weight*(10*temporal+.25*temporal_angle)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise RuntimeError('Nonfinite training loss/gradient')
    optimizer.step()
    return dict(loss=loss.detach(),supervised=supervised.detach(),temporal=temporal.detach(),
                temporal_angular=temporal_angle.detach(),direction=direction.detach())


@torch.no_grad()
def validation_score(model,data,weight,device):
    model.eval();scores=[]
    validation=set(data.validation_ids)
    for a,b,ia,ib in data.pairs:
        if a not in validation and b not in validation:continue
        take=np.linspace(0,len(ia)-1,min(512,len(ia)),dtype=int)
        indices=[ia[take],ib[take]]
        q=torch.tensor(np.stack([data.frames[f]['labels']['query'][i] for f,i in zip((a,b),indices)]),device=device)
        target=torch.tensor(np.stack([data.frames[f]['labels']['field'][i] for f,i in zip((a,b),indices)]),device=device)
        prediction=model(q,**data.inputs([a,b]))
        yaw=torch.tensor([data.frames[f]['pose']['heading_rad'] for f in (a,b)],device=device)
        mse,angular=temporal_terms(prediction,yaw,target)
        valid=torch.linalg.vector_norm(target,dim=-1)>.01
        direction=(1-F.cosine_similarity(prediction,target,dim=-1))[valid].mean() if valid.any() else mse*0
        scores.append(float(10*F.mse_loss(prediction,target)+.1*direction+weight*(10*mse+.25*angular)))
    model.train()
    return float(np.mean(scores))


def train(root, output, *, steps=250, temporal_weight=2., seed=17, device='cpu', queries=512,
          image_size=224, d_model=64):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Training output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    data = SequenceData(root, device, image_size)
    model = CofMPolicy(CofMConfig(image_size=image_size, d_model=d_model)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    history = [];best_score=float('inf');best_step=0
    model.train()
    for step in range(steps):
        values=training_step(model,data,data.train_pairs[int(rng.integers(len(data.train_pairs)))],
                             rng,queries,device,optimizer,temporal_weight)
        if step % 25 == 0 or step == steps-1:
            item = dict(step=step+1,**{k:float(v) for k,v in values.items()})
            history.append(item); print(json.dumps(item), flush=True)
        if (step+1)%100==0 or step==steps-1:
            score=validation_score(model,data,temporal_weight,device)
            print(json.dumps(dict(step=step+1,validation_score=score)),flush=True)
            if score<best_score:
                best_score,best_step=score,step+1;model.save(output/'policy.pt')
    model.eval();model.save(output/'policy-last.pt')
    model=CofMPolicy.load(output/'policy.pt').to(device).eval()
    report = dict(format='cofm_temporal_training_v1', seed=seed, steps=steps, temporal_weight=temporal_weight,
        train_frames=len(data.train_ids), heldout_frames=len(data.validation_ids), train_pairs=len(data.train_pairs),
        pretrained=False, encoder_trained=True, image_size=image_size, d_model=d_model,
        split='every fifth frame (index mod 5 == 3) held out; same-scene smoke test, not scene generalization',
        gradient_clip_norm=1., history=history)
    report.update(optimizer='AdamW',learning_rate=3e-4,weight_decay=1e-4,queries_per_frame=queries,
                  shared_queries_per_pair=queries//2)
    report.update(best_step=best_step,best_validation_score=best_score,
                  checkpoint_selection='paired held-out-frame validation objective every 100 steps; not independent test data',
                  temporal_objective='10 * world-vector MSE + .25 * world-direction cosine loss (nonterminal shared points)')
    (output/'training.json').write_text(json.dumps(report,indent=2))
    return model, data, report


@torch.no_grad()
def predict_all(model, data, output):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    device = next(model.parameters()).device
    predictions = []; timings=[]
    for i, frame in enumerate(data.frames):
        q = torch.tensor(frame['labels']['query'],device=device)[None]
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter()
        context = model.decoder.prepare_context(model.encode(**data.inputs([i])))
        chunks = [model.decoder.decode(c,context)[0].cpu().numpy() for c in q.split(4096,dim=1)]
        prediction = np.concatenate(chunks)
        if device.type=='cuda':torch.cuda.synchronize(device)
        timings.append(dict(frame=i,queries=len(prediction),seconds=time.perf_counter()-start))
        predictions.append(prediction)
        np.savez_compressed(output/f'{i:06d}.npz', field=prediction,
            world_xy=frame['labels']['world_xy'], world_index=frame['labels']['world_index'])
    (output/'inference-timing.json').write_text(json.dumps(dict(samples=timings,
        median_s=float(np.median([t['seconds'] for t in timings])),
        p95_s=float(np.percentile([t['seconds'] for t in timings],95)),
        scope='FP32 encoder+all visible queries, GPU synchronization included; input loading and planning excluded'),indent=2))
    return predictions


def metrics(data, predictions):
    ids = data.validation_ids
    error = np.concatenate([predictions[i]-data.frames[i]['labels']['field'] for i in ids])
    angles = []
    for i in ids:
        p, t = predictions[i], data.frames[i]['labels']['field']
        good = (np.linalg.norm(t,axis=1)>1e-4)&(np.linalg.norm(p,axis=1)>1e-4)
        cos = (p[good]*t[good]).sum(1)/(np.linalg.norm(p[good],axis=1)*np.linalg.norm(t[good],axis=1))
        angles.extend(np.rad2deg(np.arccos(np.clip(cos,-1,1))))
    diffs, teacher_diffs, pair_angles = [], [], []
    for a,b,ia,ib in data.pairs:
        if a not in ids and b not in ids:
            continue
        wa = predictions[a][ia] @ rotation(data.frames[a]['pose']['heading_rad']).T
        wb = predictions[b][ib] @ rotation(data.frames[b]['pose']['heading_rad']).T
        ta = data.frames[a]['labels']['field'][ia] @ rotation(data.frames[a]['pose']['heading_rad']).T
        tb = data.frames[b]['labels']['field'][ib] @ rotation(data.frames[b]['pose']['heading_rad']).T
        diffs.extend(wa-wb); teacher_diffs.extend(ta-tb)
        norm = np.linalg.norm(wa,axis=1)*np.linalg.norm(wb,axis=1)
        good = norm>1e-8
        pair_angles.extend(np.rad2deg(np.arccos(np.clip((wa[good]*wb[good]).sum(1)/norm[good],-1,1))))
    if not len(diffs):
        raise ValueError('No validation frame overlaps')
    return dict(heldout_field_rmse=float(np.sqrt(np.mean(np.square(error)))),
        heldout_direction_error_deg=float(np.mean(angles)),
        aligned_pair_rmse=float(np.sqrt(np.mean(np.square(diffs)))),
        aligned_pair_angle_mean_deg=float(np.mean(pair_angles)),
        aligned_pair_angle_p95_deg=float(np.percentile(pair_angles,95)),
        teacher_pair_rmse=float(np.sqrt(np.mean(np.square(teacher_diffs)))),
        field_units='canonical geodesic magnitude / 5; not m/s',
        evaluation='held-out frames within collected routes; no unseen-scene claim')
