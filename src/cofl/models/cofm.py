"""Language-free RGB-D DFormer policy. Random weights unless explicitly loaded."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.dformer import DFormer_Tiny
from .cofl_s import CoFLSFieldDecoder


@dataclass(frozen=True)
class CofMConfig:
    image_size: int = 224
    max_depth_m: float = 15.0
    d_model: int = 256


class CofMPolicy(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or CofMConfig()
        c = self.config
        if c.image_size < 32 or c.image_size % 32 or c.d_model % 8 or c.d_model < 8:
            raise ValueError('image_size must be a multiple of 32; d_model a positive multiple of 8')
        if not math.isfinite(c.max_depth_m) or c.max_depth_m <= 0:
            raise ValueError('max_depth_m must be finite and positive')
        self.backbone = DFormer_Tiny(drop_path_rate=0)
        self.projection = nn.Conv2d(128, c.d_model, 1)
        self.position = nn.Linear(2, c.d_model)
        self.condition = nn.Linear(2, c.d_model)  # goal XY; geometry is point-only
        self.norm = nn.LayerNorm(c.d_model)
        self.decoder = CoFLSFieldDecoder(d_model=c.d_model, num_heads=8, num_layers=2, dropout=0)
        self.register_buffer('rgb_mean', torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('rgb_std', torch.tensor([.229,.224,.225])[None,:,None,None])

    def encode(self, rgb, depth, goal_ego, valid_mask=None):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or depth.shape != (rgb.shape[0],1,*rgb.shape[-2:]):
            raise ValueError('Aligned RGB [B,3,H,W] and metric depth [B,1,H,W] required')
        if not rgb.is_floating_point() or not torch.isfinite(rgb).all() or (rgb<0).any() or (rgb>1).any():
            raise ValueError('RGB must be float in [0,1]')
        if goal_ego.shape != (rgb.shape[0],2) or not torch.isfinite(goal_ego).all():
            raise ValueError('goal_ego must be finite [B,2]')
        good=torch.isfinite(depth)&(depth>0)&(depth<self.config.max_depth_m)
        if valid_mask is not None:
            if valid_mask.shape != depth.shape or valid_mask.dtype != torch.bool:
                raise ValueError('valid_mask must be bool and match depth')
            good &= valid_mask
        size=(self.config.image_size,)*2
        rgb=F.interpolate(rgb,size,mode='bilinear',align_corners=False)
        depth=F.interpolate(torch.where(good,depth,0)/self.config.max_depth_m,size,mode='nearest')
        maps,_=self.backbone((rgb-self.rgb_mean)/self.rgb_std,depth)
        f=self.projection(maps[2]);h,w=f.shape[-2:]
        y,x=torch.meshgrid(torch.linspace(-1,1,h,device=f.device),torch.linspace(-1,1,w,device=f.device),indexing='ij')
        tokens=f.flatten(2).transpose(1,2)+self.position(torch.stack((x,y),-1).reshape(1,-1,2))
        goal=self.condition(goal_ego/5)[:,None]
        return self.norm(torch.cat((tokens,goal),1))

    def forward(self, query, rgb, depth, goal_ego, valid_mask=None, hfov_rad=None):
        context=self.encode(rgb,depth,goal_ego,valid_mask)
        return self.decoder(query,context,hfov_rad=hfov_rad)

    def save(self,path):
        torch.save(dict(format='cofm_point_rgbd_v2',config=asdict(self.config),state_dict=self.state_dict()),path)

    @classmethod
    def load(cls,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('format')!='cofm_point_rgbd_v2':
            raise ValueError('Expected point-only RGB-D v2 checkpoint; old q checkpoints require retraining')
        model=cls(CofMConfig(**data['config']));model.load_state_dict(data['state_dict'],strict=True)
        return model
