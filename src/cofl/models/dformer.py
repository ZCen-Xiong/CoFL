"""Depth-only DFormer context and goal-conditioned CoFL-S field policy.

This experimental policy has no RGB, language, SigLIP or action-head dependency.
Weights are randomly initialized unless an explicit policy checkpoint is loaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .cofl_s import CoFLSFieldDecoder
from .vendor.dformer import DFormer_Tiny


@dataclass(frozen=True)
class DepthPolicyConfig:
    image_size: int = 224
    max_depth_m: float = 15.0
    d_model: int = 256
    num_heads: int = 8
    decoder_layers: int = 2
    hidden_dim: int = 512
    bev_x_max: float = 5.0
    hfov_rad: float = math.pi / 2
    v_norm: float = 5.0

    def validate(self):
        for name in ("image_size", "d_model", "num_heads", "decoder_layers", "hidden_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size < 32 or self.image_size % 32:
            raise ValueError("image_size must be a multiple of 32 and at least 32")
        if self.d_model % 4 or self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by 4 and num_heads")
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.max_depth_m, self.bev_x_max, self.v_norm)):
            raise ValueError("Depth, query and field scales must be finite and positive")
        if not math.isfinite(self.hfov_rad) or not 0 < self.hfov_rad <= math.pi:
            raise ValueError("hfov_rad must lie in (0, pi]")


def preprocess_depth(depth, valid_mask, *, size, max_depth_m):
    """Metric [B,1,H,W] -> masked, nearest-resized depth in [0,1].

    Invalid samples remain zero; nearest interpolation avoids synthesizing
    surfaces at obstacle edges. The original metric depth stays outside the model.
    """
    if depth.ndim != 4 or depth.shape[1] != 1 or not depth.is_floating_point():
        raise ValueError("depth must be a floating tensor [B,1,H,W] in metres")
    valid = torch.isfinite(depth) & (depth > 0) & (depth < max_depth_m)
    if valid_mask is not None:
        if valid_mask.shape != depth.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with the same shape as depth")
        valid = valid & valid_mask
    depth = torch.where(valid, depth, 0.0) / max_depth_m
    return F.interpolate(depth, (size, size), mode="nearest")


def spatial_position_encoding(height, width, channels, *, device, dtype):
    """Fixed 2-D sin/cos encoding, aligned to row-major feature flattening."""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device, dtype=dtype),
        torch.linspace(-1, 1, width, device=device, dtype=dtype), indexing="ij",
    )
    frequency = torch.exp(
        -math.log(10000) * torch.arange(channels // 4, device=device, dtype=dtype)
        / max(channels // 4 - 1, 1)
    )
    x, y = x.reshape(-1, 1) * frequency * math.pi, y.reshape(-1, 1) * frequency * math.pi
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1).unsqueeze(0)


class DFormerDepthEncoder(nn.Module):
    def __init__(self, config: DepthPolicyConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = DFormer_Tiny(drop_path_rate=0.0)
        self.projection = nn.Conv2d(128, config.d_model, kernel_size=1)
        self.goal_projection = nn.Sequential(
            nn.Linear(2, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model)
        )
        self.spatial_type = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.goal_type = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, depth, goal_ego, valid_mask=None, *, return_features=False):
        if goal_ego.shape != (depth.shape[0], 2) or not torch.isfinite(goal_ego).all():
            raise ValueError("goal_ego must be finite [B,2] (forward,left) metres")
        depth = preprocess_depth(
            depth, valid_mask, size=self.config.image_size,
            max_depth_m=self.config.max_depth_m,
        )
        # Both streams observe exactly the same depth; neither receives RGB.
        features, _ = self.backbone(depth, depth)
        feature = self.projection(features[2])  # stride 16, 128 channels
        height, width = feature.shape[-2:]
        spatial = feature.flatten(2).transpose(1, 2)
        spatial = spatial + spatial_position_encoding(
            height, width, self.config.d_model, device=feature.device, dtype=feature.dtype,
        ) + self.spatial_type
        goal = self.goal_projection(goal_ego / self.config.bev_x_max).unsqueeze(1)
        context = self.norm(torch.cat((spatial, goal + self.goal_type), dim=1))
        return (context, features) if return_features else context


class DFormerDepthPolicy(nn.Module):
    def __init__(self, config: DepthPolicyConfig | None = None):
        super().__init__()
        self.config = config or DepthPolicyConfig()
        self.config.validate()
        self.encoder = DFormerDepthEncoder(self.config)
        self.decoder = CoFLSFieldDecoder(
            d_model=self.config.d_model, hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads, num_layers=self.config.decoder_layers,
            dropout=0.0, bev_x_max=self.config.bev_x_max,
            hfov_rad=self.config.hfov_rad, v_norm=self.config.v_norm,
        )

    def encode(self, depth, goal_ego, valid_mask=None):
        return self.encoder(depth, goal_ego, valid_mask)

    def prepare_query_context(self, context):
        return self.decoder.prepare_context(context)

    def query(self, query, query_context, *, hfov_rad=None):
        return self.decoder.decode(query, query_context, hfov_rad=hfov_rad)

    def forward(self, query, depth, goal_ego, valid_mask=None, *, hfov_rad=None):
        context = self.encode(depth, goal_ego, valid_mask)
        return self.query(query, self.prepare_query_context(context), hfov_rad=hfov_rad)

    def save(self, path: Path, *, metadata=None):
        torch.save(dict(format="cofl_depth_dformer", version=1, config=asdict(self.config),
                        state_dict=self.state_dict(), metadata=metadata or {}), path)

    @classmethod
    def load(cls, path: Path, *, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if checkpoint.get("format") != "cofl_depth_dformer" or checkpoint.get("version") != 1:
            raise ValueError("Expected a cofl_depth_dformer v1 policy checkpoint")
        model = cls(DepthPolicyConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model.to(map_location)
