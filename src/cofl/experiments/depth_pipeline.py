"""Generate Habitat depth and smoke-test a depth-only DFormer -> CoFL-S policy.

Example: python -m cofl.experiments.depth_pipeline --output artifacts/cofl-m/dformer-smoke
         --habitat-python .habitat/coflm/bin/python --backward-check --save-checkpoint
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from cofl.generation import obstacle_scene
from cofl.models.dformer import DFormerDepthPolicy, DepthPolicyConfig


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args):
    if args.frames < 1 or args.threads < 1 or args.query_r < 1 or args.query_theta < 1:
        raise ValueError("Frame, thread and query counts must be positive")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.scene is None:
        scene = output / "scene"
        metadata = obstacle_scene.generate_scene(scene, obstacle_scene.SceneConfig(
            seed=args.seed, frames=args.frames,
        ))
        if args.habitat_python:
            command = [str(args.habitat_python.resolve()), str(Path(obstacle_scene.__file__)),
                       "--output", str(scene), "--render-existing",
                       "--gpu-device-id", str(args.gpu_device_id)]
            result = subprocess.run(command, capture_output=True, text=True)
            (output / "habitat.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"Habitat rendering failed ({result.returncode}); see {output / 'habitat.log'}")
        else:
            obstacle_scene.render_depth(scene, metadata, gpu_device_id=args.gpu_device_id)
    else:
        scene = args.scene.resolve()
    metadata = json.loads((scene / "scene.json").read_text(encoding="utf-8"))
    if (metadata.get("format") != "cofl_m_depth_scene" or metadata.get("version") != 1
            or not metadata.get("depth_rendered") or not metadata.get("frames")):
        raise ValueError("Expected a rendered obstacle_scene v1 manifest with frames")
    scene_config = obstacle_scene.SceneConfig(**metadata["config"])
    scene_config.validate()
    if any(frame["rotation_xyzw_habitat"] != [0, 0, 0, 1] for frame in metadata["frames"]):
        raise ValueError("This experiment requires the generator's identity camera orientation")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu or enable GPU access")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = DepthPolicyConfig(
        image_size=args.image_size, hfov_rad=math.radians(scene_config.hfov_deg),
        max_depth_m=scene_config.far,
    )
    model = (DFormerDepthPolicy.load(args.checkpoint, map_location=device)
             if args.checkpoint else DFormerDepthPolicy(config).to(device))
    config = model.config
    if not math.isclose(config.max_depth_m, scene_config.far):
        raise ValueError("Checkpoint max_depth_m differs from scene far; choose matching preprocessing")
    model.eval()
    query = model.decoder.build_sector_query_lattice(
        args.query_r, args.query_theta, device=device,
    ).unsqueeze(0)
    hfov = torch.tensor(math.radians(scene_config.hfov_deg), device=device)
    query_xy = model.decoder.polar_to_cartesian(
        model.decoder.denormalize_to_polar(query, hfov_rad=float(hfov))
    )
    # A fixed world point; with the optional gap wall it lies 1m beyond the wall.
    goal_world = np.asarray(args.goal_world or [scene_config.gap_x + 1, 0], dtype=np.float32)
    if not np.isfinite(goal_world).all():
        raise ValueError("World goal must be finite")
    predictions = output / "predictions"
    predictions.mkdir()
    frames = []
    last_inputs = None
    with torch.inference_mode():
        for frame in metadata["frames"][:args.frames]:
            depth_np = np.load(scene / frame["depth"], allow_pickle=False)
            valid_np = np.load(scene / frame["valid"], allow_pickle=False)
            if (depth_np.shape != (scene_config.image_height, scene_config.image_width)
                    or valid_np.shape != depth_np.shape or valid_np.dtype != np.bool_):
                raise ValueError("Invalid depth/valid array shape or mask dtype")
            if not np.isfinite(depth_np[valid_np]).all() or not valid_np.any():
                raise ValueError("Frame must contain finite valid depth samples")
            depth = torch.from_numpy(depth_np.astype(np.float32)).to(device)[None, None]
            valid = torch.from_numpy(valid_np).to(device)[None, None]
            goal = torch.from_numpy(goal_world - np.asarray(
                frame["position_planning"][:2], dtype=np.float32,
            )).to(device)[None]
            if last_inputs is None:
                # Exclude one warm-up pass from the per-frame timings.
                model(query, depth, goal, valid, hfov_rad=hfov)
            synchronize(device)
            start = time.perf_counter()
            context, features = model.encoder(depth, goal, valid, return_features=True)
            synchronize(device)
            encoded = time.perf_counter()
            field = model.query(query, model.prepare_query_context(context), hfov_rad=hfov)
            synchronize(device)
            decoded = time.perf_counter()
            if not torch.isfinite(context).all() or not torch.isfinite(field).all():
                raise RuntimeError("Nonfinite model output")
            path = predictions / f"{frame['index']:06d}.npz"
            np.savez_compressed(
                path, field=field[0].cpu().numpy(), context=context[0].cpu().numpy(),
                query=query[0].cpu().numpy(), query_ego_m=query_xy[0].cpu().numpy(),
                goal_ego_m=goal[0].cpu().numpy(),
            )
            frames.append(dict(
                index=frame["index"], prediction=str(path.relative_to(output)),
                valid_fraction=float(valid_np.mean()), goal_ego_m=goal[0].tolist(),
                feature_shapes=[list(feature.shape) for feature in features],
                context_shape=list(context.shape), field_shape=list(field.shape),
                field_abs_max=float(field.abs().max()),
                encode_ms=(encoded - start) * 1000, decode_ms=(decoded - encoded) * 1000,
            ))
            last_inputs = depth, valid, goal
    # Recreate ordinary tensors: inference tensors cannot be saved for backward.
    depth, valid, goal = [value.clone() for value in last_inputs]
    query, hfov = query.clone(), hfov.clone()
    checks = {}
    with torch.no_grad():
        baseline = model(query, depth, goal, valid, hfov_rad=hfov)
        cached = model.query(query, model.prepare_query_context(model.encode(depth, goal, valid)),
                             hfov_rad=hfov)
        changed_goal = model(query, depth, goal + goal.new_tensor([[0, 1]]), valid, hfov_rad=hfov)
        changed_depth = model(query, depth * 0.5, goal, valid, hfov_rad=hfov)
        checks.update(
            cached_query_max_error=float((baseline - cached).abs().max()),
            goal_change_max_delta=float((baseline - changed_goal).abs().max()),
            depth_change_max_delta=float((baseline - changed_depth).abs().max()),
        )
    if checks["cached_query_max_error"] > 1e-6:
        raise RuntimeError("Cached context decoding disagrees with forward")
    if min(checks["goal_change_max_delta"], checks["depth_change_max_delta"]) <= 0:
        raise RuntimeError("Model output is insensitive to depth or goal")
    if args.backward_check:
        model.zero_grad(set_to_none=True)
        field = model(query, depth, goal, valid, hfov_rad=hfov)
        # Numerical gradient check only: this scalar is not a navigation loss.
        field.square().mean().backward()
        names = ["encoder.backbone.downsample_layers.0.0.weight",
                 "encoder.backbone.downsample_layers_e.0.0.weight",
                 "encoder.projection.weight", "encoder.goal_projection.0.weight",
                 "decoder.dir_head.0.weight"]
        parameters = dict(model.named_parameters())
        gradient_norms = {}
        for name in names:
            grad = parameters[name].grad
            if grad is None or not torch.isfinite(grad).all() or float(grad.norm()) == 0:
                raise RuntimeError(f"Missing, zero or nonfinite gradient: {name}")
            gradient_norms[name] = float(grad.norm())
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite parameter gradient")
        checks["backward_gradient_norms"] = gradient_norms
    if args.save_checkpoint:
        path = output / "policy.pt"
        model.save(path, metadata=dict(
            trained=False if args.checkpoint is None else "unknown",
            purpose="integration smoke test", source_checkpoint=str(args.checkpoint),
        ))
        restored = DFormerDepthPolicy.load(path, map_location=device).eval()
        with torch.no_grad():
            reloaded = restored(query, depth, goal, valid, hfov_rad=hfov)
        error = float((baseline - reloaded).abs().max())
        checks["checkpoint_reload_max_error"] = error
        if error > 1e-6:
            raise RuntimeError("Checkpoint reload changed model predictions")
    report = dict(
        format="cofl_depth_pipeline_smoke", version=1, status="passed",
        purpose="Wiring and numerical checks; no navigation-quality evaluation or training",
        initialization="random" if args.checkpoint is None else "explicit_policy_checkpoint",
        checkpoint=None if args.checkpoint is None else str(args.checkpoint.resolve()),
        seed=args.seed, device=str(device), torch_version=str(torch.__version__),
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        habitat_sim_version=metadata.get("habitat_sim_version"), scene=str(scene),
        visual_input="single-channel metric depth and validity only; no RGB",
        goal_world_planning_m=goal_world.tolist(), model_config=asdict(config),
        parameters=sum(p.numel() for p in model.parameters()),
        upstream_commit="e3273009b759b578945483828ff315d560be94c9",
        checks=checks, frames=frames,
    )
    obstacle_scene.save_json(output / "report.json", report)
    print(f"Passed: {len(frames)} depth frames -> DFormer -> {frames[0]['context_shape']} context"
          f" -> {frames[0]['field_shape']} field; {output / 'report.json'}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, help="Reuse a rendered obstacle_scene directory")
    parser.add_argument("--habitat-python", type=Path, help="Separate Habitat interpreter")
    parser.add_argument("--checkpoint", type=Path, help="Explicit depth-policy v1 checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--query-r", type=int, default=16)
    parser.add_argument("--query-theta", type=int, default=32)
    parser.add_argument("--goal-world", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--backward-check", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
