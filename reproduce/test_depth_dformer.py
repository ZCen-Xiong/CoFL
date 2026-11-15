"""Run with: .venv/bin/python -m pytest reproduce/test_depth_dformer.py -q"""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from cofl.experiments.depth_pipeline import run
from cofl.models.dformer import DFormerDepthPolicy, DepthPolicyConfig, preprocess_depth


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    return DFormerDepthPolicy(DepthPolicyConfig(image_size=64, d_model=64)).eval()


def inputs():
    depth = torch.linspace(0.1, 12, 64 * 64).reshape(1, 1, 64, 64)
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[..., :8, :8] = False
    return depth, valid, torch.tensor([[8.0, 0.0]])


def test_invalid_measurements_do_not_change_predictions(model):
    depth, valid, goal = inputs()
    corrupted = depth.clone()
    corrupted[~valid] = float("nan")
    query = model.decoder.build_sector_query_lattice(3, 5)[None]
    with torch.no_grad():
        original = model(query, depth, goal, valid)
        altered = model(query, corrupted, goal, valid)
    torch.testing.assert_close(original, altered, rtol=0, atol=0)


def test_missing_mask_still_removes_invalid_and_far_depth():
    depth = torch.tensor([float("nan"), float("inf"), -1, 0, 15, 30, 7.5, 3.0])
    actual = preprocess_depth(depth.reshape(1, 1, 2, 4), None, size=4, max_depth_m=15)
    assert torch.isfinite(actual).all()
    assert actual[..., :2, :].count_nonzero() == 0
    torch.testing.assert_close(actual[0, 0, 2], torch.tensor([0, 0, 0.5, 0.2]))


def test_goal_and_depth_conditioning_cached_query_and_backward(model):
    depth, valid, goal = inputs()
    query = model.decoder.build_sector_query_lattice(3, 5)[None]
    context = model.encode(depth, goal, valid)
    field = model.query(query, model.prepare_query_context(context))
    assert context.shape == (1, 17, 64)
    assert field.shape == (1, 15, 2)
    with torch.no_grad():
        torch.testing.assert_close(field, model(query, depth, goal, valid))
        assert not torch.equal(field, model(query, depth / 2, goal, valid))
        assert not torch.equal(field, model(query, depth, goal + 1, valid))
    model.zero_grad(set_to_none=True)
    field.square().mean().backward()
    for parameter in (model.encoder.backbone.downsample_layers[0][0].weight,
                      model.encoder.backbone.downsample_layers_e[0][0].weight,
                      model.encoder.goal_projection[0].weight,
                      model.decoder.mag_head[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_checkpoint_round_trip_and_strict_architecture(model, tmp_path):
    path = tmp_path / "policy.pt"
    model.save(path)
    restored = DFormerDepthPolicy.load(path).eval()
    depth, valid, goal = inputs()
    query = model.decoder.build_sector_query_lattice(3, 5)[None]
    with torch.no_grad():
        torch.testing.assert_close(model(query, depth, goal, valid),
                                   restored(query, depth, goal, valid), rtol=0, atol=0)
    checkpoint = torch.load(path, weights_only=True)
    del checkpoint["state_dict"]["encoder.projection.weight"]
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        DFormerDepthPolicy.load(path)


@pytest.mark.parametrize("change", [dict(image_size=63), dict(d_model=62),
                                   dict(max_depth_m=float("nan")), dict(decoder_layers=0)])
def test_configuration_rejects_invalid_geometry(change):
    with pytest.raises(ValueError):
        replace(DepthPolicyConfig(), **change).validate()


def test_pipeline_rejects_geometry_without_rendered_depth(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "scene.json").write_text(json.dumps(dict(
        format="cofl_m_depth_scene", version=1, depth_rendered=False,
    )))
    args = SimpleNamespace(frames=1, threads=2, query_r=3, query_theta=5,
                           output=tmp_path / "output", scene=scene)
    with pytest.raises(ValueError, match="rendered"):
        run(args)
