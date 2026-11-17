"""Build a small scene-disjoint cof-m Habitat observation set and Studio catalog.

This is scene/observation preparation, not a supervised CoFLDataset release.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess

import numpy as np

from cofl.generation import obstacle_scene


def build(output: Path, habitat_python: Path, *, frames=16):
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    if not habitat_python.is_file():
        raise FileNotFoundError(habitat_python)
    if frames < 1:
        raise ValueError("frames must be positive")
    output.mkdir(parents=True, exist_ok=True)
    recipes = [
        ("train", 100, "poles", 45, 0),
        ("train", 101, "mixed", 25, 8),
        ("train", 102, "walls", 0, 20),
        ("train", 103, "poles", 45, 0),
        ("val", 200, "poles", 45, 0),
        ("val", 201, "mixed", 25, 8),
    ]
    scenes, catalog = [], dict(device="cpu", scenes={})
    # Persist configuration before rendering, so incomplete output is inspectable.
    for split, seed, kind, poles, walls in recipes:
        scene_id = f"{split}-{kind}-{seed}"
        relative = Path(split) / scene_id
        config = obstacle_scene.SceneConfig(seed=seed, poles=poles, walls=walls, frames=frames)
        obstacle_scene.generate_scene(output / relative, config)
        entry = dict(id=scene_id, split=split, seed=seed, kind=kind,
                     path=relative.as_posix(), config=asdict(config), rendered=False)
        scenes.append(entry)
        catalog["scenes"][scene_id] = dict(
            path=(relative / "obstacles.glb").as_posix(),
            label=f"cof-m {split} / {kind} / seed {seed}", domain_id="coflm", up_axis="y",
            floors=[dict(id="fixed-xy", label="XOY z=1.2m (obstacle cutaway)",
                         elevation=0, min_height=-0.2, max_height=1.5)],
        )
    manifest = dict(
        format="cofl_m_habitat_observation_set", version=1, status="rendering",
        supervised_training_ready=False, labels_included=False,
        split_unit="scene", planning_plane="XOY", planning_height_m=1.2,
        min_surface_clearance_m=0.65, sensors=["depth"], scenes=scenes,
    )
    obstacle_scene.save_json(output / "manifest.json", manifest)
    import yaml
    (output / "studio.yaml").write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    for entry in scenes:
        scene = output / entry["path"]
        result = subprocess.run([
            str(habitat_python.resolve()), str(Path(obstacle_scene.__file__).resolve()),
            "--output", str(scene), "--render-existing",
        ], capture_output=True, text=True)
        (scene / "habitat.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Habitat failed: {scene / 'habitat.log'}")
        import json
        metadata = json.loads((scene / "scene.json").read_text())
        for frame in metadata["frames"]:
            depth = np.load(scene / frame["depth"], allow_pickle=False)
            valid = np.load(scene / frame["valid"], allow_pickle=False)
            if depth.shape != (240, 320) or valid.shape != depth.shape or valid.dtype != np.bool_:
                raise RuntimeError("Invalid rendered depth/mask shape or dtype")
            if not valid.any() or not np.isfinite(depth).all() or np.any(depth[~valid] != 0):
                raise RuntimeError("Invalid rendered metric depth")
        entry.update(rendered=True, frames=len(metadata["frames"]),
                     geometry_audit=metadata["geometry_audit"],
                     valid_fraction_range=[min(f["valid_fraction"] for f in metadata["frames"]),
                                           max(f["valid_fraction"] for f in metadata["frames"])])
        obstacle_scene.save_json(output / "manifest.json", manifest)
        print(f"Rendered {entry['id']}: {entry['frames']} depth frames", flush=True)
    manifest.update(status="complete", total_frames=sum(s["frames"] for s in scenes))
    obstacle_scene.save_json(output / "manifest.json", manifest)
    (output / "README.md").write_text(
        "# cof-m Habitat scene/observation set\n\n"
        "4 train scenes, 2 val scenes; split by scene, no shared seeds. "
        "Full-height vertical obstacles; fixed planning height z=1.2m.\n\n"
        "Contains GLB, metric depth, valid masks and poses. "
        "No flow-field supervision or trained model is included.\n\n"
        "Open the existing CoFL Studio with:\n\n```bash\n"
        f".venv/bin/python -m cofl app --config {output / 'studio.yaml'} "
        "--no-browser --port 8788\n```\n\n"
        "Visit http://127.0.0.1:8788, choose cof-m → cof-m Playground, "
        "then choose a scene. The new mode validates docking points and generates a teacher; "
        "teacher artifacts are separate from this observation set.\n",
        encoding="utf-8",
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--habitat-python", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args(argv)
    result = build(args.output, args.habitat_python, frames=args.frames)
    print(f"Complete: {len(result['scenes'])} scenes / {result['total_frames']} depth frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
