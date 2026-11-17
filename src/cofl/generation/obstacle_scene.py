"""Full-height vertical obstacles for fixed-height XOY planning and Habitat depth.

Run as a module in the project environment, or execute this file directly in a
separate Habitat environment. Only NumPy is needed for --generate-only.
No RGB sensor, scene download, Bullet dynamics, navmesh or learned model is used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import struct
import sys
import zlib

# Direct execution otherwise lets sibling random.py shadow Python's random,
# which NumPy imports via secrets. Module execution does not need this fix.
if not __package__:
    sys.path = [entry for entry in sys.path
                if Path(entry or ".").resolve() != Path(__file__).resolve().parent]

import numpy as np


@dataclass(frozen=True)
class SceneConfig:
    seed: int = 42
    length: float = 16.0
    width: float = 10.0
    height: float = 3.0
    poles: int = 45
    walls: int = 0
    rings: int = 0  # Legacy option: only zero is accepted.
    min_clearance: float = 0.65
    gap_wall: int = 0
    gap_x: float = 7.0
    gap_width: float = 0.70
    wall_thickness: float = 0.25
    camera_height: float = 1.2
    camera_end: float = 5.5
    camera_clearance: float = 0.5
    frames: int = 16
    image_width: int = 320
    image_height: int = 240
    hfov_deg: float = 90.0
    near: float = 0.05
    far: float = 15.0

    def validate(self):
        values = asdict(self)
        if not all(math.isfinite(float(v)) for v in values.values()):
            raise ValueError("All configuration values must be finite")
        for name in ("seed", "poles", "walls", "rings", "gap_wall", "frames", "image_width", "image_height"):
            if type(values[name]) is not int:
                raise ValueError(name + " must be an integer")
        if self.seed < 0 or min(self.poles, self.walls) < 0:
            raise ValueError("seed and obstacle counts must be nonnegative")
        if self.rings != 0:
            raise ValueError("Rings are disabled: regenerate a vertical-extrusion scene with --rings 0")
        if self.gap_wall not in (0, 1):
            raise ValueError("gap_wall must be 0 or 1")
        if self.min_clearance < 0.65:
            raise ValueError("min_clearance must be at least 0.65 metres")
        if self.gap_wall and self.gap_width <= self.min_clearance + 1e-6:
            raise ValueError("gap_width must exceed min_clearance; default is 0.70m")
        if min(self.frames, self.image_width, self.image_height) < 1:
            raise ValueError("Frame count and image dimensions must be positive")
        if min(self.length, self.width, self.height, self.wall_thickness,
               self.camera_clearance) <= 0:
            raise ValueError("Scene dimensions and clearance must be positive")
        if not 0 < self.gap_width < self.width:
            raise ValueError("gap_width must lie between zero and scene width")
        if not 0 < self.camera_height < self.height:
            raise ValueError("camera_height must lie between ground and wall top")
        if not 1 < self.gap_x < self.length - 1:
            raise ValueError("gap_x must lie inside the scene, with 1m margins")
        path_limit = self.gap_x - self.wall_thickness / 2 - 0.1 if self.gap_wall else self.length
        if not 0 <= self.camera_end < path_limit:
            raise ValueError("Camera path must stop at least 0.1m before the wall")
        if not 0 < self.near < self.far or not 1 < self.hfov_deg < 179:
            raise ValueError("Require 0 < near < far and 1 < hfov_deg < 179")


# Planning coordinates: x forward, y left, z up (metres).
# Habitat coordinates: X right, Y up, forward is -Z.
PLANNING_TO_HABITAT = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)


def footprint_gap(a, b):
    """Surface distance in XY for circles and axis-aligned rectangles (metres).

    Cylinders use enclosing analytic circles, conservative for the GLB polygon.
    Intersecting footprints return zero or negative distance.
    """
    delta = np.abs(np.asarray(a["center"][:2]) - np.asarray(b["center"][:2]))
    if a["kind"] == b["kind"] == "cylinder":
        return float(np.linalg.norm(delta) - a["radius"] - b["radius"])
    if a["kind"] == b["kind"] == "box":
        half = (np.asarray(a["size"][:2]) + np.asarray(b["size"][:2])) / 2
        return float(np.linalg.norm(np.maximum(delta - half, 0)))
    circle, box = (a, b) if a["kind"] == "cylinder" else (b, a)
    return float(np.linalg.norm(np.maximum(delta - np.asarray(box["size"][:2]) / 2, 0))
                 - circle["radius"])


def audit_obstacles(cfg, obstacles):
    """Independently validate pair gaps and constant horizontal cross-sections."""
    solids = [o for o in obstacles if o["id"] != "floor"]
    minimum = None
    for index, a in enumerate(solids):
        if a["kind"] not in ("box", "cylinder"):
            raise ValueError("Only vertical cylinders and walls are supported")
        height = a["height"] if a["kind"] == "cylinder" else a["size"][2]
        if not math.isclose(height, cfg.height) or not math.isclose(a["center"][2], height / 2):
            raise ValueError("Every obstacle must span z=0 to scene height")
        for b in solids[:index]:
            gap = footprint_gap(a, b)
            minimum = gap if minimum is None else min(minimum, gap)
            if gap <= cfg.min_clearance + 1e-6:
                raise ValueError(f"Insufficient surface gap: {a['id']} / {b['id']}: {gap}")
    return dict(min_surface_gap_m=minimum, checked_pairs=len(solids)*(len(solids)-1)//2,
                constant_cross_section_z_open_interval=[0, cfg.height],
                floor_excluded=True, clearance_geometry="analytic circles and axis-aligned rectangles")


def build_obstacles(cfg: SceneConfig) -> list[dict]:
    """Deterministic non-overlapping full-height obstacles for planar planning."""
    cfg.validate()
    obstacles = [dict(id="floor", kind="box", center=[cfg.length / 2 - 1, 0, -0.1],
                      size=[cfg.length + 2, cfg.width, 0.2])]
    if cfg.gap_wall:
        half_panel = (cfg.width - cfg.gap_width) / 2
        for sign, name in ((1, "gap_left"), (-1, "gap_right")):
            obstacles.append(dict(id=name, kind="box", center=[
                cfg.gap_x, sign * (cfg.gap_width / 2 + half_panel / 2), cfg.height / 2,
            ], size=[cfg.wall_thickness, half_panel, cfg.height]))
    rng = np.random.default_rng(cfg.seed)
    for kind, count in (("cylinder", cfg.poles), ("box", cfg.walls)):
        placed = 0
        for _ in range(max(100, count * 2000)):
            if placed == count:
                break
            if kind == "cylinder":
                radius = float(rng.uniform(0.12, 0.32))
                hx = hy = radius
            else:
                long_side = float(rng.uniform(0.6, 1.4))
                sx, sy = ((long_side, cfg.wall_thickness) if rng.integers(2)
                          else (cfg.wall_thickness, long_side))
                hx, hy = sx / 2, sy / 2
            # Keep at least the requested gap to the sampling-domain boundary.
            margin = cfg.min_clearance + 2e-6
            if cfg.length <= 2*(hx + margin) or cfg.width <= 2*(hy + margin):
                continue
            x = float(rng.uniform(hx + margin, cfg.length - hx - margin))
            y = float(rng.uniform(-cfg.width / 2 + hy + margin, cfg.width / 2 - hy - margin))
            if x <= cfg.camera_end + hx and abs(y) < hy + cfg.camera_clearance:
                continue
            if kind == "cylinder":
                candidate = dict(id=f"pole_{placed:03d}", kind=kind,
                                 center=[x, y, cfg.height / 2], radius=radius, height=cfg.height)
            else:
                candidate = dict(id=f"wall_{placed:03d}", kind=kind,
                                 center=[x, y, cfg.height / 2], size=[sx, sy, cfg.height])
            if any(footprint_gap(candidate, other) <= margin for other in obstacles[1:]):
                continue
            obstacles.append(candidate)
            placed += 1
        if placed != count:
            raise ValueError(f"Cannot place {count} {kind} obstacles with surface gap > "
                             f"{cfg.min_clearance}m; enlarge scene or reduce counts")
    audit_obstacles(cfg, obstacles)
    return obstacles


def obstacle_mesh(obstacle):
    """Return vertices and outward-facing triangle indices in planning coordinates."""
    kind = obstacle["kind"]
    if kind == "box":
        vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                             [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
        vertices *= np.asarray(obstacle["size"]) / 2
        faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                 (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                 (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    elif kind == "cylinder":
        n = 24
        vertices = [[obstacle["radius"] * math.cos(2 * math.pi * i / n),
                     obstacle["radius"] * math.sin(2 * math.pi * i / n),
                     z * obstacle["height"] / 2] for z in (-1, 1) for i in range(n)]
        vertices.extend([[0, 0, -obstacle["height"] / 2], [0, 0, obstacle["height"] / 2]])
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.extend([(i, j, j + n), (i, j + n, i + n),
                          (2 * n, j, i), (2 * n + 1, i + n, j + n)])
        vertices = np.asarray(vertices)
    else:
        raise ValueError("Unsupported obstacle kind: " + str(kind))
    return np.asarray(vertices) + obstacle["center"], np.asarray(faces, dtype=np.int64)


def write_glb(path: Path, obstacles):
    """Write a portable self-contained glTF 2 binary, without mesh dependencies."""
    document = dict(asset={"version": "2.0", "generator": "cof-m obstacle scene"},
                    scene=0, scenes=[{"nodes": []}], nodes=[], meshes=[],
                    materials=[{"doubleSided": True, "pbrMetallicRoughness": {
                        "baseColorFactor": [0.6, 0.6, 0.6, 1],
                        "metallicFactor": 0, "roughnessFactor": 1}}],
                    buffers=[], bufferViews=[], accessors=[])
    binary = bytearray()

    def accessor(array, data_type, component_type, target):
        binary.extend(b"\0" * (-len(binary) % 4))
        view_index = len(document["bufferViews"])
        payload = array.tobytes()
        document["bufferViews"].append(dict(buffer=0, byteOffset=len(binary),
                                           byteLength=len(payload), target=target))
        binary.extend(payload)
        entry = dict(bufferView=view_index, componentType=component_type,
                     count=len(array), type=data_type)
        if data_type == "VEC3":
            entry.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        document["accessors"].append(entry)
        return len(document["accessors"]) - 1

    for obstacle in obstacles:
        vertices, faces = obstacle_mesh(obstacle)
        vertices = vertices @ PLANNING_TO_HABITAT.T
        triangles = vertices[faces]
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        if not np.isfinite(triangles).all() or np.any(norm < 1e-12):
            raise ValueError("Invalid mesh geometry for " + obstacle["id"])
        position = accessor(triangles.reshape(-1, 3).astype("<f4"), "VEC3", 5126, 34962)
        normals = accessor(np.repeat(normal / norm, 3, axis=0).astype("<f4"), "VEC3", 5126, 34962)
        indices = accessor(np.arange(len(triangles) * 3, dtype="<u4"), "SCALAR", 5125, 34963)
        document["meshes"].append({"primitives": [{"attributes": {
            "POSITION": position, "NORMAL": normals}, "indices": indices, "material": 0}]})
        document["nodes"].append({"name": obstacle["id"], "mesh": len(document["meshes"]) - 1})
        document["scenes"][0]["nodes"].append(len(document["nodes"]) - 1)
    document["buffers"] = [{"byteLength": len(binary)}]
    header = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    header += b" " * (-len(header) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    total = 12 + 8 + len(header) + 8 + len(binary)
    with path.open("wb") as stream:
        stream.write(struct.pack("<III", 0x46546C67, 2, total))
        stream.write(struct.pack("<II", len(header), 0x4E4F534A))
        stream.write(header)
        stream.write(struct.pack("<II", len(binary), 0x004E4942))
        stream.write(binary)


def camera_frames(cfg: SceneConfig):
    """An observation sampling line ending before the gap, not a flight plan."""
    for index, x in enumerate(np.linspace(0, cfg.camera_end, cfg.frames)):
        planning = np.array([x, 0, cfg.camera_height])
        yield dict(index=index, position_planning=planning.tolist(),
                   position_habitat=(PLANNING_TO_HABITAT @ planning).tolist(),
                   rotation_xyzw_habitat=[0, 0, 0, 1])


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_depth_preview(path, depth, far):
    """16-bit grayscale PNG using only the standard library; zero stays black."""
    pixels = (np.clip(depth / far, 0, 1) * 65535).astype(">u2")
    height, width = pixels.shape

    def chunk(name, payload):
        return (struct.pack(">I", len(payload)) + name + payload
                + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF))

    scanlines = b"".join(b"\0" + row.tobytes() for row in pixels)
    with path.open("wb") as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")
        stream.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)))
        stream.write(chunk(b"IDAT", zlib.compress(scanlines)))
        stream.write(chunk(b"IEND", b""))


def generate_scene(output: Path, cfg: SceneConfig):
    cfg.validate()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    obstacles = build_obstacles(cfg)
    output.mkdir(parents=True, exist_ok=True)
    write_glb(output / "obstacles.glb", obstacles)
    save_json(output / "obstacles.stage_config.json", dict(
        render_asset="obstacles.glb", collision_asset="obstacles.glb",
        up=[0, 1, 0], front=[0, 0, -1], units_to_meters=1.0))
    focal = cfg.image_width / (2 * math.tan(math.radians(cfg.hfov_deg) / 2))
    metadata = dict(
        format="cofl_m_depth_scene", version=1, seed=cfg.seed, config=asdict(cfg),
        coordinates=dict(planning="x forward, y left, z up; metres",
                         habitat="X right, Y up, camera forward -Z; metres",
                         planning_to_habitat=PLANNING_TO_HABITAT.tolist()),
        camera=dict(width=cfg.image_width, height=cfg.image_height, hfov_deg=cfg.hfov_deg,
                    intrinsic=[[focal, 0, (cfg.image_width - 1) / 2],
                               [0, focal, (cfg.image_height - 1) / 2], [0, 0, 1]],
                    near_m=cfg.near, far_m=cfg.far,
                    depth_type="pinhole optical-axis Z depth in metres, not ray range",
                    invalid_value=0, intrinsic_pixel_convention="integer pixel centres"),
        gap=(dict(center_planning=[cfg.gap_x, 0, cfg.camera_height],
                 normal_planning=[1, 0, 0], clear_width_m=cfg.gap_width,
                 wall_thickness_m=cfg.wall_thickness, height_m=cfg.height) if cfg.gap_wall else None),
        geometry_profile="vertical_extrusion_xy_v1",
        planning=dict(plane="XOY", fixed_height_m=cfg.camera_height,
                      min_surface_clearance_m=cfg.min_clearance,
                      scope="Obstacle separation; not an arbitrary-body feasibility certificate"),
        geometry_audit=audit_obstacles(cfg, obstacles),
        obstacles=obstacles, frames=list(camera_frames(cfg)),
        depth_rendered=False,
        note="Scene geometry and sampling poses only; not CoFLDataset or flow-field labels.")
    save_json(output / "scene.json", metadata)
    return metadata


def render_depth(output: Path, metadata, *, gpu_device_id=0):
    """Load static GLB in Habitat; do not create a color sensor or use navmesh."""
    try:
        import habitat_sim
        import magnum as mn
    except ImportError as error:
        raise RuntimeError("Habitat-Sim is not installed in this interpreter. "
                           "Use a Habitat environment or --generate-only.") from error
    cfg = SceneConfig(**metadata["config"])
    if any((output / name).exists() for name in ("depth", "valid", "preview")):
        raise FileExistsError("Depth output already exists; use a new output directory")
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str((output / "obstacles.stage_config.json").resolve())
    backend.enable_physics = False
    backend.gpu_device_id = gpu_device_id
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "depth"
    sensor.sensor_type = habitat_sim.SensorType.DEPTH
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor.resolution = [cfg.image_height, cfg.image_width]
    sensor.position = mn.Vector3(0.0)  # AgentState is the optical centre, not ground foot.
    sensor.hfov = mn.Deg(cfg.hfov_deg)
    sensor.near, sensor.far = cfg.near, cfg.far
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    try:
        for name in ("depth", "valid", "preview"):
            (output / name).mkdir()
        for frame in metadata["frames"]:
            state = habitat_sim.AgentState()
            state.position = np.asarray(frame["position_habitat"], dtype=np.float32)
            # Identity rotation: image looks along -Z, or planning +x.
            simulator.get_agent(0).set_state(state, reset_sensors=True)
            raw = np.asarray(simulator.get_sensor_observations()["depth"], dtype=np.float32)
            valid = np.isfinite(raw) & (raw >= cfg.near) & (raw < cfg.far * (1 - 1e-6))
            depth = np.where(valid, raw, 0).astype(np.float32)
            stem = f"{frame['index']:06d}"
            np.save(output / "depth" / (stem + ".npy"), depth, allow_pickle=False)
            np.save(output / "valid" / (stem + ".npy"), valid, allow_pickle=False)
            # Metric NPY is authoritative; grayscale PNG is for viewing only.
            save_depth_preview(output / "preview" / (stem + ".png"), depth, cfg.far)
            frame.update(depth=f"depth/{stem}.npy", valid=f"valid/{stem}.npy",
                         preview=f"preview/{stem}.png", valid_fraction=float(valid.mean()))
        metadata.update(depth_rendered=True,
                        habitat_sim_version=str(getattr(habitat_sim, "__version__", "unknown")))
        save_json(output / "scene.json", metadata)
    finally:
        simulator.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true",
                        help="Export geometry without importing Habitat")
    parser.add_argument("--render-existing", action="store_true",
                        help="Render an existing scene.json created with --generate-only")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    for name, default in asdict(SceneConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args(argv)
    try:
        if args.render_existing and args.generate_only:
            raise ValueError("--render-existing and --generate-only are mutually exclusive")
        if args.gpu_device_id < 0:
            raise ValueError("gpu-device-id must be nonnegative")
        output = args.output.resolve()
        if args.render_existing:
            metadata = json.loads((output / "scene.json").read_text())
            if metadata.get("format") != "cofl_m_depth_scene" or metadata.get("version") != 1:
                raise ValueError("Unsupported scene manifest")
            SceneConfig(**metadata["config"]).validate()
        else:
            cfg = SceneConfig(**{name: getattr(args, name) for name in asdict(SceneConfig())})
            metadata = generate_scene(output, cfg)
        print(f"Scene: {output / 'obstacles.glb'} ({len(metadata['obstacles'])} objects)")
        if not args.generate_only:
            render_depth(output, metadata, gpu_device_id=args.gpu_device_id)
            print(f"Rendered {len(metadata['frames'])} depth frames; no RGB sensor enabled")
        return 0
    except (ValueError, FileExistsError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
