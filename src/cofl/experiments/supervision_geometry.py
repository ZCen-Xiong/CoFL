"""Virtual geometry checks for a coupled-configuration field teacher.

Run: python -m cofl.experiments.supervision_geometry --output artifacts/cofl-m/teacher
Use --habitat-python .habitat/coflm/bin/python to also render the occlusion pair.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from cofl.generation import obstacle_scene
from cofl.generation.configuration_field import (
    Box2D, ConfigurationTeacher, CoupledCrossProxy, PlanarWorld,
)


def make_worlds():
    bounds = Box2D(0, 10, -3, 3)

    def gap(name, width):
        return PlanarWorld(name, bounds, (
            Box2D(4.9, 5.1, -3, -width / 2), Box2D(4.9, 5.1, width / 2, 3),
        ), (1, 0), (9, 0))

    return dict(
        empty=PlanarWorld("empty", bounds, (), (1, 0), (9, 0)),
        narrow_065=gap("narrow_065", 0.65),
        narrow_059=gap("narrow_059", 0.59),
        u_pocket=PlanarWorld("u_pocket", bounds, (
            Box2D(5.8, 6.1, -1.6, 1.6), Box2D(2.8, 6.1, 1.4, 1.6),
            Box2D(2.8, 6.1, -1.6, -1.4),
        ), (4, 0), (8, 0)),
        two_slots=PlanarWorld("two_slots", bounds, (
            Box2D(3.9, 4.1, -3, -1.525), Box2D(3.9, 4.1, -0.875, 3),
            Box2D(0, 6.675, 1.1, 1.3), Box2D(7.325, 10, 1.1, 1.3),
        ), (1, -1.2), (7, 2.4)),
        upper_blocked=PlanarWorld("upper_blocked", bounds, (
            Box2D(3, 4, -2, 2), Box2D(4, 6, 0, 3),
        ), (1, 0), (9, 0)),
        lower_blocked=PlanarWorld("lower_blocked", bounds, (
            Box2D(3, 4, -2, 2), Box2D(4, 6, -3, 0),
        ), (1, 0), (9, 0)),
    )


def optical_depth_scan(world, angles):
    """Analytic horizontal pinhole scan, Z depth (not ray range)."""
    ox, oy = world.start_xy
    slope = np.tan(angles)
    depth = np.full_like(slope, np.inf)
    for box in world.obstacles:
        with np.errstate(divide="ignore", invalid="ignore"):
            low = (box.ymin - oy) / slope
            high = (box.ymax - oy) / slope
        tmin = np.maximum(box.xmin - ox, np.minimum(low, high))
        tmax = np.minimum(box.xmax - ox, np.maximum(low, high))
        parallel = abs(slope) < 1e-12
        y_inside = box.ymin <= oy <= box.ymax
        tmin = np.where(parallel, box.xmin - ox if y_inside else np.inf, tmin)
        tmax = np.where(parallel, box.xmax - ox if y_inside else -np.inf, tmax)
        hit = (tmin <= tmax) & (tmax > 0)
        depth = np.minimum(depth, np.where(hit, np.maximum(tmin, 0), np.inf))
    return depth


def export_habitat_fixture(output, world):
    """Export full-height extruded 2-D fixtures using the existing GLB writer."""
    output.mkdir()
    cfg = obstacle_scene.SceneConfig(length=10, width=6, poles=0, rings=0,
                                     gap_x=5, camera_end=0, frames=1)
    obstacles = [dict(id="floor", kind="box", center=[5, 0, -0.1], size=[10, 6, 0.2])]
    # Inner faces coincide with the planner bounds, so the rendered environment
    # and collision teacher share the same enclosing arena.
    for index, (center, size) in enumerate([
        ([-0.1, 0, 1.5], [0.2, 6.4, 3]), ([10.1, 0, 1.5], [0.2, 6.4, 3]),
        ([5, -3.1, 1.5], [10, 0.2, 3]), ([5, 3.1, 1.5], [10, 0.2, 3]),
    ]):
        obstacles.append(dict(id=f"arena_{index}", kind="box", center=center, size=size))
    for index, box in enumerate(world.obstacles):
        obstacles.append(dict(id=f"wall_{index}", kind="box", center=[
            (box.xmin + box.xmax) / 2, (box.ymin + box.ymax) / 2, 1.5,
        ], size=[box.xmax - box.xmin, box.ymax - box.ymin, 3]))
    obstacle_scene.write_glb(output / "obstacles.glb", obstacles)
    obstacle_scene.save_json(output / "obstacles.stage_config.json", dict(
        render_asset="obstacles.glb", collision_asset="obstacles.glb",
        up=[0, 1, 0], front=[0, 0, -1], units_to_meters=1.0,
    ))
    position = np.asarray([*world.start_xy, cfg.camera_height])
    metadata = dict(
        format="cofl_m_depth_scene", version=1, config=asdict(cfg),
        obstacles=obstacles, depth_rendered=False, gap=None,
        note="Custom teacher fixture; frames override procedural sampling; yaw=0, camera height=1.2m.",
        frames=[dict(index=0, position_planning=position.tolist(),
                     position_habitat=(obstacle_scene.PLANNING_TO_HABITAT @ position).tolist(),
                     rotation_xyzw_habitat=[0, 0, 0, 1])],
    )
    obstacle_scene.save_json(output / "scene.json", metadata)


def evaluate(output, world, *, name, q_values=None, start_q=0.0, robot=None, resolution=0.05):
    teacher = ConfigurationTeacher(world, robot, resolution=resolution, q_values=q_values).solve(goal_q=start_q)
    path = teacher.rollout(start_q=start_q)
    field, reachable, terminal = teacher.field()
    advancing = reachable.ravel() & (teacher.predecessor >= 0)
    value = teacher.value.ravel()
    if not np.all(value[teacher.predecessor[advancing]] < value[advancing]):
        raise RuntimeError(f"Non-descending successor in {name}")
    minimum = None
    first_delta = None
    if len(path):
        clearance = teacher.robot.swept_clearance(world, path[:-1], path[1:])
        minimum = float(clearance.min()) if len(clearance) else float(teacher.robot.clearance(world, path[0]))
        if minimum <= teacher.robot.margin:
            raise RuntimeError(f"Unsafe teacher edge in {name}")
        # Independent denser state sampling as a numerical cross-check of the swept bound.
        if len(path) > 1:
            t = np.linspace(0, 1, 21)[None, :, None]
            states = path[:-1, None] * (1 - t) + path[1:, None] * t
            sampled = teacher.robot.clearance(world, states)
            if not (sampled >= clearance[:, None] - 1e-9).all():
                raise RuntimeError("Swept clearance bound overestimated sampled clearance")
        deltas = np.diff(path[:, :2], axis=0)
        translations = deltas[np.linalg.norm(deltas, axis=1) > 1e-8]
        first_delta = translations[0].tolist() if len(translations) else [0, 0]
    start_index = teacher.locate((*world.start_xy, start_q))
    result = dict(
        name=name, world=world.name, reachable=bool(len(path)),
        start_q_m=start_q, start_tangent=field[start_index].tolist(),
        first_translation_delta_m=first_delta,
        path_length_m=float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()) if len(path) else None,
        q_min_m=float(path[:, 2].min()) if len(path) else None,
        q_max_m=float(path[:, 2].max()) if len(path) else None,
        min_swept_clearance_m=minimum, path_nodes=len(path),
        total_nodes=teacher.free.size, reachable_nodes=int(reachable.sum()),
        unreachable_labels_masked=bool(not np.any(reachable & ~teacher.free)),
        terminal_nodes=int(terminal.sum()),
        start_value=float(teacher.value[start_index]) if len(path) else None,
    )
    teacher.save(output / f"{name}.npz")
    np.save(output / f"{name}-path.npy", path, allow_pickle=False)
    # Retain only the nominal slice for compact overview plots.
    iq = int(np.argmin(abs(teacher.q - start_q)))
    plot_data = dict(world=world, path=path, x=teacher.x, y=teacher.y,
                     field=field[iq], reachable=reachable[iq], result=result)
    return result, plot_data


def plot_overview(path, panels):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cofl-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for axis, (title, data) in zip(axes.flat, panels):
        world, route = data["world"], data["path"]
        for box in world.obstacles:
            axis.add_patch(Rectangle((box.xmin, box.ymin), box.xmax-box.xmin,
                                     box.ymax-box.ymin, facecolor="0.3"))
        step = max(1, int(round(0.45 / (data["x"][1] - data["x"][0]))))
        sl = (slice(None, None, step), slice(None, None, step))
        x, y = np.meshgrid(data["x"], data["y"])
        field = data["field"]
        mask = data["reachable"]
        axis.quiver(x[sl][mask[sl]], y[sl][mask[sl]], field[sl][mask[sl], 0],
                    field[sl][mask[sl], 1], color="#99b8cc", angles="xy", scale_units="xy", scale=4)
        if len(route) > 1:
            axis.plot(route[:, 0], route[:, 1], color="0.2", linewidth=4.5, alpha=0.8)
            segments = np.stack((route[:-1, :2], route[1:, :2]), axis=1)
            lines = LineCollection(segments, cmap="coolwarm", norm=plt.Normalize(-0.1, 0.1), linewidth=3)
            lines.set_array((route[:-1, 2] + route[1:, 2]) / 2)
            axis.add_collection(lines)
            # Draw conservative axis-aligned footprint extents at several poses.
            robot = CoupledCrossProxy()
            for index in np.linspace(0, len(route)-1, 5, dtype=int):
                px, py, q = route[index]
                hx, hy = robot.half_extents(q)
                axis.plot([px-hx, px+hx], [py, py], color="#dd8d20", linewidth=3, alpha=0.7)
                axis.plot([px, px], [py-hy, py+hy], color="#dd8d20", linewidth=3, alpha=0.7)
        axis.scatter(*world.start_xy, color="#1460aa", marker="o", s=60, zorder=4, label="start")
        axis.scatter(*world.goal_xy, color="#14883a", marker="*", s=130, zorder=4, label="goal")
        axis.set(xlim=(0, 10), ylim=(-3, 3), aspect="equal", title=title,
                 xlabel="x [m]", ylabel="y [m]")
        axis.grid(alpha=0.15)
    figure.suptitle("Coupled (x,y,q) teacher: graph successors and collision-checked paths\n"
                   "Path blue: q=-0.10m; white: q=0; red: q=+0.10m. Arrows: nominal-q slice.")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_alias(path, output, worlds, plots):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cofl-matplotlib")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for column, name in enumerate(("upper_blocked", "lower_blocked")):
        world, route = worlds[name], plots[name]["path"]
        axis = axes[0, column]
        for box in world.obstacles:
            axis.add_patch(Rectangle((box.xmin, box.ymin), box.xmax-box.xmin,
                                     box.ymax-box.ymin, facecolor="0.3"))
        for angle in np.linspace(-math.pi/4, math.pi/4, 13):
            axis.plot([1, 3], [0, 2*math.tan(angle)], color="#e5a32c", alpha=0.55)
        axis.plot(route[:, 0], route[:, 1], color="#176dad", linewidth=2)
        axis.scatter(*world.start_xy, color="#176dad", s=50)
        axis.scatter(*world.goal_xy, color="#208632", marker="*", s=120)
        axis.set(xlim=(0, 10), ylim=(-3, 3), aspect="equal", xlabel="x [m]", ylabel="y [m]",
                 title="Upper route blocked: go down" if column == 0 else "Lower route blocked: go up")
        image = np.load(output / f"scene-{name}" / "depth/000000.npy", allow_pickle=False)
        axes[1, column].imshow(image, vmin=0, vmax=15, cmap="viridis")
        axes[1, column].set_title("Habitat metric depth: identical pixel values")
        axes[1, column].set_axis_off()
    figure.suptitle("Same depth, goal and q; opposite optimal teacher labels\n"
                   "90-degree FoV is fully occluded by the common front wall")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run(args):
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    worlds, results, plots = make_worlds(), {}, {}
    cases = [
        ("empty", "empty", [0.0], 0.0),
        ("u_pocket", "u_pocket", [0.0], 0.0),
        ("narrow_065_fixed", "narrow_065", [0.0], 0.0),
        ("narrow_065_morph", "narrow_065", None, 0.0),
        ("narrow_059_morph", "narrow_059", None, 0.0),
        ("two_slots_morph", "two_slots", None, 0.0),
        ("two_slots_fixed_negative", "two_slots", [-0.1], -0.1),
        ("two_slots_fixed_zero", "two_slots", [0.0], 0.0),
        ("two_slots_fixed_positive", "two_slots", [0.1], 0.1),
        ("upper_blocked", "upper_blocked", [0.0], 0.0),
        ("lower_blocked", "lower_blocked", [0.0], 0.0),
    ]
    for name, world, q_values, start_q in cases:
        results[name], plots[name] = evaluate(output, worlds[world], name=name,
                                             q_values=q_values, start_q=start_q,
                                             resolution=args.resolution)
        print(name, "reachable" if results[name]["reachable"] else "unreachable", flush=True)
    # A point gets through the 0.59m slit; a full body at fixed yaw cannot.
    point = CoupledCrossProxy(e0=0, horizontal_radius=0, vertical_radius=0,
                              body_radius=0, q_limit=0, margin=0.01)
    results["narrow_059_point"], _ = evaluate(output, worlds["narrow_059"],
                                             name="narrow_059_point", robot=point,
                                             q_values=[0.0], resolution=args.resolution)
    angles = np.linspace(-math.pi/4, math.pi/4, 321)
    scan_a = optical_depth_scan(worlds["upper_blocked"], angles)
    scan_b = optical_depth_scan(worlds["lower_blocked"], angles)
    alias = dict(scan_max_error_m=float(abs(scan_a - scan_b).max()),
                 upper_blocked_tangent=results["upper_blocked"]["start_tangent"],
                 lower_blocked_tangent=results["lower_blocked"]["start_tangent"])
    robot = CoupledCrossProxy()
    with np.load(output / "narrow_065_morph.npz", allow_pickle=False) as labels:
        ix = int(np.argmin(abs(labels["x"] - 1)))
        iy = int(np.argmin(abs(labels["y"])))
        iq = int(np.argmin(abs(labels["q"])))
        shape_conditioning = dict(
            query_xy_m=[1, 0], q0_tangent=labels["field"][iq, iy, ix].tolist(),
            q_positive_limit_tangent=labels["field"][-1, iy, ix].tolist(),
        )
    fixed_yaw_feasible = [float(q) for q in np.linspace(-0.1, 0.1, 401)
                          if np.all(robot.half_extents(q) + robot.margin < 0.325)]
    yaw45_half = robot.half_extents(0.0, math.pi/4)
    square = dict(side_m=0.65, feasible_q_at_yaw_zero=fixed_yaw_feasible,
                  yaw45_q0_half_extents_m=yaw45_half.tolist(),
                  yaw45_q0_feasible=bool(np.all(yaw45_half + robot.margin < 0.325)))
    # Independent attraction v_i = goal - p_i shrinks BOTH arm pairs.
    coupling = dict(independent_tip_attraction_implied_qdot_x=-0.30,
                    independent_tip_attraction_implied_qdot_y=0.30,
                    compatible=False, units="unit-gain illustrative attraction, not hardware commands")
    thin_world = PlanarWorld("thin_wall", Box2D(0, 2, -1, 1),
                             (Box2D(0.99, 1.01, -1, 1),), (0.9, 0), (1.1, 0))
    endpoints = np.asarray([[0.9, 0, 0], [1.1, 0, 0]])
    swept_check = dict(endpoints_free=bool(np.all(point.clearance(thin_world, endpoints) > point.margin)),
                       transition_rejected=bool(point.swept_clearance(thin_world, *endpoints) <= point.margin))
    # Spatial refinement of the two slit feasibility decisions.
    refinement = {}
    for world_name in ("narrow_065", "narrow_059"):
        teacher = ConfigurationTeacher(worlds[world_name], resolution=args.resolution / 2).solve()
        refinement[world_name] = dict(resolution_m=args.resolution / 2,
                                      reachable=bool(len(teacher.rollout())))
    for name in worlds:
        export_habitat_fixture(output / f"scene-{name}", worlds[name])
    if args.habitat_python:
        for name in ("upper_blocked", "lower_blocked"):
            scene = output / f"scene-{name}"
            result = subprocess.run([
                str(args.habitat_python.resolve()), str(Path(obstacle_scene.__file__).resolve()),
                "--output", str(scene), "--render-existing",
            ], capture_output=True, text=True)
            (scene / "habitat.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"Habitat failed; see {scene / 'habitat.log'}")
        depths = [np.load(output / f"scene-{name}" / "depth/000000.npy", allow_pickle=False)
                  for name in ("upper_blocked", "lower_blocked")]
        masks = [np.load(output / f"scene-{name}" / "valid/000000.npy", allow_pickle=False)
                 for name in ("upper_blocked", "lower_blocked")]
        alias.update(habitat_depth_max_error_m=float(abs(depths[0] - depths[1]).max()),
                     habitat_masks_identical=bool(np.array_equal(*masks)),
                     habitat_depth_shape=list(depths[0].shape))
    checks = dict(
        empty_reaches_goal=results["empty"]["reachable"],
        u_requires_initial_retreat=results["u_pocket"]["reachable"] and results["u_pocket"]["first_translation_delta_m"][0] < 0,
        nominal_cannot_pass_065=not results["narrow_065_fixed"]["reachable"],
        morph_passes_065=results["narrow_065_morph"]["reachable"],
        body_cannot_pass_059=not results["narrow_059_morph"]["reachable"],
        point_falsely_passes_059=results["narrow_059_point"]["reachable"],
        two_slots_require_both_q_signs=(results["two_slots_morph"]["reachable"] and
            results["two_slots_morph"]["q_min_m"] < 0 < results["two_slots_morph"]["q_max_m"]),
        fixed_shapes_fail_two_slots=all(not results[name]["reachable"] for name in results if "two_slots_fixed" in name),
        identical_scan_conflicting_labels=(alias["scan_max_error_m"] == 0 and
            alias["upper_blocked_tangent"][1] * alias["lower_blocked_tangent"][1] < 0),
        square_requires_yaw_in_this_proxy=(not fixed_yaw_feasible and square["yaw45_q0_feasible"]),
        swept_test_catches_thin_wall=all(swept_check.values()),
        refinement_agrees=(refinement["narrow_065"]["reachable"] and not refinement["narrow_059"]["reachable"]),
        shape_must_condition_policy=(shape_conditioning["q0_tangent"][2] > 0 and
                                    shape_conditioning["q_positive_limit_tangent"][0] > 0),
    )
    if args.habitat_python:
        checks["habitat_confirms_observation_alias"] = alias["habitat_depth_max_error_m"] < 1e-6 and alias["habitat_masks_identical"]
    plot_overview(output / "geometry-overview.png", [
        ("A. Open space: q=0", plots["empty"]),
        ("B. U pocket: first move away from goal", plots["u_pocket"]),
        ("C. 0.65m slit: q=0 fails; morph succeeds", plots["narrow_065_morph"]),
        ("D. 0.59m slit: point passes; body fails", plots["narrow_059_morph"]),
        ("E. Orthogonal slits: +q then -q", plots["two_slots_morph"]),
        ("F. Occluded upper route: initial move down", plots["upper_blocked"]),
    ])
    if args.habitat_python:
        plot_alias(output / "observation-alias.png", output, worlds, plots)
    report = dict(format="cofl_configuration_teacher_geometry", version=1,
                  status="passed" if all(checks.values()) else "failed",
                  assumptions="Fixed altitude and yaw; capsule proxy; no dynamics or network training",
                  robot=asdict(robot), resolution_m=args.resolution, q_step_m=0.025,
                  morph_weight=2.0, safety_weight=0.2, safety_band_m=0.25,
                  checks=checks, cases=results, occlusion_alias=alias, square_yaw=square,
                  shape_conditioning_counterexample=shape_conditioning,
                  independent_tip_counterexample=coupling, swept_edge_counterexample=swept_check,
                  refinement=refinement)
    obstacle_scene.save_json(output / "report.json", report)
    print(f"{report['status']}: {sum(checks.values())}/{len(checks)} checks; {output / 'report.json'}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--habitat-python", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (ValueError, FileExistsError, RuntimeError) as error:
        parser.exit(2, str(error) + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
