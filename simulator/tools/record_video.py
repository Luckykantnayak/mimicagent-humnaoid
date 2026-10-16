"""
simulator/tools/record_video.py

Offscreen video recorder. --trajectory renders evaluated qpos directly, without
rerunning IK. Otherwise replay uses the same home keyframe and runner as the sim.

Run from the repo root:
  python simulator/tools/record_video.py --robot humanoid --mode SKILL_CARRYBOX \
      --scene assets/scenes/carry.xml --grounding off \
      --out /tmp/carrybox.mp4
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import mujoco
import imageio

PROJECT_ROOT = Path(__file__).resolve().parents[1]      # simulator/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.simulator import MujocoSimulator
from core.controller import MotionControllerRunner
from core.recording import recording_window, prepare_recording_step
from utils.robot_config import load_robot_spec
from utils.file_utils import load_yaml


def recorded_frame_indices(count, source_fps, output_fps):
    """Sample by time, without speeding up clips when FPS do not divide evenly."""
    if count < 1 or source_fps <= 0 or output_fps <= 0:
        raise ValueError("Trajectory length and frame rates must be positive")
    return np.minimum(count - 1, np.floor(
        np.arange(int(np.ceil(count / source_fps * output_fps)))
        * source_fps / output_fps).astype(int))


def apply_recorded_frame(model, data, trajectory, index):
    """Render the evaluated qpos directly: no skill execution or second IK solve."""
    data.qpos[:3] = trajectory["root_pos"][index]
    data.qpos[3:7] = trajectory["root_quat"][index]
    data.qpos[7:] = trajectory["joints"][index]
    for name, positions in (trajectory.get("object_pos") or {}).items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0 or model.body_mocapid[bid] < 0:
            raise ValueError(f"Recorded object {name!r} is not in the rendering scene")
        mid = model.body_mocapid[bid]
        data.mocap_pos[mid] = positions[index]
        quat = (trajectory.get("object_quat") or {}).get(name)
        if quat is not None:
            data.mocap_quat[mid] = quat[index]
    mujoco.mj_forward(model, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--grounding", default="off")
    ap.add_argument("--robots_dir", default="robots")
    ap.add_argument("--sim_freq", type=float, default=200)
    default_key = load_yaml(PROJECT_ROOT / "config.yaml")["run"]["init_position"]
    ap.add_argument("--init_position", default=default_key, help="Named model keyframe")
    ap.add_argument("--trajectory", type=Path, help="Render this evaluated PKL without rerunning IK")
    ap.add_argument("--cycles", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cam_lookat", default=None, help="x,y,z")
    ap.add_argument("--cam_dist", type=float, default=None)
    ap.add_argument("--cam_az", type=float, default=None)
    ap.add_argument("--cam_el", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = load_robot_spec(args.robot, robots_dir=args.robots_dir)
    scene = args.scene or spec.scene_xml
    sim = MujocoSimulator(scene, init_position=args.init_position,
                          sim_freq=int(args.sim_freq), base_body=spec.base_body,
                          camera=spec.camera, headless=True,
                          robot_mjcf=spec.mjcf)   # compose robot + agnostic scene at load
    runner = None if args.trajectory else MotionControllerRunner(sim, args)
    model, data = sim.model, sim.data
    trajectory = None
    if args.trajectory:
        with args.trajectory.open("rb") as f:
            trajectory = pickle.load(f)
        if trajectory.get("robot") != args.robot:
            raise ValueError("Recorded trajectory robot does not match --robot")
        count = len(trajectory["root_pos"])
        for key, width in (("root_pos", 3), ("root_quat", 4), ("joints", model.nq - 7)):
            values = np.asarray(trajectory[key])
            if values.shape != (count, width) or not np.isfinite(values).all():
                raise ValueError(f"Invalid recorded {key} shape or values")
        for key, width in (("object_pos", 3), ("object_quat", 4)):
            for name, values in (trajectory.get(key) or {}).items():
                if np.asarray(values).shape != (count, width) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid recorded {key} for {name}")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    c = spec.camera
    lk = ([float(v) for v in args.cam_lookat.split(",")] if args.cam_lookat
          else c.get("lookat", [0.0, 0.0, 0.9]))
    cam.lookat[:] = lk
    cam.distance = args.cam_dist if args.cam_dist is not None else c.get("distance", 4.0)
    cam.azimuth = args.cam_az if args.cam_az is not None else c.get("azimuth", 90)
    cam.elevation = args.cam_el if args.cam_el is not None else c.get("elevation", -15)

    # Frame the entire recorded path and scene. A fixed home camera loses the
    # robot on approach/carry motions; explicit camera overrides still win.
    trajectory_path = args.trajectory or Path("skill_output") / args.robot / args.mode / (args.mode + ".pkl")
    if args.cam_lookat is None and args.cam_dist is None and trajectory_path.exists():
        with trajectory_path.open("rb") as f:
            framing = pickle.load(f)
        clouds = [np.asarray(framing["root_pos"]).reshape(-1, 3)]
        if framing.get("ee_positions_achieved") is not None:
            clouds.append(np.asarray(framing["ee_positions_achieved"]).reshape(-1, 3))
        clouds.extend(np.asarray(track).reshape(-1, 3)
                      for track in (framing.get("object_pos") or {}).values())
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.base_body)
        for geom in range(model.ngeom):
            if (model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
                    and model.body_rootid[model.geom_bodyid[geom]] != model.body_rootid[base]):
                half = np.abs(data.geom_xmat[geom].reshape(3, 3)) @ model.geom_size[geom]
                clouds.append(np.array([data.geom_xpos[geom] - half, data.geom_xpos[geom] + half]))
        points = np.concatenate(clouds)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            lo, hi = points.min(axis=0) - .15, points.max(axis=0) + .15
            cam.lookat[:] = (lo + hi) / 2
            radius = np.linalg.norm(hi - lo) / 2
            cam.distance = max(cam.distance, 1.1 * radius / np.sin(np.deg2rad(model.vis.global_.fovy / 2)))

    frames = []
    if trajectory is not None:
        indices = recorded_frame_indices(count, trajectory["fps"], args.fps)
        for index in indices:
            apply_recorded_frame(model, data, trajectory, index)
            renderer.update_scene(data, cam)
            frames.append(renderer.render())
    else:
        skill_freq = getattr(runner.motion_gen, "freq", None) or 1.0
        start_frame, end_frame = recording_window(args.sim_freq, skill_freq, args.cycles)
        indices = recorded_frame_indices(end_frame - start_frame, args.sim_freq, args.fps)
        counts = np.bincount(indices, minlength=end_frame - start_frame)
        while runner.frame < end_frame:
            prepare_recording_step(runner, start_frame)
            runner.step()
            recorded_index = runner.frame - start_frame - 1
            if 0 <= recorded_index < len(counts) and counts[recorded_index]:
                renderer.update_scene(data, cam)
                frames.extend([renderer.render()] * int(counts[recorded_index]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if str(args.out).lower().endswith(".gif"):
        # GIF: play FORWARD on an infinite loop (loop=0), never ping-pong/backward —
        # frames are appended in time order, so this always reads forward.
        # GIF stores centiseconds; alternate rounded durations instead of
        # rounding every frame and changing the speed (e.g. 33 fps -> 30 ms).
        durations = np.diff(np.rint(np.arange(len(frames) + 1) * 100 / args.fps)) * 10
        imageio.mimwrite(args.out, frames, duration=durations.tolist(), loop=0)
    else:
        imageio.mimwrite(args.out, frames, fps=args.fps, quality=8, macro_block_size=None)
    print(f"[record_video] wrote {len(frames)} frames ({len(frames)/args.fps:.1f}s) -> {args.out}")
    if args.trajectory:
        metadata = {"source": "evaluated_trajectory", "trajectory": str(args.trajectory),
                    "trajectory_sha256": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
                    "frame_indices": indices.tolist(), "fps": args.fps,
                    "missing_object_orientation_tracks": sorted(set(trajectory.get("object_pos", {}))
                        - set(trajectory.get("object_quat", {})))}
        Path(args.out).with_suffix(".render.json").write_text(json.dumps(metadata, indent=2) + "\n")
    renderer.close()
    sim.close()


if __name__ == "__main__":
    main()
