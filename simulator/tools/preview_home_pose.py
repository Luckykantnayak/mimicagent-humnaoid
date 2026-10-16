"""Render the exact model-home posture and export the data sent to MimicAgent.

Run from the repo root:
  python simulator/tools/preview_home_pose.py --robot g1 --out /tmp/g1_home.png
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.home_pose import home_pose_from_model
from utils.robot_config import load_robot_spec
from utils.scene_compose import compose_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    spec = load_robot_spec(args.robot)
    model = compose_model(spec.mjcf, spec.scene_xml)
    home = home_pose_from_model(model, spec)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = spec.camera.get("lookat", [0, 0, home["root_pos"][2]])
    camera.distance = spec.camera.get("distance", 3.0)
    camera.elevation = spec.camera.get("elevation", -15)
    sheet = Image.new("RGB", (1440, 520), "white")
    draw = ImageDraw.Draw(sheet)
    with mujoco.Renderer(model, height=480, width=480) as renderer:
        for index, angle in enumerate((0, 90, 150)):
            camera.azimuth = angle
            renderer.update_scene(data, camera)
            sheet.paste(Image.fromarray(renderer.render()), (480 * index, 40))
            draw.text((480 * index + 12, 12),
                      f"{spec.name} home | view {angle} deg", fill="black")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(home, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
