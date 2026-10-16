# Adding a Robot

[README](../README.md) · [Architecture](ARCHITECTURE.md)

Add a MuJoCo model, a reviewed robot specification, and a small interface example.
Run all commands from the repository root; replace `my_robot` with your robot name.

## 1. Add the model

Place the model and meshes under `assets/robots/my_robot/`.

- Use a floating base with a free joint and scalar joints for the remaining DoFs.
- Define joint limits, inertial properties, and actuator/torque data where available.
- Add a neutral `<key name="home" .../>` inside `<keyframe>`. Avoid locked,
  fully extended limbs; this pose supplies the shared posture reference.
- Add `ee_<name>` sites for automatic end-effector detection. Without a site,
  IK targets the body origin.
- Keep the environment separate in `assets/scenes/`. Mesh paths resolve from the
  composed scene wrapper; for meshes in the robot's `assets/` folder, use
  `meshdir="../robots/my_robot/assets/"`.

## 2. Create and review the specification

```bash
python simulator/tools/gen_robot_spec.py \
  --robot my_robot --mjcf assets/robots/my_robot/my_robot.xml
```

Review `robots/my_robot/robot.yaml`, especially every `# REVIEW` marker:

| Field | What to verify |
| --- | --- |
| `mjcf`, `base_body` | Correct model path and floating-base body. Paths are repository-root-relative. |
| `joints` | Exact model `qpos[7:]` order, limits, and passive-joint flags. |
| `end_effectors` | Names, body/site mappings, and contact offsets. |
| `contact_bodies` | Additional usable contacts such as knees, forearms, pelvis, and head. |
| `ground_offset` | Contact radius proxy when no explicit body-local contact point is supplied. |
| `robot_card` | Kinematic facts: joint axes, signs, ranges, and dimensions. |
| `example_skill`, `camera`, `base_height_range` | Correct example path and sensible model-specific settings. |

End-effectors and `contact_bodies` form the usable IK/contact roster. A hand marked
`contact: false` is excluded from default support, but a skill can still explicitly
declare it as a contact. Do not duplicate the model's home joint angles in YAML.

Optional `environments` entries select scenes; quote `grounding: "off"` in YAML.
Curated skill exemplars are optional.

## 3. Add an example and generate summaries

Create `robots/my_robot/example.py` using
[`robots/g1/example.py`](../robots/g1/example.py) as a minimal starting point.
Adapt the end-effector names to your specification. The class subclasses
[`BaseMotionGenerator`](../simulator/motion/base.py); it is also shown to the agent
as an interface example. A stationary supported pose is sufficient.

```bash
python simulator/tools/gen_robot_summary.py --robot my_robot
```

This generates `summary.md` and `dynamics.json`. Review reported home-pose and
reachability issues before generating skills.

## 4. Validate and generate

```bash
# Check loading, joint order, and the example trajectory
python simulator/run.py --robot my_robot --mode example --headless True --record True
python simulator/tools/reach_lint.py --robot my_robot --mode example --grounding off

# Inspect the pose and contacts in the viewer
python simulator/run.py --robot my_robot --mode example

# Generate a first skill
(cd agent && ROBOT_NAME=my_robot python run.py \
  --skill "The robot walks forward and stops." --name SKILL_WALK)
```

Check the home pose, limb alignment, and achieved contacts visually, then inspect
the generated skill's report. Passing a loading check does not establish motion
quality or controller tracking. Results appear in `skill_output/my_robot/`.
