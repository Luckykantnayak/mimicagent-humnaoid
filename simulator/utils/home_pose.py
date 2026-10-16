"""Robot posture data derived from the model's home keyframe, never a skill."""
import mujoco
import numpy as np


def home_pose_from_model(model, spec):
    """Return JSON-safe joint and FK references without changing simulation state.

    The root is in world coordinates; roster positions are relative to the base
    body's frame. A site is used when registered, just as in the IK solver.
    """
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key < 0:
        raise ValueError(f"{spec.name}: model needs a named 'home' keyframe")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_fwdPosition(model, data)
    base = model.body(spec.base_body).id
    rotation = data.xmat[base].reshape(3, 3)
    joints = {}
    for name in spec.actuated_joint_names:
        joint = model.joint(name)
        if int(joint.type[0]) not in (mujoco.mjtJoint.mjJNT_HINGE,
                                    mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"{spec.name}: home reference needs scalar joint {name}")
        angle = float(data.qpos[int(joint.qposadr[0])])
        if not np.isfinite(angle) or (joint.limited[0] and
                not joint.range[0] - 1e-8 <= angle <= joint.range[1] + 1e-8):
            raise ValueError(f"{spec.name}: invalid home angle for {name}: {angle}")
        joints[name] = angle
    positions = {}
    for name, body, site in zip(spec.ik_names, spec.ik_bodies, spec.ik_sites):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) if site else -1
        point = data.site_xpos[sid] if sid >= 0 else data.xpos[model.body(body).id]
        positions[name] = (rotation.T @ (point - data.xpos[base])).tolist()
    return {"source": f"{spec.mjcf}#home", "root_pos": data.qpos[:3].tolist(),
            "root_quat": data.qpos[3:7].tolist(), "joints": joints,
            "ee_positions_body": positions}
