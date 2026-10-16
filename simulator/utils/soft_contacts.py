"""Soft kinematic contact fitting, independent of skill names and control channels.

All weights are finite. Joint limits bound the search, but no dynamics or contact
force constraints are imposed. The caller must record achieved residuals, and the
fitter records its own corrections so that none of its edits is silent.
"""
import os

import numpy as np
import mujoco

# ---- finite penalty weights (squared-residual weights) -------------------------------
W_CONTACT = float(os.environ.get("SOFT_W_CONTACT", "160.0"))  # per m^2 of contact error
W_FREE = 4.0        # per m^2 of error to an authored free-limb IK target
W_PENETRATION = 100.0
SELF_COLLISION_ALLOWANCE = .01  # tolerate a small graze, below the gross-overlap check
W_POSE = float(os.environ.get("SOFT_W_POSE", "0.15"))   # intended pose, post-IK fallback
# Root continuity follows the authored root increment. Joint continuity approaches
# the current IK/override pose from the last achieved pose, without extrapolating
# differences between IK solutions. This avoids both feedback oscillation and
# freezing free limbs during contact. GAP_TAU_S sets the finite continuity weight.
#
# (It used to hold the joints at the previous frame with a weight ~670x the pose
# preference, i.e. a ~3.3 s low-pass filter on every joint whenever any contact was
# declared: the reference walk's swing foot lifted 0.9 cm instead of 10 cm and a
# generated side flip landed and never stood back up.)
GAP_TAU_S = float(os.environ.get("SOFT_GAP_TAU_S", "0.15"))
# Authored joint motion carried into the continuity reference per frame (rad). A
# larger single-frame step is an IK configuration flip or a channel-handoff snap,
# not motion; it is spread over several frames instead of teleporting the joint.
JOINT_STEP_MAX = float(os.environ.get("SOFT_JOINT_STEP_MAX", "0.15"))
# A contact prefers the SUPPORT surface beneath it (an upward-facing top) over a
# nearer wall when its contact point is within this height of that top. Otherwise
# a foot beside a stair riser anchors to the riser, and a foot floating above the
# tread then reads as a perfect contact against the wall.
SUPPORT_PREFERENCE_M = 0.08

MODES = ("stick", "slide", "roll", "pivot")
MODE_ALIASES = {
    "planted": "stick", "plant": "stick", "sticking": "stick", "stuck": "stick",
    "fixed": "stick", "static": "stick", "stance": "stick", "pinned": "stick",
    "anchored": "stick", "hold": "stick", "held": "stick", "grip": "stick",
    "sliding": "slide", "slip": "slide", "slipping": "slide", "skate": "slide",
    "skating": "slide", "drag": "slide", "dragging": "slide", "glide": "slide",
    "rolling": "roll", "wheel": "roll",
    "pivoting": "pivot", "spin": "pivot", "spinning": "pivot",
}
SURFACE_SUFFIXES = ("_top", "_surface", "_face", "_front", "_side", "_geom", "_plane", "_edge")


def _resolve_surface(value, geom_names, contact, notes):
    if geom_names is None or value in geom_names:
        return value
    lowered = {g.lower(): g for g in geom_names}
    if value.lower() in lowered:
        notes.append(f"contact '{contact}': surface {value!r} read as {lowered[value.lower()]!r}")
        return lowered[value.lower()]
    for suffix in SURFACE_SUFFIXES:
        if value.lower().endswith(suffix) and value.lower()[:-len(suffix)] in lowered:
            real = lowered[value.lower()[:-len(suffix)]]
            notes.append(f"contact '{contact}': surface {value!r} read as geom {real!r} "
                         f"(surfaces are named by their scene geom)")
            return real
    raise ValueError(
        f"contact '{contact}': unknown scene surface {value!r}. The scene's surfaces are "
        f"{sorted(geom_names)}; name one of them, or omit 'surface' to use the surface "
        f"beneath the contact.")


def normalize_contact_descriptors(descriptors, geom_names=None):
    """Interpret compute_contact_constraints() output as {name: {surface, point, mode}}.

    ONE implementation for the runtime fitter and the pre-simulation lint, so both
    accept exactly the same inputs. Unambiguous variants are normalized and reported
    in `notes` (a list of dicts, a mode alias such as 'planted', 'slab_top' for the
    geom 'slab'); anything that cannot be interpreted raises ValueError with a precise
    message. Returns (descriptors, notes).
    """
    notes = []
    if descriptors is None:
        return {}, notes
    if isinstance(descriptors, (list, tuple)):
        converted = {}
        for item in descriptors:
            if not isinstance(item, dict):
                raise ValueError("compute_contact_constraints must return a dict "
                                 "{contact_name: {'surface': ..., 'mode': ...}}; got a list "
                                 f"containing {type(item).__name__}")
            key = next((k for k in ("name", "contact", "body", "ee") if isinstance(item.get(k), str)), None)
            if key is None:
                raise ValueError("compute_contact_constraints returned a list whose entries have no "
                                 "'name'; return a dict {contact_name: {...}} instead")
            converted[item[key]] = {k: v for k, v in item.items() if k != key}
        notes.append("compute_contact_constraints returned a list; read it as "
                     "{name: descriptor}. Return a dict.")
        descriptors = converted
    if not isinstance(descriptors, dict):
        raise ValueError("compute_contact_constraints must return a dict {contact_name: {...}} "
                         f"(or {{}}), got {type(descriptors).__name__}")
    out = {}
    for name, desc in descriptors.items():
        if desc is None:
            continue
        if isinstance(desc, str):
            desc = {"surface": desc}
        if not isinstance(desc, dict):
            raise ValueError(f"contact descriptor for '{name}' must be a dict such as "
                             f"{{'surface': <scene geom>, 'mode': 'stick'}}, got {type(desc).__name__}")
        clean = {}
        for key, val in desc.items():
            if key == "mode":
                raw = str(val).strip().lower()
                mode = MODE_ALIASES.get(raw, raw)
                if mode not in MODES:
                    raise ValueError(f"contact '{name}': unsupported mode {val!r}; use one of {list(MODES)}")
                if mode != raw:
                    notes.append(f"contact '{name}': mode {val!r} read as {mode!r}")
                clean["mode"] = mode
            elif key == "surface":
                if val is None:
                    continue
                if not isinstance(val, str):
                    raise ValueError(f"contact '{name}': surface must be a scene geom name, got {val!r}")
                clean["surface"] = _resolve_surface(val, geom_names, name, notes)
            elif key == "point":
                try:
                    arr = np.asarray(val, dtype=float)
                except (TypeError, ValueError):
                    arr = np.zeros(0)
                if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                    raise ValueError(f"contact '{name}': point must be a finite body-local "
                                     f"[x, y, z] in metres, got {val!r}")
                clean["point"] = [float(x) for x in arr]
            else:
                notes.append(f"contact '{name}': ignored unknown descriptor key {key!r}")
        out[name] = clean
    return out, notes


def surface_candidates(model, data, point, geom_ids):
    """For each supported primitive: (closest point, outward normal, signed gap, geom id).

    Uses current world transforms, including moving objects. Unsupported shapes
    (meshes, height fields) are skipped rather than silently treated as the floor.
    """
    out = []
    for gid in geom_ids:
        rotation = data.geom_xmat[gid].reshape(3, 3)
        p = rotation.T @ (point - data.geom_xpos[gid])
        size, kind = model.geom_size[gid], model.geom_type[gid]
        normal = np.zeros(3)
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            normal[2] = 1.
            gap = p[2]
            nearest = p - gap * normal
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            nearest = np.clip(p, -size, size)
            delta = p - nearest
            gap = np.linalg.norm(delta)
            if gap > 1e-10:
                normal = delta / gap
            else:
                axis = np.argmin(size - np.abs(p))
                normal[axis] = 1. if p[axis] >= 0 else -1.
                gap = -(size[axis] - abs(p[axis]))
                nearest = p - gap * normal
        elif kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE):
            center = np.zeros(3)
            if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                center[2] = np.clip(p[2], -size[1], size[1])
            delta = p - center
            length = np.linalg.norm(delta)
            normal = delta / length if length > 1e-10 else np.array([1., 0., 0.])
            gap = length - size[0]
            nearest = center + size[0] * normal
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius, half = size[0], size[1]
            radial = np.linalg.norm(p[:2])
            direction = p[:2] / radial if radial > 1e-10 else np.array([1., 0.])
            if radial <= radius and abs(p[2]) <= half:          # inside: nearest face
                side, cap = radius - radial, half - abs(p[2])
                if side < cap:
                    normal[:2] = direction
                    gap = -side
                else:
                    normal[2] = 1. if p[2] >= 0 else -1.
                    gap = -cap
                nearest = p - gap * normal
            else:
                nearest = np.array([*(direction * min(radial, radius)), np.clip(p[2], -half, half)])
                delta = p - nearest
                gap = np.linalg.norm(delta)
                normal = delta / gap if gap > 1e-10 else np.array([0., 0., 1.])
        else:
            continue
        out.append((data.geom_xpos[gid] + rotation @ nearest,
                    rotation @ normal, float(gap), int(gid)))
    return out


def closest_surface(model, data, point, geom_ids):
    """Closest point, outward normal, signed gap, geom id; None if nothing supported."""
    candidates = surface_candidates(model, data, point, geom_ids)
    return min(candidates, key=lambda c: abs(c[2])) if candidates else None


def select_contact_surface(model, data, point, geom_ids, radius=0.):
    """The surface a contact at `point` rests on. The upward-facing support surface
    beneath it wins when the contact point is near it; otherwise the nearest surface
    (a wall brace, an underside, a bar)."""
    candidates = surface_candidates(model, data, point, geom_ids)
    if not candidates:
        return None
    support = [c for c in candidates if c[1][2] > 0.5 and c[2] - radius <= SUPPORT_PREFERENCE_M]
    if support:
        return min(support, key=lambda c: abs(c[2] - radius))
    return min(candidates, key=lambda c: abs(c[2]))


class SoftContactSolver:
    """Refine the final authored pose using every active contact declaration.

    Optional per-contact descriptors: surface (geom name), point (body-local
    contact point), mode (stick/slide/roll/pivot). Without a point, ground_offset
    is a radius proxy about the body origin, preserving the robot spec convention.
    Without a surface, select the support surface under the contact at touchdown.
    Stick anchors are stored in surface coordinates, not recreated each frame.
    """
    def __init__(self, model, data, names, body_ids, offsets, joint_names,
                 scene_geoms, robot_geoms, dt, sliding=(), pivots=(), rolling=(),
                 target_points=None, self_collision_pairs=()):
        self.model, self.data = model, data
        self.names, self.bodies = list(names), list(body_ids)
        self.offsets = np.asarray(offsets)
        self.scene_geoms, self.robot_geoms = list(scene_geoms), list(robot_geoms)
        self.self_collision_pairs = np.asarray(self_collision_pairs, dtype=int).reshape(-1, 2)
        self.geom_names = {}
        for gid in self.scene_geoms:
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
            if gname:
                self.geom_names[gname] = int(gid)
        # Body-local point that IK targets for each roster member (its site, when the
        # spec gives one), so the free-limb term pulls the same point IK solved for.
        self.target_points = {n: np.asarray(v, dtype=float) for n, v in (target_points or {}).items()}
        self.dt = dt
        self.modes = {n: 'slide' for n in sliding}
        self.modes.update({n: 'pivot' for n in pivots})
        self.modes.update({n: 'roll' for n in rolling})
        # Floating base plus actuated joints. Passive joints retain their pose.
        self.dofs = list(range(6))
        self.limits = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.dofs.append(int(model.jnt_dofadr[jid]))
            if model.jnt_limited[jid]:
                self.limits.append((int(model.jnt_qposadr[jid]), *model.jnt_range[jid]))
        self.dofs = np.asarray(self.dofs)
        self.w_continuity = W_POSE * GAP_TAU_S / dt
        self._reported = set()
        self.reset()

    def reset(self):
        self.anchors = {}
        self.previous = None
        self.previous_nominal = None
        self.previous_posture_reference = {}
        self.previous_motion_joint_reference = {}
        self.last_fit = {"fit_root_shift": 0., "fit_root_rotation": 0.,
                         "fit_joint_shift": 0., "authored_root_dev": 0.}

    def _note(self, message):
        if message not in self._reported:
            self._reported.add(message)
            print(f"[soft-contacts] {message}", flush=True)

    def _point(self, index, descriptor):
        bid = self.bodies[index]
        local = np.asarray(descriptor.get('point', [0., 0., 0.]), dtype=float)
        if local.shape != (3,) or not np.all(np.isfinite(local)):
            raise ValueError('Contact point must be a finite body-local 3-vector')
        return self.data.xpos[bid] + self.data.xmat[bid].reshape(3, 3) @ local

    def _clip_limits(self, q):
        for address, lo, hi in self.limits:
            q[address] = np.clip(q[address], lo, hi)
        return q

    def _jacobian(self, point, bid):
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jac, None, point, bid)
        return jac[:, self.dofs]

    def _record_fit(self, nominal, reference_root):
        delta = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, delta, 1., nominal, self.data.qpos)
        joints = self.dofs[6:]
        self.last_fit = {
            "fit_root_shift": float(np.linalg.norm(delta[:3])),
            "fit_root_rotation": float(np.linalg.norm(delta[3:6])),
            "fit_joint_shift": float(np.max(np.abs(delta[joints]))) if len(joints) else 0.,
            "authored_root_dev": (float(np.linalg.norm(self.data.qpos[:3] - reference_root[:3]))
                                  if reference_root is not None else 0.),
        }

    def solve(self, active, targets, descriptors=None, reference_root=None, exact_root=False,
              posture_reference=None, orientation_jacobian=None, motion_joint_reference=None):
        descriptors, notes = normalize_contact_descriptors(descriptors, self.geom_names)
        for note in notes:
            self._note(note)
        active = list(active)
        unknown = set(active) - set(self.names)
        if unknown:
            raise ValueError(f'Unknown declared contacts: {sorted(unknown)}; '
                             f'contact names are {self.names}')
        nominal = self.data.qpos.copy()
        # IK is an intermediate fit, not the source of intended posture. Preserve
        # explicit joint references through the final whole-body fit, which can
        # adjust the root as well as the limbs to reconcile posture and contact.
        intended_pose = nominal.copy()
        for name, value in (posture_reference or {}).items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0 and int(self.model.jnt_dofadr[jid]) in self.dofs:
                intended_pose[self.model.jnt_qposadr[jid]] = float(value)
        self._clip_limits(intended_pose)
        motion_reference = nominal.copy()
        if reference_root is not None:
            motion_reference[:7] = reference_root
        # Transport the previous correction along the newly authored motion.
        # This smooths corrections without damping intentional root travel/turns.
        smooth_reference = nominal.copy()
        if self.previous is not None:
            correction = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model, correction, 1.,
                                     self.previous_nominal, self.previous)
            if exact_root:
                correction[:3] = 0.   # ballistic flight owns the root translation exactly
            mujoco.mj_integratePos(self.model, smooth_reference, correction,
                                  np.exp(-self.dt / .08))
        if active and self.previous is not None:
            # Advance the root by its authored increment. For joints, approach
            # the CURRENT IK/override solution from the previous achieved pose.
            # On joints with authored posture references, unfiltered IK-result
            # differences feed posture corrections back through the warm start.
            # Separate their task motion from redundant posture motion below.
            increment = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model, increment, 1.,
                                     self.previous_nominal, motion_reference)
            joint_step = np.zeros(self.model.nv)
            mujoco.mj_differentiatePos(self.model, joint_step, 1., self.previous, nominal)
            # Preserve task-visible IK changes, but let redundant motion follow
            # the intended posture rather than repeatedly restoring an IK twist.
            references = posture_reference or {}
            selected, authored = [], []
            for name, value in references.items():
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    continue
                dof = int(self.model.jnt_dofadr[jid])
                if dof not in self.dofs[6:]:
                    continue
                selected.append(dof)
                authored.append(value - self.previous_posture_reference.get(name, value))
            if selected:
                mujoco.mj_comPos(self.model, self.data)
                rows = []
                columns = [int(np.flatnonzero(self.dofs == d)[0]) for d in selected]
                if orientation_jacobian is not None:
                    rows.append(np.asarray(orientation_jacobian)[:, selected])
                for name in sorted(set(active) | {n for n, t in targets.items() if t is not None}):
                    if name not in self.names:
                        continue
                    i = self.names.index(name)
                    desc = descriptors.get(name, {}) if name in active else {}
                    if name not in active and name in self.target_points:
                        desc = {'point': self.target_points[name]}
                    jac = self._jacobian(self._point(i, desc), self.bodies[i])
                    rows.append(jac[:, columns])
                positional = rows[1:] if orientation_jacobian is not None else rows
                if positional:
                    point_jac = np.vstack(positional)
                    retained = np.zeros(self.model.nv)
                    mujoco.mj_differentiatePos(self.model, retained, 1.,
                                             self.previous_nominal, self.previous)
                    joint_step[selected] += point_jac.T @ np.linalg.solve(
                        point_jac @ point_jac.T + 1e-6 * np.eye(len(point_jac)),
                        point_jac @ retained[selected])
                if rows:
                    jac = np.vstack(rows)
                    authored = np.asarray(authored)
                    visible = jac @ (joint_step[selected] - authored)
                    joint_step[selected] = authored + jac.T @ np.linalg.solve(
                        jac @ jac.T + 1e-6 * np.eye(len(jac)), visible)
                else:
                    joint_step[selected] = authored
            # Unreferenced joints have no competing authored-posture objective.
            # Preserve their existing correction transport (including nonlinear
            # contact/collision corrections that a local task Jacobian can miss).
            unreferenced = [d for d in self.dofs[6:] if d not in selected]
            if unreferenced:
                retained = np.zeros(self.model.nv)
                mujoco.mj_differentiatePos(self.model, retained, 1.,
                                         self.previous_nominal, self.previous)
                joint_step[unreferenced] += retained[unreferenced]
            # Direct joint channels have genuine authored increments, independent
            # of IK warm starts. Keep their existing correction transport, so a
            # contact elsewhere cannot interrupt an intentional free-limb swing.
            for name, value in (motion_joint_reference or {}).items():
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    dof = int(self.model.jnt_dofadr[jid])
                    if dof in self.dofs[6:]:
                        joint_step[dof] = value - self.previous_motion_joint_reference.get(name, value)
            increment[6:] = np.clip(joint_step[6:], -JOINT_STEP_MAX, JOINT_STEP_MAX)
            smooth_reference = self.previous.copy()
            mujoco.mj_integratePos(self.model, smooth_reference, increment, 1.)
        self.previous_posture_reference = dict(posture_reference or {})
        self.previous_motion_joint_reference = dict(motion_joint_reference or {})
        # Previous pose + authored increment can cross a joint limit when the previous
        # frame carried a correction; the reference must itself be a legal pose.
        smooth_reference = self._clip_limits(smooth_reference)
        # Touchdown starts from the achieved continuous pose, not from the raw
        # joint-override pose, which can change abruptly at a channel handoff.
        self.data.qpos[:] = smooth_reference
        mujoco.mj_kinematics(self.model, self.data)
        self.anchors = {n: a for n, a in self.anchors.items() if n in active}
        for name in active:
            index = self.names.index(name)
            desc = dict(descriptors.get(name, {}))
            mode = desc.get('mode', self.modes.get(name, 'stick'))
            # Code may not grant itself a no-slip exemption absent the fixed plan/goal
            # or a wheel declaration (the pre-sim lint reports this before any rollout).
            if mode in {'slide', 'roll', 'pivot'} and self.modes.get(name) != mode:
                raise ValueError(f"contact '{name}': mode {mode!r} requires a matching declaration "
                                 f"(goal sliding_contacts, a plan contact with mode 'pivot', or a "
                                 f"wheel in the robot spec); use 'stick'")
            identity = (desc.get('surface'), tuple(desc.get('point', ())), mode)
            anchor = self.anchors.get(name)
            if anchor is None or anchor['identity'] != identity:
                point = self._point(index, desc)
                # Existing world targets specify the intended vicinity. Explicit
                # local points are already patch coordinates, unlike body targets.
                seed = targets.get(name)
                probe = point if seed is None or 'point' in desc else np.asarray(seed)
                if desc.get('surface') is not None:
                    hit = closest_surface(self.model, self.data, probe,
                                          [self.geom_names[desc['surface']]])
                else:
                    radius = 0. if 'point' in desc else float(self.offsets[index])
                    hit = select_contact_surface(self.model, self.data, probe,
                                                 self.scene_geoms, radius)
                if hit is None:
                    raise ValueError(f'{name}: no supported contact surface (plane, box, sphere, '
                                     f'capsule or cylinder) near the contact')
                surface, normal, _, gid = hit
                rotation = self.data.geom_xmat[gid].reshape(3, 3)
                anchor = dict(identity=identity, descriptor=desc, gid=gid, mode=mode,
                              local=rotation.T @ (surface - self.data.geom_xpos[gid]),
                              normal=rotation.T @ normal, seed=point.copy(), age=0.)
                self.anchors[name] = anchor
            anchor['age'] += self.dt

        if not active:
            self.data.qpos[:] = smooth_reference
            mujoco.mj_kinematics(self.model, self.data)
            self._record_fit(nominal, reference_root)
            self.previous = self.data.qpos.copy()
            self.previous_nominal = motion_reference
            return self.measure()

        self.data.qpos[:] = smooth_reference

        def residuals(with_jacobian=True):
            mujoco.mj_kinematics(self.model, self.data)
            # mj_jac uses subtree inertial positions populated by mj_comPos.
            mujoco.mj_comPos(self.model, self.data)
            rows, errors = [], []
            for name, anchor in self.anchors.items():
                i = self.names.index(name)
                point = self._point(i, anchor['descriptor'])
                rotation = self.data.geom_xmat[anchor['gid']].reshape(3, 3)
                normal = rotation @ anchor['normal']
                surface = self.data.geom_xpos[anchor['gid']] + rotation @ anchor['local']
                radius = 0. if 'point' in anchor['descriptor'] else self.offsets[i]
                projector = np.eye(3)
                if anchor['mode'] in {'slide', 'roll'}:
                    surface, normal, _, _ = closest_surface(self.model, self.data, point, [anchor['gid']])
                    projector = np.outer(normal, normal)
                elif anchor['mode'] == 'pivot':
                    projector = .2 * np.eye(3) + .8 * np.outer(normal, normal)
                blend = min(1., anchor['age'] / .08)
                blend = blend * blend * (3. - 2. * blend)
                target = ((1. - blend) * anchor['seed']
                          + blend * (surface + radius * normal))
                weight = np.sqrt(W_CONTACT)
                errors.append(weight * projector @ (point - target))
                if with_jacobian:
                    rows.append(weight * projector @ self._jacobian(point, self.bodies[i]))
            # Preserve authored free endpoints while permitting contact fitting to
            # trade against them. Joint-only limbs retain the posture preference.
            weight = np.sqrt(W_FREE)
            for name, target in targets.items():
                if name in self.anchors or target is None or name not in self.names:
                    continue
                i = self.names.index(name)
                local = self.target_points.get(name)
                p = self._point(i, {'point': local} if local is not None else {})
                errors.append(weight * (p - np.asarray(target)))
                if with_jacobian:
                    rows.append(weight * self._jacobian(p, self.bodies[i]))
            # Geometric penetration regularization, same rule for every body.
            weight = np.sqrt(W_PENETRATION)
            for gid in self.robot_geoms:
                p = self.data.geom_xpos[gid]
                hit = closest_surface(self.model, self.data, p, self.scene_geoms)
                if hit is not None and hit[2] < 0:
                    errors.append(np.array([weight * hit[2]]))
                    if with_jacobian:
                        rows.append(weight * hit[1][None, :] @ self._jacobian(p, int(self.model.geom_bodyid[gid])))
            # Posture references can themselves intersect the body. Use the same
            # non-adjacent collision pairs as recording, with a finite gross-overlap
            # penalty, so preserving intent cannot silently push a limb through it.
            if len(self.self_collision_pairs):
                pairs = self.self_collision_pairs
                positions = self.data.geom_xpos
                bound = self.model.geom_rbound
                nearby = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
                nearby = pairs[nearby < bound[pairs[:, 0]] + bound[pairs[:, 1]]]
                points = np.zeros(6)
                weight = np.sqrt(W_PENETRATION)
                for first, second in nearby:
                    distance = mujoco.mj_geomDistance(self.model, self.data, first, second, .1, points)
                    if distance >= -SELF_COLLISION_ALLOWANCE:
                        continue
                    errors.append(np.array([weight * (distance + SELF_COLLISION_ALLOWANCE)]))
                    if with_jacobian:
                        # Signed distance is negative in overlap; dividing the
                        # witness-point vector by it retains the separating normal.
                        normal = (points[3:] - points[:3]) / distance
                        j1 = self._jacobian(points[:3], int(self.model.geom_bodyid[first]))
                        j2 = self._jacobian(points[3:], int(self.model.geom_bodyid[second]))
                        rows.append(weight * normal[None, :] @ (j2 - j1))
            # Pose and continuity are finite preferences, including all base DoFs.
            for reference, w in ((intended_pose, W_POSE), (smooth_reference, self.w_continuity)):
                delta = np.zeros(self.model.nv)
                mujoco.mj_differentiatePos(self.model, delta, 1., reference, self.data.qpos)
                errors.append(np.sqrt(w) * delta[self.dofs])
                if with_jacobian:
                    rows.append(np.sqrt(w) * np.eye(len(self.dofs)))
            return np.concatenate(errors), np.vstack(rows) if with_jacobian else None

        for _ in range(16):
            residual, jac = residuals()
            step = np.linalg.solve(jac.T @ jac + .005 * np.eye(len(self.dofs)), -jac.T @ residual)
            step *= min(1., .10 / max(1e-10, np.max(np.abs(step))))
            if np.linalg.norm(step) < 1e-5:
                break
            before, loss = self.data.qpos.copy(), float(residual @ residual)
            velocity = np.zeros(self.model.nv)
            velocity[self.dofs] = step
            accepted = False
            for scale in (1., .5, .25, .125):
                self.data.qpos[:] = before
                mujoco.mj_integratePos(self.model, self.data.qpos, velocity, scale)
                for address, lo, hi in self.limits:
                    self.data.qpos[address] = np.clip(self.data.qpos[address], lo, hi)
                trial, _ = residuals(False)
                if float(trial @ trial) < loss:
                    accepted = True
                    break
            if not accepted:
                self.data.qpos[:] = before
                break
        self._clip_limits(self.data.qpos)
        mujoco.mj_kinematics(self.model, self.data)
        self._record_fit(nominal, reference_root)
        self.previous = self.data.qpos.copy()
        self.previous_nominal = motion_reference
        return self.measure()

    def measure(self):
        gap = np.zeros(len(self.names))
        drift = gap.copy()
        for name, anchor in self.anchors.items():
            i = self.names.index(name)
            p = self._point(i, anchor['descriptor'])
            hit = closest_surface(self.model, self.data, p, [anchor['gid']])
            radius = 0. if 'point' in anchor['descriptor'] else self.offsets[i]
            gap[i] = hit[2] - radius
            rotation = self.data.geom_xmat[anchor['gid']].reshape(3, 3)
            delta = p - self.data.geom_xpos[anchor['gid']] - rotation @ anchor['local']
            normal = rotation @ anchor['normal']
            drift[i] = np.linalg.norm(delta - normal * np.dot(delta, normal))
        return {'contact_gap': gap, 'contact_drift': drift, **self.last_fit}
