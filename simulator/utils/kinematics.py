"""
utils/kinematics.py

Multi-link gradient-descent IK solver.

Morphology-agnostic: works with any number of end-effectors and joints —
both come explicitly from the robot spec.
"""

import numpy as np
import mujoco
import os


# Backtrack against the complete soft objective, including posture and continuity.
LS_STEP0 = float(os.environ.get("IK_LS_STEP0", "0.9"))
LS_BACKTRACK = 8
# Relative residual weight of previous achieved pose versus intended posture.
# This is a soft preference, not a velocity limit. Tasks can override it.
CONTINUITY_WEIGHT = float(os.environ.get("IK_CONTINUITY_WEIGHT", "4.0"))

# Weight on the rotational task rows relative to position. Position error is in
# metres and orientation in radians, so they are not comparable raw; this keeps
# position dominant (a contact must land in the RIGHT PLACE first) while still
# observing the twist DoFs. IK_ORI_WEIGHT=0 disables the rotational task entirely.
ORI_WEIGHT = float(os.environ.get("IK_ORI_WEIGHT", "0.4"))
# Surface-normal alignment is inferred by the framework, not requested by the
# skill. Let a tilted contact compete with posture instead of twisting a whole
# limb to flatten it. Explicit quaternion targets retain their full weight.
CONTACT_ORI_SCALE = float(os.environ.get("IK_CONTACT_ORI_SCALE", "0.125"))


class MultiLinkGradientDescentIK:
    """
    Multi-link Gradient Descent IK solver using stacked Jacobians.

    Parameters
    ----------
    model       : mujoco.MjModel
    data        : mujoco.MjData
    body_names  : list[str] — IK target bodies (XML names), one per end-effector
    sites       : list[str] — end-effector tip site names, same order as
                  body_names. A missing site falls back to the body position.
    joint_names : list[str] — actuated joint names the IK may move
    step_size   : float
    tol         : float
    """

    def __init__(
        self,
        model,
        data,
        body_names,
        sites,
        joint_names,
        step_size=0.01,
        tol=5e-3,
        null_space_gain=0.1,
    ):
        self.model = model
        self.data = data
        self.body_names = body_names
        self.step_size = step_size
        self.tol = tol
        # Strength of the pull that keeps redundant DoFs near the standing pose.
        # Lower it for high-DoF robots so end-effectors can reach far from home
        # (e.g. a humanoid raising its arms overhead) instead of being tugged back.
        self.null_space_gain = null_space_gain
        # CONTINUITY_REF=0 disables the separate temporal objective (A/B).
        self._continuity_ref = os.environ.get("CONTINUITY_REF", "1") != "0"

        self.qpos_ids = []
        joint_lo, joint_hi = [], []
        for jn in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                # jnt_qposadr is the correct index for qpos array
                self.qpos_ids.append(model.jnt_qposadr[jid])
                if model.jnt_limited[jid]:
                    joint_lo.append(model.jnt_range[jid][0])
                    joint_hi.append(model.jnt_range[jid][1])
                else:
                    joint_lo.append(-np.inf)
                    joint_hi.append(np.inf)
        self.qpos_ids = np.array(self.qpos_ids, dtype=int)
        self.joint_lo = np.array(joint_lo)
        self.joint_hi = np.array(joint_hi)

        # Read the actual model home even if this solver is created mid-motion.
        # Standalone test models without a named home retain their initial pose.
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.q0 = (model.key_qpos[home_id].copy() if home_id >= 0
                   else data.qpos.copy())
        home_data = mujoco.MjData(model)
        home_data.qpos[:] = self.q0
        mujoco.mj_fwdPosition(model, home_data)

        # ── Body IDs ────────────────────────────────────────────────────────
        self.body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in body_names
        ]

        # ── Site IDs (end-effector tips) ─────────────────────────────────────
        self.site_ids = []
        for site_name in sites:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                print(f"⚠️  Site '{site_name}' not found; falling back to body position")
                sid = None
            self.site_ids.append(sid)

        # Per-EE "up" axis in its own BODY frame: the direction that points along world
        # +z at the HOME pose. Aligning it to a surface normal makes the end-effector
        # sit on that surface exactly the way it sits flat on the floor at home — which
        # avoids having to assume anything about which way a given robot's sole faces.
        # Used by the CONTACT-FLAT orientation task (see calculate). Must come after
        # site_ids/body_ids are resolved.
        self._axis_body = []
        for sid, bid in zip(self.site_ids, self.body_ids):
            R = (home_data.site_xmat[sid] if sid is not None else home_data.xmat[bid]).reshape(3, 3)
            self._axis_body.append(R.T @ np.array([0.0, 0.0, 1.0]))

        # ── DOF indices ──────────────────────────────────────────────────────
        self.dof_ids = self._dofs_from_names(joint_names)

        print(f"[IK] Robot bodies : {body_names}")
        print(f"[IK] DOFs found   : {len(self.dof_ids)}")
        print(f"[IK] Sites found  : {sum(s is not None for s in self.site_ids)}/{len(self.site_ids)}")

    # ── DOF helpers ─────────────────────────────────────────────────────────

    def _dofs_from_names(self, joint_names):
        """Build DOF index array from an explicit list of joint names."""
        dof_ids = []
        for jn in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                dof_ids.append(self.model.jnt_dofadr[jid])
            else:
                print(f"⚠️  Joint '{jn}' not found in model — skipping")
        return np.array(dof_ids, dtype=int)

    # ── Forward kinematics ───────────────────────────────────────────────────

    def get_ee_positions(self):
        """Return current end-effector positions in world frame (N × 3)."""
        pos = []
        for sid, bid in zip(self.site_ids, self.body_ids):
            if sid is not None:
                pos.append(self.data.site_xpos[sid].copy())
            else:
                pos.append(self.data.xpos[bid].copy())
        return np.array(pos)

    def _ee_rotmat(self, i):
        """World rotation matrix of end-effector i (site frame if it has one)."""
        sid, bid = self.site_ids[i], self.body_ids[i]
        m = self.data.site_xmat[sid] if sid is not None else self.data.xmat[bid]
        return np.asarray(m, dtype=float).reshape(3, 3)

    def _ori_error(self, i, spec):
        """Rotational task error (3,) for end-effector i, or None.

        Two modes, because they answer different questions:

        ('align', axis_world) — put this EE's home-'up' axis along `axis_world`,
            leaving rotation ABOUT that axis free. The error is the cross product
            a_current x a_target, whose magnitude is sin(angle) and which vanishes
            for any twist once aligned. This is what a contact needs: a sole flat on
            a tread or a palm flat on a wall, WITHOUT pinning the foot's yaw — a full
            orientation target would over-constrain that and fight the skill.

        ('quat', q) — full orientation, for a skill that genuinely needs all three
            (gripping an object at a specific attitude).
        """
        mode, val = spec
        R = self._ee_rotmat(i)
        if mode == "align":
            a_tgt = np.asarray(val, dtype=float)
            n = np.linalg.norm(a_tgt)
            if n < 1e-9:
                return None
            a_tgt = a_tgt / n
            a_cur = R @ self._axis_body[i]
            return np.cross(a_cur, a_tgt)
        if mode == "quat":
            q_cur = np.zeros(4)
            mujoco.mju_mat2Quat(q_cur, R.flatten())
            err = np.zeros(3)
            mujoco.mju_subQuat(err, np.asarray(val, dtype=float), q_cur)
            return err
        return None

    # ── IK solver ────────────────────────────────────────────────────────────

    def _surface_errors(self, box_aabb, robot_geoms):
        """Existing floor/static-box center test, shared by fitting and cleanup."""
        for gid in robot_geoms:
            p = self.data.geom_xpos[gid]
            depth, d = 0.0, None
            if -float(p[2]) > depth:                      # geom center below the floor
                depth, d = -float(p[2]), np.array([0.0, 0.0, 1.0])
            for c, h in box_aabb:
                dx, dy, dz = h[0]-abs(p[0]-c[0]), h[1]-abs(p[1]-c[1]), h[2]-abs(p[2]-c[2])
                if dx > 0 and dy > 0 and dz > 0:          # center inside this box
                    if dz <= dx and dz <= dy:
                        m, dr = dz, np.array([0.0, 0.0, (1.0 if p[2] >= c[2] else -1.0)])
                    elif dx <= dy:
                        m, dr = dx, np.array([(1.0 if p[0] >= c[0] else -1.0), 0.0, 0.0])
                    else:
                        m, dr = dy, np.array([0.0, (1.0 if p[1] >= c[1] else -1.0), 0.0])
                    if m > depth:
                        depth, d = m, dr
            yield gid, depth, d

    def _repel(self, box_aabb, robot_geoms):
        """Soft joint gradient pushing penetrating geom centers out of surfaces."""
        dq = np.zeros(len(self.dof_ids))
        jacp = np.zeros((3, self.model.nv))
        for gid, depth, direction in self._surface_errors(box_aabb, robot_geoms):
            if direction is not None and depth > 1e-4:
                mujoco.mj_jacGeom(self.model, self.data, jacp, None, gid)
                dq += jacp[:, self.dof_ids].T @ (direction * depth)
        return dq

    def resolve_penetration(self, box_aabb, robot_geoms, max_iter=12, repel_gain=3.0):
        """Push any ROBOT GEOM out of a surface (floor/box) by adjusting the JOINTS,
        run AFTER joint_overrides have been applied. The in-`calculate` repulsion
        acts during IK, but a JOINT-DRIVEN limb's angles are set by
        compute_joint_overrides AFTER IK — overwriting that fix — so a joint-driven
        knee/shin can still penetrate. This closes that gap generally: it bends
        whatever joints move a penetrating geom out (overridden joints included),
        never touching the base. INERT when nothing penetrates (early-exit)."""
        if box_aabb is None or robot_geoms is None:
            return
        for _ in range(max_iter):
            mujoco.mj_fwdPosition(self.model, self.data)
            dq = self._repel(box_aabb, robot_geoms)
            if not dq.any():
                break
            dq_full = np.zeros(self.model.nv)
            dq_full[self.dof_ids] = dq * repel_gain * self.step_size
            mujoco.mj_integratePos(self.model, self.data.qpos, dq_full, 1)
            self.data.qpos[self.qpos_ids] = np.clip(
                self.data.qpos[self.qpos_ids], self.joint_lo, self.joint_hi)
        mujoco.mj_fwdPosition(self.model, self.data)

    def _ori_jacobian(self, i, spec, jacr):
        """Negative derivative of the orientation error used by the solve.

        An aligned normal constrains two axes; its unobserved twist must remain
        available to the posture objective. A full angular Jacobian freezes it.
        """
        mode, value = spec
        if mode == "align":
            target = np.asarray(value, dtype=float)
            target = target / np.linalg.norm(target)
            current = self._ee_rotmat(i) @ self._axis_body[i]
            return (np.dot(current, target) * np.eye(3)
                    - np.outer(current, target)) @ jacr
        return jacr

    def calculate(self, target_ee_pos, active_mask=None, max_iter=60, damping=1e-2, debug=False,
                  box_aabb=None, robot_geoms=None, repel_gain=3.0, ref_overrides=None,
                  ori_targets=None):
        """Fit endpoints, orientation, intended posture and continuity together.

        All objectives have finite weights. Posture defaults to the model home;
        phase-specific joint references replace it. The previous achieved pose
        is a separate continuity objective, never a replacement for the intended
        posture. No velocity/torque constraints or skill-specific rules are used.
        """
        active = (list(range(len(self.body_ids))) if active_mask is None
                  else list(np.flatnonzero(active_mask)))
        if not active:
            return
        previous = self.data.qpos[self.qpos_ids].copy()
        reference = self.q0[self.qpos_ids].copy()
        # Only change joints on active kinematic chains. Uncommanded free limbs
        # are not silently drawn back to home while a different limb uses IK.
        chain_bodies = set()
        for i in active:
            body = int(self.body_ids[i])
            while body > 0:
                chain_bodies.add(body)
                body = int(self.model.body_parentid[body])
        owned = np.array([int(self.model.dof_bodyid[d]) in chain_bodies
                          for d in self.dof_ids])
        reference[~owned] = previous[~owned]
        for name, value in (ref_overrides or {}).items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                indices = np.flatnonzero(self.qpos_ids == self.model.jnt_qposadr[jid])
                if indices.size and owned[indices[0]]:
                    reference[indices[0]] = float(value)
        reference = np.clip(reference, self.joint_lo, self.joint_hi)
        posture_weight = max(0., float(self.null_space_gain))
        continuity_weight = max(0., CONTINUITY_WEIGHT) * posture_weight if self._continuity_ref else 0.
        identity = np.eye(len(self.dof_ids))

        def residuals(with_jacobian=False):
            mujoco.mj_fwdPosition(self.model, self.data)
            current = self.get_ee_positions()
            errors, rows = [], []
            for i in active:
                errors.append(np.asarray(target_ee_pos[i]) - current[i])
                spec = (ori_targets or {}).get(i)
                orientation_error = self._ori_error(i, spec) if spec is not None else None
                orientation_weight = ORI_WEIGHT * (
                    max(0., CONTACT_ORI_SCALE) if spec is not None and spec[0] == "align" else 1.)
                if orientation_error is not None:
                    errors.append(orientation_weight * orientation_error)
                if with_jacobian:
                    jp = np.zeros((3, self.model.nv)); jr = jp.copy()
                    sid = self.site_ids[i]
                    if sid is not None:
                        mujoco.mj_jacSite(self.model, self.data, jp, jr, sid)
                    else:
                        mujoco.mj_jacBody(self.model, self.data, jp, jr, self.body_ids[i])
                    rows.append(jp[:, self.dof_ids].copy())
                    if orientation_error is not None:
                        rows.append(orientation_weight * self._ori_jacobian(i, spec, jr[:, self.dof_ids]))
            # Keep penetration avoidance inside the same finite objective.
            # A fixed row per geom keeps line-search losses comparable as contact changes.
            if box_aabb is not None and robot_geoms is not None:
                weight = np.sqrt(max(0., repel_gain))
                for gid, depth, direction in self._surface_errors(box_aabb, robot_geoms):
                    errors.append(np.array([weight * depth]))
                    if with_jacobian:
                        row = np.zeros((1, len(self.dof_ids)))
                        if direction is not None:
                            jp = np.zeros((3, self.model.nv))
                            mujoco.mj_jacGeom(self.model, self.data, jp, None, gid)
                            row[0] = weight * direction @ jp[:, self.dof_ids]
                        rows.append(row)
            q = self.data.qpos[self.qpos_ids]
            for target, weight in ((reference, posture_weight), (previous, continuity_weight)):
                errors.append(weight * (target - q))
                if with_jacobian:
                    rows.append(weight * identity)
            return np.concatenate(errors), np.vstack(rows) if with_jacobian else None

        # A locked, straight chain can have zero first-order authority along
        # the requested displacement. Retain the small interior seed used by
        # the endpoint solver, limited to the chains participating in this solve.
        initial_error, _ = residuals()
        if np.linalg.norm(initial_error) >= self.tol:
            q = self.data.qpos[self.qpos_ids]
            span = self.joint_hi - self.joint_lo
            margin = np.where(np.isfinite(span), np.minimum(.05, .02 * span), 0.)
            seeded = np.clip(q, self.joint_lo + margin, self.joint_hi - margin)
            self.data.qpos[self.qpos_ids] = np.where(owned, seeded, q)

        for it in range(max_iter):
            error, jacobian = residuals(True)
            gradient = jacobian.T @ error
            if np.max(np.abs(gradient)) < 1e-7:
                break
            step = np.linalg.solve(jacobian.T @ jacobian + damping ** 2 * identity, gradient)
            q = self.data.qpos[self.qpos_ids]
            step[(q >= self.joint_hi - 1e-6) & (step > 0)] = 0.
            step[(q <= self.joint_lo + 1e-6) & (step < 0)] = 0.
            if np.max(np.abs(step)) < 1e-5:
                break
            before = self.data.qpos.copy()
            loss = error @ error
            accepted = False
            for scale in LS_STEP0 * .5 ** np.arange(LS_BACKTRACK):
                self.data.qpos[:] = before
                velocity = np.zeros(self.model.nv)
                velocity[self.dof_ids] = scale * step
                mujoco.mj_integratePos(self.model, self.data.qpos, velocity, 1.)
                self.data.qpos[self.qpos_ids] = np.clip(
                    self.data.qpos[self.qpos_ids], self.joint_lo, self.joint_hi)
                trial, _ = residuals()
                if trial @ trial < loss:
                    accepted = True
                    break
            if not accepted:
                self.data.qpos[:] = before
                break
        mujoco.mj_fwdPosition(self.model, self.data)
        if debug:
            error = np.linalg.norm((np.asarray(target_ee_pos)[active]
                                   - self.get_ee_positions()[active]).ravel())
            print(f"[IK] it={it:3d} | endpoint err={error * 1000:.2f}mm")


class MinkIK(MultiLinkGradientDescentIK):
    """
    mink-backed IK — MuJoCo differential-IK as a per-frame QP. SAME public interface
    as the DLS solver (swap via IK_BACKEND), so the controller is unchanged. It is the
    STRUCTURAL cure for the DLS drive-through-floor under-reach on extreme poses:
      * joint ranges are HARD QP constraints (ConfigurationLimit), not a post-hoc clip
        that freezes DoFs at their walls and stalls a near-singular leg;
      * the null-space POSTURE task biases redundant DoFs toward the AUTHORED pose
        (ref_overrides / the crouch), NOT toward standing (the DLS's q0 pull) — so a
        deep crouch / handstand / mantle is regularized toward itself.
    FAST: warm-started from the previous frame + early-exit on a tiny solve velocity,
    so normal motion converges in 1-3 QP iterations (~0.1-0.5 ms each, daqp backend).
    """

    def __init__(self, model, data, body_names, sites, joint_names,
                 step_size=0.01, tol=5e-3, null_space_gain=0.1,
                 velocity_ceilings=None, self_collision=True,
                 collision_margin=0.004):
        super().__init__(model, data, body_names, sites, joint_names,
                         step_size=step_size, tol=tol, null_space_gain=null_space_gain)
        import mink
        self._mink = mink
        self.configuration = mink.Configuration(model)
        # one POSITION-only FrameTask per union member (EE or contact body). A site is
        # a precise tip; without one we drive the body frame (matches the DLS fallback).
        self._tasks = []
        for i, bn in enumerate(body_names):
            if self.site_ids[i] is not None:
                name, ftype = sites[i], "site"
            else:
                name, ftype = bn, "body"
            self._tasks.append(mink.FrameTask(
                frame_name=name, frame_type=ftype,
                position_cost=1.0, orientation_cost=0.0, lm_damping=1e-2))
        # posture (null-space) regularizer — target is set per frame to the authored pose
        self._posture = mink.PostureTask(model, cost=max(float(null_space_gain), 1e-3))

        # ── HARD CONSTRAINTS in the QP (this is mink's real advantage over the DLS:
        #    the solution is FEASIBLE by construction, not fixed up after the fact) ──
        self._limits = [mink.ConfigurationLimit(model)]     # (1) joint POSITION ranges

        # (2) joint VELOCITY ceilings — no actuated joint may step faster than its
        #     torque-derived ceiling (rad/s), so an infeasible slew is impossible.
        vc = dict(velocity_ceilings or {})
        if vc:
            self._limits.append(mink.VelocityLimit(
                model, {jn: float(v) for jn, v in vc.items() if jn in self._qadr_names(joint_names)}))

        # (3) COLLISION AVOIDANCE — object AND body penetration as a hard inequality:
        #     the foot CANNOT be solved through the floor/a box, and (self_collision)
        #     the legs CANNOT be solved into each other. Robot geoms = every geom on a
        #     non-worldbody body; surface geoms = the worldbody collidables (floor + boxes).
        robot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0
                       and model.geom_contype[g]]
        surface_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0
                         and model.geom_contype[g]]
        # CONTACT geoms — those on the union IK roster's bodies (soles, palms, and any
        # declared contact body) — are EXCLUDED from surface avoidance. This simulator
        # is KINEMATIC: a planted contact is *meant* to rest on its surface at ~0
        # clearance, so a hard "stay 4 mm off every surface" constraint over those geoms
        # is a contradiction — it holds the feet off the floor and the robot reads as
        # unsupported for most of the clip (support_or_ballistic 0.875 on a plain walk).
        # Everything ELSE still avoids surfaces, so a shin cannot cut through a slab.
        contact_geoms = {g for g in range(model.ngeom)
                         if int(model.geom_bodyid[g]) in set(int(b) for b in self.body_ids)}
        avoid_geoms = [g for g in robot_geoms if g not in contact_geoms]
        pairs = []
        if avoid_geoms and surface_geoms:
            pairs.append((avoid_geoms, surface_geoms))       # non-contact links vs floor/boxes
        if self_collision and len(robot_geoms) > 1:
            pairs.append((robot_geoms, robot_geoms))         # robot vs itself (adjacency
                                                             # auto-excluded via the model)
        self._collision = None
        if pairs:
            try:
                self._collision = mink.CollisionAvoidanceLimit(
                    model, geom_pairs=pairs, gain=0.85,
                    minimum_distance_from_collisions=collision_margin,
                    collision_detection_distance=max(0.02, collision_margin * 3))
                self._limits.append(self._collision)
            except Exception as e:
                print(f"[IK] mink collision-avoidance disabled ({e})")

        # the base freejoint's tangent DoFs (and any passive DoFs, e.g. go2w wheels) are
        # FROZEN — the skill authors the base absolutely; IK only bends the actuated joints.
        self.base_dof = np.setdiff1d(np.arange(model.nv), self.dof_ids)
        # joint name -> qpos address, to overlay ref_overrides into the posture target.
        self._qadr = {}
        for jn in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                self._qadr[jn] = model.jnt_qposadr[jid]
        print(f"[IK] backend: MINK (QP) | limits: joint-pos"
              f"{' + vel' if vc else ''}"
              f"{' + collision(obj' + (', self' if (self_collision and self._collision) else '') + ')' if self._collision else ''}"
              f" | posture->authored-pose")

    @staticmethod
    def _qadr_names(joint_names):
        return set(joint_names)

    def calculate(self, target_ee_pos, active_mask=None, max_iter=20, damping=1e-2,
                  debug=False, box_aabb=None, robot_geoms=None, repel_gain=3.0,
                  ref_overrides=None):
        mink = self._mink
        # 1. warm-start the QP config from the CURRENT pose (includes the authored base)
        self.configuration.update(self.data.qpos.copy())
        # 2. posture target = the AUTHORED pose (home qpos overlaid with ref_overrides) —
        #    this is the fix: redundant DoFs pull toward the crouch, not toward standing.
        q_ref = self.data.qpos.copy()
        for jn, ang in (ref_overrides or {}).items():
            a = self._qadr.get(jn)
            if a is not None:
                q_ref[a] = float(ang)
        self._posture.set_target(q_ref)
        # 3. only ACTIVE EEs get a task (inactive ones contribute nothing to the QP)
        active = (list(range(len(self._tasks))) if active_mask is None
                  else [i for i in range(len(self._tasks)) if active_mask[i]])
        for i in active:
            self._tasks[i].set_target(
                mink.SE3.from_translation(np.asarray(target_ee_pos[i], dtype=float)))
        tasks = [self._tasks[i] for i in active] + [self._posture]
        # 4. iterate differential solve -> integrate, FREEZING the base; early-exit fast.
        dt, it = 1.0, 0
        for it in range(max_iter):
            vel = mink.solve_ik(self.configuration, tasks, dt, "daqp",
                                damping=damping, limits=self._limits)
            vel[self.base_dof] = 0.0
            if np.abs(vel[self.dof_ids]).max() < 1e-4:      # converged (target met or stuck)
                break
            self.configuration.integrate_inplace(vel, dt)
        # 5. write back; belt-and-suspenders clamp (mink #104 runaway on unreachable
        #    targets) keeps a genuinely-unreachable target at the joint-limited best pose
        #    (a clean LARGE residual for the reach nets) instead of a wild config; FK.
        self.data.qpos[:] = self.configuration.q
        self.data.qpos[self.qpos_ids] = np.clip(
            self.data.qpos[self.qpos_ids], self.joint_lo, self.joint_hi)
        mujoco.mj_fwdPosition(self.model, self.data)
        if debug and active:
            new = self.get_ee_positions()
            e = np.linalg.norm((np.asarray(target_ee_pos)[active] - new[active]).flatten())
            print(f"[IK-mink] it={it} err={e*1000:.2f}mm")


def make_ik(model, data, body_names, sites, joint_names, null_space_gain=0.1,
            velocity_ceilings=None):
    """Construct the IK backend selected by IK_BACKEND.

    'dls'  (DEFAULT) — damped least squares. Tracks the authored targets tightly,
           which is what the contact checks measure; joint limits are enforced by
           clamping each iteration, everything else by the surrounding safety nets.
    'mink' (opt-in)  — the same problem as a per-frame QP, with joint POSITION and
           VELOCITY limits and self/object collision avoidance as HARD constraints,
           so an infeasible solution cannot be produced in the first place.

    Which is better is not obvious and was measured, not assumed. mink's constraints
    are genuinely stronger, but its first-order velocity solve TRACKS more loosely,
    and the contact laws are scored on ACHIEVED positions — so on the bundled skills
    mink currently trades a class of violation it makes impossible for a tracking lag
    that shows up as contact drift. DLS therefore stays the default until the tracking
    gap is closed; use IK_BACKEND=mink to A/B.

    velocity_ceilings: {joint: rad/s} for mink's VelocityLimit (ignored by DLS).
    """
    backend = os.environ.get("IK_BACKEND", "dls").lower()
    if backend == "mink":
        try:
            return MinkIK(model, data, body_names, sites, joint_names,
                          null_space_gain=null_space_gain,
                          velocity_ceilings=velocity_ceilings)
        except ImportError as e:
            # An OPTIONAL dependency: degrade loudly rather than crash, and say what
            # is lost, so a missing package is never mistaken for a working backend.
            print(f"[IK] ⚠️  IK_BACKEND=mink requested but unavailable ({e}).\n"
                  f"[IK]     Falling back to DLS — hard joint-velocity limits and QP "
                  f"collision avoidance are OFF.\n"
                  f"[IK]     Install with:  pip install mink 'qpsolvers[daqp]'")
    return MultiLinkGradientDescentIK(model, data, body_names, sites, joint_names,
                                      null_space_gain=null_space_gain)
