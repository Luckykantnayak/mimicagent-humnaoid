"""Contact behavior must follow declarations, geometry and modes, never labels."""
import importlib.util
from pathlib import Path
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
module_spec = importlib.util.spec_from_file_location('soft_contacts', ROOT / 'simulator/utils/soft_contacts.py')
soft = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(soft)


class SoftContactsTest(unittest.TestCase):
    def fixture(self, names=('alpha', 'beta'), sliding=()):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="floor" type="plane" size="5 5 .1"/>
          <geom name="wall" type="box" pos="1 0 .7" size=".05 2 2"/>
          <body name="root" pos="0 0 .8"><freejoint/><geom size=".04"/>
            <body name="part_a" pos="-.2 0 -.5"><geom size=".02"/></body>
            <body name="part_b" pos=".2 0 -.5"><geom size=".02"/></body>
          </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        solver = soft.SoftContactSolver(model, data, names, [2, 3], [.02, .02], [],
                                        [0, 1], [2, 3, 4], .01, sliding=sliding)
        return model, data, solver, data.qpos.copy()

    def rollout(self, model, data, solver, nominal, active, descriptors=None, frames=35):
        states = []
        for _ in range(frames):
            data.qpos[:] = nominal
            mujoco.mj_forward(model, data)
            report = solver.solve(active, {}, descriptors)
            states.append(data.qpos.copy())
        return report, np.asarray(states)

    def test_targetless_contacts_move_root_and_do_not_teleport(self):
        model, data, solver, nominal = self.fixture()
        report, states = self.rollout(model, data, solver, nominal, ['alpha', 'beta'])
        self.assertLess(np.max(np.abs(report['contact_gap'])), .006)
        self.assertLess(data.qpos[2], nominal[2] - .25)
        self.assertLess(np.max(np.abs(np.diff(states[:, 2]))), .06)

    def test_body_labels_do_not_change_solution(self):
        positions = []
        for names in [('alpha', 'beta'), ('contact_37', 'contact_92')]:
            model, data, solver, nominal = self.fixture(names)
            self.rollout(model, data, solver, nominal, list(names))
            positions.append(data.qpos.copy())
        np.testing.assert_allclose(*positions, atol=1e-10)

    def test_explicit_wall_and_local_patch(self):
        model, data, solver, nominal = self.fixture()
        report, _ = self.rollout(model, data, solver, nominal, ['beta'],
                                {'beta': {'surface': 'wall', 'point': [.02, 0, 0]}})
        self.assertLess(abs(report['contact_gap'][1]), .008)
        self.assertGreater(data.qpos[0], .3)

    def test_sliding_preserves_normal_contact_without_sticking(self):
        results = []
        for sliding in [(), ('alpha', 'beta')]:
            model, data, solver, nominal = self.fixture(sliding=sliding)
            self.rollout(model, data, solver, nominal, ['alpha', 'beta'])
            for x in np.linspace(0., .3, 50):
                nominal[0] = x
                data.qpos[:] = nominal
                mujoco.mj_forward(model, data)
                report = solver.solve(['alpha', 'beta'], {})
            results.append(data.qpos[0])
            self.assertLess(np.max(np.abs(report['contact_gap'])), .01)
        self.assertLess(abs(results[0]), .02)
        self.assertGreater(results[1], .25)

    def test_no_contact_is_inert_and_release_clears_anchors(self):
        model, data, solver, nominal = self.fixture()
        solver.solve([], {})
        np.testing.assert_array_equal(data.qpos, nominal)
        self.rollout(model, data, solver, nominal, ['alpha', 'beta'])
        before = data.qpos.copy()
        data.qpos[:] = nominal
        solver.solve([], {})
        self.assertEqual(solver.anchors, {})
        self.assertLess(abs(data.qpos[2] - before[2]), .05)
        solver.reset()
        data.qpos[:] = nominal
        solver.solve([], {})
        np.testing.assert_array_equal(data.qpos, nominal)

    def test_unknown_body_surface_and_unauthorized_slide_fail_explicitly(self):
        for active, descriptors in [(['missing'], {}),
                                    (['alpha'], {'alpha': {'surface': 'missing'}}),
                                    (['alpha'], {'alpha': {'mode': 'slide'}})]:
            model, data, solver, _ = self.fixture()
            with self.assertRaises(ValueError):
                solver.solve(active, {}, descriptors)

    def test_joint_channel_handoff_starts_from_achieved_pose(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="floor" type="plane" size="5 5 .1"/>
          <body name="root" pos="0 0 .6"><freejoint/><geom size=".04"/>
            <body name="branch"><joint name="hinge" axis="0 1 0"/>
              <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".02"/>
              <body name="tip" pos="0 0 -.4"><geom size=".02"/></body>
            </body></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        solver = soft.SoftContactSolver(model, data, ['part'], [3], [.02], ['hinge'],
                                        [0], [1, 2, 3], .005)
        solver.solve([], {})
        previous = data.qpos.copy()
        data.qpos[7] = 1.2  # abrupt authored joint-channel change at touchdown
        mujoco.mj_forward(model, data)
        solver.solve(['part'], {})
        self.assertLess(np.linalg.norm(data.qpos[:3] - previous[:3]), .02)
        # The authored 1.2 rad snap is carried as a bounded ramp (at most JOINT_STEP_MAX
        # plus a small relaxation per frame), never applied in one frame.
        self.assertLess(abs(data.qpos[7] - previous[7]), soft.JOINT_STEP_MAX + .08)
        self.assertGreater(abs(data.qpos[7] - 1.2), .8)

    def test_conflicting_contacts_remain_finite_with_visible_residual(self):
        model, data, solver, nominal = self.fixture()
        report, _ = self.rollout(model, data, solver, nominal, ['alpha', 'beta'],
                                {'alpha': {'surface': 'floor', 'point': [0, 0, 0]},
                                 'beta': {'surface': 'wall', 'point': [0, 0, 0]}})
        # A soft solve always returns a pose and measurements; there is no
        # "solver success means semantic success" shortcut.
        self.assertTrue(np.all(np.isfinite(data.qpos)))
        self.assertTrue(np.all(np.isfinite(report['contact_gap'])))
        self.assertGreater(max(np.max(np.abs(report['contact_gap'])),
                               np.max(report['contact_drift'])), .10)

    def test_final_contact_fit_retains_intended_posture_after_ik(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
          <worldbody><geom name="floor" type="plane" size="5 5 .1"/>
          <body name="root" pos="0 0 .5"><freejoint/><geom size=".04"/>
            <body name="tip" pos="0 0 -.48">
              <joint name="twist" axis="0 0 1" range="-3 3"/>
              <geom size=".02"/>
            </body></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        data.qpos[7] = 1.2  # a twisted intermediate IK solution
        mujoco.mj_forward(model, data)
        nominal = data.qpos.copy()
        solver = soft.SoftContactSolver(model, data, ['part'], [2], [.02], ['twist'],
                                        [0], [1, 2], .005)
        for _ in range(180):
            data.qpos[:] = nominal
            mujoco.mj_forward(model, data)
            report = solver.solve(['part'], {}, posture_reference={'twist': .2})
        self.assertLess(abs(data.qpos[7] - .2), .01)
        self.assertLess(abs(report['contact_gap'][0]), .005)

    def test_posture_preference_cannot_hide_gross_body_overlap(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
          <worldbody><geom name="floor" type="plane" size="5 5 .1"/>
          <body name="root" pos="0 0 .4"><freejoint/><geom name="torso" size=".1"/>
            <body name="spacer"><body name="limb" pos=".12 0 0">
              <joint name="separation" type="slide" axis="1 0 0" range="-.1 .2"/>
              <geom name="limb_geom" size=".1"/>
            </body></body>
            <body name="support" pos="0 0 -.38"><geom size=".02"/></body>
          </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        nominal = data.qpos.copy()
        pair = (model.geom('torso').id, model.geom('limb_geom').id)
        solver = soft.SoftContactSolver(model, data, ['part'], [model.body('support').id],
                                        [.02], ['separation'], [0], [1, 2, 3], .005,
                                        self_collision_pairs=[pair])
        for _ in range(60):
            data.qpos[:] = nominal
            mujoco.mj_forward(model, data)
            report = solver.solve(['part'], {}, posture_reference={'separation': 0.})
        distance = mujoco.mj_geomDistance(model, data, *pair, .1, None)
        self.assertGreater(distance, -.032)
        self.assertGreater(data.qpos[7], .045)
        self.assertLess(abs(report['contact_gap'][0]), .005)

    def twist_fixture(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
          <worldbody><geom name="floor" type="plane" size="5 5 .1"/>
          <body name="root" pos="0 0 .5"><freejoint/><geom size=".04"/>
            <body name="tip" pos="0 0 -.48">
              <joint name="twist" axis="0 0 1" range="-3 3"/>
              <geom size=".02"/>
            </body></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        solver = soft.SoftContactSolver(model, data, ['arbitrary_body'], [2], [.02],
                                        ['twist'], [0], [1, 2], .005)
        return model, data, solver

    def test_feedback_from_warm_started_ik_does_not_create_oscillation(self):
        for reference in ({'twist': .2}, None):
            with self.subTest(posture_reference=reference):
                model, data, solver = self.twist_fixture()
                data.qpos[7] = 1.
                values = []
                for _ in range(240):
                    # IK and contact fitting agree on .2 rad. IK closes 5% of the
                    # remaining error from its warm start. The old extrapolation still
                    # overshot by .41 rad and oscillated despite these consistent goals.
                    data.qpos[7] = .2 + .95 * (data.qpos[7] - .2)
                    mujoco.mj_forward(model, data)
                    report = solver.solve(['arbitrary_body'], {},
                                          posture_reference=reference)
                    values.append(data.qpos[7])
                self.assertGreaterEqual(min(values), .19)
                self.assertLess(np.ptp(values[-80:]), .005)
                self.assertLess(abs(values[-1] - .2), .005)
                self.assertLess(abs(report['contact_gap'][0]), .005)

    def test_authored_joint_motion_is_not_low_pass_filtered(self):
        for channel in ('posture_reference', 'motion_joint_reference'):
            with self.subTest(channel=channel):
                model, data, solver = self.twist_fixture()
                values = []
                references = .6 * np.sin(np.arange(240) * .005 * 2 * np.pi)
                for target in references:
                    data.qpos[7] = target  # explicit joint channel, as applied by controller
                    mujoco.mj_forward(model, data)
                    solver.solve(['arbitrary_body'], {}, **{channel: {'twist': target}})
                    values.append(data.qpos[7])
                np.testing.assert_allclose(values, references, atol=.002)

    def test_orientation_task_keeps_twist_observable_during_contact(self):
        model, data, solver = self.twist_fixture()
        orientation = np.zeros((1, model.nv))
        orientation[0, model.joint('twist').dofadr[0]] = 1.
        for _ in range(80):
            data.qpos[7] = 1.2  # upstream explicit orientation requires this twist
            mujoco.mj_forward(model, data)
            solver.solve(['arbitrary_body'], {}, posture_reference={'twist': .2},
                         orientation_jacobian=orientation)
        # A posture preference must not erase an observed orientation task.
        self.assertGreater(data.qpos[7], 1.1)



if __name__ == '__main__':
    unittest.main()
