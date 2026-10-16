"""Regressions for the 2026-09-22 framework fixes (see CLAUDE.md, 'Soft contact fitter',
'Ballistic flight', 'Goal contract'). Each test encodes a failure that was measured on
real skills before the fix."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))


def _load(name, rel):
    spec_ = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    return module


soft = _load('soft_contacts_fix_test', 'simulator/utils/soft_contacts.py')
ballistics = _load('ballistics_fix_test', 'simulator/utils/ballistics.py')

from utils import test_motion_quality as q                          # noqa: E402
from utils.semantic_contract import validate_invariants              # noqa: E402
from utils.robot_spec import load_robot_spec                         # noqa: E402
from utils.tools import extract_python_code_with_report, dropped_code_lines  # noqa: E402
from agents.code_gen.run_unit_test import candidate_rank             # noqa: E402
from agents.code_gen.goal_contract_agent import make_goal_contract_node  # noqa: E402
from agents.code_gen.code_executer_agent import make_code_executer_node  # noqa: E402
from prompts.prompt_loader import load_yaml_config                   # noqa: E402

LEG_ARM = '''<mujoco><compiler angle="radian"/><worldbody>
  <geom name="floor" type="plane" size="5 5 .1"/>
  <geom name="riser" type="box" pos="1 0 .3" size=".3 1 .3"/>
  <body name="root" pos="0 0 .73"><freejoint/><geom size=".05"/>
    <body name="leg"><joint name="leg_j" axis="0 1 0" range="-2 2"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.7" size=".03"/>
      <body name="foot" pos="0 0 -.7"><geom size=".03"/></body></body>
    <body name="arm" pos="0 .2 0"><joint name="arm_j" axis="1 0 0" range="-3 3"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".02"/></body>
  </body></worldbody></mujoco>'''


class SoftFitterTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(LEG_ARM)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.solver = soft.SoftContactSolver(self.model, self.data, ['foot'], [3], [.03],
                                             ['leg_j', 'arm_j'], [0, 1], [2, 3, 4, 5], .005)

    def test_authored_joint_motion_passes_through_while_a_contact_is_held(self):
        # Measured before the fix: a declared contact froze every joint behind a ~3.3 s
        # low-pass (the reference walk's swing foot lifted 0.9 cm instead of 10 cm).
        base = self.data.qpos.copy()
        errors = []
        for k in range(300):
            authored = 0.8 * np.sin(2 * np.pi * k * 0.005)     # 1 Hz arm gesture
            self.data.qpos[:] = base
            self.data.qpos[8] = authored
            mujoco.mj_forward(self.model, self.data)
            self.solver.solve(['foot'], {}, reference_root=base[:7].copy())
            errors.append(abs(self.data.qpos[8] - authored))
        self.assertLess(max(errors[100:]), 0.1)

    def test_fitter_records_its_own_corrections(self):
        self.solver.solve(['foot'], {})
        fit = self.solver.measure()
        for key in ('fit_root_shift', 'fit_root_rotation', 'fit_joint_shift', 'authored_root_dev'):
            self.assertIn(key, fit)
            self.assertGreaterEqual(fit[key], 0.0)

    def test_support_surface_beats_a_nearer_riser(self):
        # A foot hovering 3 cm over the floor, 4 cm from a riser face: the nearest
        # surface is the riser, but the contact rests on the floor.
        point = np.array([0.66, 0.0, 0.06])
        nearest = soft.closest_surface(self.model, self.data, point, [0, 1])
        chosen = soft.select_contact_surface(self.model, self.data, point, [0, 1], radius=0.03)
        self.assertEqual(nearest[3], 1)
        self.assertEqual(chosen[3], 0)
        self.assertGreater(chosen[1][2], 0.9)
        # Well above the floor, beside the riser: a brace against the wall is the contact.
        high = soft.select_contact_surface(self.model, self.data, np.array([0.66, 0., 0.45]), [0, 1], radius=0.03)
        self.assertEqual(high[3], 1)

    def test_descriptor_normalization_is_lenient_only_when_unambiguous(self):
        names = {'floor': 0, 'riser': 1}
        out, notes = soft.normalize_contact_descriptors(
            {'a': {'mode': 'planted', 'surface': 'riser_top'}}, names)
        self.assertEqual(out, {'a': {'mode': 'stick', 'surface': 'riser'}})
        self.assertEqual(len(notes), 2)
        out, notes = soft.normalize_contact_descriptors([{'name': 'a', 'mode': 'sticking'}], names)
        self.assertEqual(out, {'a': {'mode': 'stick'}})
        for bad in ({'a': {'surface': 'ledge_top'}}, {'a': {'point': [1, 2]}},
                    {'a': {'mode': 'hover'}}, 'floor', [3]):
            with self.assertRaises(ValueError):
                soft.normalize_contact_descriptors(bad, names)

    def test_unauthorized_relaxation_fails_explicitly(self):
        with self.assertRaises(ValueError):
            self.solver.solve(['foot'], {}, {'foot': {'mode': 'pivot'}})


class BallisticFlightTest(unittest.TestCase):
    def test_short_window_becomes_the_gravity_parabola(self):
        dt, T = 0.005, 0.4
        window = ballistics.plan_window([0, 0, 0.88], (int(T / dt), np.array([0.4, 0, 0.88])), dt, 5.5, 800)
        self.assertTrue(window['ok'])
        zs = [ballistics.position(window, k, dt)[2] for k in range(int(T / dt) + 1)]
        self.assertAlmostEqual(max(zs) - 0.88, 9.81 * T * T / 8, places=2)
        acc = np.diff(zs, 2) / dt ** 2
        np.testing.assert_allclose(acc, -9.81, atol=1e-6)

    def test_hovering_window_is_rejected_with_the_physical_duration(self):
        # Measured before the fix: a generated side flip hovered 2.4 s at -0.7 m/s^2.
        dt = 0.005
        window = ballistics.plan_window([0, 0, 0.86], (482, np.array([0, 0, 0.76])), dt, 5.5, 800)
        self.assertFalse(window['ok'])
        self.assertGreater(window['violation'], 5.0)
        self.assertLess(window['tmax'], 1.3)
        self.assertFalse(ballistics.plan_window([0, 0, .9], None, dt, 5.5, 800)['ok'])

    def test_violation_fails_the_floor_check(self):
        spec = load_robot_spec(load_yaml_config(str(ROOT / 'agent/cfg/config.yaml')))
        traj = {'flight_violation': np.r_[np.zeros(10), np.full(20, 6.0), np.zeros(10)],
                'flight_tmax': np.full(40, 1.1)}
        self.assertFalse(q.eval_ballistic_flight(traj, spec, 0.005)['passed'])
        traj['flight_violation'][:] = 0
        self.assertTrue(q.eval_ballistic_flight(traj, spec, 0.005)['passed'])


class SemanticPredicateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_yaml_config(str(ROOT / 'agent/cfg/config.yaml'))
        cls.spec = load_robot_spec(cls.cfg)

    def trajectory(self, frames=400):
        names = list(self.spec.ee_names) + list(self.spec.contact_body_names)
        return {'root_pos': np.tile([0., 0., .88], (frames, 1)), 'root_quat': np.tile([1., 0., 0., 0.], (frames, 1)),
                'joints': np.zeros((frames, len(self.spec.joint_names))), 'ee_names': names,
                'ee_positions_achieved': np.zeros((frames, len(names), 3)),
                'contact_mask': np.zeros((frames, len(names)), bool),
                'surface_clearance': np.full((frames, len(names)), 0.3)}

    def check(self, traj, inv):
        return q.eval_declared_invariants(traj, self.spec, 0.01, [inv])[0]

    def test_signal_window_measures_posture_and_end_state(self):
        traj = self.trajectory()
        inverted = {'name': 'inverted', 'kind': 'signal_window',
                    'params': {'signal': 'upright', 'max': -0.8, 'min_duration_s': 0.5}}
        self.assertFalse(self.check(traj, inverted)['passed'])
        traj['root_quat'][100:200] = [0., 0., 1., 0.]            # upside down for 1 s
        self.assertTrue(self.check(traj, inverted)['passed'])
        ends_up = {'name': 'up', 'kind': 'signal_window',
                   'params': {'signal': 'upright', 'window': 'end', 'stat': 'min', 'min': 0.9}}
        self.assertTrue(self.check(traj, ends_up)['passed'])
        traj['root_quat'][-5:] = [0., 0., 1., 0.]
        self.assertFalse(self.check(traj, ends_up)['passed'])

    def test_contact_absence_and_ordered_landing(self):
        traj = self.trajectory()
        hand = traj['ee_names'].index('left_hand')
        feet = [traj['ee_names'].index(n) for n in ('left_foot', 'right_foot')]
        never = {'name': 'no_hand', 'kind': 'signal_window',
                 'params': {'signal': 'touching:left_hand', 'window': 'all', 'stat': 'max', 'max': 0}}
        self.assertTrue(self.check(traj, never)['passed'])
        traj['surface_clearance'][50, hand] = 0.01
        self.assertFalse(self.check(traj, never)['passed'])
        land = {'name': 'land', 'kind': 'event_order', 'params': {'events': [
            {'signal': 'airborne', 'min': 1, 'min_duration_s': 0.2},
            {'signal': 'contacts:left_foot,right_foot', 'min': 2, 'min_duration_s': 0.1}]}}
        for i in feet:                                   # stand, fly 0.3 s, land together
            traj['contact_mask'][:100, i] = True
            traj['contact_mask'][130:, i] = True
            traj['surface_clearance'][:100, i] = 0.0
            traj['surface_clearance'][130:, i] = 0.0
        self.assertTrue(self.check(traj, land)['passed'])
        # The hand touches only after the flight: "touch, then fly" is out of order.
        traj['surface_clearance'][:, hand] = 0.3
        traj['surface_clearance'][350:, hand] = 0.0
        touch = {'signal': 'touching:left_hand', 'min': 1}
        fly = {'signal': 'airborne', 'min': 1, 'min_duration_s': 0.2}
        in_order = {'name': 'ok', 'kind': 'event_order', 'params': {'events': [fly, touch]}}
        out_of_order = {'name': 'bad', 'kind': 'event_order', 'params': {'events': [touch, fly]}}
        self.assertTrue(self.check(traj, in_order)['passed'])
        self.assertFalse(self.check(traj, out_of_order)['passed'])

    def test_knees_are_valid_bodies_and_bad_signals_are_rejected(self):
        validate_invariants([{'name': 'knees', 'kind': 'contact_alternation',
                              'params': {'ee_a': 'left_knee', 'ee_b': 'right_knee'}}], self.spec)
        validate_invariants([{'name': 'legs_over', 'kind': 'signal_window',
                              'params': {'signal': 'above:left_foot:pelvis', 'min': 0.2}}], self.spec)
        for params in ({'signal': 'above:left_foot:tail', 'min': 0.2}, {'signal': 'wobble', 'min': 1},
                       {'signal': 'upright', 'window': 'later', 'min': 0.9}, {'signal': 'upright'}):
            with self.assertRaises(ValueError):
                validate_invariants([{'name': 'x', 'kind': 'signal_window', 'params': params}], self.spec)

    def test_semantic_quartile_beats_a_cleaner_empty_motion_unless_physics_is_gross(self):
        def res(name, category, passed, **metrics):
            return dict(name=name, category=category, passed=passed, skipped=False, metrics=metrics)
        goals = lambda n_pass: [res(f'fidelity:goal/g{i}', 'fidelity', i < n_pass) for i in range(10)]
        empty = {'tests': goals(0)}
        crawling = {'tests': goals(5) + [res('contact_grounded', 'discrete', False, contact_gap_rms=0.03)]}
        self.assertGreater(candidate_rank(crawling), candidate_rank(empty))
        wrecked = {'tests': goals(10) + [res('surface_penetration', 'discrete', False, max_penetration=0.4)]}
        self.assertGreater(candidate_rank(empty), candidate_rank(wrecked))


class ScriptedAgent:
    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, messages):
        return AIMessage(self.replies.pop(0), response_metadata={'usage': {'input_tokens': 1, 'output_tokens': 1}})


class GoalContractRobustnessTest(unittest.TestCase):
    def run_node(self, replies):
        cfg = load_yaml_config(str(ROOT / 'agent/cfg/config.yaml'))
        tmp = tempfile.mkdtemp()
        cfg['paths']['skill_output_dir'] = tmp
        cfg['goal_contract'] = {'max_attempts': 2}
        node = make_goal_contract_node(ScriptedAgent(replies), {'system': 's', 'human': '{USER_SKILL}'}, cfg)
        return node({'skill_desc': {'user_request': 'turn around once'}, 'skill_name': 'T', 'token_usage': {}})

    contract = json.dumps({'environment': 'flat', 'requirements': [{'id': 'r1', 'text': 'turn around once'}],
                           'invariants': [{'name': 'turn', 'kind': 'net_progress', 'params': {'axis': 'yaw', 'min': 6.0},
                                           'requirement_ids': ['r1']}], 'sliding_contacts': []})

    def test_accepted_with_notes_is_accepted(self):
        out = self.run_node([self.contract, json.dumps({'accepted': True, 'issues': ['minor wording']})])
        self.assertTrue(out['goal_review_accepted'])
        self.assertIn('goal_contract', out)

    def test_never_accepted_falls_back_to_the_valid_contract(self):
        reject = json.dumps({'accepted': False, 'issues': ['x']})
        out = self.run_node([self.contract, reject, self.contract, reject])
        self.assertFalse(out['goal_review_accepted'])
        self.assertIn('goal_contract', out)

    def test_no_valid_contract_does_not_crash_generation(self):
        out = self.run_node(['not json', '{"environment": "moon"}'])
        self.assertNotIn('goal_contract', out)
        self.assertEqual(out['outcome'], 'in_progress')


class CodeIntakeTest(unittest.TestCase):
    def state(self, message):
        return {'messages': [message], 'skill_name': 'T', 'skill_version': 'v', 'num_critic_iters': 0,
                'rewrite_mode': '', 'skill_desc': {}}

    def test_truncated_or_cut_code_is_never_run(self):
        cfg = load_yaml_config(str(ROOT / 'agent/cfg/config.yaml'))
        node = make_code_executer_node(cfg)
        out = node(self.state(AIMessage('class A: pass', response_metadata={'truncated': True})))
        self.assertFalse(out['success'])
        self.assertIn('CUT OFF', out['messages'][0].content)
        broken = "```python\nclass A:\n    def f(self):\n        return 1\n    def contacts(self, phase):\n        a = p[\"x%d\" % (\n```"
        out = node(self.state(AIMessage(broken)))
        self.assertFalse(out['success'])
        self.assertIn('NOT VALID PYTHON', out['messages'][0].content)

    def test_trailing_prose_is_still_salvaged(self):
        code, dropped = extract_python_code_with_report(
            "```python\nclass A:\n    pass\nThis version fixes the knees.\n```")
        self.assertEqual(code.strip(), "class A:\n    pass")
        self.assertEqual(dropped_code_lines(dropped), [])


if __name__ == '__main__':
    unittest.main()
