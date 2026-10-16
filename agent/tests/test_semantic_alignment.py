"""Behavioral counterexamples for goal preservation and trajectory semantics."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))
from utils import test_motion_quality as q
from utils.semantic_contract import validate_contract, validate_invariants, goal_invariants, digest
from utils.robot_spec import load_robot_spec
from agents.code_gen.run_unit_test import candidate_rank, evaluate_candidate
from agents.code_gen.code_executer_agent import run_skill
from agents.code_gen.goal_contract_agent import make_goal_contract_node
from agents.code_gen.motion_planner_agent import make_motion_plan_node
from agents.skill_gen.skill_formatter import prepare_skill
from prompts.prompt_loader import load_yaml_config, load_prompt
from pipeline import build_graph


def contract():
    return {'environment': 'flat', 'requirements': [{'id': 'spin', 'text': 'Turn one full rotation'}],
            'invariants': [{'name': 'rotation', 'kind': 'net_progress',
                            'params': {'axis': 'yaw', 'min': 6., 'max': 6.6}, 'requirement_ids': ['spin']}],
            'sliding_contacts': [], 'user_request': 'Turn one full rotation'}


class SemanticAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_yaml_config(str(ROOT / 'agent/cfg/config.yaml'))
        cls.spec = load_robot_spec(cls.cfg)

    def trajectory(self, frames=401):
        n = len(self.spec.ee_names)
        return {'root_pos': np.tile([0., 0., 1.], (frames, 1)),
                'root_quat': np.tile([1., 0., 0., 0.], (frames, 1)),
                'joints': np.zeros((frames, len(self.spec.joint_names))), 'ee_names': self.spec.ee_names,
                'ee_positions_achieved': np.zeros((frames, n, 3)),
                'contact_mask': np.ones((frames, n), bool), 'surface_clearance': np.zeros((frames, n))}

    def test_all_original_counterexamples_are_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'audit.json'
            subprocess.run([sys.executable, str(ROOT / 'agent/tools/audit_degeneracies.py'), '--out', str(output)],
                           check=True, capture_output=True, text=True)
            evidence = json.loads(output.read_text())
            self.assertEqual(evidence['gaps_reproduced'], 0, evidence)

    def test_true_ballistic_com_passes_while_root_moves_during_tuck(self):
        traj = self.trajectory()
        t = np.arange(401) * .01
        traj['contact_mask'][:] = False
        traj['surface_clearance'][:] = 10
        traj['com_pos'] = np.column_stack([2 * t, -t, 30 + 5 * t - .5 * 9.81 * t*t])
        traj['root_pos'] = traj['com_pos'] + np.column_stack([.1*np.sin(10*t), 0*t, .2*np.cos(8*t)])
        self.assertTrue(q.eval_support_or_ballistic(traj, self.spec, .01)['passed'])
        traj['com_pos'][:, 0] += 10*t*t
        self.assertFalse(q.eval_support_or_ballistic(traj, self.spec, .01)['passed'])

    def test_dynamics_are_advisory_for_kinematic_references(self):
        traj = self.trajectory()
        traj['contact_mask'][:] = False
        traj['surface_clearance'][:] = 10.
        feedback = q.evaluate_trajectory(traj, self.spec, .01)
        support = next(t for t in feedback['tests'] if t['name'] == 'support_or_ballistic')
        self.assertFalse(support['passed'])
        self.assertTrue(support['advisory'])
        without = dict(feedback, tests=[t for t in feedback['tests'] if not t.get('advisory')])
        self.assertEqual(candidate_rank(feedback), candidate_rank(without))
        strict = q.evaluate_trajectory(traj, self.spec, .01, require_dynamics=True)
        self.assertFalse(strict['summary']['passed_all_discrete'])
        self.assertEqual(strict['summary']['validation_mode'], 'dynamic')

    def test_sliding_does_not_authorize_floating_or_penetrating(self):
        for gap in [.25, -.25]:
            traj = self.trajectory()
            traj['surface_clearance'][:, 0] = gap
            report = q.eval_contact_grounded(traj, self.spec, .01,
                                            exempt_names=[self.spec.ee_names[0]])
            self.assertFalse(report['passed'])
            self.assertGreater(report['metrics']['contact_gap_rms'], 0.)

    def test_soft_contact_reward_resolves_small_geometric_improvements(self):
        traj = self.trajectory()
        traj['surface_clearance'][:] = .06
        worse = q.eval_contact_grounded(traj, self.spec, .01)
        traj['surface_clearance'][:] = .02
        better = q.eval_contact_grounded(traj, self.spec, .01)
        self.assertTrue(worse['passed'] and better['passed'])
        self.assertGreater(better['reward'], worse['reward'])

    def test_surface_relative_drift_supports_moving_contacts(self):
        traj = self.trajectory()
        traj['ee_positions_achieved'][:, :, 0] = np.arange(401)[:, None] * .005
        traj['contact_drift'] = np.zeros_like(traj['surface_clearance'])
        self.assertTrue(q.eval_contact_no_slide(traj, self.spec, .01)['passed'])
        traj['contact_drift'][:, 0] = .25
        self.assertFalse(q.eval_contact_no_slide(traj, self.spec, .01)['passed'])

    def test_signed_translation_with_only_upper_bound(self):
        for axis, column in [('x', 0), ('y', 1), ('z', 2)]:
            traj = self.trajectory()
            traj['root_pos'][:, column] -= np.linspace(0, .82, 401)
            params = {'axis': axis, 'max': -.55}
            self.assertTrue(q._inv_net_progress(traj, self.spec, .01, params)[0])
            # Explicit lower bounds still apply.
            self.assertFalse(q._inv_net_progress(
                traj, self.spec, .01, dict(params, min=-.7))[0])
            self.assertFalse(q._inv_net_progress(
                self.trajectory(), self.spec, .01, params)[0])
            traj['root_pos'][:, column] += np.linspace(0, 1.64, 401)
            self.assertFalse(q._inv_net_progress(traj, self.spec, .01, params)[0])

    def test_full_pitch_flip_in_initial_frame_is_not_yaw(self):
        traj = self.trajectory()
        angles = np.linspace(0, 2*np.pi, 401)
        heading = np.array([np.cos(.6), 0, 0, np.sin(.6)])
        traj['root_quat'] = np.array([q.quat_multiply(heading, [np.cos(a/2), 0, np.sin(a/2), 0]) for a in angles])
        self.assertTrue(q._inv_net_progress(traj, self.spec, .01, {'axis': 'pitch', 'min': 6, 'max': 6.5})[0])
        self.assertFalse(q._inv_net_progress(traj, self.spec, .01, {'axis': 'yaw', 'min': 6, 'max': 6.5})[0])
        traj['root_quat'][::2] *= -1
        self.assertTrue(q._inv_net_progress(traj, self.spec, .01, {'axis': 'pitch', 'min': 6, 'max': 6.5})[0])

    def test_surface_requires_dwell_and_correct_height(self):
        traj = self.trajectory()
        i = self.spec.ee_names.index('left_foot')
        offset = self.spec.ee_ground_offsets[i]
        traj['ee_positions_achieved'][:, i] = [1.5, 0, .5 + offset]
        params = {'ee': 'left_foot', 'x_range': [1, 2], 'y_range': [-.2, .2], 'surface_z': .5, 'min_dwell_s': .1}
        self.assertTrue(q._inv_contact_over_surface(traj, self.spec, .01, params)[0])
        traj['contact_mask'][:, i] = False
        traj['contact_mask'][10, i] = True
        self.assertFalse(q._inv_contact_over_surface(traj, self.spec, .01, params)[0])
        traj['contact_mask'][:, i] = True
        traj['ee_positions_achieved'][:, i, 2] = offset
        self.assertFalse(q._inv_contact_over_surface(traj, self.spec, .01, params)[0])

    def test_gait_cannot_stop_alternating_after_first_cycle(self):
        traj = self.trajectory()
        a, b = [self.spec.ee_names.index(x) for x in ('left_foot', 'right_foot')]
        phase = np.arange(401) % 100
        traj['contact_mask'][:, a] = (phase >= 20) & (phase < 60)
        traj['contact_mask'][:, b] = phase >= 70
        params = {'ee_a': 'left_foot', 'ee_b': 'right_foot', 'expected': .5, 'tol': .1}
        self.assertTrue(q._inv_phase_offset(traj, self.spec, .01, params)[0])
        traj['contact_mask'][160:290, b] = False
        self.assertFalse(q._inv_phase_offset(traj, self.spec, .01, params)[0])

    def test_alternation_allows_variable_timing_but_rejects_hops(self):
        traj = self.trajectory()
        a, b = [self.spec.ee_names.index(x) for x in ('left_foot', 'right_foot')]
        traj['contact_mask'][:, [a, b]] = False
        for start, end, limb in [(10, 80, a), (90, 115, b), (130, 180, a), (250, 350, b)]:
            traj['contact_mask'][start:end, limb] = True
        params = {'ee_a': 'left_foot', 'ee_b': 'right_foot', 'min_pairs': 2}
        self.assertTrue(q._inv_contact_alternation(traj, self.spec, .01, params)[0])
        traj['contact_mask'][:, b] = traj['contact_mask'][:, a]
        self.assertFalse(q._inv_contact_alternation(traj, self.spec, .01, params)[0])

    def test_nested_execution_plan_preserves_top_level_metadata(self):
        class Planner:
            def generate(self, prompt):
                return AIMessage(json.dumps({'environment': 'flat', 'duration_s': 2, 'repetitions': 2,
                                             'motion_plan': {'phase_definition': {}}}))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(self.cfg); cfg['paths']['skill_output_dir'] = tmp
            state = dict(skill_desc={}, skill_name='X', skill_version='test', success=True,
                         num_critic_iters=0, rewrite_mode='', token_usage={})
            result = make_motion_plan_node(Planner(), {'system': '', 'human': ''}, cfg)(state)
            self.assertEqual(result['environment']['name'], 'flat')
            saved = json.loads(next(Path(tmp).rglob('motion_plan_*.json')).read_text())
            self.assertEqual(saved['duration_s'], 2)
            self.assertIn('motion_plan', saved)

    def test_feasibility_retries_invalid_boolean_instead_of_crashing(self):
        from agents.code_gen.skill_filter import make_skill_filter_node
        class Filter:
            def __init__(self): self.calls = 0
            def generate(self, prompt):
                self.calls += 1
                return AIMessage('{"feasible": True}' if self.calls == 1 else '{"feasible": true}')
        agent = Filter()
        with patch.dict('os.environ', {'SKIP_SKILL_FILTER': ''}):
            route = make_skill_filter_node(agent, {'system': '', 'human': '{SKILL_DESCRIPTION_JSON}'})(
                {'skill_desc': {'user_request': 'walk'}})
        self.assertEqual(route, 'continue')
        self.assertEqual(agent.calls, 2)

    def test_capture_starts_at_phase_zero_and_excludes_final_wrap(self):
        import importlib.util
        from types import SimpleNamespace
        module_spec = importlib.util.spec_from_file_location('capture_timing', ROOT / 'simulator/core/recording.py')
        module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
        class Runner:
            frame = 0
            resets = 0
            motion_gen = SimpleNamespace(t=0.0)
            def reset_bridge(self): self.resets += 1
        runner = Runner()
        start, end = module.recording_window(200, 1.25, 2)
        recorded = []
        while runner.frame < end:
            module.prepare_recording_step(runner, start)
            sample = runner.motion_gen.t
            runner.motion_gen.t += .005
            runner.frame += 1
            if runner.frame > start: recorded.append(sample)
        self.assertEqual(runner.resets, 1)
        self.assertEqual(len(recorded), 320)
        self.assertEqual(recorded[0], 0.0)
        self.assertAlmostEqual(recorded[-1], 1.595)

    def test_contract_missing_coverage_or_invalid_check_is_rejected(self):
        for mutate in [lambda c: c['invariants'].clear(),
                       lambda c: c['invariants'][0]['params'].update(axis='twist'),
                       lambda c: c['invariants'][0]['params'].update(threshold=1),
                       lambda c: c['invariants'][0]['params'].update(min=float('nan'))]:
            c = contract(); mutate(c)
            with self.assertRaises(ValueError):
                validate_contract(c, self.spec)

    def test_mutation_is_rejected_and_unsupported_is_a_failure(self):
        c = contract(); state = {'goal_contract': c, 'goal_contract_id': digest(c)}
        self.assertEqual(len(goal_invariants(state)), 1)
        c['invariants'][0]['params']['min'] = 0
        with self.assertRaises(ValueError):
            goal_invariants(state)
        c = contract(); c['requirements'].append({'id': 'style', 'text': 'look joyful', 'unsupported_reason': 'style not measured'})
        state = {'goal_contract': c, 'goal_contract_id': digest(c)}
        result = q.eval_declared_invariants(self.trajectory(), self.spec, .01, goal_invariants(state))
        self.assertFalse(result[-1]['passed'])
        self.assertFalse(result[-1]['skipped'])

    def test_weaker_replan_cannot_turn_translation_into_a_successful_spin(self):
        import pickle
        c = contract()
        state = {'skill_name': 'X', 'goal_contract': c, 'goal_contract_id': digest(c)}
        traj = self.trajectory()
        traj['fps'] = 100
        traj['root_pos'][:, 0] = np.linspace(0, 6.3, len(traj['root_pos']))
        traj['surface_penetration'] = np.zeros(len(traj['root_pos']))
        traj['self_collision'] = np.zeros(len(traj['root_pos']))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(self.cfg); cfg['paths']['skill_output_dir'] = tmp
            folder = Path(tmp) / 'humanoid' / 'X'; folder.mkdir(parents=True)
            with (folder / 'X.pkl').open('wb') as f: pickle.dump(traj, f)
            plan = {'environment': 'flat', 'fidelity_invariants': [
                {'name': 'goal/rotation', 'kind': 'net_progress', 'params': {'axis': 'x', 'min': 5}}]}
            before, _ = evaluate_candidate(state, cfg, plan=plan)
            after, _ = evaluate_candidate(state, cfg, plan={'environment': 'flat', 'fidelity_invariants': []})
            self.assertFalse(before['summary']['goal_passed'])
            self.assertFalse(after['summary']['verified_complete'])
            self.assertEqual(before['summary']['num_goal_total'], 1)
            self.assertEqual(after['summary']['num_goal_total'], 1)
            self.assertEqual(candidate_rank(before), candidate_rank(after))

    def test_worse_physics_cannot_win_on_more_semantic_passes(self):
        def result(name, category, passed):
            return dict(name=name, category=category, passed=passed, skipped=False, metrics={})
        better = {'tests': [result('contact_no_slide', 'discrete', False), result('fidelity:goal/spin', 'fidelity', False)]}
        worse = {'tests': [result('contact_no_slide', 'discrete', False), result('surface_penetration', 'discrete', False),
                           result('fidelity:goal/spin', 'fidelity', True)]}
        self.assertGreater(candidate_rank(better), candidate_rank(worse))
        extra = copy.deepcopy(better)
        extra['tests'] += [result('fidelity:plan/easy' + str(i), 'fidelity', True) for i in range(100)]
        self.assertEqual(candidate_rank(better), candidate_rank(extra))

    def test_formatter_preserves_original_request(self):
        class Formatter:
            def generate(self, prompt):
                return AIMessage(json.dumps({'skill_id': 'X', 'skill_description': 'walk'}))
        skill = prepare_skill(Formatter(), {'system': '', 'human': '{USER_SKILL}'}, 'Spin, do not walk', 'X')
        self.assertEqual(skill['user_request'], 'Spin, do not walk')

    def test_timeout_is_a_repairable_execution_failure(self):
        with patch('agents.code_gen.code_executer_agent.subprocess.run', side_effect=subprocess.TimeoutExpired('sim', 1)):
            result = run_skill('X', 'test', self.cfg, {'name': 'flat'})
        self.assertFalse(result['success'])
        self.assertIn('time budget', result['stderr'])

    def test_shared_graph_has_goal_replan_and_finalization(self):
        class Dummy:
            def invoke(self, messages):
                raise AssertionError('Building a graph must not make API calls')
        graph = build_graph(Dummy(), self.cfg).get_graph()
        self.assertIn('goal_contract_agent', graph.nodes)
        self.assertIn('finalize_agent', graph.nodes)
        self.assertTrue(any(e.source == 'critic_router' and e.target == 'motion_planner_agent' for e in graph.edges))

    def test_contract_review_retries_incomplete_request_coverage(self):
        class Responses:
            def __init__(self): self.calls = 0
            def generate(self, prompt):
                self.calls += 1
                values = [contract(), {'accepted': False, 'issues': ['missing explicit condition']},
                          contract(), {'accepted': True, 'issues': []}]
                return AIMessage(json.dumps(values[self.calls-1]))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(self.cfg); cfg['paths']['skill_output_dir'] = tmp
            agent = Responses()
            result = make_goal_contract_node(agent, {'system': '', 'human': '{USER_SKILL}'}, cfg)(
                {'skill_desc': {'user_request': 'Turn one full rotation'}, 'skill_name': 'X'})
            self.assertEqual(agent.calls, 4)
            self.assertEqual(result['goal_contract_id'], digest(result['goal_contract']))

    def test_replan_cannot_change_scene_or_authorize_sliding(self):
        class Planner:
            def generate(self, prompt):
                return AIMessage(json.dumps({'environment': 'flat', 'duration_s': 3, 'repetitions': 1,
                                             'sliding_contacts': ['left_foot']}))
        c = contract()
        state = dict(skill_desc={'user_request': c['user_request']}, goal_contract=c,
                     goal_contract_id=digest(c), token_usage={}, stuck=True, skill_name='X')
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(self.cfg); cfg['paths']['skill_output_dir'] = tmp
            with self.assertRaises(RuntimeError):
                make_motion_plan_node(Planner(), {'system': '', 'human': ''}, cfg)(state)
            self.assertFalse(list(Path(tmp).rglob('motion_plan_*.json')))


if __name__ == '__main__':
    unittest.main()
