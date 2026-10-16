"""Offline regressions: python -m unittest discover -s agent/tests -v."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import get_buffer_string, AIMessage
from agents.code_gen.code_executer_agent import make_finalize_node, write_to_file, render_skill_gif
from agents.code_gen.critic_coder_agent import make_critic_coder_node
from agents.code_gen.motion_diagnosis_agent import make_motion_diagnosis_agent_node
from utils.plan_io import load_plan_dict
from utils.repair_context import repair_prompt
from utils.semantic_contract import digest


class RepairContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = {"paths": {"skill_output_dir": self.temp.name},
                    "robot": {"name": "humanoid"}}
        self.folder = Path(self.temp.name) / "humanoid" / "SKILL_TEST"
        self.folder.mkdir(parents=True)
        self.plan = {"environment": "ledge", "repetitions": 1,
                     "fidelity_invariants": [{"kind": "net_progress", "params": {"min": .3}}]}
        self.environment = {"name": "ledge", "scene": "assets/scenes/slab.xml", "grounding": "off"}
        (self.folder / "motion_plan_0001.json").write_text(json.dumps(self.plan))
        self.state = {"skill_name": "SKILL_TEST", "skill_version": "test",
                      "environment": self.environment, "success": True,
                      "num_critic_iters": 1, "rewrite_mode": "structural",
                      "skill_desc": {"description": "climb the ledge"}, "token_usage": {},
                      "messages": [AIMessage("OLD HISTORY SHOULD NOT BE RESENT")],
                      "evaluation_feedback": {"motion_readout": "missed the ledge"},
                      "last_diagnosis": "move the base forward", "last_prompt": "winning prompt"}
        goal = {'environment': 'ledge', 'user_request': 'climb the ledge',
                'requirements': [{'id': 'climb', 'text': 'climb the ledge'}],
                'invariants': [{'name': 'rise', 'kind': 'net_progress', 'params': {'axis': 'z', 'min': .3},
                                'requirement_ids': ['climb']}]}
        self.state.update(goal_contract=goal, goal_contract_id=digest(goal))
        (self.folder / 'SKILL_TEST.pkl').write_bytes(b'fake trajectory for mocked evaluator')
        result = write_to_file(self.state, "x = 1\n", self.cfg)
        self.assertTrue(result["success"])
        self.best = Path(result["file_path"])
        self.state.update(current_code_path=str(self.best), best_code_path=str(self.best))

    def replan(self):
        (self.folder / "motion_plan_0002.json").write_text(json.dumps({"environment": "flat", "repetitions": 6}))
        self.state["environment"] = {"name": "flat", "grounding": "contact"}
        result = write_to_file(self.state, "x = 2\n", self.cfg)
        return Path(result["file_path"])

    def test_winning_contract_is_restored_before_replay(self):
        self.replan()
        def replay(*args, **kwargs):
            self.assertEqual(kwargs["environment"], self.environment)
            self.assertEqual(load_plan_dict(str(self.folder)), self.plan)
            latest = max(self.folder.glob("SKILL_TEST_[0-9]*.py"))
            self.assertEqual(latest.read_text(), "x = 1\n")
            return {"success": True}
        with patch("agents.code_gen.code_executer_agent.run_skill", side_effect=replay), \
             patch("agents.code_gen.code_executer_agent.render_skill_gif"), \
             patch("agents.code_gen.code_executer_agent.evaluate_candidate", return_value=({'summary': {'verified_complete': True}}, {})):
            result = make_finalize_node(self.cfg)(self.state)
        self.assertTrue(result["success"])
        self.assertEqual(json.loads((self.folder / "SKILL_TEST_plan.json").read_text()), self.plan)
        self.assertEqual((self.folder / "SKILL_TEST.py").read_text(), "x = 1\n")
        self.assertEqual(json.loads((self.folder / "SKILL_TEST_context.json").read_text())["environment"], self.environment)
        self.assertIn("winning prompt", (self.folder / "SKILL_TEST_prompt.md").read_text())
        self.assertFalse(list(self.folder.glob("motion_plan_*.json")))

    def test_failed_replay_preserves_attempts_and_does_not_publish(self):
        latest = self.replan()
        with patch("agents.code_gen.code_executer_agent.run_skill", return_value={"success": False}), \
             patch("agents.code_gen.code_executer_agent.render_skill_gif") as render:
            result = make_finalize_node(self.cfg)(self.state)
        self.assertFalse(result["success"])
        self.assertTrue(self.best.exists())
        self.assertTrue(latest.exists())
        self.assertFalse((self.folder / "SKILL_TEST.py").exists())
        render.assert_not_called()

    def test_missing_provenance_is_not_guessed(self):
        self.best.with_name(self.best.stem + "_context.json").unlink()
        with patch("agents.code_gen.code_executer_agent.run_skill") as replay:
            result = make_finalize_node(self.cfg)(self.state)
        self.assertFalse(result["success"])
        replay.assert_not_called()

    def test_successful_replay_with_failed_goal_is_not_completed(self):
        with patch("agents.code_gen.code_executer_agent.run_skill", return_value={"success": True}), \
             patch("agents.code_gen.code_executer_agent.render_skill_gif"), \
             patch("agents.code_gen.code_executer_agent.evaluate_candidate", return_value=({'summary': {'verified_complete': False}}, {})):
            result = make_finalize_node(self.cfg)(self.state)
        self.assertFalse(result['success'])
        self.assertEqual(result['outcome'], 'best_effort')
        self.assertTrue(self.best.exists())
        self.assertFalse((self.folder / 'SKILL_TEST.py').exists())
        report = json.loads((self.folder / 'SKILL_TEST_report.json').read_text())
        self.assertEqual(report['outcome'], 'best_effort')
        self.assertEqual(report['artifacts']['goal_contract_id'], self.state['goal_contract_id'])

    def test_goal_from_different_run_cannot_be_replayed(self):
        self.state['goal_contract']['user_request'] = 'walk instead'
        self.state['goal_contract_id'] = digest(self.state['goal_contract'])
        with patch("agents.code_gen.code_executer_agent.run_skill") as replay:
            result = make_finalize_node(self.cfg)(self.state)
        self.assertFalse(result['success'])
        replay.assert_not_called()

    def test_preview_uses_full_plan_repetitions_and_headless_backend(self):
        self.plan['repetitions'] = 4
        (self.folder / 'motion_plan_0001.json').write_text(json.dumps(self.plan))
        cfg = dict(self.cfg, simulator={'python_bin': 'python'})
        cfg['paths'] = dict(self.cfg['paths'], simulator_dir=str(Path(__file__).resolve().parents[2]))
        with patch.dict('os.environ', {'MUJOCO_GL': 'egl'}), \
             patch('agents.code_gen.code_executer_agent.subprocess.run') as run:
            run.return_value.returncode = 0
            render_skill_gif('SKILL_TEST', self.environment, cfg)
        args, kwargs = run.call_args
        command = args[0]
        self.assertEqual(command[command.index('--cycles') + 1], '4')
        self.assertEqual(command[command.index('--trajectory') + 1],
                         str(self.folder / 'SKILL_TEST.pkl'))
        self.assertEqual(kwargs['env']['MUJOCO_GL'], 'egl')

    def test_real_templates_receive_current_evidence(self):
        class Capture:
            def generate(self, prompt):
                # Nodes pass [SystemMessage, HumanMessage] so the system prompt can be cached.
                self.prompt = prompt if isinstance(prompt, str) else get_buffer_string(prompt)
                return AIMessage("diagnosis", response_metadata={"usage": {"input_tokens": 1, "output_tokens": 1}})
        for name, factory in [("critic_coder", make_critic_coder_node),
                              ("motion_failure_diagnosis", make_motion_diagnosis_agent_node)]:
            text = (Path(__file__).resolve().parents[1] / "prompts" / "templates" / (name + ".txt")).read_text()
            _, text = text.split("=== SYSTEM ===")
            system, human = text.split("=== HUMAN ===")
            capture = Capture()
            factory(capture, {"system": system, "human": human}, self.cfg)(self.state)
            self.assertIn("x = 1", capture.prompt)
            self.assertIn("missed the ledge", capture.prompt)
            self.assertIn("climb the ledge", capture.prompt)
            self.assertIn("fidelity_invariants", capture.prompt)
            self.assertNotIn("OLD HISTORY SHOULD NOT BE RESENT", capture.prompt)
            self.assertNotRegex(capture.prompt, r"\{(?:CURRENT_SKILL_CODE|SKILL_DESCRIPTION_JSON|UNIT_TEST_FEEDBACK_JSON|EVALUATION_FEEDBACK_JSON|MOTION_PLAN_JSON|MOTION_FAILURE_DIAGNOSIS_FEEDBACK)\}")

    def test_runtime_failure_uses_current_error_instead_of_old_diagnosis(self):
        self.state.update(success=False, messages=[AIMessage("NameError: missing_helper")])
        prompt = repair_prompt("{UNIT_TEST_FEEDBACK_JSON}\n{MOTION_FAILURE_DIAGNOSIS_FEEDBACK}", self.state, self.cfg)
        self.assertIn("missing_helper", prompt)
        self.assertNotIn("missed the ledge", prompt)
        self.assertNotIn("move the base forward", prompt)

    def test_source_placeholders_are_literal_data(self):
        self.best.write_text('literal = "{SKILL_DESCRIPTION_JSON}"\n')
        prompt = repair_prompt("{CURRENT_SKILL_CODE}", self.state, self.cfg)
        self.assertIn('literal = "{SKILL_DESCRIPTION_JSON}"', prompt)


if __name__ == "__main__":
    unittest.main()
