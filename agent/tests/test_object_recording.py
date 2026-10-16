"""Scene objects remain observable before, during and after sparse commands."""
from pathlib import Path
import subprocess
import sys
import unittest


class ObjectRecordingTest(unittest.TestCase):
    def test_sparse_commands_produce_complete_achieved_tracks(self):
        root = Path(__file__).resolve().parents[2]
        script = '''
import sys
import numpy as np
sys.path.insert(0, 'simulator')
from core.config import load_config
from core.simulator import MujocoSimulator
from core.controller import MotionControllerRunner
from core.recorder import MimicKitRecorder
from utils.robot_config import load_robot_spec
sys.argv = ['test', '--robot', 'humanoid', '--mode', 'example', '--grounding', 'off']
args = load_config()
spec = load_robot_spec('humanoid')
sim = MujocoSimulator('assets/scenes/carry.xml', args.init_position, 200,
                      base_body=spec.base_body, headless=True, robot_mjcf=spec.mjcf)
runner = MotionControllerRunner(sim, args)
recorder = MimicKitRecorder(200)
commands = [{}, {'box': ([1.11, 0, .12], [1, 0, 0, 0])}, {}]
for command in commands:
    runner.motion_gen.compute_object_poses = lambda phase: command
    state = runner.step()
    np.testing.assert_allclose(state['object_poses']['box'][0], sim.data.mocap_pos[runner._mocap_ids['box']])
    recorder.record(state)
positions = np.asarray(recorder.object_pos['box'])
assert positions.shape == (len(recorder.root_pos), 3) == (3, 3)
np.testing.assert_allclose(positions[0], [1.10, 0, .12])
np.testing.assert_allclose(positions[1:], [[1.11, 0, .12], [1.11, 0, .12]])
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=root,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
