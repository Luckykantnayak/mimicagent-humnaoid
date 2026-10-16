"""G1 model registration and visual/collision geometry separation."""
from pathlib import Path
import os
import subprocess
import sys
import unittest


class G1IntegrationTest(unittest.TestCase):
    def test_home_pose_and_collision_surfaces(self):
        script = '''
import sys
import numpy as np
import mujoco
sys.path.insert(0, 'simulator')
from core.config import load_config
from core.simulator import MujocoSimulator
from core.controller import MotionControllerRunner
from utils.robot_config import load_robot_spec
sys.argv = ['test', '--robot', 'g1', '--mode', 'example', '--grounding', 'off']
args = load_config()
spec = load_robot_spec('g1')
sim = MujocoSimulator(spec.scene_xml, args.init_position, 200,
                      base_body=spec.base_body, headless=True, robot_mjcf=spec.mjcf)
runner = MotionControllerRunner(sim, args)
m = sim.model
assert m.nq == 36 and m.nv == 35 and m.nu == 29
assert len(runner.motion_gen.ee_names) == 15
for g, h in runner._self_pairs:
    assert m.geom_contype[g] or m.geom_conaffinity[g]
    assert m.geom_contype[h] or m.geom_conaffinity[h]
# Both shin collision meshes must still be checked against one another.
shins = {m.body('left_knee_link').id, m.body('right_knee_link').id}
assert any({int(m.geom_bodyid[g]), int(m.geom_bodyid[h])} == shins
           for g, h in runner._self_pairs)
for _ in range(5):
    state = runner.step()
    assert state['self_collision'] < .006, state['collision_pair']
    assert np.isfinite(state['root_pos']).all()
sim.close()
'''
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, '-c', script], cwd=root,
                                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
