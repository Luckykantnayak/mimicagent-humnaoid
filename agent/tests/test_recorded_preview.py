"""Previews must preserve evaluated poses instead of solving IK a second time."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class RecordedPreviewTest(unittest.TestCase):
    def run_script(self, script):
        result = subprocess.run(
            [sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[2],
            env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_exact_recorded_pose_and_object_orientation(self):
        self.run_script('''
import sys, tempfile, pickle
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, 'simulator')
from tools.record_video import apply_recorded_frame, recorded_frame_indices
from core.recorder import MimicKitRecorder
xml = """<mujoco><worldbody>
<body name="base"><freejoint/><geom size=".1"/>
<body name="limb"><joint name="bend"/><geom size=".05"/></body></body>
<body name="object" mocap="true"><geom size=".1 .2 .3" type="box"/></body>
</worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
q = np.array([.9238795325, 0., 0., .3826834324]); q /= np.linalg.norm(q)
trajectory = dict(root_pos=np.array([[1., 2., 3.]]), root_quat=q[None],
    joints=np.array([[.71]]), object_pos={'object': np.array([[4., 5., 6.]])},
    object_quat={'object': q[None]})
apply_recorded_frame(m, d, trajectory, 0)
np.testing.assert_array_equal(d.qpos, np.r_[trajectory['root_pos'][0], q, .71])
np.testing.assert_array_equal(d.mocap_pos[0], [4., 5., 6.])
np.testing.assert_array_equal(d.mocap_quat[0], q)
rec = MimicKitRecorder(200, robot='test', ee_names=['limb'])
rec.record(dict(root_pos=trajectory['root_pos'][0], root_quat=q,
    joints=np.array([.71]), ee_positions_world=np.zeros((1, 3)),
    object_poses={'object': (trajectory['object_pos']['object'][0], q)}))
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'motion.pkl'; rec.save(path)
    saved=pickle.loads(path.read_bytes())
    np.testing.assert_array_equal(saved['object_quat']['object'][0], q)
indices=recorded_frame_indices(3200, 200, 33)
assert len(indices) == 528 and indices[-1] < 3200
np.testing.assert_allclose(indices / 200, np.arange(528) / 33, atol=1/200)
assert len(recorded_frame_indices(10, 20, 40)) == 20
''')

    def test_unknown_keyframe_cannot_silently_zero_robot(self):
        self.run_script('''
import sys
sys.path.insert(0, 'simulator')
from core.simulator import MujocoSimulator
from utils.robot_config import load_robot_spec
from utils.file_utils import load_yaml
spec=load_robot_spec('g1')
key=load_yaml('simulator/config.yaml')['run']['init_position']
sim=MujocoSimulator(spec.scene_xml, key, 200, headless=True, robot_mjcf=spec.mjcf)
assert abs(sim.data.qpos[sim.model.joint('left_knee_joint').qposadr[0]] - .3) < 1e-8
try:
    sim._reset('0 0 0')
except ValueError as exc:
    assert 'keyframe' in str(exc)
else:
    raise AssertionError('Invalid keyframe silently reset the robot')
sim.close()
''')

    def test_readout_exposes_twist_without_velocity_violation(self):
        self.run_script('''
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, 'agent')
from utils.trajectory_readout import _joint_posture_readout
spec=SimpleNamespace(joint_names=['swivel', 'wheel'], passive_joint_indices=[1])
text=str(_joint_posture_readout(np.tile([2.5, 0.], (100, 1)), spec))
assert 'swivel: +2.50 / +2.50 / +2.50' in text
assert 'wheel:' not in text
assert 'within limits does not imply' in text
''')
