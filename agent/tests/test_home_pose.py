"""Home posture must agree across prompts, model FK, and generated-code access."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


class HomePoseTest(unittest.TestCase):
    def run_script(self, script):
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'),
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_home_bias_still_allows_contact_driven_bending(self):
        self.run_script('''
import sys, numpy as np, mujoco
sys.path.insert(0, 'simulator')
from utils.scene_compose import compose_model
from utils.robot_config import load_robot_spec
from utils.home_pose import home_pose_from_model
from utils.kinematics import MultiLinkGradientDescentIK
for robot in ('humanoid', 'g1'):
    spec = load_robot_spec(robot)
    model = compose_model(spec.mjcf, spec.scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    home = home_pose_from_model(model, spec)
    ik = MultiLinkGradientDescentIK(model, data, spec.ik_bodies, spec.ik_sites,
                                   spec.actuated_joint_names, null_space_gain=spec.null_space_gain)
    targets = ik.get_ee_positions().copy()
    mask = np.array([name in ('left_foot', 'right_foot') for name in spec.ik_names])
    data.qpos[2] -= .15
    for _ in range(5):
        ik.calculate(targets, active_mask=mask, ref_overrides=home['joints'])
    assert np.max(np.linalg.norm(ik.get_ee_positions()[mask] - targets[mask], axis=1)) < .005
    assert abs(data.qpos[2] - (home['root_pos'][2] - .15)) < 1e-8
''')

    def test_both_robot_prompts_use_model_home(self):
        self.run_script('''
import json, sys
import numpy as np
sys.path.insert(0, 'agent')
from prompts.prompt_loader import load_yaml_config, load_prompt
from utils.robot_spec import load_robot_spec, load_home_pose
cfg = load_yaml_config('agent/cfg/config.yaml')
for robot, count in [('humanoid', 28), ('g1', 29)]:
    cfg['robot']['name'] = robot
    spec = load_robot_spec(cfg)
    home = load_home_pose(cfg, spec)
    assert len(home['joints']) == count
    assert len(home['ee_positions_body']) == len(spec.ik_names)
    assert np.isfinite(list(home['joints'].values())).all()
    for name in ('motion_planner', 'coder', 'critic_coder', 'motion_failure_diagnosis'):
        prompt = load_prompt(name, cfg)['system']
        assert json.dumps(home, indent=2) in prompt
        assert 'home_joint_pose' in prompt
        assert '{home_pose}' not in prompt
    if robot == 'g1':
        assert home['joints']['left_knee_joint'] == .3
        assert home['joints']['left_hip_pitch_joint'] == -.1
''')

    def test_runtime_copy_and_intentional_non_home_pose(self):
        self.run_script('''
import sys
import numpy as np
sys.path.insert(0, 'simulator')
from core.config import load_config
from core.simulator import MujocoSimulator
from core.controller import MotionControllerRunner
from utils.robot_config import load_robot_spec
for robot in ('g1', 'humanoid'):
    sys.argv = ['test', '--robot', robot, '--mode', 'example', '--grounding', 'off']
    args = load_config()
    spec = load_robot_spec(robot)
    sim = MujocoSimulator(spec.scene_xml, args.init_position, 200,
                          base_body=spec.base_body, headless=True, robot_mjcf=spec.mjcf)
    runner = MotionControllerRunner(sim, args)
    gen = runner.motion_gen
    for name, point in gen.home_pose['ee_positions_body'].items():
        np.testing.assert_allclose(point, gen.init_ee_pos[name], atol=1e-8)
    if robot == 'g1':
        name = 'left_elbow_joint'
        original = gen.home_joint_pose([name])
        altered = gen.home_joint_pose([name])
        altered[name] = 1.1
        assert gen.home_joint_pose([name]) == original
        gen.compute_joint_overrides = lambda phase: altered.copy()
        for _ in range(5):
            runner.step()
        joint = sim.model.joint(name)
        assert abs(sim.data.qpos[joint.qposadr[0]] - 1.1) < .03
    sim.close()
''')

    def test_fk_frame_joint_order_and_invalid_home(self):
        self.run_script('''
import sys
from types import SimpleNamespace
import mujoco
import numpy as np
sys.path.insert(0, 'simulator')
from utils.home_pose import home_pose_from_model
xml = """<mujoco><compiler angle="radian"/><worldbody>
<body name="base"><freejoint/><geom size=".1"/>
<body name="limb" pos="0 0 -.3"><joint name="bend" axis="0 1 0" range="-2 2"/>
<geom size=".1"/><site name="tip" pos="0 0 -.2"/></body></body>
</worldbody><keyframe><key name="home" qpos="2 3 4 .7071067811865476 .7071067811865476 0 0 .4"/></keyframe></mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
spec = SimpleNamespace(name='test', mjcf='test.xml', base_body='base',
    actuated_joint_names=['bend'], ik_names=['end', 'body'],
    ik_bodies=['limb', 'limb'], ik_sites=['tip', 'missing'])
before = model.key_qpos.copy()
home = home_pose_from_model(model, spec)
assert home['joints'] == {'bend': .4}
np.testing.assert_allclose(home['ee_positions_body']['body'], [0, 0, -.3], atol=1e-8)
np.testing.assert_allclose(home['ee_positions_body']['end'],
    [-.2*np.sin(.4), 0, -.3-.2*np.cos(.4)], atol=1e-8)
np.testing.assert_array_equal(model.key_qpos, before)
model.key_qpos[0, -1] = 2.5
try:
    home_pose_from_model(model, spec)
except ValueError as error:
    assert 'invalid home angle' in str(error)
else:
    raise AssertionError('Invalid home should not reach prompts or skills')
missing = mujoco.MjModel.from_xml_string(xml.replace('name="home"', 'name="other"'))
try:
    home_pose_from_model(missing, spec)
except ValueError as error:
    assert "named 'home'" in str(error)
else:
    raise AssertionError('Do not silently substitute a zero pose')
''')
