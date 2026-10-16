"""Regression tests for unconstrained twist, home references and task freedom."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


class IKPostureTest(unittest.TestCase):
    def run_script(self, source):
        result = subprocess.run([sys.executable, '-c', source], cwd=ROOT,
                                env=dict(os.environ, OPENBLAS_NUM_THREADS='1'),
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_small_joint_channel_handoff_is_blended(self):
        self.run_script('''
import sys
sys.path.insert(0, 'simulator')
from core.config import load_config
from core.simulator import MujocoSimulator
from core.controller import MotionControllerRunner
from utils.robot_config import load_robot_spec
sys.argv = ['test','--robot','g1','--mode','example','--grounding','off']
args = load_config(); spec = load_robot_spec('g1')
sim = MujocoSimulator(spec.scene_xml,args.init_position,200,base_body=spec.base_body,
                      headless=True,robot_mjcf=spec.mjcf)
runner = MotionControllerRunner(sim,args)
runner.motion_gen.compute_joint_overrides = lambda phase: {}
for _ in range(20): runner.step()
name = 'left_elbow_joint'; index = sim.model.joint(name).qposadr[0]
before = float(sim.data.qpos[index]); target = before + .17
runner.motion_gen.compute_joint_overrides = lambda phase: {name:target}
runner.step()
assert abs(sim.data.qpos[index]-before) < .02, (before,sim.data.qpos[index])
for _ in range(20): runner.step()
assert abs(sim.data.qpos[index]-target) < .005
sim.close()
''')

    def test_redundant_twist_uses_home_or_intent_without_moving_free_branch(self):
        self.run_script('''
import sys, mujoco, numpy as np
sys.path.insert(0, 'simulator')
from utils.kinematics import MultiLinkGradientDescentIK
model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
<worldbody><body name="tip"><joint name="twist" axis="0 0 1" range="-3 3"/>
<geom size=".1"/><site name="tip_site"/></body>
<body name="free" pos="1 0 0"><joint name="free_twist" axis="0 0 1" range="-3 3"/>
<geom size=".1"/><site name="free_site"/></body></worldbody>
<keyframe><key name="home" qpos=".2 0"/></keyframe></mujoco>""")
data = mujoco.MjData(model)
data.qpos[:] = [1.2, .7]  # construct mid-motion, not at home
mujoco.mj_forward(model, data)
ik = MultiLinkGradientDescentIK(model, data, ['tip','free'], ['tip_site','free_site'],
                               ['twist','free_twist'], null_space_gain=.02)
assert ik.q0[0] == .2
target = ik.get_ee_positions().copy()
for _ in range(100):
    ik.calculate(target, active_mask=[True,False], ori_targets={0:('align',[0,0,1])})
assert abs(data.qpos[0] - .2) < .01, data.qpos
assert data.qpos[1] == .7, data.qpos
for _ in range(100):
    ik.calculate(target, active_mask=[True,False], ref_overrides={'twist':1.1},
                 ori_targets={0:('align',[0,0,1])})
assert abs(data.qpos[0] - 1.1) < .01, data.qpos
np.testing.assert_allclose(ik.get_ee_positions(), target, atol=1e-9)
# An explicit orientation remains stronger than the default home preference.
q = [np.cos(-.5/2),0,0,np.sin(-.5/2)]
for _ in range(100):
    ik.calculate(target,active_mask=[True,False],ori_targets={0:('quat',q)})
assert abs(data.qpos[0] + .5) < .01, data.qpos
''')

    def test_inferred_flat_contact_does_not_twist_a_tilted_pose(self):
        self.run_script('''
import sys, mujoco, numpy as np
sys.path.insert(0, 'simulator')
from utils.kinematics import MultiLinkGradientDescentIK
from utils.scene_compose import compose_model
from utils.robot_config import load_robot_spec
spec = load_robot_spec('g1'); model = compose_model(spec.mjcf,spec.scene_xml)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
mujoco.mj_forward(model,data)
ik = MultiLinkGradientDescentIK(model,data,spec.ik_bodies,spec.ik_sites,
                               spec.actuated_joint_names,null_space_gain=spec.null_space_gain)
pitch = np.deg2rad(70)
data.qpos[3:7] = [np.cos(pitch/2),0,np.sin(pitch/2),0]
reference = {}
for side in ('left','right'):
    for joint,value in [('hip_pitch',-1.8),('knee',1.9),('ankle_pitch',-.6),
                        ('hip_roll',0.),('hip_yaw',0.),('ankle_roll',0.)]:
        name = side+'_'+joint+'_joint'
        reference[name] = value
        data.qpos[model.joint(name).qposadr[0]] = value
mujoco.mj_forward(model,data)
target = ik.get_ee_positions().copy()
mask = np.array([n.endswith('_foot') for n in spec.ik_names])
orientation = {i:('align',[0,0,1]) for i in np.flatnonzero(mask)}
for _ in range(100):
    ik.calculate(target,active_mask=mask,ref_overrides=reference,ori_targets=orientation)
for side in ('left','right'):
    for joint in ('hip_roll','hip_yaw'):
        value = data.qpos[model.joint(side+'_'+joint+'_joint').qposadr[0]]
        assert abs(value) < .1, (side,joint,value)
assert np.max(np.linalg.norm(ik.get_ee_positions()[mask]-target[mask],axis=1)) < .005
''')

    def test_alignment_jacobian_matches_residual_at_tilted_pose(self):
        self.run_script('''
import sys, mujoco, numpy as np
sys.path.insert(0, 'simulator')
from utils.kinematics import MultiLinkGradientDescentIK
from utils.scene_compose import compose_model
from utils.robot_config import load_robot_spec
spec = load_robot_spec('g1')
model = compose_model(spec.mjcf, spec.scene_xml)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
mujoco.mj_forward(model,data)
ik = MultiLinkGradientDescentIK(model,data,spec.ik_bodies,spec.ik_sites,
                               spec.actuated_joint_names,null_space_gain=.02)
data.qpos[model.joint('left_hip_roll_joint').qposadr[0]] = .25
data.qpos[model.joint('left_hip_yaw_joint').qposadr[0]] = .4
mujoco.mj_forward(model,data)
i = spec.ik_names.index('left_foot')
orientation = ('align', np.array([.1,.2,1.]))
jp = np.zeros((3,model.nv)); jr = jp.copy()
mujoco.mj_jacSite(model,data,jp,jr,ik.site_ids[i])
analytic = ik._ori_jacobian(i,orientation,jr[:,ik.dof_ids])
before = data.qpos.copy(); error = ik._ori_error(i,orientation)
numeric = np.zeros_like(analytic)
for k,dof in enumerate(ik.dof_ids):
    data.qpos[:] = before
    velocity = np.zeros(model.nv); velocity[dof] = 1e-6
    mujoco.mj_integratePos(model,data.qpos,velocity,1.)
    mujoco.mj_fwdPosition(model,data)
    numeric[:,k] = (error - ik._ori_error(i,orientation))/1e-6
np.testing.assert_allclose(analytic,numeric,atol=1e-6)
# Jacobian evaluation must not mutate the caller's surface normal.
np.testing.assert_array_equal(orientation[1], [.1,.2,1.])
''')


if __name__ == '__main__':
    unittest.main()
