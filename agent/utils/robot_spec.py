"""
Bridge to the simulator's robot spec loader.

`agent/utils` shadows the simulator's `utils` package, so we load
simulator/utils/robot_config.py by file path instead of importing it.
"""
import importlib.util
import os
import sys


def load_home_pose(cfg, spec):
    """Use the same compiled-model posture reader as the simulator.

    File imports avoid the agent/simulator `utils` package name collision.
    """
    root = cfg["paths"]["simulator_dir"]
    modules = {}
    for name in ("scene_compose", "home_pose"):
        path = os.path.join(root, "simulator", "utils", name + ".py")
        module_spec = importlib.util.spec_from_file_location("sim_" + name, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        modules[name] = module
    model = modules["scene_compose"].compose_model(
        os.path.join(root, spec.mjcf), os.path.join(root, spec.scene_xml))
    return modules["home_pose"].home_pose_from_model(model, spec)


def load_robot_spec(cfg):
    """Load the RobotSpec for cfg['robot']['name'] from the repo-root robots dir."""
    repo_root = cfg["paths"]["simulator_dir"]
    module_path = os.path.join(repo_root, "simulator", "utils", "robot_config.py")

    if "sim_robot_config" in sys.modules:
        module = sys.modules["sim_robot_config"]
    else:
        spec_ = importlib.util.spec_from_file_location("sim_robot_config", module_path)
        module = importlib.util.module_from_spec(spec_)
        sys.modules["sim_robot_config"] = module
        spec_.loader.exec_module(module)

    robots_dir = cfg["paths"].get("robots_dir", os.path.join(repo_root, "robots"))
    return module.load_robot_spec(cfg["robot"]["name"], robots_dir=robots_dir)
