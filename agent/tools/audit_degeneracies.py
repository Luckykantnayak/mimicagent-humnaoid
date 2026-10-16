"""Reproduce evaluator counterexamples without an LLM or simulator rollout.

Run from the repository root:
    python agent/tools/audit_degeneracies.py --out docs/framework_review_evidence.json

These are deliberately invalid or mis-scored inputs, not candidate skill data.
Each probe tests one evaluator in isolation; passing it does not imply that the
entire evaluator would accept the trajectory.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
from utils import test_motion_quality as quality
from utils.robot_spec import load_robot_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    spec = load_robot_spec({"paths": {"robots_dir": str(ROOT / "robots"), "simulator_dir": str(ROOT)},
                           "robot": {"name": "humanoid"}})
    names = list(spec.ee_names)
    left, right = names.index("left_foot"), names.index("right_foot")

    def trajectory(frames=101):
        return {"root_pos": np.tile([0., 0., 1.], (frames, 1)),
                "root_quat": np.tile([1., 0., 0., 0.], (frames, 1)),
                "joints": np.zeros((frames, len(spec.joint_names))),
                "ee_names": names,
                "ee_positions_achieved": np.zeros((frames, len(names), 3)),
                "contact_mask": np.ones((frames, len(names)), dtype=bool),
                "surface_clearance": np.zeros((frames, len(names)))}

    probes = []

    def record(name, expected, actual, detail):
        probes.append({"probe": name, "expected": expected, "actual": actual,
                       "exposes_gap": expected != actual, "detail": detail})

    traj = trajectory()
    t = np.linspace(0, 1, 101)
    traj["ee_positions_achieved"][:, left, 0] = .5 * np.sin(np.pi * t)
    result = quality.eval_contact_no_slide(traj, spec, .01)
    record("planted_foot_moves_50cm_and_returns", False, result["passed"], result["metrics"])

    traj = trajectory()
    traj["ee_positions_achieved"][:, right, 0] = .3 * t
    result = quality._inv_stance_no_drift(traj, spec, .01, {"ee": "left_foot", "max_drift": .1})
    record("stationary_left_foot_invariant_with_sliding_right_foot", True, result[0], result[3])

    traj = trajectory()
    traj["contact_mask"][:] = False
    traj["surface_clearance"][:] = 10
    traj["root_pos"][:, 2] = 100 - .5 * 74 * t ** 2
    result = quality.eval_support_or_ballistic(traj, spec, .01)
    record("flight_at_minus_74_m_per_s_squared", False, result["passed"], result["feedback"])

    traj = trajectory()
    traj["contact_mask"][:] = False
    traj["ee_positions_achieved"][:, left] = [1.5, 0, 5]
    result = quality._inv_contact_over_surface(traj, spec, .01,
        {"ee": "left_foot", "x_range": [1, 2], "y_range": [-.2, .2]})
    record("contact_over_surface_five_metres_in_air", False, result[0], result[3])

    traj = trajectory()
    traj["root_quat"][:, 0] = np.cos(6 * t)
    traj["root_quat"][:, 3] = np.sin(6 * t)
    result = quality._inv_net_progress(traj, spec, .01, {"axis": "yaw", "min": 10, "max": 15})
    record("genuine_12rad_yaw_spin", True, result[0], result[3])
    traj["root_quat"][:] = [1, 0, 0, 0]
    traj["root_pos"][:, 0] = 12 * t
    result = quality._inv_net_progress(traj, spec, .01, {"axis": "yaw", "min": 10, "max": 15})
    record("12metre_walk_passes_yaw_spin_invariant", False, result[0], result[3])

    traj = trajectory()
    traj["root_quat"][-1] = [0, 0, 0, 1]
    result = quality._inv_end_state(traj, spec, .01, {"mode": "returns"})
    record("return_pose_ignores_180deg_orientation_error", False, result[0], result[3])

    for cycles in (1, 4):
        traj = trajectory(100 * cycles + 1)
        phase = (np.arange(len(traj["root_pos"])) % 100) / 100
        traj["contact_mask"][:, left] = (phase >= .2) & (phase < .6)
        traj["contact_mask"][:, right] = (phase >= .7) & (phase < 1.)
        result = quality._inv_phase_offset(traj, spec, .01,
            {"ee_a": "left_foot", "ee_b": "right_foot", "expected": .5, "tol": .1})
        record(f"alternating_gait_{cycles}_cycles", True, result[0], result[3])

    result = quality.eval_declared_invariants(trajectory(), spec, .01,
        [{"kind": "not_a_real_primitive"}, {"kind": "base_height_band", "params": {}}])
    counted = [r for r in result if r["category"] == "fidelity" and not r["skipped"]]
    record("malformed_invariants_cannot_disappear_from_success_denominator", 2, len(counted),
           [{"passed": r["passed"], "skipped": r["skipped"], "feedback": r["feedback"]} for r in result])

    evidence = {"scope": "isolated evaluator counterexamples; no full-pipeline success claim",
                "probes": probes, "gaps_reproduced": sum(p["exposes_gap"] for p in probes)}
    text = json.dumps(evidence, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
