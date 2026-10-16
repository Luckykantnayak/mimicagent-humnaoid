import json
import os

from agents.code_gen.state import AgentState
from utils.test_motion_quality import evaluate_trajectory, save_report
from utils.trajectory_readout import summarize_trajectory
from utils.plan_io import load_plan_dict, extract_invariants
from utils.robot_spec import load_robot_spec
from utils.semantic_contract import goal_invariants, digest

from langchain_core.messages import AIMessage

import pickle as pkl

# The universal PHYSICS FLOOR (always on; passing all of these is the primary term
# of the version-selection score). base_height_envelope was DEMOTED to an
# agent-declared fidelity invariant; self_collision is now SOFT (gross-only).
GENERIC_CORE = {
    "surface_penetration", "self_collision",
    "support_or_ballistic", "contact_grounded", "contact_no_slide", "joint_limits", "no_teleport",
    "ee_no_teleport", "joint_velocity_envelope", "framework_interventions",
    "ballistic_flight",
}

# magnitude of a core violation that counts as "fully bad" (normalized -> 1.0).
# Used only to rank versions by SEVERITY when none of them pass — so a nearly-fixed
# version (8 cm penetration) outranks a badly-broken one (45 cm) that fails the
# SAME number of checks. Keys are the metric names those evaluators emit.
_SEVERITY_SCALE = {
    "surface_penetration": ("max_penetration", 0.30),
    "self_collision":      ("max_overlap", 0.15),
    "contact_no_slide":    ("max_stance_drift", 0.50),
    "no_teleport":         ("worst_step", 1.5),
    "ee_no_teleport":      ("worst_step", 1.0),
    "joint_velocity_envelope": ("worst_ratio", 5.0),
    "framework_interventions": ("clamp_max", 0.45),
    "ballistic_flight": ("excess_launch_m_s", 10.0),
}


def _core_severity(tests):
    """Mean normalized violation magnitude over the FAILED generic-core checks,
    clamped to [0, 1). A continuous 'how far from passing' signal: two versions
    that fail the same COUNT of checks are separated by HOW BADLY they fail, so
    the repair loop's progress (e.g. 45 cm -> 8 cm penetration) is not thrown away.
    Capped below 1.0 so it only ever breaks ties WITHIN a whole-number reward tier."""
    sev = []
    for t in tests:
        if t["name"] not in GENERIC_CORE or t["passed"] or t.get("skipped") or t.get('advisory'):
            continue
        m = t.get("metrics", {})
        if t["name"] in _SEVERITY_SCALE:
            key, scale = _SEVERITY_SCALE[t["name"]]
            sev.append(min(1.0, float(m.get(key, scale)) / scale))
        elif t["name"] == "contact_grounded":
            # track the PASS criterion (float_frac > 0.30), not just the clearance —
            # else shaving depth looks like progress while the frac (what passes) is stuck.
            sev.append(min(1., float(m.get('contact_gap_rms', m.get('float_clearance', .3))) / .30))
        elif t["name"] == "support_or_ballistic":
            sev.append(min(1.0, 1.0 - float(m.get("rest_frac", 0.0))))
        elif t["name"] == "joint_limits":
            sev.append(min(1.0, len(m.get("violated_joints", [1])) / 5.0))
        else:
            sev.append(0.5)
    return min(0.99, sum(sev) / len(sev)) if sev else 0.0


def evaluate_candidate(state, cfg, plan=None, spec=None):
    """The same checks for a loop attempt and the exact final replay."""
    skill_dir = os.path.join(cfg['paths']['skill_output_dir'], cfg['robot']['name'], state['skill_name'])
    with open(os.path.join(skill_dir, state['skill_name'] + '.pkl'), 'rb') as f:
        traj = pkl.load(f)
    spec = spec or load_robot_spec(cfg)
    plan = plan if plan is not None else load_plan_dict(skill_dir, state['skill_name'])
    invariants, sliders, pivots = extract_invariants(plan)
    invariants = [dict(inv, name="plan/" + str(inv.get("name") or i)) if isinstance(inv, dict) else inv
                  for i, inv in enumerate(invariants)]
    goals = goal_invariants(state)
    if state.get('goal_contract'):
        sliders = state['goal_contract'].get('sliding_contacts', [])
    traj['cycle_duration_s'] = plan.get('duration_s') if plan.get('cyclic') else None
    feedback = evaluate_trajectory(traj, spec, 1.0 / traj['fps'],
                                   invariants=goals + invariants, sliding_contacts=sliders,
                                   pivot_contacts=pivots, cyclic=plan.get('cyclic'))
    feedback['goal_contract_id'] = state.get('goal_contract_id')
    goal_tests = [t for t in feedback['tests'] if t['name'].startswith('fidelity:goal/')]
    feedback['summary']['goal_passed'] = bool(goal_tests) and all(t['passed'] and not t['skipped'] for t in goal_tests)
    feedback['summary']['num_goal_passed'] = sum(t['passed'] for t in goal_tests)
    feedback['summary']['num_goal_total'] = len(goal_tests)
    required = {'root_pos', 'root_quat', 'joints', 'ee_positions_achieved', 'contact_mask',
                'surface_clearance', 'surface_penetration', 'self_collision'}
    feedback['missing_measurements'] = sorted(k for k in required if traj.get(k) is None)
    # The digest-locked goal contract is the verdict; planner-owned invariants are
    # supplementary (they drive repair via `fid_unmet` but cannot veto completion — a
    # single mis-declared plan band used to block `complete` regardless of the goal).
    feedback['summary']['verified_complete'] = (
        bool(state.get('goal_contract')) and feedback['summary']['goal_passed']
        and feedback['summary']['passed_all_discrete'] and not feedback['missing_measurements'])
    return feedback, traj


def candidate_rank(feedback):
    """Fixed task criteria remain comparable when supplemental plan checks change."""
    tests = feedback['tests']
    core_failures = [t for t in tests if t['name'] in GENERIC_CORE and not t['passed'] and not t.get('advisory')]
    discrete_failures = [t for t in tests if t['category'] == 'discrete' and not t['passed'] and not t['skipped']]
    goals = [t for t in tests if t['name'].startswith('fidelity:goal/')]
    fraction = sum(t['passed'] and not t['skipped'] for t in goals) / len(goals) if goals else 0.0
    contact_rms = next((float(t['metrics'].get('contact_gap_rms', 0.)) for t in tests
                        if t['name'] == 'contact_grounded'), 0.)
    # Neither plan-owned check counts nor their rewards can promote an easier plan.
    # Among candidates WITHOUT a gross physics failure, semantics dominate at a coarse
    # level: meeting a higher QUARTILE of the fixed goal outranks being merely cleaner (a
    # folded, stationary "crawl" with no physics failure must not beat a crawl that
    # crawls with one small residual). A grossly broken candidate never wins on
    # semantics. Within a tier: physics first, then exact goal coverage.
    severity = _core_severity(tests)
    gross = bool(core_failures) and severity >= 0.5
    quartile = 0 if gross else int(fraction * 4 + 1e-9)
    return (int(not gross), quartile, int(not core_failures), -len(core_failures),
            -severity, int(not discrete_failures), -len(discrete_failures), fraction, -contact_rms)


def make_unit_test_node(cfg):

    spec = load_robot_spec(cfg)

    def run_unit_test(state: AgentState) -> AgentState:
        skill_name = state["skill_name"]
        # All artifacts for a skill live in skill_output/<robot>/<skill>/
        skill_dir = os.path.join(
            cfg["paths"]["skill_output_dir"], cfg["robot"]["name"], skill_name
        )
        feedback, traj = evaluate_candidate(state, cfg, spec=spec)
        dt = 1.0 / traj['fps']
        summary = feedback["summary"]

        # PERCEPTION BEFORE VERDICT: a geometric readout of what the motion ACTUALLY
        # did (base path, per-limb contact timeline + lift + drift, gait phase
        # offsets, object path). Fed to diagnosis + critic so the loop reasons about
        # the MOTION, not just the pass/fail lamps — the fix for the blindness that
        # made the checks oscillate.
        readout = summarize_trajectory(traj, spec, dt)

        # The loop continues while EITHER the physics floor OR the fidelity layer
        # is unmet — fidelity must DRIVE repair, not merely rank versions. Without
        # the fidelity tier here, the loop exited the instant the floor was clean
        # (a carrybox that had picked the box up but let it drift from the hands
        # shipped at 4/6 because passed_all_discrete was already true).
        # Missing evaluator capabilities cannot be repaired by changing motion code.
        # They remain failed in the report and block completion, but do not cause
        # futile rewrites once all measurable criteria have been met.
        fid_unmet = any(t['category'] == 'fidelity' and not t['passed']
                        and not t.get('unverifiable') for t in feedback['tests'])
        if not summary["passed_all_discrete"]:
            rewrite_mode = "structural"
        elif fid_unmet:
            rewrite_mode = "fidelity"     # floor clean, but it doesn't yet LOOK like the skill
        else:
            rewrite_mode = "none"

        # Extract failures
        discrete_failures = [
            t for t in feedback["tests"]
            if t["category"] == "discrete"
               and not t["passed"]
               and not t["skipped"]
        ]
        fidelity_failures = [
            t for t in feedback["tests"]
            if t["category"] == "fidelity"
               and not t["passed"]
               and not t["skipped"]
        ]
        # Form feedback for the diagnosis + critic agents (they read state["messages"]).
        critic_payload = dict(
            rewrite_mode=rewrite_mode,
            motion_readout=readout,        # <-- what the robot ACTUALLY did

            summary=dict(
                passed_all=summary["passed_all"],
                passed_all_discrete=summary["passed_all_discrete"],
                total_reward=summary["total_reward"],
                fidelity=f"{summary.get('num_fidelity_passed',0)}/{summary.get('num_fidelity_total',0)} invariants met",
            ),

            physics_floor_failures=[
                {"name": t["name"], "metrics": t["metrics"], "feedback": t["feedback"]}
                for t in discrete_failures
            ],
            fidelity_failures=[
                {"name": t["name"], "metrics": t["metrics"], "feedback": t["feedback"]}
                for t in fidelity_failures
            ],
        )

        os.makedirs(skill_dir, exist_ok=True)
        save_report(feedback, skill_name="report", out_dir=skill_dir)

        # Keep the historical scalar for readable logs. Selection below uses a
        # lexicographic rank with fixed task criteria, independent of this value.
        generic_pass = all(
            t["passed"] for t in feedback["tests"] if t["name"] in GENERIC_CORE and not t.get('advisory')
        )
        fid_total = max(1, summary.get("num_fidelity_total", 0))
        fid_frac = summary.get("num_fidelity_passed", 0) / fid_total
        fid_weight = 40.0 if generic_pass else 2.0
        score = (summary["total_reward"]
                 + fid_weight * fid_frac
                 + (100.0 if summary["passed_all_discrete"] else 0.0)
                 + (1000.0 if generic_pass else 0.0)
                 - _core_severity(feedback["tests"]))
        rank = candidate_rank(feedback)
        prev_rank = state.get("best_rank")
        cur_path = state.get("current_code_path")
        improved = prev_rank is None or rank > tuple(prev_rank)
        if prev_rank is None or rank >= tuple(prev_rank):
            best_score, best_code_path, best_rank = score, cur_path, rank
        else:
            best_score, best_code_path, best_rank = state.get("best_score"), state.get("best_code_path"), prev_rank
        # No early stop-loss: with fidelity DRIVING the loop, a version that clears
        # the floor but scores below a prior best (e.g. a fidelity fix that briefly
        # regressed something) must still be allowed to keep experimenting toward
        # higher fidelity. Termination is handled by the hard iteration cap, the
        # one-shot replan when stuck, and finalize (which always re-instates the
        # best-scoring version) — so a late regression can never ship.

        # Stuck detector (for the re-plan path): count consecutive scored versions
        # that did NOT improve the score.
        no_improve = 0 if improved else (state.get("no_improve_count") or 0) + 1
        last_failures = ", ".join(
            t["name"] for t in discrete_failures + fidelity_failures
        )
        # PER-TEST stall: the OVERALL score can tick up (a severity change on an
        # unrelated check) while a SPECIFIC core test stays stuck failing — so
        # no_improve resets and the loop never replans, even though one check has been
        # dead for iterations (carrybox's object_grip, crawl's contact_grounded ×6). A
        # whole-SET signature never fires either (fidelity churn resets it every
        # version). Track per-CORE-test consecutive-fail STREAKS: a stall = any core
        # test failing >= 3 versions in a row.
        core_now = {t["name"] for t in discrete_failures if t["name"] in GENERIC_CORE}
        prev_streaks = state.get("core_fail_streaks") or {}
        core_fail_streaks = {n: prev_streaks.get(n, 0) + 1 for n in core_now}
        per_test_stall = any(v >= 3 for v in core_fail_streaks.values())
        # STUCK — the single flag BOTH the critic-router (replan decision) and the
        # planner (replan context) read, so they can never disagree: the overall score
        # plateaued for >=2 iters, OR a core test has been stuck failing for >=3.
        stuck = (no_improve >= 2) or per_test_stall

        # Cross-version readout history (a compact oscillation-visible table). The
        # 5-iteration crawl thrash was invisible frame-by-frame and obvious across
        # versions — give the critic the whole trajectory of attempts.
        it = (state.get("num_critic_iters") or 0)
        floor_fail = ",".join(t["name"] for t in discrete_failures) or "floor OK"
        fid_note = f"{summary.get('num_fidelity_passed',0)}/{summary.get('num_fidelity_total',0)} fidelity"
        hist_line = (f"v{it}: score {score:.1f} | {floor_fail} | {fid_note} | "
                     f"reward {summary['total_reward']:.1f}")
        readout_history = list(state.get("readout_history") or []) + [hist_line]
        critic_payload["version_history"] = readout_history

        return {
            "messages": [AIMessage(json.dumps(critic_payload))],
            "evaluation_feedback": critic_payload,
            'skill_name': state['skill_name'],
            'skill_version': state['skill_version'],
            'success': state['success'],
            "num_critic_iters": (1 if state['num_critic_iters'] is None else state['num_critic_iters'] + 1),
            'rewrite_mode': rewrite_mode,
            'skill_desc': state['skill_desc'],
            'best_score': best_score,
            'best_rank': best_rank,
            'best_code_path': best_code_path,
            'no_improve_count': no_improve,
            'core_fail_streaks': core_fail_streaks,
            'stuck': stuck,
            'last_failures': last_failures,
            'readout_history': readout_history,
        }

    return run_unit_test
