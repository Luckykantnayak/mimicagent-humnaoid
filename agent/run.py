import argparse
import json
import os

from prompts.prompt_loader import load_yaml_config, load_prompt

from agents.base_agent import BaseAgent
from langchain_core.messages import AIMessage
from agents.skill_gen.skill_formatter import prepare_skill
from llm.load_claude import load_llm
from pipeline import build_graph

import time

#----------------------------------------------------
#                    LOAD CONFIG
#----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(BASE_DIR, "cfg", "config.yaml")
cfg = load_yaml_config(yaml_path)
# Per-process robot override (so a second robot can run in PARALLEL without editing
# the shared config.yaml). Everything
# downstream reads cfg["robot"]["name"], so overriding it here propagates fully.
if os.environ.get("ROBOT_NAME"):
    cfg["robot"]["name"] = os.environ["ROBOT_NAME"]

skill_folder = "skills_v1"


def print_stream(stream, seen=None):
    terminal_state = None
    for s in stream:
        terminal_state = s
        if seen is not None:
            seen["state"] = s  # survives an exception raised by a later node
        messages = s.get("messages", [])
        if not messages:
            continue  # nothing to print

        message = messages[-1]

        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

    return terminal_state


def run_all_skills(skill_version, skills=None, llm=None):
    app = build_graph(llm or load_llm(), cfg)

    # Mermaid source is local; a third-party image service cannot block generation.
    with open("graph_coder.mmd", "w") as f:
        f.write(app.get_graph().draw_mermaid())

    all_tokens = []
    all_time = []
    all_skills = []

    for idx, skill in enumerate(skills):

        t1 = time.time()

        print("\n" + "=" * 60)
        print(f"Running skill {idx}: {skill['skill_id']}")
        print("=" * 60)

        skill_desc = {k: v for (k, v) in skill.items() if k != "think"}

        inputs = {
            "messages": [AIMessage(json.dumps(skill_desc))],
            "skill_name": skill["skill_id"],
            "skill_desc": skill_desc,
            "success": True,
            "skill_version": skill_version,
            "num_critic_iters": 0,
            "rewrite_mode": "",
            "token_usage": {},
        }

        seen = {}
        try:
            terminal_state = print_stream(app.stream(inputs, stream_mode="values",
                                                     config={"recursion_limit": 150}), seen)
        except Exception as exc:  # keep the run's cost/time/outcome on disk, then report
            import traceback
            traceback.print_exc()
            # Tokens spent by the nodes that finished before the failure are real spend;
            # reporting {} made a mid-run API error (e.g. an exhausted balance) look free.
            last = seen.get("state") or {}
            terminal_state = {"token_usage": last.get("token_usage") or {}, "outcome": "error",
                              "error": f"{type(exc).__name__}: {exc}"}
        terminal_state = terminal_state or {"token_usage": {}, "outcome": "error"}

        token_usage = terminal_state.get("token_usage") or {}
        skill["outcome"] = terminal_state.get("outcome", "unverified")
        skill["error"] = terminal_state.get("error")
        skill["final_report_path"] = terminal_state.get("final_report_path")
        print("Outcome:", skill["outcome"])

        t2 = time.time()
        print("Time taken for skill:", t2 - t1)
        print("token usage:", token_usage)

        all_tokens.append(token_usage)
        all_time.append(t2 - t1)
        all_skills.append(skill)

    return all_tokens, all_time, all_skills


def save_token_usage_time(cfg, all_token_usage, all_time_taken, all_skills):
    robot = cfg["robot"]["name"]

    for token_usage, time_taken, skill in zip(all_token_usage, all_time_taken, all_skills):
        skill_dir = os.path.join(
            cfg["paths"]["skill_output_dir"], robot, skill["skill_id"]
        )
        os.makedirs(skill_dir, exist_ok=True)

        metadata = {
            "token_usage": token_usage,
            "time_taken": time_taken,
            "outcome": skill.get("outcome", "unverified"),
            "final_report_path": skill.get("final_report_path"),
            "error": skill.get("error"),
        }

        with open(os.path.join(skill_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the motion-prior synthesis pipeline.")
    parser.add_argument("--skill", help="Natural-language skill description. Runs just this skill instead of the JSON library.")
    parser.add_argument("--name", default="SKILL_CUSTOM", help="Identifier for the --skill run (used for filenames and the simulator --mode).")
    parser.add_argument("--version", default=skill_folder, help="Output/skill-library version folder.")
    args = parser.parse_args()

    if not args.skill:
        # Batch mode read a bundled input skill library (archive/), which has been
        # removed — pass a single skill with --skill "<description>" --name <NAME>.
        parser.error("--skill is required: the batch input library was removed. "
                     'Run e.g.  python run.py --skill "the robot trots forward" --name SKILL_TROT')

    llm = load_llm()
    formatter_prompt = load_prompt("skill_formatter", cfg)
    skill = prepare_skill(BaseAgent(llm), formatter_prompt, args.skill, name=args.name)
    print("\nSkill:\n" + json.dumps(skill, indent=2))
    skills = [skill]

    token_usage, time_taken, skills = run_all_skills(skill_version=args.version, skills=skills, llm=llm)
    save_token_usage_time(cfg, token_usage, time_taken, skills)

    raise SystemExit(0 if all(s.get("outcome") == "complete" for s in skills) else 1)
