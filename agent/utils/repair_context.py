"""Explicit, bounded evidence for diagnosis and code repair."""
import json
import os
import re

from langchain_core.messages import get_buffer_string
from utils.plan_io import load_plan_dict
from utils.semantic_contract import goal_prompt
from utils.tools import extract_json_from_message


def repair_prompt(template, state, cfg):
    path = state.get("current_code_path")
    code = "(No implementation was written successfully.)"
    if path and os.path.isfile(path):
        with open(path) as f:
            code = f.read()

    feedback = state.get("evaluation_feedback")
    if not state.get("success", False):
        feedback = {"execution_failed": True, "error": get_buffer_string(state.get("messages", [])[-1:])}
    elif feedback is None:
        # Compatibility with states saved before explicit feedback was recorded.
        for message in reversed(state.get("messages", [])):
            try:
                candidate = extract_json_from_message(get_buffer_string([message]))
            except (ValueError, TypeError):
                continue
            if isinstance(candidate, dict) and "motion_readout" in candidate:
                feedback = candidate
                break
    skill_dir = os.path.join(cfg["paths"]["skill_output_dir"],
                             cfg["robot"]["name"], state["skill_name"])
    plan = load_plan_dict(skill_dir, state["skill_name"])
    values = {
        "CURRENT_SKILL_CODE": code,
        "SKILL_DESCRIPTION_JSON": json.dumps(state["skill_desc"], indent=2),
        "UNIT_TEST_FEEDBACK_JSON": json.dumps(feedback or {}, indent=2),
        "EVALUATION_FEEDBACK_JSON": json.dumps(feedback or {}, indent=2),
        "MOTION_FAILURE_DIAGNOSIS_FEEDBACK": (
            state.get("last_diagnosis") or "(No diagnosis for this attempt.)"
        ) if state.get("success", False) else "(Execution failed before diagnosis.)",
        "MOTION_PLAN_JSON": json.dumps(plan, indent=2),
    }
    # One substitution pass: braces in generated source or user text are data.
    prompt = re.sub(r"\{(" + "|".join(values) + r")\}",
                    lambda match: values[match.group(1)], template)
    if "{MOTION_PLAN_JSON}" not in template:
        prompt += "\n\nCurrent planner-owned motion plan:\n" + values["MOTION_PLAN_JSON"]
    history = list(state.get("readout_history") or [])[-8:]
    if history:
        prompt += "\n\nRecent attempt scores:\n" + "\n".join(history)
    return prompt + goal_prompt(state)
