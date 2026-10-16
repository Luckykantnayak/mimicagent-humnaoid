from agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string
from agents.code_gen.state import AgentState

from utils.tools import extract_json_from_message

import os
import json


def make_skill_filter_node(agent: BaseAgent, data):
    def skill_filter(state: AgentState) -> str:
        """
        This function looks at the skill description and decides whether it's a feasible skill or not.
        Returns the edge 'continue' if skill is valid, else 'end' --> proceeds to next skill
        :param state:
        :return:
        """
        # Explicit user-requested skills (SKIP_SKILL_FILTER=1) bypass the gate —
        # the user has already decided they want it (e.g. aerial/expressive moves).
        if os.environ.get("SKIP_SKILL_FILTER"):
            return "continue"

        system_message = data["system"]  # str
        human_message = data["human"]  # str

        skill_description = json.dumps(state["skill_desc"])   # str
        human_message = human_message.replace('{SKILL_DESCRIPTION_JSON}', skill_description)

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]

        error = ""
        for attempt in range(3):
            reply = agent.generate(messages[:-1] + [HumanMessage(content=messages[-1].content + error)])
            try:
                output = extract_json_from_message(get_buffer_string([reply]))
                feasible = output["feasible"]
                if not isinstance(feasible, bool):
                    raise ValueError("feasible must be a JSON boolean")
                break
            except (ValueError, TypeError, KeyError) as exc:
                error = "\nInvalid response: " + str(exc) + ". Return JSON with lowercase true/false."
        else:
            raise ValueError("Feasibility response remained invalid after 3 attempts")

        if feasible:
            return "continue"

        return "end_skill"

    return skill_filter
