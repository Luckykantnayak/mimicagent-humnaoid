from agents.base_agent import BaseAgent, merge_usage
from langchain_core.messages import SystemMessage, HumanMessage, get_buffer_string
from agents.code_gen.state import AgentState
from utils.repair_context import repair_prompt


def make_motion_diagnosis_agent_node(coder_agent: BaseAgent, data, cfg):
    def node(state: AgentState) -> AgentState:
        system_message = data["system"]  # str
        human_message = repair_prompt(data["human"], state, cfg)

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]

        output = coder_agent.generate(messages)  # output type: AIMessage
        # Get Token Usage
        token_dict = merge_usage(state["token_usage"], output)


        return {
            "messages": [output],
            "last_diagnosis": output.content,
            'skill_name': state['skill_name'],
            'skill_version': state['skill_version'],
            'success': state['success'],
            "num_critic_iters": state['num_critic_iters'],
            'rewrite_mode': state['rewrite_mode'],
            'skill_desc': state['skill_desc'],
            'token_usage': token_dict,
        }

    return node
