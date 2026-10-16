"""The graph used by both command line and Streamlit entry points."""
from prompts.prompt_loader import load_prompt

from langgraph.graph import StateGraph, START, END
from agents.base_agent import BaseAgent
from agents.code_gen.state import AgentState

from agents.code_gen.coder_agent import make_coder_node
from agents.code_gen.motion_planner_agent import make_motion_plan_node
from agents.code_gen.code_executer_agent import make_code_executer_node, make_finalize_node
from agents.code_gen.critic_coder_agent import make_critic_coder_node
from agents.code_gen.routers import make_critic_router, make_unit_test_router
from agents.code_gen.run_unit_test import make_unit_test_node
from agents.code_gen.skill_filter import make_skill_filter_node
from agents.code_gen.motion_diagnosis_agent import make_motion_diagnosis_agent_node

from agents.code_gen.goal_contract_agent import make_goal_contract_node


def build_graph(llm, cfg):
    coder_prompt = load_prompt("coder", cfg)
    mp_prompt = load_prompt("motion_planner", cfg)
    critic_prompt = load_prompt("critic_coder", cfg)
    skill_filter_prompt = load_prompt("skill_filter", cfg)
    motion_diagnosis_prompt = load_prompt("motion_failure_diagnosis", cfg)
    graph = StateGraph(AgentState)

    graph.add_node("skill_filter_agent", lambda state: state)
    graph.add_node("rejected_agent", lambda state: {"success": False, "outcome": "rejected"})
    graph.add_edge("rejected_agent", END)
    graph.add_node("goal_contract_agent", make_goal_contract_node(BaseAgent(llm), load_prompt("goal_contract", cfg), cfg))
    graph.add_node("motion_planner_agent", make_motion_plan_node(BaseAgent(llm), data=mp_prompt, cfg=cfg))
    graph.add_node("code_generator_agent", make_coder_node(BaseAgent(llm), data=coder_prompt, cfg=cfg))
    graph.add_node("code_executer_agent", make_code_executer_node(cfg))
    graph.add_node("unit_test_router", lambda state: state)
    graph.add_node("unit_test_agent", make_unit_test_node(cfg))
    graph.add_node("critic_router", lambda state: state)
    graph.add_node("motion_diagnosis_agent", make_motion_diagnosis_agent_node(BaseAgent(llm), data=motion_diagnosis_prompt, cfg=cfg))
    graph.add_node("critic_coder_agent", make_critic_coder_node(BaseAgent(llm), data=critic_prompt, cfg=cfg))
    graph.add_node("finalize_agent", make_finalize_node(cfg))
    graph.add_edge("finalize_agent", END)

    #----------------------------------------------------
    #                    ADD EDGES
    #----------------------------------------------------
    graph.add_edge(START, "goal_contract_agent")
    graph.add_edge("goal_contract_agent", "skill_filter_agent")
    graph.add_edge("motion_planner_agent", "code_generator_agent")
    graph.add_edge("code_generator_agent", "code_executer_agent")
    graph.add_edge("code_executer_agent", "unit_test_router")
    graph.add_edge("unit_test_agent", "critic_router")
    graph.add_edge("motion_diagnosis_agent", "critic_coder_agent")
    graph.add_edge("critic_coder_agent", "code_executer_agent")

    #----------------------------------------------------
    #                ADD CONDITIONAL EDGES
    #----------------------------------------------------

    graph.add_conditional_edges(
        "skill_filter_agent",
        make_skill_filter_node(agent=BaseAgent(llm), data=skill_filter_prompt),
        {
            "continue": "motion_planner_agent",
            "end_skill": "rejected_agent",
        }
    )

    graph.add_conditional_edges(
        "unit_test_router",
        make_unit_test_router(cfg),
        {
            "continue": "unit_test_agent",
            "repair": "critic_coder_agent",
            "replan": "motion_planner_agent",   # repeated identical lint hard-fail = a plan bug
            "end_skill": "finalize_agent",
        }
    )

    graph.add_conditional_edges(
        "critic_router",
        make_critic_router(cfg),
        {
            "loop": "motion_diagnosis_agent",
            "replan": "motion_planner_agent",   # step back and re-decide the interface
            "end_skill": "finalize_agent",
        }
    )

    return graph.compile()

