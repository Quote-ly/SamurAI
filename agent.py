"""LangGraph agent wired to Vertex AI Gemini with GCP and GitHub tools."""

import os

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

from tools.gcp_logging import query_cloud_logs
from tools.gcp_monitoring import check_gcp_metrics
from tools.gcp_cloudrun import list_cloud_run_services
from tools.github import github_list_prs, github_get_pr_details, github_list_recent_commits

TOOLS = [
    query_cloud_logs,
    list_cloud_run_services,
    check_gcp_metrics,
    github_list_prs,
    github_get_pr_details,
    github_list_recent_commits,
]

SYSTEM_PROMPT = (
    "You are SamurAI, a DevOps assistant in Microsoft Teams. "
    "You help the team check Google Cloud infrastructure, read logs, "
    "monitor services, and review GitHub activity. "
    "Be concise and use markdown formatting when it helps readability."
)


def _build_graph():
    llm = ChatVertexAI(
        model_name="gemini-3.1-pro",
        project=os.environ.get("GCP_PROJECT_ID"),
    )
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def call_model(state: MessagesState):
        messages = state["messages"]
        # Prepend system prompt if this is the first turn
        if len(messages) == 1:
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


_app = _build_graph()


async def run_agent(user_message: str) -> str:
    result = await _app.ainvoke(
        {"messages": [HumanMessage(content=user_message)]}
    )
    return result["messages"][-1].content
