"""LangGraph agent wired to Vertex AI Gemini with GCP, GitHub, and VirtualDojo CRM tools."""

import os

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage

from tools.gcp_logging import query_cloud_logs
from tools.gcp_monitoring import check_gcp_metrics
from tools.gcp_cloudrun import list_cloud_run_services
from tools.github import (
    github_list_prs, github_get_pr_details, github_list_recent_commits,
    github_list_issues, github_get_issue_details, github_create_issue,
    github_list_workflow_runs, github_get_workflow_run_details,
)
from tools.virtualdojo_mcp import create_virtualdojo_tool, create_virtualdojo_list_tools

# Static tools — always available
STATIC_TOOLS = [
    query_cloud_logs,
    list_cloud_run_services,
    check_gcp_metrics,
    github_list_prs,
    github_get_pr_details,
    github_list_recent_commits,
    github_list_issues,
    github_get_issue_details,
    github_create_issue,
    github_list_workflow_runs,
    github_get_workflow_run_details,
]

SYSTEM_PROMPT = (
    "You are SamurAI, a DevOps and CRM assistant in Microsoft Teams. "
    "You help the team check Google Cloud infrastructure, read logs, "
    "monitor services, review GitHub activity, and query VirtualDojo CRM data. "
    "Be concise and use markdown formatting when it helps readability.\n\n"
    "IMPORTANT — GCP project IDs you have access to:\n"
    "- virtualdojo-samurai (this bot)\n"
    "- virtualdojo-fedramp-dev (FedRAMP dev environment)\n"
    "- virtualdojo-fedramp-prod (FedRAMP production environment)\n"
    "When the user mentions 'fedramp dev' or 'dev', use project_id='virtualdojo-fedramp-dev'. "
    "When they mention 'fedramp prod' or 'prod', use project_id='virtualdojo-fedramp-prod'. "
    "Always use the exact project IDs above — never guess or construct project IDs.\n\n"
    "GitHub organization: Quote-ly\n"
    "Key repositories:\n"
    "- Quote-ly/quotely-data-service (main data service)\n"
    "- Quote-ly/virtualdojo_cli (VirtualDojo CLI tool)\n"
    "- Quote-ly/SamurAI (this bot's repo)\n"
    "When the user mentions a repo name without the org prefix, assume it's under Quote-ly/. "
    "When the user says 'data service' or 'quotely', use Quote-ly/quotely-data-service. "
    "When they say 'CLI' or 'vdojo cli', use Quote-ly/virtualdojo_cli.\n\n"
    "VirtualDojo CRM:\n"
    "You can query CRM data (contacts, accounts, opportunities, quotes, compliance records) "
    "using the virtualdojo_crm tool. Use virtualdojo_list_tools to discover available operations. "
    "Common tool_name values: 'search_records', 'list_objects', 'describe_object', "
    "'create_record', 'update_record', 'get_record'. "
    "If the user asks about CRM data and is not signed in, tell them to say 'connect to VirtualDojo' to authenticate.\n\n"
    "Each message includes the user's name and timezone in brackets at the start. "
    "Use their timezone when displaying times — convert UTC timestamps to their local time. "
    "For example, if the user is in America/New_York, show times in ET."
)


def _build_graph(user_id: str = "default"):
    """Build a LangGraph agent with user-specific CRM tools."""
    llm = ChatVertexAI(
        model_name="gemini-3-flash-preview",
        project=os.environ.get("GCP_PROJECT_ID"),
        location="global",
    )

    # Combine static tools with user-specific VirtualDojo tools
    user_tools = STATIC_TOOLS + [
        create_virtualdojo_tool(user_id),
        create_virtualdojo_list_tools(user_id),
    ]

    llm_with_tools = llm.bind_tools(user_tools)
    tool_node = ToolNode(user_tools)

    def call_model(state: MessagesState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
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

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Cache of per-user graphs to avoid rebuilding on every message
_user_graphs: dict[str, object] = {}


def _get_graph(user_id: str):
    """Get or create a LangGraph agent for a specific user."""
    if user_id not in _user_graphs:
        _user_graphs[user_id] = _build_graph(user_id)
    return _user_graphs[user_id]


def _extract_text(content) -> str:
    """Extract plain text from Gemini's content blocks."""
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content


async def run_agent(
    user_message: str,
    conversation_id: str = "default",
    user_id: str = "default",
    user_name: str = "",
    user_timezone: str = "",
) -> str:
    # Build context prefix so the LLM knows who it's talking to
    context_parts = []
    if user_name:
        context_parts.append(f"User: {user_name}")
    if user_timezone:
        from datetime import datetime
        import zoneinfo
        try:
            tz = zoneinfo.ZoneInfo(user_timezone)
            local_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M %Z")
            context_parts.append(f"Timezone: {user_timezone} (current time: {local_time})")
        except Exception:
            context_parts.append(f"Timezone: {user_timezone}")

    message = user_message
    if context_parts:
        message = f"[{' | '.join(context_parts)}]\n{user_message}"

    graph = _get_graph(user_id)
    config = {"configurable": {"thread_id": conversation_id}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    return _extract_text(result["messages"][-1].content)
