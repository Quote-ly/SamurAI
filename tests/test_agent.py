"""Tests for agent.py — LangGraph agent graph and run_agent()."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_llm():
    """Patch ChatVertexAI before importing agent so _build_graph() doesn't hit Vertex."""
    with patch("langchain_google_vertexai.ChatVertexAI") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.bind_tools.return_value = mock_instance
        mock_instance.invoke.return_value = MagicMock(
            content="Hello from SamurAI!", tool_calls=[]
        )
        mock_cls.return_value = mock_instance
        # Force reimport so _build_graph() picks up the mock
        import agent
        importlib.reload(agent)
        yield mock_instance, agent


def test_tools_list_contains_all_six(mock_llm):
    _, agent = mock_llm
    assert len(agent.TOOLS) == 6
    tool_names = {t.name for t in agent.TOOLS}
    assert "query_cloud_logs" in tool_names
    assert "list_cloud_run_services" in tool_names
    assert "check_gcp_metrics" in tool_names
    assert "github_list_prs" in tool_names
    assert "github_get_pr_details" in tool_names
    assert "github_list_recent_commits" in tool_names


def test_system_prompt_defined(mock_llm):
    _, agent = mock_llm
    assert "SamurAI" in agent.SYSTEM_PROMPT
    assert "DevOps" in agent.SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_run_agent_returns_final_message(mock_llm):
    _, agent = mock_llm
    # Patch the compiled graph's ainvoke
    agent._app.ainvoke = AsyncMock(
        return_value={"messages": [MagicMock(content="Here are your logs.")]}
    )

    result = await agent.run_agent("show me recent errors")
    assert result == "Here are your logs."


@pytest.mark.asyncio
async def test_run_agent_passes_human_message(mock_llm):
    _, agent = mock_llm
    agent._app.ainvoke = AsyncMock(
        return_value={"messages": [MagicMock(content="ok")]}
    )

    await agent.run_agent("check cloud run services")

    call_args = agent._app.ainvoke.call_args[0][0]
    messages = call_args["messages"]
    assert len(messages) == 1
    assert messages[0].content == "check cloud run services"


@pytest.mark.asyncio
async def test_run_agent_graph_routes_to_end_without_tools(mock_llm):
    """Full integration: LLM returns no tool_calls → graph goes straight to END."""
    llm_mock, agent = mock_llm

    # LLM returns a message with no tool calls
    from langchain_core.messages import AIMessage
    llm_mock.invoke.return_value = AIMessage(content="All good, no tools needed.")

    result = await agent.run_agent("how are things?")
    assert result == "All good, no tools needed."
