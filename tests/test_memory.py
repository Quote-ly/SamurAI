"""Tests for memory.py — LangMem InMemoryStore with SQLite persistence."""

import json
import os
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset memory module singletons between tests."""
    import memory

    memory._store = None
    memory._checkpointer = None
    memory._checkpoint_conn = None
    memory._background_executor = None
    yield
    memory._store = None
    memory._checkpointer = None
    memory._checkpoint_conn = None
    memory._background_executor = None


# --- InMemoryStore ---


def test_get_memory_store_returns_store():
    from memory import get_memory_store

    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 1536 for _ in texts]):
        store = get_memory_store()
    assert store is not None


def test_get_memory_store_is_singleton():
    from memory import get_memory_store

    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 1536 for _ in texts]):
        store1 = get_memory_store()
        store2 = get_memory_store()
    assert store1 is store2


def test_store_put_and_search():
    from memory import get_memory_store

    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 1536 for _ in texts]):
        store = get_memory_store()

    store.put(("memories", "user1"), "mem1", {"content": "Devin likes Python"})
    results = store.search(("memories", "user1"), query="Python")
    assert len(results) > 0
    assert results[0].value["content"] == "Devin likes Python"


def test_store_user_isolation():
    from memory import get_memory_store

    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 1536 for _ in texts]):
        store = get_memory_store()

    store.put(("memories", "user-a"), "m1", {"content": "secret A"})
    store.put(("memories", "user-b"), "m2", {"content": "secret B"})

    results_a = store.search(("memories", "user-a"), query="secret")
    results_b = store.search(("memories", "user-b"), query="secret")

    assert all(r.value["content"] == "secret A" for r in results_a)
    assert all(r.value["content"] == "secret B" for r in results_b)


# --- SQLite Persistence ---


def test_persist_and_load_memories(tmp_path):
    import memory

    memory.MEMORY_DB_PATH = str(tmp_path / "test_memories.sqlite")
    memory.DATA_DIR = str(tmp_path)

    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 1536 for _ in texts]):
        store = memory.get_memory_store()

    store.put(("memories", "user1"), "m1", {"content": "fact one"})
    store.put(("memories", "user1"), "m2", {"content": "fact two"})

    memory.persist_memories()

    # Verify SQLite has the data
    conn = sqlite3.connect(memory.MEMORY_DB_PATH)
    rows = conn.execute("SELECT * FROM memories").fetchall()
    conn.close()
    assert len(rows) == 2

    # Reset store and reload
    memory._store = None
    with patch("memory._create_embed_fn", return_value=lambda texts: [[0.1] * 1536 for _ in texts]):
        store2 = memory.get_memory_store()

    results = store2.search(("memories", "user1"), query="fact")
    assert len(results) == 2


def test_persist_no_op_when_no_store():
    from memory import persist_memories

    persist_memories()  # Should not raise


# --- Checkpointer ---


@pytest.mark.asyncio
async def test_get_checkpointer_creates_sqlite_saver(tmp_path):
    import memory

    memory.CHECKPOINT_DB_PATH = str(tmp_path / "test_checkpoints.sqlite")
    memory._checkpointer = None
    memory._checkpoint_conn = None

    ckpt = await memory.get_checkpointer()
    assert ckpt is not None


@pytest.mark.asyncio
async def test_get_checkpointer_is_singleton(tmp_path):
    import memory

    memory.CHECKPOINT_DB_PATH = str(tmp_path / "test_checkpoints.sqlite")
    memory._checkpointer = None
    memory._checkpoint_conn = None

    ckpt1 = await memory.get_checkpointer()
    ckpt2 = await memory.get_checkpointer()
    assert ckpt1 is ckpt2


@pytest.mark.asyncio
async def test_get_checkpointer_falls_back_to_memory_saver():
    import memory

    memory.CHECKPOINT_DB_PATH = "/nonexistent/path/checkpoints.sqlite"
    memory._checkpointer = None
    memory._checkpoint_conn = None

    ckpt = await memory.get_checkpointer()
    from langgraph.checkpoint.memory import MemorySaver

    assert isinstance(ckpt, MemorySaver)


# --- LangMem Memory Tools ---


def test_create_memory_tools_returns_two_tools():
    from memory import create_memory_tools

    with patch("memory.get_memory_store", return_value=MagicMock()):
        tools = create_memory_tools("test-user")

    assert len(tools) == 2
    names = {t.name for t in tools}
    assert "manage_memory" in names
    assert "search_memory" in names


# --- Auto-Retrieval ---


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_with_results():
    from memory import retrieve_relevant_memories

    mock_store = MagicMock()
    mock_result = MagicMock()
    mock_result.value = {"content": "Devin prefers short responses"}
    mock_store.search.return_value = [mock_result]

    with patch("memory.get_memory_store", return_value=mock_store):
        result = await retrieve_relevant_memories("user1", "what does Devin prefer?")

    assert result is not None
    assert "Devin prefers short responses" in result


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_empty():
    from memory import retrieve_relevant_memories

    mock_store = MagicMock()
    mock_store.search.return_value = []

    with patch("memory.get_memory_store", return_value=mock_store):
        result = await retrieve_relevant_memories("user1", "anything")

    assert result is None


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_handles_error():
    from memory import retrieve_relevant_memories

    with patch("memory.get_memory_store", side_effect=Exception("boom")):
        result = await retrieve_relevant_memories("user1", "anything")

    assert result is None


# --- Background Extractor ---


def test_get_background_extractor_returns_executor():
    from memory import get_background_extractor

    with (
        patch("memory.get_memory_store", return_value=MagicMock()),
        patch("langmem.create_memory_store_manager", return_value=MagicMock()),
        patch("langmem.ReflectionExecutor") as mock_executor_cls,
    ):
        mock_executor_cls.return_value = MagicMock()
        executor = get_background_extractor()

    assert executor is not None


def test_get_background_extractor_is_singleton():
    from memory import get_background_extractor

    with (
        patch("memory.get_memory_store", return_value=MagicMock()),
        patch("langmem.create_memory_store_manager", return_value=MagicMock()),
        patch("langmem.ReflectionExecutor") as mock_executor_cls,
    ):
        mock_executor_cls.return_value = MagicMock()
        e1 = get_background_extractor()
        e2 = get_background_extractor()

    assert e1 is e2
