"""Tests for memory.py — VectorMemoryStore, checkpointer, and memory tools."""

import importlib
import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


def _word_embedding(text: str, dim: int = 1536) -> np.ndarray:
    """Deterministic embedding based on word hashing — texts with
    overlapping words produce similar vectors (high cosine similarity)."""
    vec = np.zeros(dim, dtype=np.float32)
    for word in text.lower().split():
        vec[hash(word) % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _make_embed_mock():
    """Return an AsyncMock embedding model whose aembed_documents uses word overlap."""

    async def fake_embed(texts):
        return [_word_embedding(t).tolist() for t in texts]

    mock = MagicMock()
    mock.aembed_documents = AsyncMock(side_effect=fake_embed)
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """Fresh VectorMemoryStore backed by a temp SQLite file."""
    import memory

    importlib.reload(memory)

    s = memory.VectorMemoryStore(str(tmp_path / "test_memory.sqlite"))
    s._embeddings = _make_embed_mock()
    return s


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset memory module singletons between tests."""
    import memory

    memory._memory_store = None
    memory._checkpointer = None
    memory._checkpoint_conn = None
    yield
    memory._memory_store = None
    memory._checkpointer = None
    memory._checkpoint_conn = None


# ---------------------------------------------------------------------------
# VectorMemoryStore — table creation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_creates_table(store):
    import aiosqlite

    await store.initialize()
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        row = await cursor.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_initialize_is_idempotent(store):
    await store.initialize()
    await store.initialize()  # should not raise


# ---------------------------------------------------------------------------
# VectorMemoryStore — add / search / list / delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_returns_id(store):
    await store.initialize()
    memory_id = await store.add("user-1", "the user likes dark mode")
    assert isinstance(memory_id, str)
    assert len(memory_id) == 8


@pytest.mark.asyncio
async def test_add_persists_to_sqlite(store):
    import aiosqlite

    await store.initialize()
    await store.add("user-1", "prefers concise answers")
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM memories")
        count = (await cursor.fetchone())[0]
    assert count == 1


@pytest.mark.asyncio
async def test_search_finds_similar_memories(store):
    await store.initialize()
    await store.add("user-1", "user prefers dark mode theme")
    await store.add("user-1", "user works on cloud infrastructure")
    await store.add("user-1", "user prefers dark mode at night")

    results = await store.search("user-1", "dark mode preference", top_k=5)
    assert len(results) >= 2
    # The two dark-mode memories should rank higher than the cloud one
    dark_results = [r for r in results if "dark mode" in r["content"]]
    cloud_results = [r for r in results if "cloud" in r["content"]]
    if dark_results and cloud_results:
        assert dark_results[0]["similarity"] > cloud_results[0]["similarity"]


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_memories(store):
    await store.initialize()
    results = await store.search("user-1", "anything")
    assert results == []


@pytest.mark.asyncio
async def test_search_scoped_by_user(store):
    await store.initialize()
    await store.add("user-a", "secret fact about user A")
    await store.add("user-b", "secret fact about user B")

    results_a = await store.search("user-a", "secret fact", top_k=10)
    results_b = await store.search("user-b", "secret fact", top_k=10)

    assert all(r["content"].endswith("user A") for r in results_a)
    assert all(r["content"].endswith("user B") for r in results_b)


@pytest.mark.asyncio
async def test_search_respects_top_k(store):
    await store.initialize()
    for i in range(10):
        await store.add("user-1", f"memory number {i}")
    results = await store.search("user-1", "memory", top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_list_memories(store):
    await store.initialize()
    await store.add("user-1", "first memory")
    await store.add("user-1", "second memory")

    memories = await store.list_memories("user-1")
    assert len(memories) == 2
    contents = {m["content"] for m in memories}
    assert "first memory" in contents
    assert "second memory" in contents


@pytest.mark.asyncio
async def test_list_memories_empty(store):
    await store.initialize()
    memories = await store.list_memories("user-1")
    assert memories == []


@pytest.mark.asyncio
async def test_list_memories_scoped_by_user(store):
    await store.initialize()
    await store.add("user-a", "A's memory")
    await store.add("user-b", "B's memory")

    assert len(await store.list_memories("user-a")) == 1
    assert len(await store.list_memories("user-b")) == 1


@pytest.mark.asyncio
async def test_delete_existing_memory(store):
    await store.initialize()
    mid = await store.add("user-1", "delete me")
    assert await store.delete(mid) is True
    assert await store.list_memories("user-1") == []


@pytest.mark.asyncio
async def test_delete_nonexistent_memory(store):
    await store.initialize()
    assert await store.delete("does-not-exist") is False


# ---------------------------------------------------------------------------
# get_checkpointer — singleton + fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_checkpointer_creates_sqlite_saver(tmp_path):
    import memory

    with patch.dict(os.environ, {"SAMURAI_DATA_DIR": str(tmp_path)}):
        memory.DATA_DIR = str(tmp_path)
        memory.CHECKPOINT_DB_PATH = os.path.join(str(tmp_path), "checkpoints.sqlite")
        ckpt = await memory.get_checkpointer()

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    assert isinstance(ckpt, AsyncSqliteSaver)


@pytest.mark.asyncio
async def test_get_checkpointer_falls_back_to_memory_saver():
    import memory

    with patch("aiosqlite.connect", side_effect=Exception("no sqlite")):
        memory._checkpointer = None
        ckpt = await memory.get_checkpointer()

    from langgraph.checkpoint.memory import MemorySaver

    assert isinstance(ckpt, MemorySaver)


@pytest.mark.asyncio
async def test_get_checkpointer_is_singleton(tmp_path):
    import memory

    memory.DATA_DIR = str(tmp_path)
    memory.CHECKPOINT_DB_PATH = os.path.join(str(tmp_path), "checkpoints.sqlite")

    ckpt1 = await memory.get_checkpointer()
    ckpt2 = await memory.get_checkpointer()
    assert ckpt1 is ckpt2


# ---------------------------------------------------------------------------
# get_memory_store — singleton
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_memory_store_creates_and_caches(tmp_path):
    import memory

    memory.DATA_DIR = str(tmp_path)
    memory.MEMORY_DB_PATH = os.path.join(str(tmp_path), "memory.sqlite")

    with patch.object(memory.VectorMemoryStore, "initialize", new_callable=AsyncMock):
        store1 = await memory.get_memory_store()
        store2 = await memory.get_memory_store()

    assert store1 is store2
    assert isinstance(store1, memory.VectorMemoryStore)


# ---------------------------------------------------------------------------
# create_memory_tools
# ---------------------------------------------------------------------------


def test_create_memory_tools_returns_four_tools():
    import memory

    tools = memory.create_memory_tools("user-1")
    assert len(tools) == 4
    names = {t.name for t in tools}
    assert names == {"save_memory", "recall_memories", "list_all_memories", "forget_memory"}


@pytest.mark.asyncio
async def test_save_memory_tool(store, tmp_path):
    import memory

    await store.initialize()
    memory._memory_store = store

    tools = memory.create_memory_tools("user-1")
    save_tool = next(t for t in tools if t.name == "save_memory")

    result = await save_tool.ainvoke({"content": "user likes Python"})
    assert "Saved to memory" in result

    memories = await store.list_memories("user-1")
    assert len(memories) == 1
    assert memories[0]["content"] == "user likes Python"


@pytest.mark.asyncio
async def test_recall_memories_tool(store, tmp_path):
    import memory

    await store.initialize()
    memory._memory_store = store
    await store.add("user-1", "user likes Python programming")

    tools = memory.create_memory_tools("user-1")
    recall_tool = next(t for t in tools if t.name == "recall_memories")

    result = await recall_tool.ainvoke({"query": "Python programming"})
    assert "user likes Python programming" in result


@pytest.mark.asyncio
async def test_recall_memories_tool_empty(store):
    import memory

    await store.initialize()
    memory._memory_store = store

    tools = memory.create_memory_tools("user-1")
    recall_tool = next(t for t in tools if t.name == "recall_memories")

    result = await recall_tool.ainvoke({"query": "anything"})
    assert "No relevant memories" in result


@pytest.mark.asyncio
async def test_list_all_memories_tool(store):
    import memory

    await store.initialize()
    memory._memory_store = store
    await store.add("user-1", "memory one")
    await store.add("user-1", "memory two")

    tools = memory.create_memory_tools("user-1")
    list_tool = next(t for t in tools if t.name == "list_all_memories")

    result = await list_tool.ainvoke({})
    assert "memory one" in result
    assert "memory two" in result
    assert "All memories (2)" in result


@pytest.mark.asyncio
async def test_forget_memory_tool(store):
    import memory

    await store.initialize()
    memory._memory_store = store
    mid = await store.add("user-1", "forget this")

    tools = memory.create_memory_tools("user-1")
    forget_tool = next(t for t in tools if t.name == "forget_memory")

    result = await forget_tool.ainvoke({"memory_id": mid})
    assert "deleted" in result

    assert await store.list_memories("user-1") == []


@pytest.mark.asyncio
async def test_forget_nonexistent_memory_tool(store):
    import memory

    await store.initialize()
    memory._memory_store = store

    tools = memory.create_memory_tools("user-1")
    forget_tool = next(t for t in tools if t.name == "forget_memory")

    result = await forget_tool.ainvoke({"memory_id": "nope"})
    assert "not found" in result


# ---------------------------------------------------------------------------
# retrieve_relevant_memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_with_context(store):
    import memory

    await store.initialize()
    memory._memory_store = store
    await store.add("user-1", "user prefers dark mode theme")

    ctx = await memory.retrieve_relevant_memories("user-1", "dark mode preference")
    assert ctx is not None
    assert "dark mode" in ctx
    assert "Relevant context" in ctx


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_returns_none_when_empty(store):
    import memory

    await store.initialize()
    memory._memory_store = store

    ctx = await memory.retrieve_relevant_memories("user-1", "anything")
    assert ctx is None


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_returns_none_on_error():
    import memory

    memory._memory_store = None

    with patch.object(
        memory, "get_memory_store", new_callable=AsyncMock, side_effect=Exception("boom")
    ):
        ctx = await memory.retrieve_relevant_memories("user-1", "query")
    assert ctx is None
