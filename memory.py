"""Memory system using LangMem + LangGraph InMemoryStore with SQLite persistence.

Provides:
1. LangGraph InMemoryStore — fast in-RAM memory with semantic search
2. LangMem tools — manage_memory + search_memory for the agent
3. Background extraction — auto-extracts memories after conversations
4. SQLite persistence — periodic flush to GCS for survival across restarts
5. AsyncSqliteSaver — LangGraph checkpointer for conversation history
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("SAMURAI_DATA_DIR", "/data")
MEMORY_DB_PATH = os.path.join(DATA_DIR, "langmem_memories.sqlite")
# Checkpoints on local SSD — too write-heavy for GCS FUSE
CHECKPOINT_DB_PATH = "/tmp/checkpoints.sqlite"

# Singletons
_store = None
_checkpointer = None
_checkpoint_conn = None
_background_executor = None


# ── Vertex AI Embedding Function ──────────────────────────────────────


def _create_embed_fn():
    """Create an embedding function using Vertex AI."""
    _embeddings = None

    def embed(texts: list[str]) -> list[list[float]]:
        nonlocal _embeddings
        if _embeddings is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            _embeddings = GoogleGenerativeAIEmbeddings(
                model="text-embedding-005",
                project=os.environ.get("GCP_PROJECT_ID"),
                task_type="RETRIEVAL_DOCUMENT",
                dimensions=1536,
            )
        return _embeddings.embed_documents(texts)

    return embed


# ── Memory Store (InMemoryStore + SQLite persistence) ─────────────────


def get_memory_store():
    """Get the singleton LangGraph InMemoryStore with embedding search."""
    global _store
    if _store is None:
        from langgraph.store.memory import InMemoryStore

        _store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": _create_embed_fn(),
            }
        )
        # Load persisted memories from SQLite
        _load_persisted_memories(_store)
        logger.info("LangMem memory store ready (InMemoryStore + SQLite backup)")
    return _store


def _load_persisted_memories(store):
    """Load memories from SQLite into the InMemoryStore on startup."""
    import sqlite3

    if not os.path.exists(MEMORY_DB_PATH):
        return

    try:
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.execute(
            "SELECT namespace, key, value_json, created_at, updated_at FROM memories"
        )
        count = 0
        for row in cursor:
            namespace = tuple(json.loads(row[0]))
            key = row[1]
            value = json.loads(row[2])
            store.put(namespace, key, value)
            count += 1
        conn.close()
        if count:
            logger.info("Loaded %d persisted memories from SQLite", count)
    except Exception as e:
        logger.warning("Failed to load persisted memories: %s", e)


def persist_memories():
    """Flush the InMemoryStore to SQLite for persistence across restarts."""
    import sqlite3

    if _store is None:
        return

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        conn = sqlite3.connect(MEMORY_DB_PATH)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (namespace, key)
            )"""
        )

        # Get all items from the store by searching known namespaces
        # InMemoryStore stores items internally — we iterate via _data
        items_saved = 0
        if hasattr(_store, '_data'):
            for namespace_tuple, keys in _store._data.items():
                ns_json = json.dumps(list(namespace_tuple))
                for key, item in keys.items():
                    value_json = json.dumps(item.value)
                    created = getattr(item, 'created_at', '')
                    updated = getattr(item, 'updated_at', '')
                    conn.execute(
                        """INSERT OR REPLACE INTO memories
                           (namespace, key, value_json, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (ns_json, key, value_json, str(created), str(updated)),
                    )
                    items_saved += 1

        conn.commit()
        conn.close()
        if items_saved:
            logger.info("Persisted %d memories to SQLite", items_saved)
    except Exception as e:
        logger.warning("Failed to persist memories: %s", e)


# ── Checkpointer ─────────────────────────────────────────────────────


async def get_checkpointer():
    """Get or create the singleton SQLite checkpointer for LangGraph."""
    global _checkpointer, _checkpoint_conn
    if _checkpointer is None:
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            _checkpoint_conn = await aiosqlite.connect(CHECKPOINT_DB_PATH)
            _checkpointer = AsyncSqliteSaver(_checkpoint_conn)
            await _checkpointer.setup()
            logger.info("SQLite checkpointer ready: %s", CHECKPOINT_DB_PATH)
        except Exception as e:
            logger.warning(
                "SQLite checkpointer unavailable, falling back to in-memory: %s", e
            )
            from langgraph.checkpoint.memory import MemorySaver

            _checkpointer = MemorySaver()
    return _checkpointer


# ── LangMem Memory Tools ─────────────────────────────────────────────


def create_memory_tools(user_id: str) -> list:
    """Create LangMem memory tools scoped to a specific user."""
    from langmem import create_manage_memory_tool, create_search_memory_tool

    store = get_memory_store()
    return [
        create_manage_memory_tool(
            namespace=("memories", "{user_id}"),
            instructions=(
                "Save important facts about users, projects, preferences, "
                "and key decisions. Update existing memories when information "
                "changes rather than creating duplicates. Delete outdated memories. "
                "Focus on: user preferences, project context, team information, "
                "infrastructure details, and recurring patterns."
            ),
            store=store,
        ),
        create_search_memory_tool(
            namespace=("memories", "{user_id}"),
            store=store,
        ),
    ]


# ── Background Memory Extraction ─────────────────────────────────────


def get_background_extractor():
    """Get the singleton background memory extractor.

    Automatically extracts and consolidates memories from conversations
    without the agent needing to explicitly call save_memory.
    """
    global _background_executor
    if _background_executor is None:
        from langmem import create_memory_store_manager, ReflectionExecutor

        store = get_memory_store()
        manager = create_memory_store_manager(
            "google_genai:gemini-2.0-flash-lite",
            namespace=("memories", "{user_id}"),
            store=store,
            enable_inserts=True,
            enable_updates=True,
            enable_deletes=True,
            instructions=(
                "Extract important facts, preferences, and decisions from the conversation. "
                "Update existing memories if information has changed. "
                "Delete memories that are contradicted by new information. "
                "Focus on: user preferences, project context, team information, "
                "infrastructure details, and recurring patterns. "
                "Do NOT save trivial information like greetings or routine status checks."
            ),
        )
        _background_executor = ReflectionExecutor(manager)
        logger.info("Background memory extractor ready")
    return _background_executor


# ── Auto-Retrieval for System Prompt Injection ────────────────────────


async def retrieve_relevant_memories(user_id: str, query: str) -> str | None:
    """Auto-retrieve relevant memories for injection into system prompt.

    Returns a formatted string of relevant memories, or None if nothing found.
    """
    try:
        store = get_memory_store()
        results = store.search(
            ("memories", user_id),
            query=query,
            limit=5,
        )
        if not results:
            return None
        lines = []
        for r in results:
            val = r.value
            if isinstance(val, dict):
                content = val.get("content", json.dumps(val))
            else:
                content = str(val)
            lines.append(f"- {content}")
        return (
            "Relevant context from past conversations:\n"
            + "\n".join(lines)
        )
    except Exception as e:
        logger.debug("Memory retrieval failed: %s", e)
        return None
