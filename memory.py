"""Persistent vector memory store using SQLite + Vertex AI embeddings.

Provides two persistence layers:
1. AsyncSqliteSaver — LangGraph checkpointer for conversation history
2. VectorMemoryStore — long-term semantic memory with embedding search
"""

import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("SAMURAI_DATA_DIR", "/data")
MEMORY_DB_PATH = os.path.join(DATA_DIR, "memory.sqlite")
CHECKPOINT_DB_PATH = os.path.join(DATA_DIR, "checkpoints.sqlite")

# Minimum cosine similarity for a memory to be considered relevant
SIMILARITY_THRESHOLD = 0.5

# Singletons
_memory_store = None
_checkpointer = None
_checkpoint_conn = None


class VectorMemoryStore:
    """SQLite-backed vector store with cosine similarity search."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._embeddings = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self._embeddings = GoogleGenerativeAIEmbeddings(
                model="text-embedding-005",
                project=os.environ.get("GCP_PROJECT_ID"),
                task_type="RETRIEVAL_DOCUMENT",
                dimensions=1536,
            )
        return self._embeddings

    async def initialize(self):
        """Create the memories table if it doesn't exist."""
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=DELETE")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"
            )
            await db.commit()

    async def _embed(self, text: str):
        """Generate an embedding vector for text."""
        import numpy as np

        vectors = await self.embeddings.aembed_documents([text])
        return np.array(vectors[0], dtype=np.float32)

    async def add(self, user_id: str, content: str) -> str:
        """Save a memory with its embedding vector."""
        import aiosqlite

        embedding = await self._embed(content)
        memory_id = str(uuid.uuid4())[:8]
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO memories (id, user_id, content, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_id, user_id, content, embedding.tobytes(), time.time()),
            )
            await db.commit()
        return memory_id

    async def search(self, user_id: str, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search over a user's memories using cosine similarity."""
        import aiosqlite
        import numpy as np

        query_embedding = await self._embed(query)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, content, embedding, created_at "
                "FROM memories WHERE user_id = ?",
                (user_id,),
            )
            rows = await cursor.fetchall()

        if not rows:
            return []

        results = []
        for row in rows:
            stored = np.frombuffer(row[2], dtype=np.float32)
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(stored)
            similarity = float(
                np.dot(query_embedding, stored) / (norm_product + 1e-10)
            )
            results.append(
                {
                    "id": row[0],
                    "content": row[1],
                    "similarity": similarity,
                    "created_at": row[3],
                }
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    async def list_memories(self, user_id: str) -> list[dict]:
        """List all memories for a user, newest first."""
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, content, created_at FROM memories "
                "WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
            rows = await cursor.fetchall()
        return [{"id": r[0], "content": r[1], "created_at": r[2]} for r in rows]

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if found and deleted."""
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            await db.commit()
            return cursor.rowcount > 0


async def get_memory_store() -> VectorMemoryStore:
    """Get or create the singleton vector memory store."""
    global _memory_store
    if _memory_store is None:
        os.makedirs(DATA_DIR, exist_ok=True)
        _memory_store = VectorMemoryStore(MEMORY_DB_PATH)
        await _memory_store.initialize()
        logger.info("Vector memory store ready: %s", MEMORY_DB_PATH)
    return _memory_store


async def get_checkpointer():
    """Get or create the singleton SQLite checkpointer for LangGraph."""
    global _checkpointer, _checkpoint_conn
    if _checkpointer is None:
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            os.makedirs(DATA_DIR, exist_ok=True)
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


def create_memory_tools(user_id: str) -> list:
    """Create memory management tools scoped to a specific user."""
    from langchain_core.tools import tool

    @tool
    async def save_memory(content: str) -> str:
        """Save an important fact or preference to long-term memory.
        Use this to remember user preferences, project context, key decisions,
        or anything that should persist across conversations.

        Args:
            content: The fact or information to remember.
        """
        store = await get_memory_store()
        memory_id = await store.add(user_id, content)
        return f"Saved to memory (id: {memory_id})"

    @tool
    async def recall_memories(query: str) -> str:
        """Search long-term memory for relevant past context.
        Use this to find previously saved information about users, projects,
        or decisions.

        Args:
            query: What to search for in memory.
        """
        store = await get_memory_store()
        results = await store.search(user_id, query, top_k=5)
        relevant = [r for r in results if r["similarity"] > SIMILARITY_THRESHOLD]
        if not relevant:
            return "No relevant memories found."
        from datetime import datetime

        lines = []
        for r in relevant:
            ts = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d")
            lines.append(
                f"- [{r['id']}] ({ts}) {r['content']} "
                f"[relevance: {r['similarity']:.0%}]"
            )
        return "Recalled memories:\n" + "\n".join(lines)

    @tool
    async def list_all_memories() -> str:
        """List all saved memories for the current user."""
        store = await get_memory_store()
        memories = await store.list_memories(user_id)
        if not memories:
            return "No memories saved yet."
        from datetime import datetime

        lines = []
        for m in memories:
            ts = datetime.fromtimestamp(m["created_at"]).strftime("%Y-%m-%d")
            lines.append(f"- [{m['id']}] ({ts}) {m['content']}")
        return f"All memories ({len(memories)}):\n" + "\n".join(lines)

    @tool
    async def forget_memory(memory_id: str) -> str:
        """Delete a specific memory by its ID.

        Args:
            memory_id: The ID of the memory to delete.
        """
        store = await get_memory_store()
        deleted = await store.delete(memory_id)
        if deleted:
            return f"Memory {memory_id} deleted."
        return f"Memory {memory_id} not found."

    return [save_memory, recall_memories, list_all_memories, forget_memory]


async def retrieve_relevant_memories(user_id: str, query: str) -> str | None:
    """Auto-retrieve relevant memories for injection into system prompt.

    Returns a formatted string of relevant memories, or None if nothing found.
    Silently returns None on any error to avoid breaking the agent.
    """
    try:
        store = await get_memory_store()
        results = await store.search(user_id, query, top_k=3)
        relevant = [r for r in results if r["similarity"] > SIMILARITY_THRESHOLD]
        if not relevant:
            return None
        lines = [f"- {r['content']}" for r in relevant]
        return (
            "Relevant context from past conversations with this user:\n"
            + "\n".join(lines)
        )
    except Exception as e:
        logger.debug("Memory retrieval failed: %s", e)
        return None
