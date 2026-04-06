"""SQLite-backed persistence for background tasks, conversation references, and team roster."""

import json
import logging
import os
import time
import uuid

import aiosqlite

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("SAMURAI_DATA_DIR", "/data")
TASK_DB_PATH = os.path.join(DATA_DIR, "tasks.sqlite")

_task_store = None


class TaskStore:
    """Manages background tasks, conversation references, and team roster in SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    async def initialize(self) -> None:
        """Create tables and indexes. Called once at startup."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=DELETE")

            await db.execute(
                """CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    user_name TEXT NOT NULL DEFAULT '',
                    user_email TEXT NOT NULL DEFAULT '',
                    user_timezone TEXT NOT NULL DEFAULT '',
                    conversation_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    cron_expression TEXT,
                    run_at TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    last_run_at REAL,
                    next_run_at TEXT,
                    run_count INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    max_failures INTEGER NOT NULL DEFAULT 3,
                    locked_until REAL NOT NULL DEFAULT 0
                )"""
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)"
            )

            # Migration: add locked_until column to existing tables
            try:
                await db.execute(
                    "ALTER TABLE tasks ADD COLUMN locked_until REAL NOT NULL DEFAULT 0"
                )
            except Exception:
                pass  # Column already exists

            await db.execute(
                """CREATE TABLE IF NOT EXISTS conversation_refs (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    ref_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )"""
            )

            await db.execute(
                """CREATE TABLE IF NOT EXISTS team_roster (
                    email TEXT PRIMARY KEY,
                    teams_id TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    service_url TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                )"""
            )

            await db.commit()
        logger.info("Task store initialized: %s", self.db_path)

    # ── Task CRUD ──────────────────────────────────────────────────────

    async def create_task(
        self,
        user_id: str,
        user_name: str,
        user_email: str,
        user_timezone: str,
        conversation_id: str,
        task_type: str,
        prompt: str,
        cron_expression: str | None = None,
        run_at: str | None = None,
    ) -> dict:
        """Insert a new task. Returns the full task dict."""
        task_id = str(uuid.uuid4())[:8]
        now = time.time()
        task = {
            "id": task_id,
            "user_id": user_id,
            "user_name": user_name,
            "user_email": user_email,
            "user_timezone": user_timezone,
            "conversation_id": conversation_id,
            "task_type": task_type,
            "prompt": prompt,
            "cron_expression": cron_expression,
            "run_at": run_at,
            "status": "active",
            "created_at": now,
            "last_run_at": None,
            "next_run_at": None,
            "run_count": 0,
            "error_count": 0,
            "last_error": None,
            "max_failures": 3,
        }
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO tasks
                   (id, user_id, user_name, user_email, user_timezone,
                    conversation_id, task_type, prompt, cron_expression,
                    run_at, status, created_at, run_count, error_count, max_failures)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    user_id,
                    user_name,
                    user_email,
                    user_timezone,
                    conversation_id,
                    task_type,
                    prompt,
                    cron_expression,
                    run_at,
                    "active",
                    now,
                    0,
                    0,
                    3,
                ),
            )
            await db.commit()
        logger.info("Created task %s: %s", task_id, prompt[:60])
        return task

    async def get_task(self, task_id: str) -> dict | None:
        """Fetch a single task by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_tasks(
        self,
        user_id: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """List tasks, optionally filtered by user and/or status."""
        query = "SELECT * FROM tasks"
        params: list = []
        conditions: list[str] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def update_task(self, task_id: str, **fields) -> bool:
        """Update arbitrary fields on a task. Returns True if row existed."""
        if not fields:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [task_id]
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"UPDATE tasks SET {set_clause} WHERE id = ?", values
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete_task(self, task_id: str) -> bool:
        """Hard-delete a task by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def try_lock(self, task_id: str, lock_duration: float = 300) -> bool:
        """Atomically try to lock a task for execution.

        Returns True if lock acquired, False if another instance already holds it.
        Uses an atomic UPDATE ... WHERE to prevent race conditions.
        """
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE tasks SET locked_until = ? WHERE id = ? AND locked_until < ?",
                (now + lock_duration, task_id, now),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def unlock(self, task_id: str) -> None:
        """Release the execution lock on a task."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE tasks SET locked_until = 0 WHERE id = ?", (task_id,)
            )
            await db.commit()

    async def record_run(
        self,
        task_id: str,
        success: bool,
        error_message: str | None = None,
    ) -> dict | None:
        """After execution: update run_count, handle errors, auto-pause if needed.

        Returns updated task dict, or None if task not found.
        """
        task = await self.get_task(task_id)
        if not task:
            return None

        now = time.time()
        updates: dict = {"last_run_at": now, "run_count": task["run_count"] + 1}

        if success:
            updates["error_count"] = 0
            updates["last_error"] = None
            if task["task_type"] == "one_shot":
                updates["status"] = "completed"
        else:
            new_error_count = task["error_count"] + 1
            updates["error_count"] = new_error_count
            updates["last_error"] = (error_message or "Unknown error")[:500]
            if new_error_count >= task["max_failures"]:
                updates["status"] = "failed"
                logger.warning(
                    "Task %s auto-paused after %d consecutive failures",
                    task_id,
                    new_error_count,
                )

        await self.update_task(task_id, **updates)
        task.update(updates)
        return task

    # ── Conversation References ────────────────────────────────────────

    async def save_conversation_ref(
        self,
        conversation_id: str,
        user_id: str,
        ref_json: str,
    ) -> None:
        """Upsert a serialized ConversationReference."""
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO conversation_refs (conversation_id, user_id, ref_json, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(conversation_id) DO UPDATE SET
                       user_id = excluded.user_id,
                       ref_json = excluded.ref_json,
                       updated_at = excluded.updated_at""",
                (conversation_id, user_id, ref_json, now),
            )
            await db.commit()

    async def get_conversation_ref(self, conversation_id: str) -> str | None:
        """Load serialized ConversationReference JSON. Returns None if not found."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT ref_json FROM conversation_refs WHERE conversation_id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            return row[0] if row else None

    # ── Team Roster ────────────────────────────────────────────────────

    async def save_team_member(
        self,
        email: str,
        teams_id: str,
        display_name: str,
        service_url: str,
        tenant_id: str = "",
    ) -> None:
        """Upsert a team member into the roster."""
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO team_roster (email, teams_id, display_name, service_url, tenant_id, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(email) DO UPDATE SET
                       teams_id = excluded.teams_id,
                       display_name = excluded.display_name,
                       service_url = excluded.service_url,
                       tenant_id = excluded.tenant_id,
                       updated_at = excluded.updated_at""",
                (email.lower(), teams_id, display_name, service_url, tenant_id, now),
            )
            await db.commit()

    async def get_team_member(self, email: str) -> dict | None:
        """Look up a team member by email."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM team_roster WHERE email = ?", (email.lower(),)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_team_members(self) -> list[dict]:
        """Return all known team members."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM team_roster ORDER BY display_name"
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def get_task_store() -> TaskStore:
    """Get or create the singleton TaskStore."""
    global _task_store
    if _task_store is None:
        os.makedirs(DATA_DIR, exist_ok=True)
        _task_store = TaskStore(TASK_DB_PATH)
        await _task_store.initialize()
    return _task_store
