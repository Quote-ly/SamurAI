"""Tests for task_store.py — TaskStore CRUD, record_run, conversation refs, team roster."""

import json
import os

import aiosqlite
import pytest

from task_store import TaskStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path):
    """Fresh TaskStore backed by a temp SQLite file."""
    s = TaskStore(str(tmp_path / "test_tasks.sqlite"))
    await s.initialize()
    return s


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_creates_tables(store):
    """All three tables should exist after initialize()."""
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in await cursor.fetchall()}
    assert "tasks" in tables
    assert "conversation_refs" in tables
    assert "team_roster" in tables


@pytest.mark.asyncio
async def test_initialize_creates_indexes(store):
    """Indexes on user_id and status should exist."""
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row[0] for row in await cursor.fetchall()}
    assert "idx_tasks_user" in indexes
    assert "idx_tasks_status" in indexes


@pytest.mark.asyncio
async def test_initialize_is_idempotent(tmp_path):
    """Calling initialize() twice should not raise."""
    s = TaskStore(str(tmp_path / "tasks.sqlite"))
    await s.initialize()
    await s.initialize()  # should not raise


# ---------------------------------------------------------------------------
# create_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_task_returns_dict_with_correct_fields(store):
    task = await store.create_task(
        user_id="user-1",
        user_name="Alice",
        user_email="alice@example.com",
        user_timezone="America/New_York",
        conversation_id="conv-1",
        task_type="recurring",
        prompt="Send daily standup reminder",
        cron_expression="0 9 * * *",
    )
    assert isinstance(task, dict)
    assert len(task["id"]) == 8
    assert task["user_id"] == "user-1"
    assert task["user_name"] == "Alice"
    assert task["user_email"] == "alice@example.com"
    assert task["user_timezone"] == "America/New_York"
    assert task["conversation_id"] == "conv-1"
    assert task["task_type"] == "recurring"
    assert task["prompt"] == "Send daily standup reminder"
    assert task["cron_expression"] == "0 9 * * *"
    assert task["run_at"] is None
    assert task["status"] == "active"
    assert task["created_at"] > 0
    assert task["last_run_at"] is None
    assert task["next_run_at"] is None
    assert task["run_count"] == 0
    assert task["error_count"] == 0
    assert task["last_error"] is None
    assert task["max_failures"] == 3


@pytest.mark.asyncio
async def test_create_task_persists_to_sqlite(store):
    await store.create_task(
        user_id="user-1",
        user_name="Alice",
        user_email="alice@example.com",
        user_timezone="UTC",
        conversation_id="conv-1",
        task_type="one_shot",
        prompt="Run once",
    )
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM tasks")
        count = (await cursor.fetchone())[0]
    assert count == 1


@pytest.mark.asyncio
async def test_create_task_with_run_at(store):
    task = await store.create_task(
        user_id="user-1",
        user_name="Bob",
        user_email="bob@example.com",
        user_timezone="UTC",
        conversation_id="conv-2",
        task_type="one_shot",
        prompt="Run at specific time",
        run_at="2026-04-07T09:00:00Z",
    )
    assert task["run_at"] == "2026-04-07T09:00:00Z"
    assert task["cron_expression"] is None


# ---------------------------------------------------------------------------
# get_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_task_returns_existing(store):
    created = await store.create_task(
        user_id="user-1",
        user_name="Alice",
        user_email="alice@example.com",
        user_timezone="UTC",
        conversation_id="conv-1",
        task_type="recurring",
        prompt="Check metrics",
    )
    fetched = await store.get_task(created["id"])
    assert fetched is not None
    assert fetched["id"] == created["id"]
    assert fetched["prompt"] == "Check metrics"


@pytest.mark.asyncio
async def test_get_task_returns_none_for_missing(store):
    result = await store.get_task("nonexistent-id")
    assert result is None


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tasks_returns_all(store):
    await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="task 1",
    )
    await store.create_task(
        user_id="user-2", user_name="B", user_email="b@test.com",
        user_timezone="UTC", conversation_id="c2", task_type="one_shot",
        prompt="task 2",
    )
    tasks = await store.list_tasks()
    assert len(tasks) == 2


@pytest.mark.asyncio
async def test_list_tasks_filter_by_user_id(store):
    await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="user-1 task",
    )
    await store.create_task(
        user_id="user-2", user_name="B", user_email="b@test.com",
        user_timezone="UTC", conversation_id="c2", task_type="recurring",
        prompt="user-2 task",
    )
    tasks = await store.list_tasks(user_id="user-1")
    assert len(tasks) == 1
    assert tasks[0]["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_list_tasks_filter_by_status(store):
    t1 = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="active task",
    )
    t2 = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c2", task_type="one_shot",
        prompt="will be paused",
    )
    await store.update_task(t2["id"], status="paused")

    active_tasks = await store.list_tasks(status="active")
    assert len(active_tasks) == 1
    assert active_tasks[0]["id"] == t1["id"]

    paused_tasks = await store.list_tasks(status="paused")
    assert len(paused_tasks) == 1
    assert paused_tasks[0]["id"] == t2["id"]


@pytest.mark.asyncio
async def test_list_tasks_filter_by_user_and_status(store):
    await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="u1 active",
    )
    t2 = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c2", task_type="recurring",
        prompt="u1 paused",
    )
    await store.update_task(t2["id"], status="paused")
    await store.create_task(
        user_id="user-2", user_name="B", user_email="b@test.com",
        user_timezone="UTC", conversation_id="c3", task_type="recurring",
        prompt="u2 active",
    )

    result = await store.list_tasks(user_id="user-1", status="active")
    assert len(result) == 1
    assert result[0]["prompt"] == "u1 active"


@pytest.mark.asyncio
async def test_list_tasks_empty(store):
    tasks = await store.list_tasks()
    assert tasks == []


@pytest.mark.asyncio
async def test_list_tasks_ordered_by_created_at_desc(store):
    t1 = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="first",
    )
    t2 = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c2", task_type="recurring",
        prompt="second",
    )
    tasks = await store.list_tasks()
    # Most recently created should come first
    assert tasks[0]["id"] == t2["id"]
    assert tasks[1]["id"] == t1["id"]


# ---------------------------------------------------------------------------
# update_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_task_changes_fields(store):
    task = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="original prompt",
    )
    result = await store.update_task(task["id"], prompt="updated prompt", status="paused")
    assert result is True

    updated = await store.get_task(task["id"])
    assert updated["prompt"] == "updated prompt"
    assert updated["status"] == "paused"


@pytest.mark.asyncio
async def test_update_task_returns_false_for_missing(store):
    result = await store.update_task("nonexistent", status="paused")
    assert result is False


@pytest.mark.asyncio
async def test_update_task_with_no_fields(store):
    result = await store.update_task("any-id")
    assert result is False


# ---------------------------------------------------------------------------
# delete_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_task_removes_row(store):
    task = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="delete me",
    )
    assert await store.delete_task(task["id"]) is True
    assert await store.get_task(task["id"]) is None


@pytest.mark.asyncio
async def test_delete_task_returns_false_for_missing(store):
    assert await store.delete_task("nonexistent") is False


@pytest.mark.asyncio
async def test_delete_task_does_not_affect_others(store):
    t1 = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="keep me",
    )
    t2 = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c2", task_type="recurring",
        prompt="delete me",
    )
    await store.delete_task(t2["id"])
    remaining = await store.list_tasks()
    assert len(remaining) == 1
    assert remaining[0]["id"] == t1["id"]


# ---------------------------------------------------------------------------
# record_run — success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_run_success_increments_run_count(store):
    task = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="run me",
    )
    updated = await store.record_run(task["id"], success=True)
    assert updated["run_count"] == 1
    assert updated["error_count"] == 0
    assert updated["last_error"] is None
    assert updated["last_run_at"] is not None


@pytest.mark.asyncio
async def test_record_run_success_resets_error_count(store):
    task = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="flaky task",
    )
    # Simulate two failures
    await store.record_run(task["id"], success=False, error_message="err1")
    await store.record_run(task["id"], success=False, error_message="err2")

    mid_state = await store.get_task(task["id"])
    assert mid_state["error_count"] == 2

    # Now succeed — error_count resets to 0
    updated = await store.record_run(task["id"], success=True)
    assert updated["error_count"] == 0
    assert updated["last_error"] is None
    assert updated["run_count"] == 3


# ---------------------------------------------------------------------------
# record_run — failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_run_failure_increments_error_count(store):
    task = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="failing task",
    )
    updated = await store.record_run(task["id"], success=False, error_message="oops")
    assert updated["run_count"] == 1
    assert updated["error_count"] == 1
    assert updated["last_error"] == "oops"
    assert updated["status"] == "active"  # not yet at max_failures


@pytest.mark.asyncio
async def test_record_run_failure_truncates_long_error(store):
    task = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="verbose errors",
    )
    long_error = "x" * 1000
    updated = await store.record_run(task["id"], success=False, error_message=long_error)
    assert len(updated["last_error"]) == 500


@pytest.mark.asyncio
async def test_record_run_failure_default_error_message(store):
    task = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="no message",
    )
    updated = await store.record_run(task["id"], success=False)
    assert updated["last_error"] == "Unknown error"


# ---------------------------------------------------------------------------
# record_run — auto-pause at max_failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_run_auto_pauses_at_max_failures(store):
    task = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="fragile task",
    )
    # Default max_failures is 3; fail three times
    await store.record_run(task["id"], success=False, error_message="err")
    await store.record_run(task["id"], success=False, error_message="err")
    updated = await store.record_run(task["id"], success=False, error_message="err")

    assert updated["status"] == "failed"
    assert updated["error_count"] == 3


@pytest.mark.asyncio
async def test_record_run_does_not_pause_before_max_failures(store):
    task = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="semi-fragile",
    )
    await store.record_run(task["id"], success=False, error_message="err")
    updated = await store.record_run(task["id"], success=False, error_message="err")
    assert updated["status"] == "active"
    assert updated["error_count"] == 2


# ---------------------------------------------------------------------------
# record_run — one_shot completes on success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_run_one_shot_completes_on_success(store):
    task = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="one_shot",
        prompt="run once",
    )
    updated = await store.record_run(task["id"], success=True)
    assert updated["status"] == "completed"


@pytest.mark.asyncio
async def test_record_run_recurring_stays_active_on_success(store):
    task = await store.create_task(
        user_id="user-1", user_name="A", user_email="a@test.com",
        user_timezone="UTC", conversation_id="c1", task_type="recurring",
        prompt="keep going",
    )
    updated = await store.record_run(task["id"], success=True)
    assert updated["status"] == "active"


# ---------------------------------------------------------------------------
# record_run — missing task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_run_returns_none_for_missing_task(store):
    result = await store.record_run("nonexistent", success=True)
    assert result is None


# ---------------------------------------------------------------------------
# Conversation References — save / get round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_get_conversation_ref(store):
    ref_data = json.dumps({"bot_id": "bot-1", "service_url": "https://example.com"})
    await store.save_conversation_ref("conv-1", "user-1", ref_data)

    loaded = await store.get_conversation_ref("conv-1")
    assert loaded is not None
    assert json.loads(loaded) == {"bot_id": "bot-1", "service_url": "https://example.com"}


@pytest.mark.asyncio
async def test_get_conversation_ref_returns_none_for_missing(store):
    result = await store.get_conversation_ref("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_save_conversation_ref_upserts_on_conflict(store):
    await store.save_conversation_ref("conv-1", "user-1", '{"v": 1}')
    await store.save_conversation_ref("conv-1", "user-1", '{"v": 2}')

    loaded = await store.get_conversation_ref("conv-1")
    assert json.loads(loaded) == {"v": 2}

    # Verify only one row exists
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM conversation_refs")
        count = (await cursor.fetchone())[0]
    assert count == 1


@pytest.mark.asyncio
async def test_save_conversation_ref_upserts_user_id(store):
    await store.save_conversation_ref("conv-1", "user-1", '{"v": 1}')
    await store.save_conversation_ref("conv-1", "user-2", '{"v": 2}')

    async with aiosqlite.connect(store.db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM conversation_refs WHERE conversation_id = ?",
            ("conv-1",),
        )
        row = dict(await cursor.fetchone())
    assert row["user_id"] == "user-2"
    assert json.loads(row["ref_json"]) == {"v": 2}


# ---------------------------------------------------------------------------
# Team Roster — save / get / list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_and_get_team_member(store):
    await store.save_team_member(
        email="Alice@Example.com",
        teams_id="tid-1",
        display_name="Alice",
        service_url="https://smba.trafficmanager.net/teams",
        tenant_id="tenant-1",
    )
    member = await store.get_team_member("alice@example.com")
    assert member is not None
    assert member["email"] == "alice@example.com"  # lowercased
    assert member["teams_id"] == "tid-1"
    assert member["display_name"] == "Alice"
    assert member["service_url"] == "https://smba.trafficmanager.net/teams"
    assert member["tenant_id"] == "tenant-1"
    assert member["updated_at"] > 0


@pytest.mark.asyncio
async def test_get_team_member_returns_none_for_missing(store):
    result = await store.get_team_member("nobody@example.com")
    assert result is None


@pytest.mark.asyncio
async def test_get_team_member_case_insensitive_lookup(store):
    await store.save_team_member(
        email="Bob@Example.COM",
        teams_id="tid-2",
        display_name="Bob",
        service_url="https://example.com",
    )
    # Lookup with different casing
    member = await store.get_team_member("BOB@EXAMPLE.COM")
    assert member is not None
    assert member["display_name"] == "Bob"


@pytest.mark.asyncio
async def test_save_team_member_upserts_on_conflict(store):
    await store.save_team_member(
        email="alice@example.com",
        teams_id="tid-1",
        display_name="Alice",
        service_url="https://old.example.com",
        tenant_id="tenant-1",
    )
    await store.save_team_member(
        email="alice@example.com",
        teams_id="tid-1-new",
        display_name="Alice Updated",
        service_url="https://new.example.com",
        tenant_id="tenant-2",
    )
    member = await store.get_team_member("alice@example.com")
    assert member["teams_id"] == "tid-1-new"
    assert member["display_name"] == "Alice Updated"
    assert member["service_url"] == "https://new.example.com"
    assert member["tenant_id"] == "tenant-2"

    # Only one row
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM team_roster")
        count = (await cursor.fetchone())[0]
    assert count == 1


@pytest.mark.asyncio
async def test_save_team_member_default_tenant_id(store):
    await store.save_team_member(
        email="carol@example.com",
        teams_id="tid-3",
        display_name="Carol",
        service_url="https://example.com",
    )
    member = await store.get_team_member("carol@example.com")
    assert member["tenant_id"] == ""


@pytest.mark.asyncio
async def test_list_team_members_empty(store):
    members = await store.list_team_members()
    assert members == []


@pytest.mark.asyncio
async def test_list_team_members_returns_all(store):
    await store.save_team_member(
        email="alice@example.com", teams_id="t1",
        display_name="Alice", service_url="https://example.com",
    )
    await store.save_team_member(
        email="bob@example.com", teams_id="t2",
        display_name="Bob", service_url="https://example.com",
    )
    await store.save_team_member(
        email="carol@example.com", teams_id="t3",
        display_name="Carol", service_url="https://example.com",
    )
    members = await store.list_team_members()
    assert len(members) == 3


@pytest.mark.asyncio
async def test_list_team_members_ordered_by_display_name(store):
    await store.save_team_member(
        email="carol@example.com", teams_id="t3",
        display_name="Carol", service_url="https://example.com",
    )
    await store.save_team_member(
        email="alice@example.com", teams_id="t1",
        display_name="Alice", service_url="https://example.com",
    )
    await store.save_team_member(
        email="bob@example.com", teams_id="t2",
        display_name="Bob", service_url="https://example.com",
    )
    members = await store.list_team_members()
    names = [m["display_name"] for m in members]
    assert names == ["Alice", "Bob", "Carol"]
