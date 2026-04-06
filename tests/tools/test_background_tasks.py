"""Tests for tools.background_tasks -- creating, listing, pausing, resuming, cancelling tasks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

CONV_ID = "test-conv-123"
USER_EMAIL = "devin@virtualdojo.com"


@pytest.fixture(autouse=True)
def clear_context():
    """Clear the pending task context before/after each test."""
    from tools.background_tasks import _pending_task_context

    _pending_task_context.clear()
    yield
    _pending_task_context.clear()


@pytest.fixture
def seed_context():
    """Seed _pending_task_context so the tools can look up user metadata."""
    from tools.background_tasks import _pending_task_context

    _pending_task_context[CONV_ID] = {
        "user_id": "u-001",
        "user_name": "Devin Henderson",
        "user_timezone": "America/New_York",
    }


# ---------------------------------------------------------------------------
# create_background_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_task_invalid_type():
    from tools.background_tasks import create_background_task

    result = await create_background_task.ainvoke(
        {
            "prompt": "Do something",
            "task_type": "invalid",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )
    assert "Error" in result
    assert "must be" in result


@pytest.mark.asyncio
async def test_create_recurring_requires_cron():
    from tools.background_tasks import create_background_task

    result = await create_background_task.ainvoke(
        {
            "prompt": "Do something",
            "task_type": "recurring",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )
    assert "cron_expression is required" in result


@pytest.mark.asyncio
async def test_create_one_shot_requires_run_at():
    from tools.background_tasks import create_background_task

    result = await create_background_task.ainvoke(
        {
            "prompt": "Do something",
            "task_type": "one_shot",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )
    assert "run_at is required" in result


@pytest.mark.asyncio
@patch("scheduler.schedule_task", new_callable=AsyncMock)
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_create_recurring_task(mock_get_store, mock_schedule, seed_context):
    from tools.background_tasks import create_background_task

    mock_store = AsyncMock()
    mock_store.create_task.return_value = {"id": "task-abc"}
    mock_get_store.return_value = mock_store

    result = await create_background_task.ainvoke(
        {
            "prompt": "Check PR status",
            "task_type": "recurring",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
            "cron_expression": "0 * * * *",
        }
    )

    assert "task-abc" in result
    assert "0 * * * *" in result
    mock_store.create_task.assert_awaited_once()
    call_kwargs = mock_store.create_task.call_args.kwargs
    assert call_kwargs["task_type"] == "recurring"
    assert call_kwargs["cron_expression"] == "0 * * * *"
    assert call_kwargs["user_id"] == "u-001"
    mock_schedule.assert_awaited_once_with({"id": "task-abc"})


@pytest.mark.asyncio
@patch("scheduler.schedule_task", new_callable=AsyncMock)
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_create_one_shot_task(mock_get_store, mock_schedule, seed_context):
    from tools.background_tasks import create_background_task

    mock_store = AsyncMock()
    mock_store.create_task.return_value = {"id": "task-xyz"}
    mock_get_store.return_value = mock_store

    result = await create_background_task.ainvoke(
        {
            "prompt": "Send reminder",
            "task_type": "one_shot",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
            "run_at": "2026-04-06T15:30:00Z",
        }
    )

    assert "task-xyz" in result
    assert "2026-04-06T15:30:00Z" in result
    call_kwargs = mock_store.create_task.call_args.kwargs
    assert call_kwargs["task_type"] == "one_shot"
    assert call_kwargs["run_at"] == "2026-04-06T15:30:00Z"
    mock_schedule.assert_awaited_once()


# ---------------------------------------------------------------------------
# list_background_tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_list_tasks_formatted(mock_get_store, seed_context):
    from tools.background_tasks import list_background_tasks

    mock_store = AsyncMock()
    mock_store.list_tasks.return_value = [
        {
            "id": "task-1",
            "status": "active",
            "prompt": "Check PR status hourly",
            "cron_expression": "0 * * * *",
            "run_count": 5,
            "created_at": 1712400000,
            "last_error": None,
        },
        {
            "id": "task-2",
            "status": "paused",
            "prompt": "Daily standup reminder",
            "cron_expression": "0 9 * * *",
            "run_count": 12,
            "created_at": 1712300000,
            "last_error": None,
        },
    ]
    mock_get_store.return_value = mock_store

    result = await list_background_tasks.ainvoke(
        {
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "task-1" in result
    assert "task-2" in result
    assert "running" in result
    assert "paused" in result
    assert "Background Tasks" in result
    assert "(2)" in result


@pytest.mark.asyncio
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_list_tasks_filters_by_status(mock_get_store, seed_context):
    from tools.background_tasks import list_background_tasks

    mock_store = AsyncMock()
    mock_store.list_tasks.return_value = [
        {
            "id": "task-1",
            "status": "active",
            "prompt": "Active task",
            "cron_expression": "0 * * * *",
            "run_count": 1,
            "created_at": 1712400000,
            "last_error": None,
        },
        {
            "id": "task-2",
            "status": "completed",
            "prompt": "Done task",
            "run_at": "2026-04-01T12:00:00Z",
            "run_count": 1,
            "created_at": 1712300000,
            "last_error": None,
        },
    ]
    mock_get_store.return_value = mock_store

    # Default: only active/paused
    result = await list_background_tasks.ainvoke(
        {
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )
    assert "task-1" in result
    assert "task-2" not in result

    # show_all=True: includes completed
    result_all = await list_background_tasks.ainvoke(
        {
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
            "show_all": True,
        }
    )
    assert "task-1" in result_all
    assert "task-2" in result_all


@pytest.mark.asyncio
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_list_tasks_empty(mock_get_store, seed_context):
    from tools.background_tasks import list_background_tasks

    mock_store = AsyncMock()
    mock_store.list_tasks.return_value = []
    mock_get_store.return_value = mock_store

    result = await list_background_tasks.ainvoke(
        {
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )
    assert "No background tasks found" in result


@pytest.mark.asyncio
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_list_tasks_shows_last_error(mock_get_store, seed_context):
    from tools.background_tasks import list_background_tasks

    mock_store = AsyncMock()
    mock_store.list_tasks.return_value = [
        {
            "id": "task-err",
            "status": "active",
            "prompt": "Broken task",
            "cron_expression": "0 * * * *",
            "run_count": 3,
            "created_at": 1712400000,
            "last_error": "Connection timeout to API",
        },
    ]
    mock_get_store.return_value = mock_store

    result = await list_background_tasks.ainvoke(
        {
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
            "show_all": True,
        }
    )
    assert "Connection timeout" in result


# ---------------------------------------------------------------------------
# pause_background_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("scheduler.pause_task", new_callable=AsyncMock)
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_pause_active_task(mock_get_store, mock_pause):
    from tools.background_tasks import pause_background_task

    mock_store = AsyncMock()
    mock_store.get_task.return_value = {"id": "task-1", "status": "active"}
    mock_get_store.return_value = mock_store

    result = await pause_background_task.ainvoke(
        {
            "task_id": "task-1",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "paused" in result.lower()
    mock_pause.assert_awaited_once_with("task-1")
    mock_store.update_task.assert_awaited_once_with("task-1", status="paused")


@pytest.mark.asyncio
@patch("scheduler.pause_task", new_callable=AsyncMock)
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_pause_non_active_rejected(mock_get_store, mock_pause):
    from tools.background_tasks import pause_background_task

    mock_store = AsyncMock()
    mock_store.get_task.return_value = {"id": "task-1", "status": "paused"}
    mock_get_store.return_value = mock_store

    result = await pause_background_task.ainvoke(
        {
            "task_id": "task-1",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "not active" in result
    mock_pause.assert_not_awaited()


@pytest.mark.asyncio
@patch("scheduler.pause_task", new_callable=AsyncMock)
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_pause_not_found(mock_get_store, mock_pause):
    from tools.background_tasks import pause_background_task

    mock_store = AsyncMock()
    mock_store.get_task.return_value = None
    mock_get_store.return_value = mock_store

    result = await pause_background_task.ainvoke(
        {
            "task_id": "task-missing",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "not found" in result
    mock_pause.assert_not_awaited()


# ---------------------------------------------------------------------------
# resume_background_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("scheduler.resume_task", new_callable=AsyncMock)
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_resume_paused_task(mock_get_store, mock_resume):
    from tools.background_tasks import resume_background_task

    mock_store = AsyncMock()
    task_data = {"id": "task-1", "status": "paused"}
    mock_store.get_task.return_value = task_data
    mock_get_store.return_value = mock_store

    result = await resume_background_task.ainvoke(
        {
            "task_id": "task-1",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "resumed" in result.lower()
    mock_store.update_task.assert_awaited_once_with(
        "task-1", status="active", error_count=0, last_error=None
    )
    mock_resume.assert_awaited_once()


@pytest.mark.asyncio
@patch("scheduler.resume_task", new_callable=AsyncMock)
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_resume_failed_task(mock_get_store, mock_resume):
    from tools.background_tasks import resume_background_task

    mock_store = AsyncMock()
    task_data = {"id": "task-2", "status": "failed"}
    mock_store.get_task.return_value = task_data
    mock_get_store.return_value = mock_store

    result = await resume_background_task.ainvoke(
        {
            "task_id": "task-2",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "resumed" in result.lower()
    mock_resume.assert_awaited_once()


@pytest.mark.asyncio
@patch("scheduler.resume_task", new_callable=AsyncMock)
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_resume_active_rejected(mock_get_store, mock_resume):
    from tools.background_tasks import resume_background_task

    mock_store = AsyncMock()
    mock_store.get_task.return_value = {"id": "task-1", "status": "active"}
    mock_get_store.return_value = mock_store

    result = await resume_background_task.ainvoke(
        {
            "task_id": "task-1",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "cannot be resumed" in result
    mock_resume.assert_not_awaited()


@pytest.mark.asyncio
@patch("scheduler.resume_task", new_callable=AsyncMock)
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_resume_not_found(mock_get_store, mock_resume):
    from tools.background_tasks import resume_background_task

    mock_store = AsyncMock()
    mock_store.get_task.return_value = None
    mock_get_store.return_value = mock_store

    result = await resume_background_task.ainvoke(
        {
            "task_id": "task-missing",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "not found" in result
    mock_resume.assert_not_awaited()


# ---------------------------------------------------------------------------
# cancel_background_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("scheduler.cancel_task", new_callable=AsyncMock)
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_cancel_task(mock_get_store, mock_cancel):
    from tools.background_tasks import cancel_background_task

    mock_store = AsyncMock()
    mock_store.get_task.return_value = {"id": "task-1", "status": "active"}
    mock_get_store.return_value = mock_store

    result = await cancel_background_task.ainvoke(
        {
            "task_id": "task-1",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "cancelled" in result.lower()
    mock_cancel.assert_awaited_once_with("task-1")
    mock_store.delete_task.assert_awaited_once_with("task-1")


@pytest.mark.asyncio
@patch("scheduler.cancel_task", new_callable=AsyncMock)
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_cancel_not_found(mock_get_store, mock_cancel):
    from tools.background_tasks import cancel_background_task

    mock_store = AsyncMock()
    mock_store.get_task.return_value = None
    mock_get_store.return_value = mock_store

    result = await cancel_background_task.ainvoke(
        {
            "task_id": "task-missing",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "not found" in result
    mock_cancel.assert_not_awaited()
