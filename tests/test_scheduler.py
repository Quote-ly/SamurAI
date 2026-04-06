"""Tests for scheduler.py — APScheduler-based background task execution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import scheduler as scheduler_module


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_task(
    task_id="abc123",
    task_type="recurring",
    cron_expression="0 9 * * *",
    run_at=None,
    status="active",
    prompt="Check deployment status",
    user_id="user-1",
    user_name="Test User",
    user_email="test@example.com",
    user_timezone="America/Chicago",
    conversation_id="conv-1",
    run_count=0,
    error_count=0,
    max_failures=3,
):
    return {
        "id": task_id,
        "task_type": task_type,
        "cron_expression": cron_expression,
        "run_at": run_at,
        "status": status,
        "prompt": prompt,
        "user_id": user_id,
        "user_name": user_name,
        "user_email": user_email,
        "user_timezone": user_timezone,
        "conversation_id": conversation_id,
        "run_count": run_count,
        "error_count": error_count,
        "max_failures": max_failures,
    }


@pytest.fixture(autouse=True)
def _reset_scheduler_globals():
    """Ensure every test starts with a clean scheduler module state."""
    original_scheduler = scheduler_module._scheduler
    original_adapter = scheduler_module._adapter
    original_app_id = scheduler_module._app_id
    yield
    scheduler_module._scheduler = original_scheduler
    scheduler_module._adapter = original_adapter
    scheduler_module._app_id = original_app_id


@pytest.fixture
def mock_scheduler():
    """Provide a MagicMock standing in for AsyncIOScheduler."""
    sched = MagicMock()
    sched.add_job = MagicMock()
    sched.get_job = MagicMock(return_value=MagicMock())
    sched.remove_job = MagicMock()
    sched.shutdown = MagicMock()
    scheduler_module._scheduler = sched
    return sched


@pytest.fixture
def mock_adapter():
    """Provide a MagicMock standing in for BotFrameworkAdapter."""
    adapter = MagicMock()
    adapter.continue_conversation = AsyncMock()
    scheduler_module._adapter = adapter
    scheduler_module._app_id = "test-app-id"
    return adapter


@pytest.fixture
def mock_store():
    """Provide an AsyncMock standing in for TaskStore."""
    store = AsyncMock()
    store.get_task = AsyncMock()
    store.record_run = AsyncMock()
    store.get_conversation_ref = AsyncMock()
    return store


# ── _register_job tests ──────────────────────────────────────────────────


class TestRegisterJob:
    """Tests for _register_job()."""

    def test_registers_recurring_job_with_cron(self, mock_scheduler):
        task = _make_task(task_type="recurring", cron_expression="0 9 * * *")
        scheduler_module._register_job(task)

        mock_scheduler.add_job.assert_called_once()
        call_kwargs = mock_scheduler.add_job.call_args
        assert call_kwargs.kwargs["id"] == "task_abc123"
        assert call_kwargs.kwargs["replace_existing"] is True
        assert call_kwargs.kwargs["args"] == ["abc123"]

    def test_registers_one_shot_job_with_run_at(self, mock_scheduler):
        task = _make_task(
            task_type="one_shot",
            cron_expression=None,
            run_at="2026-06-15T14:00:00",
        )
        scheduler_module._register_job(task)

        mock_scheduler.add_job.assert_called_once()
        call_kwargs = mock_scheduler.add_job.call_args
        assert call_kwargs.kwargs["id"] == "task_abc123"

    def test_invalid_cron_does_not_raise(self, mock_scheduler):
        task = _make_task(task_type="recurring", cron_expression="bad cron")
        # Should log an error but not raise
        scheduler_module._register_job(task)
        mock_scheduler.add_job.assert_not_called()

    def test_invalid_run_at_does_not_raise(self, mock_scheduler):
        task = _make_task(
            task_type="one_shot",
            cron_expression=None,
            run_at="not-a-date",
        )
        scheduler_module._register_job(task)
        mock_scheduler.add_job.assert_not_called()

    def test_missing_trigger_config_skips(self, mock_scheduler):
        task = _make_task(task_type="recurring", cron_expression=None)
        scheduler_module._register_job(task)
        mock_scheduler.add_job.assert_not_called()

    def test_noop_when_scheduler_is_none(self):
        scheduler_module._scheduler = None
        task = _make_task()
        # Should silently return without error
        scheduler_module._register_job(task)


# ── _execute_task tests ──────────────────────────────────────────────────


class TestExecuteTask:
    """Tests for _execute_task()."""

    @pytest.mark.asyncio
    async def test_success_calls_run_agent_and_records(
        self, mock_adapter, mock_store
    ):
        task = _make_task()
        mock_store.get_task.return_value = task
        mock_store.record_run.return_value = task
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
                return_value="Deployment healthy",
            ) as mock_run_agent,
        ):
            await scheduler_module._execute_task("abc123")

        mock_run_agent.assert_called_once_with(
            user_message="Check deployment status",
            conversation_id="bg_task_abc123",
            user_id="user-1",
            user_name="Test User",
            user_timezone="America/Chicago",
            user_email="test@example.com",
        )
        mock_store.record_run.assert_called_once_with("abc123", success=True)

    @pytest.mark.asyncio
    async def test_success_sends_proactive_message(
        self, mock_adapter, mock_store
    ):
        task = _make_task()
        mock_store.get_task.return_value = task
        mock_store.record_run.return_value = task
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
                return_value="All good",
            ),
        ):
            await scheduler_module._execute_task("abc123")

        mock_adapter.continue_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_records_error(self, mock_adapter, mock_store):
        task = _make_task()
        mock_store.get_task.return_value = task
        # Not auto-paused yet (status stays active)
        mock_store.record_run.return_value = {**task, "status": "active", "error_count": 1}
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API timeout"),
            ),
        ):
            await scheduler_module._execute_task("abc123")

        mock_store.record_run.assert_called_once_with(
            "abc123", success=False, error_message="API timeout"
        )

    @pytest.mark.asyncio
    async def test_failure_auto_pause_sends_notification(
        self, mock_scheduler, mock_adapter, mock_store
    ):
        task = _make_task(error_count=2)
        mock_store.get_task.return_value = task
        # Simulate auto-pause: status becomes "failed"
        mock_store.record_run.return_value = {**task, "status": "failed", "error_count": 3}
        mock_store.get_conversation_ref.return_value = '{"conversationId": "conv-1"}'

        with patch(
            "task_store.get_task_store",
            new_callable=AsyncMock,
            return_value=mock_store,
        ), patch(
            "agent.run_agent",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            await scheduler_module._execute_task("abc123")

        # Failure notification sends a proactive message
        # continue_conversation is called for the failure notification
        mock_adapter.continue_conversation.assert_called()
        # The job should also be removed from the scheduler (via pause_task)
        mock_scheduler.remove_job.assert_called_with("task_abc123")

    @pytest.mark.asyncio
    async def test_skips_inactive_task(self, mock_store):
        task = _make_task(status="paused")
        mock_store.get_task.return_value = task

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
            ) as mock_run_agent,
        ):
            await scheduler_module._execute_task("abc123")

        mock_run_agent.assert_not_called()
        mock_store.record_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_missing_task(self, mock_store):
        mock_store.get_task.return_value = None

        with (
            patch(
                "task_store.get_task_store",
                new_callable=AsyncMock,
                return_value=mock_store,
            ),
            patch(
                "agent.run_agent",
                new_callable=AsyncMock,
            ) as mock_run_agent,
        ):
            await scheduler_module._execute_task("nonexistent")

        mock_run_agent.assert_not_called()


# ── Public API tests ─────────────────────────────────────────────────────


class TestPublicAPI:
    """Tests for schedule_task, pause_task, resume_task, cancel_task."""

    @pytest.mark.asyncio
    async def test_schedule_task_registers_job(self, mock_scheduler):
        task = _make_task()
        await scheduler_module.schedule_task(task)
        mock_scheduler.add_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_pause_task_removes_job(self, mock_scheduler):
        mock_scheduler.get_job.return_value = MagicMock()
        await scheduler_module.pause_task("abc123")
        mock_scheduler.remove_job.assert_called_once_with("task_abc123")

    @pytest.mark.asyncio
    async def test_pause_task_noop_when_no_job(self, mock_scheduler):
        mock_scheduler.get_job.return_value = None
        await scheduler_module.pause_task("abc123")
        mock_scheduler.remove_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_pause_task_noop_when_no_scheduler(self):
        scheduler_module._scheduler = None
        # Should not raise
        await scheduler_module.pause_task("abc123")

    @pytest.mark.asyncio
    async def test_resume_task_registers_job(self, mock_scheduler):
        task = _make_task()
        await scheduler_module.resume_task(task)
        mock_scheduler.add_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_task_removes_job(self, mock_scheduler):
        mock_scheduler.get_job.return_value = MagicMock()
        await scheduler_module.cancel_task("abc123")
        mock_scheduler.remove_job.assert_called_once_with("task_abc123")


# ── shutdown_scheduler tests ─────────────────────────────────────────────


class TestShutdownScheduler:
    """Tests for shutdown_scheduler()."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_scheduler_shutdown(self, mock_scheduler):
        await scheduler_module.shutdown_scheduler()
        mock_scheduler.shutdown.assert_called_once_with(wait=False)

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_no_scheduler(self):
        scheduler_module._scheduler = None
        # Should not raise
        await scheduler_module.shutdown_scheduler()
