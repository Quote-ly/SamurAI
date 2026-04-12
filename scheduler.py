"""Background task scheduler using APScheduler + in-process AsyncIOScheduler."""

import json
import logging
import time
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from botbuilder.core import TurnContext
from botbuilder.schema import Activity, ConversationReference

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None

# Set by init_scheduler() — shared with tools/teams_messaging.py
_adapter = None
_app_id: str = ""


async def init_scheduler(adapter, app_id: str) -> AsyncIOScheduler:
    """Initialize the scheduler, load persisted tasks, and start.

    Called once from app.py on_startup.
    """
    global _scheduler, _adapter, _app_id
    _adapter = adapter
    _app_id = app_id

    # Also configure the Teams messaging module with adapter access
    import tools.teams_messaging as teams_msg

    teams_msg._adapter = adapter
    teams_msg._app_id = app_id

    _scheduler = AsyncIOScheduler(
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 300,
        }
    )

    from task_store import get_task_store

    store = await get_task_store()
    tasks = await store.list_tasks(status="active")
    for task in tasks:
        _register_job(task)

    _scheduler.start()
    logger.info("Scheduler started with %d active tasks", len(tasks))
    return _scheduler


def _register_job(task: dict) -> None:
    """Register a single task as an APScheduler job."""
    if not _scheduler:
        return

    job_id = f"task_{task['id']}"

    if task["task_type"] == "recurring" and task.get("cron_expression"):
        try:
            trigger = CronTrigger.from_crontab(task["cron_expression"])
        except ValueError as e:
            logger.error("Invalid cron for task %s: %s", task["id"], e)
            return
    elif task["task_type"] == "one_shot" and task.get("run_at"):
        try:
            run_date = datetime.fromisoformat(task["run_at"])
        except ValueError as e:
            logger.error("Invalid run_at for task %s: %s", task["id"], e)
            return
        trigger = DateTrigger(run_date=run_date)
    else:
        logger.warning("Task %s has no valid trigger config, skipping", task["id"])
        return

    _scheduler.add_job(
        _execute_task,
        trigger=trigger,
        id=job_id,
        args=[task["id"]],
        replace_existing=True,
    )
    logger.info("Registered job %s for task %s", job_id, task["id"])


def _reschedule_one_shot(task_id: str, run_date: datetime) -> None:
    """Re-register a one-shot task with a new DateTrigger for retry."""
    if not _scheduler:
        return
    job_id = f"task_{task_id}"
    _scheduler.add_job(
        _execute_task,
        trigger=DateTrigger(run_date=run_date),
        id=job_id,
        args=[task_id],
        replace_existing=True,
    )
    logger.info("Rescheduled job %s for %s", job_id, run_date.isoformat())


async def _resolve_conversation_ref(store, task: dict) -> str | None:
    """Look up a conversation ref, following the bg_task_ parent chain if needed.

    When a background task spawns sub-tasks, those sub-tasks get a synthetic
    ``bg_task_<parent_id>`` conversation ID that has no saved ref.  This helper
    walks the parent chain until it finds a real (non-synthetic) ref, then
    caches the result so future lookups succeed directly.
    """
    ref_json = await store.get_conversation_ref(task["conversation_id"])
    if ref_json:
        return ref_json

    # Follow bg_task_ → parent task → parent's conversation_id chain
    conv_id = task["conversation_id"]
    visited: set[str] = set()
    while conv_id.startswith("bg_task_") and conv_id not in visited:
        visited.add(conv_id)
        parent_task_id = conv_id.removeprefix("bg_task_")
        parent_task = await store.get_task(parent_task_id)
        if not parent_task:
            break
        ref_json = await store.get_conversation_ref(parent_task["conversation_id"])
        if ref_json:
            # Cache so future lookups succeed directly
            await store.save_conversation_ref(
                conversation_id=task["conversation_id"],
                user_id=task["user_id"],
                ref_json=ref_json,
            )
            return ref_json
        conv_id = parent_task["conversation_id"]

    return None


async def _execute_task(task_id: str) -> None:
    """Execute a background task: run the agent, send results to Teams."""
    from task_store import get_task_store
    from agent import run_agent

    store = await get_task_store()
    task = await store.get_task(task_id)
    if not task or task["status"] != "active":
        return

    # Acquire execution lock — prevents duplicate runs across instances
    if not await store.try_lock(task_id):
        logger.info("Task %s already locked by another instance, skipping", task_id)
        return

    logger.info("Executing task %s: %s", task_id, task["prompt"][:80])

    try:
        # Use a dedicated thread_id so background history stays separate
        bg_conversation_id = f"bg_task_{task_id}"

        # Propagate the original conversation ref to the bg conversation ID
        # so any sub-tasks created during execution can deliver results
        ref_json = await _resolve_conversation_ref(store, task)
        if ref_json:
            await store.save_conversation_ref(
                conversation_id=bg_conversation_id,
                user_id=task["user_id"],
                ref_json=ref_json,
            )

        response = await run_agent(
            user_message=task["prompt"],
            conversation_id=bg_conversation_id,
            user_id=task["user_id"],
            user_name=task["user_name"],
            user_timezone=task["user_timezone"],
            user_email=task["user_email"],
            recursion_limit=50,
            is_background_task=True,
        )

        await _send_task_result(task, response)
        await store.record_run(task_id, success=True)
        logger.info("Task %s completed successfully (run #%d)", task_id, task["run_count"] + 1)

    except Exception as e:
        logger.error("Task %s failed: %s", task_id, e, exc_info=True)
        updated = await store.record_run(
            task_id, success=False, error_message=str(e)
        )

        if updated and updated["status"] == "failed":
            await _send_failure_notification(task, str(e))
            # Remove from scheduler since it's auto-paused
            await pause_task(task_id)
        elif (
            updated
            and task["task_type"] == "one_shot"
            and updated["status"] == "active"
        ):
            # One-shot tasks lose their DateTrigger after firing once.
            # Reschedule with a 60-second delay so the retry has a chance.
            retry_at = datetime.now(timezone.utc) + timedelta(seconds=60)
            logger.info(
                "Rescheduling one-shot task %s for retry at %s",
                task_id,
                retry_at.isoformat(),
            )
            _reschedule_one_shot(task_id, retry_at)
    finally:
        await store.unlock(task_id)


async def _send_task_result(task: dict, response: str) -> None:
    """Send the agent's response to the original Teams conversation."""
    from task_store import get_task_store

    store = await get_task_store()
    ref_json = await _resolve_conversation_ref(store, task)
    if not ref_json:
        logger.error(
            "No conversation ref for task %s, conv %s",
            task["id"],
            task["conversation_id"],
        )
        return

    conv_ref = ConversationReference().deserialize(json.loads(ref_json))

    # Keep the message clean — just deliver the content
    message_text = response

    async def _notify(turn_context: TurnContext):
        await turn_context.send_activity(
            Activity(type="message", text=message_text)
        )

    try:
        await _adapter.continue_conversation(conv_ref, _notify, _app_id)
    except Exception as e:
        logger.error("Proactive message failed for task %s: %s", task["id"], e)


async def _send_failure_notification(task: dict, error: str) -> None:
    """Notify the user that a task has been auto-paused due to repeated failures."""
    from task_store import get_task_store

    store = await get_task_store()
    ref_json = await _resolve_conversation_ref(store, task)
    if not ref_json:
        return

    conv_ref = ConversationReference().deserialize(json.loads(ref_json))

    error_count = task.get("error_count", 0) + 1
    message = (
        f"**Task Auto-Paused** `{task['id']}`\n\n"
        f"Task: _{task['prompt'][:80]}_\n"
        f"Failed {error_count} consecutive times.\n"
        f"Last error: `{error[:200]}`\n\n"
        f"Say **resume task {task['id']}** to retry, or "
        f"**cancel task {task['id']}** to remove it."
    )

    async def _notify(turn_context: TurnContext):
        await turn_context.send_activity(Activity(type="message", text=message))

    try:
        await _adapter.continue_conversation(conv_ref, _notify, _app_id)
    except Exception as e:
        logger.error(
            "Failure notification failed for task %s: %s", task["id"], e
        )


# ── Public API for agent tools ─────────────────────────────────────────


async def schedule_task(task: dict) -> None:
    """Register a newly created task with the scheduler."""
    _register_job(task)


async def pause_task(task_id: str) -> None:
    """Pause a scheduled task (remove from APScheduler, keep in DB)."""
    job_id = f"task_{task_id}"
    if _scheduler and _scheduler.get_job(job_id):
        _scheduler.remove_job(job_id)


async def resume_task(task: dict) -> None:
    """Resume a paused task (re-register with APScheduler)."""
    _register_job(task)


async def cancel_task(task_id: str) -> None:
    """Fully remove a task from the scheduler."""
    await pause_task(task_id)


async def shutdown_scheduler() -> None:
    """Gracefully shut down the scheduler. Called from app.py on_cleanup."""
    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down")
