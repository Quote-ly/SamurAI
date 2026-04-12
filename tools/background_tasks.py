"""Agent tools for creating, listing, and managing autonomous background tasks."""

import logging
from datetime import datetime

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Context dict populated by app.py before agent runs, so tools can access user metadata.
# Key: conversation_id, Value: {user_id, user_name, user_timezone}
_pending_task_context: dict[str, dict] = {}


@tool
async def create_background_task(
    prompt: str,
    task_type: str,
    conversation_id: str,
    user_email: str,
    cron_expression: str = "",
    run_at: str = "",
) -> str:
    """Create a background task that the agent will run autonomously.

    For recurring tasks, provide a cron_expression. For one-shot deferred tasks, provide run_at.

    Examples:
    - Recurring hourly: task_type="recurring", cron_expression="0 * * * *"
    - Every 30 min: task_type="recurring", cron_expression="*/30 * * * *"
    - Monday 9am UTC: task_type="recurring", cron_expression="0 9 * * 1"
    - Daily 9am UTC: task_type="recurring", cron_expression="0 9 * * *"
    - One-shot: task_type="one_shot", run_at="2026-04-06T15:30:00Z"

    IMPORTANT for communication tasks: Write the prompt to FIRST CHECK if the action
    is still necessary before acting. Example: "Check if John has already reviewed
    PR #42. If not, send him a Teams message reminding him. If yes, skip."

    Args:
        prompt: The instruction for the agent to execute. Be specific and include
                conditions for when to act vs skip.
        task_type: Either 'recurring' or 'one_shot'.
        conversation_id: The current conversation ID (from context brackets).
        user_email: The user's email (from context brackets).
        cron_expression: Cron schedule for recurring tasks. Required if recurring.
        run_at: ISO 8601 datetime for one-shot tasks. Required if one_shot.
    """
    if task_type not in ("recurring", "one_shot"):
        return "Error: task_type must be 'recurring' or 'one_shot'."
    if task_type == "recurring" and not cron_expression:
        return "Error: cron_expression is required for recurring tasks."
    if task_type == "one_shot" and not run_at:
        return "Error: run_at is required for one-shot tasks."

    ctx = _pending_task_context.get(conversation_id, {})

    from task_store import get_task_store
    from scheduler import schedule_task

    store = await get_task_store()
    # One-shot tasks get 1 retry (2 total attempts); recurring tasks get 3
    max_failures = 2 if task_type == "one_shot" else 3
    task = await store.create_task(
        user_id=ctx.get("user_id", "unknown"),
        user_name=ctx.get("user_name", ""),
        user_email=user_email,
        user_timezone=ctx.get("user_timezone", ""),
        conversation_id=conversation_id,
        task_type=task_type,
        prompt=prompt,
        cron_expression=cron_expression or None,
        run_at=run_at or None,
        max_failures=max_failures,
    )

    await schedule_task(task)

    if task_type == "recurring":
        return (
            f"Background task created (ID: {task['id']})\n"
            f"Schedule: {cron_expression}\n"
            f"Prompt: {prompt}\n"
            f"I'll run this automatically and report results here."
        )
    return (
        f"Background task created (ID: {task['id']})\n"
        f"Scheduled for: {run_at}\n"
        f"Prompt: {prompt}\n"
        f"I'll report the results here when it runs."
    )


@tool
async def list_background_tasks(
    conversation_id: str,
    user_email: str,
    show_all: bool = False,
) -> str:
    """List background tasks for the current user.

    Args:
        conversation_id: The current conversation ID (from context brackets).
        user_email: The user's email (from context brackets).
        show_all: If True, show tasks in all statuses. Default shows only active/paused.
    """
    ctx = _pending_task_context.get(conversation_id, {})
    user_id = ctx.get("user_id", "unknown")

    from task_store import get_task_store

    store = await get_task_store()
    tasks = await store.list_tasks(user_id=user_id)

    if not show_all:
        tasks = [t for t in tasks if t["status"] in ("active", "paused")]

    if not tasks:
        return "No background tasks found."

    lines = [f"**Background Tasks** ({len(tasks)})\n"]
    for t in tasks:
        status_icon = {
            "active": "running",
            "paused": "paused",
            "completed": "done",
            "failed": "error",
        }.get(t["status"], t["status"])
        created = datetime.fromtimestamp(t["created_at"]).strftime("%Y-%m-%d %H:%M")
        schedule = t.get("cron_expression") or t.get("run_at") or "N/A"
        line = (
            f"- **{t['id']}** [{status_icon}] -- {t['prompt'][:60]}\n"
            f"  Schedule: `{schedule}` | Runs: {t['run_count']} | Created: {created}"
        )
        if t.get("last_error"):
            line += f"\n  Last error: `{t['last_error'][:80]}`"
        lines.append(line)

    return "\n".join(lines)


@tool
async def pause_background_task(
    task_id: str, conversation_id: str, user_email: str
) -> str:
    """Pause an active background task. It can be resumed later.

    Args:
        task_id: The task ID to pause.
        conversation_id: The current conversation ID (from context brackets).
        user_email: The user's email (from context brackets).
    """
    from task_store import get_task_store
    from scheduler import pause_task

    store = await get_task_store()
    task = await store.get_task(task_id)
    if not task:
        return f"Task {task_id} not found."
    if task["status"] != "active":
        return f"Task {task_id} is not active (current status: {task['status']})."

    await pause_task(task_id)
    await store.update_task(task_id, status="paused")
    return f"Task {task_id} paused. Say 'resume task {task_id}' to restart it."


@tool
async def resume_background_task(
    task_id: str, conversation_id: str, user_email: str
) -> str:
    """Resume a paused or failed background task.

    Args:
        task_id: The task ID to resume.
        conversation_id: The current conversation ID (from context brackets).
        user_email: The user's email (from context brackets).
    """
    from task_store import get_task_store
    from scheduler import resume_task

    store = await get_task_store()
    task = await store.get_task(task_id)
    if not task:
        return f"Task {task_id} not found."
    if task["status"] not in ("paused", "failed"):
        return f"Task {task_id} cannot be resumed (current status: {task['status']})."

    await store.update_task(task_id, status="active", error_count=0, last_error=None)
    task["status"] = "active"
    await resume_task(task)
    return f"Task {task_id} resumed."


@tool
async def cancel_background_task(
    task_id: str, conversation_id: str, user_email: str
) -> str:
    """Permanently cancel and delete a background task.

    Args:
        task_id: The task ID to cancel.
        conversation_id: The current conversation ID (from context brackets).
        user_email: The user's email (from context brackets).
    """
    from task_store import get_task_store
    from scheduler import cancel_task

    store = await get_task_store()
    task = await store.get_task(task_id)
    if not task:
        return f"Task {task_id} not found."

    await cancel_task(task_id)
    await store.delete_task(task_id)
    return f"Task {task_id} cancelled and removed."


BACKGROUND_TASK_TOOLS = [
    create_background_task,
    list_background_tasks,
    pause_background_task,
    resume_background_task,
    cancel_background_task,
]
