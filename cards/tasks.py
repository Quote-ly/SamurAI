"""Adaptive Card builders for background task notifications and management."""

from datetime import datetime


def build_task_result_card(
    task_id: str,
    prompt: str,
    result: str,
    run_count: int,
    task_type: str,
    schedule: str,
) -> dict:
    """Build an Adaptive Card for a background task result notification."""
    body = [
        {
            "type": "TextBlock",
            "text": "Background Task Result",
            "weight": "bolder",
            "size": "medium",
            "wrap": True,
        },
        {
            "type": "FactSet",
            "facts": [
                {"title": "Task ID", "value": task_id},
                {"title": "Task", "value": prompt[:100]},
                {"title": "Schedule", "value": schedule},
                {"title": "Run #", "value": str(run_count)},
            ],
        },
        {
            "type": "TextBlock",
            "text": "",
            "spacing": "small",
            "separator": True,
        },
        {
            "type": "TextBlock",
            "text": result[:2000],
            "wrap": True,
        },
    ]

    actions = []
    if task_type == "recurring":
        actions = [
            {
                "type": "Action.Submit",
                "title": "Pause Task",
                "data": {"action": "task_pause", "task_id": task_id},
            },
            {
                "type": "Action.Submit",
                "title": "Cancel Task",
                "data": {"action": "task_cancel", "task_id": task_id},
            },
        ]

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.3",
        "body": body,
        "actions": actions,
    }


def build_task_failure_card(
    task_id: str,
    prompt: str,
    error: str,
    error_count: int,
    max_failures: int,
) -> dict:
    """Build an Adaptive Card for a task failure/auto-pause notification."""
    body = [
        {
            "type": "TextBlock",
            "text": "Task Auto-Paused",
            "weight": "bolder",
            "size": "medium",
            "color": "attention",
            "wrap": True,
        },
        {
            "type": "FactSet",
            "facts": [
                {"title": "Task ID", "value": task_id},
                {"title": "Task", "value": prompt[:100]},
                {
                    "title": "Consecutive Failures",
                    "value": f"{error_count}/{max_failures}",
                },
            ],
        },
        {
            "type": "TextBlock",
            "text": f"Last error: {error[:500]}",
            "wrap": True,
            "color": "attention",
        },
    ]

    actions = [
        {
            "type": "Action.Submit",
            "title": "Resume Task",
            "data": {"action": "task_resume", "task_id": task_id},
        },
        {
            "type": "Action.Submit",
            "title": "Cancel Task",
            "data": {"action": "task_cancel", "task_id": task_id},
        },
    ]

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.3",
        "body": body,
        "actions": actions,
    }
