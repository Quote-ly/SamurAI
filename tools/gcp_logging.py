"""Tool for querying Google Cloud Logging."""

import os

from langchain_core.tools import tool


@tool
def query_cloud_logs(filter_query: str, project_id: str | None = None) -> str:
    """Query Google Cloud Logging entries.

    Args:
        filter_query: A Cloud Logging filter string, e.g.
            'resource.type="cloud_run_revision" severity>=ERROR'
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
    """
    from google.cloud import logging as cloud_logging

    pid = project_id or os.environ["GCP_PROJECT_ID"]
    client = cloud_logging.Client(project=pid)
    entries = list(client.list_entries(filter_=filter_query, max_results=50))

    if not entries:
        return "No log entries found for that filter."

    lines = []
    for entry in entries:
        ts = entry.timestamp.isoformat() if entry.timestamp else "?"
        lines.append(f"[{ts}] {entry.severity}: {entry.payload}")
    return "\n".join(lines)
