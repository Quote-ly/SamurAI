"""Tool for inspecting Google Cloud Run services."""

import os

from langchain_core.tools import tool


@tool
def list_cloud_run_services(
    project_id: str | None = None, region: str = "-"
) -> str:
    """List Cloud Run services and their current status.

    Args:
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
        region: GCP region, or '-' for all regions (default).
    """
    from google.cloud import run_v2

    pid = project_id or os.environ["GCP_PROJECT_ID"]
    client = run_v2.ServicesClient()
    parent = f"projects/{pid}/locations/{region}"
    services = client.list_services(parent=parent)

    results = []
    for svc in services:
        name = svc.name.split("/")[-1]
        condition = svc.terminal_condition
        status = condition.state.name if condition else "UNKNOWN"
        results.append(f"{name} | Status: {status} | URL: {svc.uri}")

    return "\n".join(results) if results else "No Cloud Run services found."
