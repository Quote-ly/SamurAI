"""Tool for querying Google Cloud Monitoring metrics."""

import os
import time

from langchain_core.tools import tool


@tool
def check_gcp_metrics(
    metric_type: str,
    resource_type: str,
    minutes: int = 30,
    project_id: str | None = None,
) -> str:
    """Query GCP Cloud Monitoring time-series metrics.

    Args:
        metric_type: The metric type, e.g. 'run.googleapis.com/request_latencies'.
        resource_type: The monitored resource type, e.g. 'cloud_run_revision'.
        minutes: How far back to look (default 30).
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
    """
    from google.cloud import monitoring_v3

    pid = project_id or os.environ["GCP_PROJECT_ID"]
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{pid}"

    now = time.time()
    interval = monitoring_v3.TimeInterval(
        {
            "end_time": {"seconds": int(now)},
            "start_time": {"seconds": int(now - minutes * 60)},
        }
    )

    results = client.list_time_series(
        name=project_name,
        filter=f'metric.type="{metric_type}" AND resource.type="{resource_type}"',
        interval=interval,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
    )

    lines = []
    for ts in results:
        for point in ts.points[-5:]:
            lines.append(
                f"{ts.metric.labels} | {point.value} @ {point.interval.end_time}"
            )

    return "\n".join(lines) if lines else "No data points found for that metric/resource."
