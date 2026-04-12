"""Tools for querying Google Cloud Monitoring metrics and billing."""

import os
import time
from datetime import datetime, timedelta, timezone

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


@tool
def gcp_billing_summary(
    days: int = 30,
    project_id: str | None = None,
) -> str:
    """Get a read-only cost summary for a GCP project from Cloud Billing.

    Queries the BigQuery billing export for cost breakdown by service.
    Returns total cost and per-service breakdown for the given period.

    Args:
        days: Number of days to look back (default 30).
        project_id: GCP project ID. Defaults to GCP_PROJECT_ID env var.
    """
    from google.cloud import bigquery

    pid = project_id or os.environ["GCP_PROJECT_ID"]

    # Standard billing export dataset — adjust if using a different export table
    billing_table = os.environ.get(
        "GCP_BILLING_TABLE",
        f"{pid}.billing_export.gcp_billing_export_v1",
    )

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    query = f"""
        SELECT
            service.description AS service,
            ROUND(SUM(cost), 2) AS total_cost,
            ROUND(SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)), 2) AS credits,
            ROUND(SUM(cost) + SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)), 2) AS net_cost
        FROM `{billing_table}`
        WHERE project.id = @project_id
          AND usage_start_time >= @start_date
          AND usage_start_time < @end_date
        GROUP BY service
        HAVING net_cost > 0.01
        ORDER BY net_cost DESC
    """

    client = bigquery.Client(project=pid)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("project_id", "STRING", pid),
            bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
            bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date),
        ]
    )

    try:
        rows = list(client.query(query, job_config=job_config).result())
    except Exception as e:
        error_msg = str(e)
        if "Not found" in error_msg or "billing" in error_msg.lower():
            return (
                f"Billing export table not found for project {pid}. "
                "Ensure BigQuery billing export is enabled and GCP_BILLING_TABLE is set."
            )
        raise

    if not rows:
        return f"No billing data found for project {pid} in the last {days} days."

    lines = [f"**GCP Cost Summary** — {pid} ({days} days)\n"]
    total_net = 0.0
    for row in rows:
        lines.append(f"- {row.service}: ${row.net_cost:.2f} (cost ${row.total_cost:.2f}, credits ${row.credits:.2f})")
        total_net += row.net_cost
    lines.append(f"\n**Total: ${total_net:.2f}**")

    return "\n".join(lines)
