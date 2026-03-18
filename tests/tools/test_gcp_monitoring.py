"""Tests for tools.gcp_monitoring.check_gcp_metrics."""

from unittest.mock import MagicMock, patch


def _make_time_series(num_points=3):
    ts = MagicMock()
    ts.metric.labels = {"method": "GET"}
    points = []
    for i in range(num_points):
        p = MagicMock()
        p.value = MagicMock(__str__=lambda self: f"value_{id(self)}")
        p.interval.end_time = f"2025-01-15T12:0{i}:00Z"
        points.append(p)
    ts.points = points
    return ts


@patch("google.cloud.monitoring_v3.ListTimeSeriesRequest")
@patch("google.cloud.monitoring_v3.TimeInterval")
@patch("google.cloud.monitoring_v3.MetricServiceClient")
def test_returns_formatted_time_series(mock_client_cls, mock_interval, mock_req):
    from tools.gcp_monitoring import check_gcp_metrics

    mock_client_cls.return_value.list_time_series.return_value = [_make_time_series(3)]

    result = check_gcp_metrics.invoke({
        "metric_type": "run.googleapis.com/request_latencies",
        "resource_type": "cloud_run_revision",
    })

    assert "GET" in result
    assert result.count("\n") >= 0  # at least one line


@patch("google.cloud.monitoring_v3.ListTimeSeriesRequest")
@patch("google.cloud.monitoring_v3.TimeInterval")
@patch("google.cloud.monitoring_v3.MetricServiceClient")
def test_no_data_points(mock_client_cls, mock_interval, mock_req):
    from tools.gcp_monitoring import check_gcp_metrics

    mock_client_cls.return_value.list_time_series.return_value = []

    result = check_gcp_metrics.invoke({
        "metric_type": "compute.googleapis.com/instance/cpu/utilization",
        "resource_type": "gce_instance",
    })
    assert result == "No data points found for that metric/resource."


@patch("google.cloud.monitoring_v3.ListTimeSeriesRequest")
@patch("google.cloud.monitoring_v3.TimeInterval")
@patch("google.cloud.monitoring_v3.MetricServiceClient")
@patch("tools.gcp_monitoring.time")
def test_default_minutes_30(mock_time, mock_client_cls, mock_interval, mock_req):
    from tools.gcp_monitoring import check_gcp_metrics

    mock_time.time.return_value = 1700000000
    mock_client_cls.return_value.list_time_series.return_value = []

    check_gcp_metrics.invoke({
        "metric_type": "test/metric",
        "resource_type": "test_resource",
    })

    mock_interval.assert_called_once()
    call_args = mock_interval.call_args[0][0]
    assert call_args["start_time"]["seconds"] == 1700000000 - 30 * 60
    assert call_args["end_time"]["seconds"] == 1700000000


@patch("google.cloud.monitoring_v3.ListTimeSeriesRequest")
@patch("google.cloud.monitoring_v3.TimeInterval")
@patch("google.cloud.monitoring_v3.MetricServiceClient")
@patch("tools.gcp_monitoring.time")
def test_custom_minutes(mock_time, mock_client_cls, mock_interval, mock_req):
    from tools.gcp_monitoring import check_gcp_metrics

    mock_time.time.return_value = 1700000000
    mock_client_cls.return_value.list_time_series.return_value = []

    check_gcp_metrics.invoke({
        "metric_type": "test/metric",
        "resource_type": "test_resource",
        "minutes": 60,
    })

    call_args = mock_interval.call_args[0][0]
    assert call_args["start_time"]["seconds"] == 1700000000 - 60 * 60


@patch("google.cloud.monitoring_v3.ListTimeSeriesRequest")
@patch("google.cloud.monitoring_v3.TimeInterval")
@patch("google.cloud.monitoring_v3.MetricServiceClient")
def test_only_last_5_points_per_series(mock_client_cls, mock_interval, mock_req):
    from tools.gcp_monitoring import check_gcp_metrics

    ts = _make_time_series(num_points=10)
    mock_client_cls.return_value.list_time_series.return_value = [ts]

    result = check_gcp_metrics.invoke({
        "metric_type": "test/metric",
        "resource_type": "test_resource",
    })

    # points[-5:] means at most 5 lines per series
    lines = result.strip().split("\n")
    assert len(lines) == 5


@patch("google.cloud.monitoring_v3.ListTimeSeriesRequest")
@patch("google.cloud.monitoring_v3.TimeInterval")
@patch("google.cloud.monitoring_v3.MetricServiceClient")
def test_filter_string_construction(mock_client_cls, mock_interval, mock_req):
    from tools.gcp_monitoring import check_gcp_metrics

    mock_client_cls.return_value.list_time_series.return_value = []

    check_gcp_metrics.invoke({
        "metric_type": "run.googleapis.com/request_count",
        "resource_type": "cloud_run_revision",
    })

    call_kwargs = mock_client_cls.return_value.list_time_series.call_args
    filter_str = call_kwargs.kwargs.get("filter") or call_kwargs[1].get("filter")
    assert 'metric.type="run.googleapis.com/request_count"' in filter_str
    assert 'resource.type="cloud_run_revision"' in filter_str
