"""Tests for tools.gcp_cloudrun.list_cloud_run_services."""

from unittest.mock import MagicMock, patch


def _make_service(name, state_name="CONDITION_SUCCEEDED", uri="https://svc.run.app"):
    svc = MagicMock()
    svc.name = name
    svc.uri = uri
    svc.terminal_condition.state.name = state_name
    return svc


@patch("google.cloud.run_v2.ServicesClient")
def test_returns_formatted_services(mock_client_cls):
    from tools.gcp_cloudrun import list_cloud_run_services

    services = [
        _make_service("projects/p/locations/us-central1/services/api", "CONDITION_SUCCEEDED", "https://api.run.app"),
        _make_service("projects/p/locations/us-central1/services/web", "CONDITION_SUCCEEDED", "https://web.run.app"),
    ]
    mock_client_cls.return_value.list_services.return_value = services

    result = list_cloud_run_services.invoke({})

    assert "api" in result
    assert "web" in result
    assert "https://api.run.app" in result
    assert "https://web.run.app" in result


@patch("google.cloud.run_v2.ServicesClient")
def test_no_services_found(mock_client_cls):
    from tools.gcp_cloudrun import list_cloud_run_services

    mock_client_cls.return_value.list_services.return_value = []

    result = list_cloud_run_services.invoke({})
    assert result == "No Cloud Run services found."


@patch("google.cloud.run_v2.ServicesClient")
def test_service_name_extracted_from_full_path(mock_client_cls):
    from tools.gcp_cloudrun import list_cloud_run_services

    services = [_make_service("projects/p/locations/us-central1/services/my-svc")]
    mock_client_cls.return_value.list_services.return_value = services

    result = list_cloud_run_services.invoke({})
    # Should show just "my-svc", not the full resource path
    assert "my-svc" in result
    assert "projects/p/" not in result


@patch("google.cloud.run_v2.ServicesClient")
def test_unknown_status_when_no_terminal_condition(mock_client_cls):
    from tools.gcp_cloudrun import list_cloud_run_services

    svc = MagicMock()
    svc.name = "projects/p/locations/us-central1/services/broken"
    svc.uri = "https://broken.run.app"
    svc.terminal_condition = None
    mock_client_cls.return_value.list_services.return_value = [svc]

    result = list_cloud_run_services.invoke({})
    assert "UNKNOWN" in result


@patch("google.cloud.run_v2.ServicesClient")
def test_region_default_is_dash(mock_client_cls):
    from tools.gcp_cloudrun import list_cloud_run_services

    mock_client_cls.return_value.list_services.return_value = []

    list_cloud_run_services.invoke({})
    mock_client_cls.return_value.list_services.assert_called_once_with(
        parent="projects/test-project/locations/-"
    )


@patch("google.cloud.run_v2.ServicesClient")
def test_explicit_region(mock_client_cls):
    from tools.gcp_cloudrun import list_cloud_run_services

    mock_client_cls.return_value.list_services.return_value = []

    list_cloud_run_services.invoke({"region": "us-east1"})
    mock_client_cls.return_value.list_services.assert_called_once_with(
        parent="projects/test-project/locations/us-east1"
    )
