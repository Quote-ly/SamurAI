"""Tests for tools/database.py — read-only AlloyDB query tools."""

from unittest.mock import MagicMock, patch

import pytest


# --- db_query ---


def test_db_query_blocks_insert():
    from tools.database import db_query

    result = db_query.invoke({"sql": "INSERT INTO users (email) VALUES ('x')"})
    assert "read-only" in result.lower() or "not allowed" in result.lower()


def test_db_query_blocks_update():
    from tools.database import db_query

    result = db_query.invoke({"sql": "UPDATE users SET is_active = false"})
    assert "read-only" in result.lower() or "not allowed" in result.lower()


def test_db_query_blocks_delete():
    from tools.database import db_query

    result = db_query.invoke({"sql": "DELETE FROM users WHERE id = 1"})
    assert "read-only" in result.lower() or "not allowed" in result.lower()


def test_db_query_blocks_drop():
    from tools.database import db_query

    result = db_query.invoke({"sql": "DROP TABLE users"})
    assert "read-only" in result.lower() or "not allowed" in result.lower()


def test_db_query_allows_select():
    from tools.database import db_query

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchmany.return_value = [("alice@test.com", "Alice", True)]
    mock_result.keys.return_value = ["email", "full_name", "is_active"]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("tools.database._get_engine", return_value=mock_engine):
        result = db_query.invoke({"sql": "SELECT email, full_name, is_active FROM users"})

    assert "alice@test.com" in result
    assert "Alice" in result


def test_db_query_allows_with_cte():
    from tools.database import db_query

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchmany.return_value = [("data",)]
    mock_result.keys.return_value = ["result"]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("tools.database._get_engine", return_value=mock_engine):
        result = db_query.invoke({"sql": "WITH cte AS (SELECT 1) SELECT * FROM cte"})

    assert "data" in result


def test_db_query_handles_no_results():
    from tools.database import db_query

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchmany.return_value = []
    mock_result.keys.return_value = ["id"]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("tools.database._get_engine", return_value=mock_engine):
        result = db_query.invoke({"sql": "SELECT * FROM users WHERE email = 'nobody'"})

    assert "No results" in result


def test_db_query_handles_permission_denied():
    from tools.database import db_query

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = Exception("permission denied for table secrets")
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("tools.database._get_engine", return_value=mock_engine):
        result = db_query.invoke({"sql": "SELECT * FROM secrets"})

    assert "Permission denied" in result


def test_db_query_handles_table_not_found():
    from tools.database import db_query

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = Exception('relation "nonexistent" does not exist')
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("tools.database._get_engine", return_value=mock_engine):
        result = db_query.invoke({"sql": "SELECT * FROM nonexistent"})

    assert "Table not found" in result


# --- db_list_tables ---


def test_db_list_tables_returns_formatted():
    from tools.database import db_list_tables

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchmany.return_value = [
        ("public", "users", 150),
        ("public", "audit_log", 5000),
        ("public", "profiles", 10),
    ]
    mock_result.keys.return_value = ["schemaname", "tablename", "approximate_row_count"]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("tools.database._get_engine", return_value=mock_engine):
        result = db_list_tables.invoke({})

    assert "users" in result
    assert "audit_log" in result
    assert "profiles" in result


# --- db_describe_table ---


def test_db_describe_table_returns_columns():
    from tools.database import db_describe_table

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchmany.return_value = [
        ("id", "integer", "NO", None, None),
        ("email", "character varying", "NO", None, 255),
        ("is_active", "boolean", "YES", "true", None),
    ]
    mock_result.keys.return_value = [
        "column_name", "data_type", "is_nullable", "column_default", "character_maximum_length"
    ]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("tools.database._get_engine", return_value=mock_engine):
        result = db_describe_table.invoke({"table_name": "users"})

    assert "email" in result
    assert "character varying" in result
    assert "is_active" in result


def test_db_describe_table_rejects_injection():
    from tools.database import db_describe_table

    result = db_describe_table.invoke({"table_name": "users; DROP TABLE users"})
    assert "Invalid table name" in result


# --- db_check_user ---


def test_db_check_user_found():
    from tools.database import db_check_user

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchmany.return_value = [
        (1, "devin@virtualdojo.com", "Devin Henderson", True, "2026-04-06", "2025-01-01")
    ]
    mock_result.keys.return_value = ["id", "email", "full_name", "is_active", "last_login", "created_at"]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("tools.database._get_engine", return_value=mock_engine):
        result = db_check_user.invoke({"email": "devin@virtualdojo.com"})

    assert "devin@virtualdojo.com" in result
    assert "Devin Henderson" in result


def test_db_check_user_not_found():
    from tools.database import db_check_user

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchmany.return_value = []
    mock_result.keys.return_value = ["id", "email", "full_name", "is_active", "last_login", "created_at"]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("tools.database._get_engine", return_value=mock_engine):
        result = db_check_user.invoke({"email": "nobody@test.com"})

    assert "No results" in result


# --- db_recent_audit_logs ---


def test_db_recent_audit_logs_all():
    from tools.database import db_recent_audit_logs

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchmany.return_value = [
        (1, "login_failed", 42, "192.168.1.1", "bad password", "2026-04-06T12:00:00"),
        (2, "login_success", 42, "192.168.1.1", None, "2026-04-06T12:01:00"),
    ]
    mock_result.keys.return_value = ["id", "event_type", "user_id", "ip_address", "details", "created_at"]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("tools.database._get_engine", return_value=mock_engine):
        result = db_recent_audit_logs.invoke({})

    assert "login_failed" in result
    assert "login_success" in result


def test_db_recent_audit_logs_filtered():
    from tools.database import db_recent_audit_logs

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchmany.return_value = [
        (1, "login_failed", 42, "10.0.0.1", "invalid creds", "2026-04-06T12:00:00"),
    ]
    mock_result.keys.return_value = ["id", "event_type", "user_id", "ip_address", "details", "created_at"]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_engine.connect.return_value = mock_conn

    with patch("tools.database._get_engine", return_value=mock_engine):
        result = db_recent_audit_logs.invoke({"event_type": "login_failed", "hours": 12})

    assert "login_failed" in result
