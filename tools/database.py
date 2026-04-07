"""Read-only database tools for AlloyDB troubleshooting via Auth Proxy sidecar."""

import logging
import os

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Connection engines per environment
_engines: dict[str, object] = {}

# Database configs read from env vars (set in service.yaml)
DB_CONFIGS = {
    "prod": {
        "host": os.environ.get("PROD_DB_HOST", "127.0.0.1"),
        "port": os.environ.get("PROD_DB_PORT", "5432"),
        "name": os.environ.get("PROD_DB_NAME", "quotely"),
        "user": os.environ.get("PROD_DB_USER", "samurai-bot@virtualdojo-samurai.iam"),
    },
    "dev": {
        "host": os.environ.get("DEV_DB_HOST", "127.0.0.1"),
        "port": os.environ.get("DEV_DB_PORT", "5433"),
        "name": os.environ.get("DEV_DB_NAME", "quotely"),
        "user": os.environ.get("DEV_DB_USER", "samurai-bot@virtualdojo-samurai.iam"),
    },
}


def _get_engine(env: str = "prod"):
    """Get or create a SQLAlchemy engine for the given environment.

    Connects via the AlloyDB Auth Proxy sidecar running on localhost.
    """
    if env in _engines:
        return _engines[env]

    import sqlalchemy

    config = DB_CONFIGS.get(env)
    if not config:
        raise ValueError(f"Unknown environment: {env}. Use 'prod' or 'dev'.")

    url = sqlalchemy.URL.create(
        drivername="postgresql+pg8000",
        username=config["user"],
        host=config["host"],
        port=int(config["port"]),
        database=config["name"],
    )

    engine = sqlalchemy.create_engine(
        url,
        pool_size=2,
        max_overflow=1,
        pool_pre_ping=True,
        connect_args={"timeout": 10},
    )
    _engines[env] = engine
    logger.info("Database engine created for %s (%s:%s)", env, config["host"], config["port"])
    return engine


def _run_readonly_query(sql: str, env: str = "prod", params: dict | None = None, limit: int = 50) -> str:
    """Execute a read-only SQL query and return formatted results."""
    import sqlalchemy

    try:
        engine = _get_engine(env)
    except Exception as e:
        return f"Error connecting to {env} database: {e}"

    try:
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SET TRANSACTION READ ONLY"))
            result = conn.execute(sqlalchemy.text(sql), params or {})
            rows = result.fetchmany(limit)
            columns = list(result.keys())

            if not rows:
                return "No results returned."

            lines = [" | ".join(columns)]
            lines.append("-" * len(lines[0]))
            for row in rows:
                lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))

            output = "\n".join(lines)
            total = len(rows)
            if total == limit:
                output += f"\n\n... [showing first {limit} rows, may be more]"

            return output

    except Exception as e:
        err = str(e)
        if "permission denied" in err.lower():
            return f"Permission denied. The bot has read-only access. Error: {err[:200]}"
        if "relation" in err.lower() and "does not exist" in err.lower():
            return f"Table not found. Use db_list_tables to see available tables. Error: {err[:200]}"
        if "connection" in err.lower() or "timeout" in err.lower():
            return f"Connection failed to {env} database. The Auth Proxy may not be running. Error: {err[:200]}"
        return f"Query error: {err[:300]}"


@tool
def db_query(
    sql: str,
    env: str = "prod",
    limit: int = 50,
) -> str:
    """Run a read-only SQL query against the AlloyDB database.

    The bot has SELECT-only access. INSERT, UPDATE, DELETE will be rejected.

    Args:
        sql: A SELECT query to run. Only SELECT statements are allowed.
        env: Environment — 'prod' or 'dev'. Default is 'prod'.
        limit: Max rows to return (default 50).
    """
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
        return "Error: Only SELECT queries are allowed. The bot has read-only access."

    for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]:
        if keyword in sql_upper.split("(")[0]:
            return f"Error: {keyword} is not allowed. The bot has read-only access."

    return _run_readonly_query(sql, env=env, limit=limit)


@tool
def db_list_tables(env: str = "prod") -> str:
    """List all tables in the database with row counts.

    Args:
        env: Environment — 'prod' or 'dev'. Default is 'prod'.
    """
    sql = """
        SELECT
            schemaname,
            tablename,
            n_live_tup AS approximate_row_count
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
    """
    return _run_readonly_query(sql, env=env, limit=100)


@tool
def db_describe_table(table_name: str, env: str = "prod") -> str:
    """Show the columns, types, and constraints for a table.

    Args:
        table_name: The table name to describe.
        env: Environment — 'prod' or 'dev'. Default is 'prod'.
    """
    if not table_name.replace("_", "").replace(".", "").isalnum():
        return "Error: Invalid table name."

    sql = """
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns
        WHERE table_name = :table_name
          AND table_schema = 'public'
        ORDER BY ordinal_position
    """
    return _run_readonly_query(sql, env=env, params={"table_name": table_name}, limit=200)


@tool
def db_check_user(email: str, env: str = "prod") -> str:
    """Look up a user by email in the database for troubleshooting.

    Args:
        email: The user's email address to look up.
        env: Environment — 'prod' or 'dev'. Default is 'prod'.
    """
    sql = """
        SELECT id, email, full_name, is_active, last_login, created_at
        FROM users
        WHERE email = :email
    """
    return _run_readonly_query(sql, env=env, params={"email": email}, limit=1)


@tool
def db_recent_audit_logs(
    event_type: str = "",
    hours: int = 24,
    env: str = "prod",
    limit: int = 20,
) -> str:
    """Query recent entries from the application audit_log table.

    Useful for troubleshooting auth failures, permission issues, and data access patterns.

    Args:
        event_type: Optional filter — 'login_failed', 'login_success', 'permission_denied',
                    'data_export', 'account_created', etc. Empty for all events.
        hours: How far back to look (default 24 hours).
        env: Environment — 'prod' or 'dev'. Default is 'prod'.
        limit: Max rows to return (default 20).
    """
    if event_type:
        sql = f"""
            SELECT id, event_type, user_id, ip_address, details, created_at
            FROM audit_log
            WHERE event_type = :event_type
              AND created_at > NOW() - INTERVAL '{int(hours)} hours'
            ORDER BY created_at DESC
        """
        return _run_readonly_query(sql, env=env, params={"event_type": event_type}, limit=limit)
    else:
        sql = f"""
            SELECT id, event_type, user_id, ip_address, details, created_at
            FROM audit_log
            WHERE created_at > NOW() - INTERVAL '{int(hours)} hours'
            ORDER BY created_at DESC
        """
        return _run_readonly_query(sql, env=env, limit=limit)


DATABASE_TOOLS = [
    db_query,
    db_list_tables,
    db_describe_table,
    db_check_user,
    db_recent_audit_logs,
]
