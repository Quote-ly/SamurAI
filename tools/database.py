"""Read-only database tools for AlloyDB troubleshooting via IAM authentication."""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# AlloyDB connection details — prod only (dev not accessible due to CIDR overlap)
PROD_INSTANCE = "projects/virtualdojo-fedramp-prod/locations/us-central1/clusters/virtualdojo-cluster/instances/virtualdojo-primary"
PROD_DB = "quotely"
PROD_IAM_USER = "samurai-bot@virtualdojo-samurai.iam"

# Connection singleton
_engine = None


def _get_engine():
    """Get or create a SQLAlchemy engine with AlloyDB IAM auth."""
    global _engine
    if _engine is not None:
        return _engine

    from google.cloud.alloydb.connector import Connector
    import sqlalchemy

    connector = Connector(refresh_strategy="lazy")

    def getconn():
        return connector.connect(
            PROD_INSTANCE,
            "pg8000",
            user=PROD_IAM_USER,
            db=PROD_DB,
            enable_iam_auth=True,
        )

    _engine = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        pool_size=2,
        max_overflow=1,
        pool_pre_ping=True,
    )
    logger.info("AlloyDB engine created for %s", PROD_INSTANCE)
    return _engine


def _run_readonly_query(sql: str, params: dict | None = None, limit: int = 50) -> str:
    """Execute a read-only SQL query and return formatted results."""
    import sqlalchemy

    engine = _get_engine()

    try:
        with engine.connect() as conn:
            # Force read-only transaction
            conn.execute(sqlalchemy.text("SET TRANSACTION READ ONLY"))
            result = conn.execute(sqlalchemy.text(sql), params or {})
            rows = result.fetchmany(limit)
            columns = list(result.keys())

            if not rows:
                return "No results returned."

            # Format as a readable table
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
        return f"Query error: {err[:300]}"


@tool
def db_query(
    sql: str,
    limit: int = 50,
) -> str:
    """Run a read-only SQL query against the production AlloyDB database.

    The bot has SELECT-only access. INSERT, UPDATE, DELETE will be rejected.
    This connects to the quotely database on virtualdojo-fedramp-prod.

    NOTE: Database access is available for PRODUCTION only. Dev database is
    not accessible due to a VPC CIDR overlap.

    Args:
        sql: A SELECT query to run. Only SELECT statements are allowed.
        limit: Max rows to return (default 50).
    """
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
        return "Error: Only SELECT queries are allowed. The bot has read-only access."

    # Block obvious write attempts even though the DB user can't write
    for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]:
        if keyword in sql_upper.split("(")[0]:
            return f"Error: {keyword} is not allowed. The bot has read-only access."

    return _run_readonly_query(sql, limit=limit)


@tool
def db_list_tables() -> str:
    """List all tables in the production database with row counts.

    Connects to the quotely database on virtualdojo-fedramp-prod.
    """
    sql = """
        SELECT
            schemaname,
            tablename,
            n_live_tup AS approximate_row_count
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
    """
    return _run_readonly_query(sql, limit=100)


@tool
def db_describe_table(table_name: str) -> str:
    """Show the columns, types, and constraints for a table.

    Args:
        table_name: The table name to describe.
    """
    # Sanitize table name to prevent injection
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
    return _run_readonly_query(sql, params={"table_name": table_name}, limit=200)


@tool
def db_check_user(email: str) -> str:
    """Look up a user by email in the production database for troubleshooting.

    Args:
        email: The user's email address to look up.
    """
    sql = """
        SELECT id, email, full_name, is_active, last_login, created_at
        FROM users
        WHERE email = :email
    """
    return _run_readonly_query(sql, params={"email": email}, limit=1)


@tool
def db_recent_audit_logs(
    event_type: str = "",
    hours: int = 24,
    limit: int = 20,
) -> str:
    """Query recent entries from the application audit_log table.

    Useful for troubleshooting auth failures, permission issues, and data access patterns.

    Args:
        event_type: Optional filter — 'login_failed', 'login_success', 'permission_denied',
                    'data_export', 'account_created', etc. Empty for all events.
        hours: How far back to look (default 24 hours).
        limit: Max rows to return (default 20).
    """
    if event_type:
        sql = """
            SELECT id, event_type, user_id, ip_address, details, created_at
            FROM audit_log
            WHERE event_type = :event_type
              AND created_at > NOW() - INTERVAL ':hours hours'
            ORDER BY created_at DESC
            LIMIT :limit
        """
        # Build the query with proper interval handling
        sql = f"""
            SELECT id, event_type, user_id, ip_address, details, created_at
            FROM audit_log
            WHERE event_type = :event_type
              AND created_at > NOW() - INTERVAL '{int(hours)} hours'
            ORDER BY created_at DESC
        """
        return _run_readonly_query(sql, params={"event_type": event_type}, limit=limit)
    else:
        sql = f"""
            SELECT id, event_type, user_id, ip_address, details, created_at
            FROM audit_log
            WHERE created_at > NOW() - INTERVAL '{int(hours)} hours'
            ORDER BY created_at DESC
        """
        return _run_readonly_query(sql, limit=limit)


DATABASE_TOOLS = [
    db_query,
    db_list_tables,
    db_describe_table,
    db_check_user,
    db_recent_audit_logs,
]
