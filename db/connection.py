"""
db/connection.py
----------------
PostgreSQL connection management via psycopg2.
Credentials are loaded from .env using python-dotenv.
The connection is cached at the Streamlit session level to avoid
reconnecting on every rerun.
"""

from __future__ import annotations

import os
import logging
from typing import Any

import psycopg2
import psycopg2.extras
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection factory (cached at the Streamlit resource level)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_connection() -> psycopg2.extensions.connection:
    """Return a cached psycopg2 connection to the PostgreSQL database.

    Raises
    ------
    psycopg2.OperationalError
        If the connection cannot be established with the supplied credentials.
    """
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "")
    user = os.getenv("DB_USER", "")
    password = os.getenv("DB_PASSWORD", "")

    if not all([dbname, user, password]):
        raise ValueError(
            "Missing database credentials. "
            "Please fill in DB_NAME, DB_USER, and DB_PASSWORD in your .env file."
        )

    logger.info("Establishing database connection to %s:%s/%s", host, port, dbname)
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=10,
    )
    conn.autocommit = True
    logger.info("Database connection established successfully.")
    return conn


def execute_query(
    sql: str,
    params: tuple | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Execute a SELECT query and return rows as a list of dicts.

    Parameters
    ----------
    sql:
        The SQL string to execute.
    params:
        Optional tuple of parameterised values.

    Returns
    -------
    rows:
        A list of row dictionaries (column_name → value).
    columns:
        A list of column name strings in result order.

    Raises
    ------
    psycopg2.Error
        On any database-level error.
    ValueError
        If the query is not a SELECT statement.
    """
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT") and not sql_stripped.startswith("WITH"):
        raise ValueError("Only SELECT / WITH queries are permitted for safety.")

    conn = get_connection()

    # Reconnect if the connection was closed
    try:
        if conn.closed:
            # Clear the cache so a new connection is established
            get_connection.clear()
            conn = get_connection()
    except Exception:
        get_connection.clear()
        conn = get_connection()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        logger.debug("Executing SQL: %s | params: %s", sql, params)
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []

    rows_as_dicts = [dict(row) for row in rows]
    logger.info("Query returned %d rows.", len(rows_as_dicts))
    return rows_as_dicts, columns


def test_connection() -> bool:
    """Return True if the database is reachable, False otherwise."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception as exc:
        logger.warning("Connection test failed: %s", exc)
        return False
