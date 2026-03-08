"""
agents/sql_query_agent.py
--------------------------
SQL Query Agent — generates, executes, and retries SQL queries.

Responsibilities
----------------
1. Use RAG on `knowledge/Bilingual README.txt` to inject ONLY the relevant
   table schema into the LLM prompt (reduces hallucinations + token cost).
2. Call OpenAI to generate a safe SELECT query.
3. Execute via psycopg2 through db/connection.py.
4. If 0 results returned → retry with refined prompt (up to MAX_SQL_RETRIES times).

Returns a typed dict:
    {
        "sql": str,
        "results": list[dict],
        "columns": list[str],
        "attempts": int,
        "error": str | None,
    }
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from db.connection import execute_query
from utils.rag import retrieve_relevant_chunks
from utils.logging_utils import get_logger

load_dotenv()
logger = get_logger(__name__)

_client: OpenAI | None = None

# Path to the bilingual schema README
_README_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "Bilingual README.txt"

_MAX_RETRIES = int(os.getenv("MAX_SQL_RETRIES", "3"))

# Known table names from the Bilingual README (used for schema prefix injection)
_KNOWN_TABLES = [
    "booking",
    "book_payment",
    "category_pattern_mapping",
    "customer",
    "purchase",
    "sale",
    "salesman",
    "stock",
]


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in your .env file.")
        _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(schema_context: str, db_schema: str) -> str:
    schema_note = (
        f"- ALWAYS prefix every table name with the schema name: "
        f'"{db_schema}.<table_name>" (e.g. {db_schema}.stock, {db_schema}.sale).\n'
        f"- Never reference a table without the \"{db_schema}.\" prefix."
    )
    return f"""You are the SQL Query Agent for Chop Kong Hin, a gold jewellery shop.

Database schema name: **{db_schema}**

You have access to the following PostgreSQL table schema (extracted from the database README):

{schema_context}

Rules:
- Generate ONLY a single valid PostgreSQL SELECT query (or WITH … SELECT).
- Do NOT use INSERT, UPDATE, DELETE, DROP, TRUNCATE, or any DDL/DML.
- {schema_note}
- Use column names exactly as shown in the schema.
- Prefer ILIKE for case-insensitive string matching.
- Always handle potential NULLs gracefully.
- Limit results to 100 rows unless the question explicitly asks for all.
- Respond ONLY with a valid JSON object:
  {{
    "sql": "<the SQL query>",
    "reasoning": "<brief explanation of why you chose these tables/columns>"
  }}
Do NOT include markdown fences or any text outside the JSON.
"""


def _extract_sql(raw: str) -> str:
    """Parse SQL from LLM JSON response."""
    try:
        data = json.loads(raw)
        return data.get("sql", "").strip()
    except json.JSONDecodeError:
        # Fallback: try to extract a SELECT statement directly
        match = re.search(r"(WITH\s+.+?SELECT\s+.+?;|SELECT\s+.+?;)", raw, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""


def _sanitise_sql(sql: str) -> str:
    """Strip any trailing semicolons and ensure query is read-only."""
    sql = sql.strip().rstrip(";").strip()
    upper = sql.upper()
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE", "GRANT"]
    for kw in forbidden:
        if re.search(rf"\b{kw}\b", upper):
            raise ValueError(f"Forbidden SQL operation detected: {kw}")
    return sql


def _apply_schema_prefix(sql: str, db_schema: str) -> str:
    """Safety net: ensure all known table names in *sql* are prefixed with *db_schema*.

    The LLM is instructed to do this, but this function acts as a deterministic
    fallback in case any table reference slips through without the schema prefix.
    """
    if not db_schema:
        return sql

    for table in _KNOWN_TABLES:
        # Match the bare table name that is NOT already preceded by "schema."
        # Negative lookbehind: not preceded by a word char or dot (i.e. not already qualified)
        pattern = rf"(?<![\w.]){re.escape(table)}(?!\w)"
        qualified = f"{db_schema}.{table}"
        sql = re.sub(pattern, qualified, sql, flags=re.IGNORECASE)

    return sql


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------

def run(
    question: str,
    previous_feedback: str | None = None,
) -> dict[str, Any]:
    """Generate and execute a SQL query for the given question.

    Parameters
    ----------
    question:
        The restructured question from the Question Understanding Agent.
    previous_feedback:
        Optional feedback from the Data Evaluation Agent for retry refinement.

    Returns
    -------
    dict with keys: sql, results, columns, attempts, error.
    """
    logger.info("[SQLQueryAgent] Generating SQL for: %s", question)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    db_schema = os.getenv("DB_SCHEMA", "").strip().strip("'\"")
    client = _get_client()

    if db_schema:
        logger.info("[SQLQueryAgent] Using DB schema: %s", db_schema)
    else:
        logger.warning("[SQLQueryAgent] DB_SCHEMA not set — table names will not be schema-qualified.")

    # RAG: retrieve relevant schema sections
    schema_context = retrieve_relevant_chunks(question, _README_PATH, top_k=6)
    logger.debug("[SQLQueryAgent] Retrieved schema context (%d chars)", len(schema_context))

    attempts = 0
    last_error: str | None = None
    generated_sql = ""

    for attempt in range(1, _MAX_RETRIES + 1):
        attempts = attempt
        logger.info("[SQLQueryAgent] Attempt %d/%d", attempt, _MAX_RETRIES)

        # Build user message — include feedback on retries
        user_msg = f"Question: {question}"
        if previous_feedback:
            user_msg += f"\n\nAdditional guidance: {previous_feedback}"
        if attempt > 1 and last_error:
            user_msg += (
                f"\n\nPrevious attempt returned no results or failed. "
                f"Error/hint: {last_error}. "
                f"Please generate a broader or corrected SQL query."
            )

        messages = [
            {"role": "system", "content": _build_system_prompt(schema_context, db_schema)},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            logger.debug("[SQLQueryAgent] Raw LLM response: %s", raw)

            generated_sql = _extract_sql(raw)
            if not generated_sql:
                last_error = "LLM did not return a valid SQL query."
                logger.warning("[SQLQueryAgent] %s", last_error)
                continue

            generated_sql = _sanitise_sql(generated_sql)
            generated_sql = _apply_schema_prefix(generated_sql, db_schema)
            logger.info("[SQLQueryAgent] Generated SQL: %s", generated_sql)

            rows, columns = execute_query(generated_sql)

            if rows:
                logger.info("[SQLQueryAgent] Query succeeded with %d rows.", len(rows))
                return {
                    "sql": generated_sql,
                    "results": rows,
                    "columns": columns,
                    "attempts": attempts,
                    "error": None,
                }
            else:
                last_error = "Query returned 0 rows."
                logger.info("[SQLQueryAgent] %s Retrying...", last_error)

        except ValueError as exc:
            last_error = str(exc)
            logger.error("[SQLQueryAgent] Safety/parse error: %s", exc)
            continue
        except Exception as exc:
            last_error = str(exc)
            logger.error("[SQLQueryAgent] DB execution error: %s", exc)
            continue

    # All retries exhausted
    logger.warning("[SQLQueryAgent] All %d attempts exhausted. Returning empty result.", _MAX_RETRIES)
    return {
        "sql": generated_sql,
        "results": [],
        "columns": [],
        "attempts": attempts,
        "error": last_error or "No results found after maximum retries.",
    }
