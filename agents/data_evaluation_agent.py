"""
agents/data_evaluation_agent.py
--------------------------------
Data Evaluation Agent — quality gate between the SQL Agent and Summary Agent.

Responsibilities
----------------
Given the user's original question and the rows returned by the SQL Agent,
this agent decides:

- "sufficient"   → the data directly answers the question.
- "insufficient" → the data is empty, irrelevant, or missing key info;
                   return feedback for the SQL Agent to refine its query.

Returns a typed dict:
    {
        "verdict": "sufficient" | "insufficient",
        "feedback": str,   # refinement instructions for SQL Agent if insufficient
        "reasoning": str,
    }
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from utils.logging_utils import get_logger

load_dotenv()
logger = get_logger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in your .env file.")
        _client = OpenAI(api_key=api_key)
    return _client


_SYSTEM_PROMPT = """You are the Data Evaluation Agent for Chop Kong Hin, \
a gold jewellery shop management system.

Your job is to evaluate whether the data retrieved from the database is \
SUFFICIENT to answer the user's original question.

Be pragmatic: if the data at least partially addresses the question or gives \
a meaningful answer, consider it sufficient. Only mark insufficient if the \
result set is empty or completely irrelevant to the question.

Respond ONLY with a valid JSON object:
{
  "verdict": "sufficient" or "insufficient",
  "feedback": "<if insufficient: precise instructions for the SQL Agent on how to broaden or fix the query; if sufficient: empty string>",
  "reasoning": "<brief explanation>"
}
Do NOT include markdown fences or any text outside the JSON.
"""


def _format_results_for_prompt(results: list[dict], columns: list[str]) -> str:
    """Format DB results as a compact text table for the LLM prompt."""
    if not results:
        return "(No rows returned)"

    # Show at most 20 rows to stay within token limits
    sample = results[:20]
    header = " | ".join(columns)
    divider = "-" * len(header)
    rows_str = "\n".join(
        " | ".join(str(row.get(col, "")) for col in columns) for row in sample
    )
    suffix = f"\n... ({len(results) - 20} more rows)" if len(results) > 20 else ""
    return f"{header}\n{divider}\n{rows_str}{suffix}"


def run(
    question: str,
    sql: str,
    results: list[dict[str, Any]],
    columns: list[str],
) -> dict[str, Any]:
    """Evaluate whether the retrieved data adequately answers the question.

    Parameters
    ----------
    question:
        The user's original (or restructured) question.
    sql:
        The SQL query that was executed.
    results:
        List of row dicts returned by the database.
    columns:
        Column names corresponding to the results.

    Returns
    -------
    dict with keys: verdict, feedback, reasoning.
    """
    logger.info("[DataEvaluationAgent] Evaluating %d rows for: %s", len(results), question)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = _get_client()

    results_text = _format_results_for_prompt(results, columns)

    user_msg = (
        f"User question: {question}\n\n"
        f"SQL executed:\n{sql}\n\n"
        f"Database results:\n{results_text}"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    logger.debug("[DataEvaluationAgent] Raw LLM response: %s", raw)

    try:
        result: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("[DataEvaluationAgent] Failed to parse JSON: %s", exc)
        result = {
            "verdict": "sufficient" if results else "insufficient",
            "feedback": "" if results else "The query returned no rows. Try broader conditions.",
            "reasoning": "JSON parse failed; defaulting based on row count.",
        }

    result.setdefault("verdict", "sufficient")
    result.setdefault("feedback", "")
    result.setdefault("reasoning", "")

    logger.info(
        "[DataEvaluationAgent] Verdict: %s | Reasoning: %s",
        result["verdict"],
        result["reasoning"],
    )
    return result
