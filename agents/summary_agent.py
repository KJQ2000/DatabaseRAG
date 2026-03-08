"""
agents/summary_agent.py
------------------------
Summary Agent — the final stage of the agentic pipeline.

Responsibilities
----------------
1. Use RAG on `knowledge/knowledge.txt` to inject relevant store-policy context.
2. Combine the database results with the store knowledge to produce a clear,
   user-facing answer.

Returns a typed dict:
    {
        "answer": str,
        "knowledge_context_used": str,
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from utils.rag import retrieve_relevant_chunks
from utils.logging_utils import get_logger

load_dotenv()
logger = get_logger(__name__)

_client: OpenAI | None = None

# Path to the store knowledge file
_KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "knowledge.txt"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in your .env file.")
        _client = OpenAI(api_key=api_key)
    return _client


_SYSTEM_PROMPT = """You are the Summary Agent for Chop Kong Hin, \
a gold jewellery shop.

You are the final step in an agentic AI pipeline. Your job is to synthesise:
1. The database results retrieved by a previous agent.
2. Relevant store knowledge/policies provided to you.

Write a clear, professional, and concise answer to the user's original question.
Follow these guidelines:
- Present any numerical data in a readable format (e.g., tables, bullet points).
- If the data is financial, include currency (RM) where appropriate.
- Reference the store knowledge only when directly relevant.
- Keep the tone helpful and customer-friendly.
- If results are partial or limited, acknowledge that honestly.
- Do NOT mention the underlying SQL, database, or technical pipeline.
"""


def _format_results_for_prompt(results: list[dict], columns: list[str]) -> str:
    """Format DB results as a readable text representation."""
    if not results:
        return "(No database records found)"

    sample = results[:50]
    header = " | ".join(columns)
    divider = "-" * max(len(header), 30)
    rows_str = "\n".join(
        " | ".join(str(row.get(col, "N/A")) for col in columns) for row in sample
    )
    suffix = f"\n... and {len(results) - 50} more rows" if len(results) > 50 else ""
    return f"{header}\n{divider}\n{rows_str}{suffix}"


def run(
    question: str,
    results: list[dict[str, Any]],
    columns: list[str],
) -> dict[str, Any]:
    """Generate the final user-facing answer.

    Parameters
    ----------
    question:
        The user's original question.
    results:
        Database rows returned by the SQL Agent.
    columns:
        Column names for the results.

    Returns
    -------
    dict with keys: answer, knowledge_context_used.
    """
    logger.info("[SummaryAgent] Generating final answer for: %s", question)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # RAG: fetch relevant store knowledge
    knowledge_context = retrieve_relevant_chunks(question, _KNOWLEDGE_PATH, top_k=4)
    logger.debug("[SummaryAgent] Knowledge context (%d chars)", len(knowledge_context))

    db_text = _format_results_for_prompt(results, columns)

    user_msg = (
        f"User's question: {question}\n\n"
        f"--- Database Results ---\n{db_text}\n\n"
        f"--- Relevant Store Knowledge ---\n{knowledge_context}"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    logger.info("[SummaryAgent] Final answer generated (%d chars).", len(answer))

    return {
        "answer": answer,
        "knowledge_context_used": knowledge_context,
    }
