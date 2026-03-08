"""
agents/question_understanding.py
---------------------------------
Question Understanding Agent — the first agent in the pipeline.

Responsibilities
----------------
1. Classify whether the question can be answered from general/store knowledge
   or requires a live database query.
2. If database access is needed, restructure the question into a precise,
   SQL-friendly form that captures the intent clearly.
3. If general knowledge, retrieve relevant policy sentences via RAG on
   knowledge.txt and return them as `policy_sources` for UI citation display.

Returns a typed dict:
    {
        "type": "general" | "database",
        "answer": str,                    # populated when type == "general"
        "restructured_question": str,     # populated when type == "database"
        "reasoning": str,
        "policy_sources": list[str],      # relevant policy sentences (general only)
    }
"""

from __future__ import annotations

import json
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

# Path to the store knowledge file (policy text)
_KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "knowledge.txt"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in your .env file.")
        _client = OpenAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Question Understanding Agent for Chop Kong Hin, \
a gold jewellery shop management system.

Your job is to analyse the user's question and decide:

1. If the question is about GENERAL STORE KNOWLEDGE — such as policies, product \
descriptions, refund rules, contact information, or anything that doesn't require \
looking up specific records (sales figures, stock counts, customer details, dates, \
financial data, etc.) — answer it directly from your understanding.

2. If the question requires LIVE DATABASE DATA — such as inventory levels, \
transaction records, customer records, sales summaries, purchase history, staff \
data, or any aggregated business metrics — classify it as "database" and rewrite \
the question in a clear, precise form suitable for SQL generation.

Respond ONLY with a valid JSON object with this exact schema:
{
  "type": "general" or "database",
  "answer": "<direct answer if type==general, else empty string>",
  "restructured_question": "<precise SQL-oriented question if type==database, else empty string>",
  "reasoning": "<brief explanation of your classification decision>"
}

Do NOT include markdown fences or any text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------

def run(question: str) -> dict[str, Any]:
    """Classify and optionally restructure the user's question.

    Parameters
    ----------
    question:
        The raw question from the user.

    Returns
    -------
    dict with keys: type, answer, restructured_question, reasoning.
    """
    logger.info("[QuestionUnderstandingAgent] Processing question: %s", question)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = _get_client()

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"User question: {question}"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    logger.debug("[QuestionUnderstandingAgent] Raw LLM response: %s", raw)

    try:
        result: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("[QuestionUnderstandingAgent] Failed to parse JSON: %s", exc)
        # Fallback: treat as database query
        result = {
            "type": "database",
            "answer": "",
            "restructured_question": question,
            "reasoning": "Failed to parse agent response; defaulting to database query.",
        }

    # Normalise
    result.setdefault("type", "database")
    result.setdefault("answer", "")
    result.setdefault("restructured_question", question)
    result.setdefault("reasoning", "")
    result["policy_sources"] = []

    # For general questions: retrieve the specific policy sentences used as evidence
    if result["type"] == "general":
        try:
            raw_context = retrieve_relevant_chunks(question, _KNOWLEDGE_PATH, top_k=5)
            # Split on the separator used by rag.py so each chunk is its own citation
            chunks = [c.strip() for c in raw_context.split("\n\n---\n\n") if c.strip()]
            result["policy_sources"] = chunks
            logger.info(
                "[QuestionUnderstandingAgent] Retrieved %d policy source chunk(s).",
                len(chunks),
            )
        except Exception as exc:
            logger.warning("[QuestionUnderstandingAgent] RAG on knowledge.txt failed: %s", exc)

    logger.info(
        "[QuestionUnderstandingAgent] Classification: %s | Reasoning: %s",
        result["type"],
        result["reasoning"],
    )
    return result
