"""
api.py
------
FastAPI entry point for the Agentic Database RAG pipeline.
Exposes a REST API that can be integrated with WhatsApp, Slack, etc.

Usage:
    uvicorn api:app --reload
"""

import os
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.logging_utils import configure_logging

# Configure logging before any imports that use it
configure_logging()

import agents.question_understanding as qu_agent
import agents.sql_query_agent as sql_agent
import agents.data_evaluation_agent as eval_agent
import agents.summary_agent as summary_agent
from db.connection import test_connection

load_dotenv()

app = FastAPI(
    title="Chop Kong Hin - Agentic Database API",
    description="REST API for querying the database and store policies using natural language.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    type: str
    answer: str
    policy_sources: list[str] = []
    sql_executed: str = ""
    rows_retrieved: int = 0
    elapsed_seconds: float


@app.get("/health")
def health_check() -> dict[str, str]:
    """Check if the API and database are reachable."""
    db_ok = test_connection()
    if db_ok:
        return {"status": "healthy", "database": "connected"}
    return {"status": "degraded", "database": "disconnected"}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest) -> AskResponse:
    """Run the 4-agent pipeline for a single question and return the synthesized answer."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start_ts = time.time()

    # ── Step 1: Question Understanding ──────────────────────────────────────
    qu_result = qu_agent.run(question)

    if qu_result["type"] == "general":
        # Direct answer
        return AskResponse(
            type="general",
            answer=qu_result["answer"],
            policy_sources=qu_result.get("policy_sources", []),
            elapsed_seconds=round(time.time() - start_ts, 2),
        )

    # ── Step 2: SQL Query Agent ─────────────────────────────────────────────
    feedback: str | None = None
    sql_result: dict[str, Any] = {}
    max_eval_retries = int(os.getenv("MAX_SQL_RETRIES", "3"))

    for _ in range(max_eval_retries):
        sql_result = sql_agent.run(
            qu_result["restructured_question"], previous_feedback=feedback
        )

        if sql_result["error"] and not sql_result["results"]:
            # Exhausted retries with no results
            break

        # ── Step 3: Data Evaluation Agent ───────────────────────────────────
        eval_result = eval_agent.run(
            question=qu_result["restructured_question"],
            sql=sql_result["sql"],
            results=sql_result["results"],
            columns=sql_result["columns"],
        )

        if eval_result["verdict"] == "sufficient":
            break
        else:
            feedback = eval_result["feedback"]

    # ── Step 4: Summary Agent ────────────────────────────────────────────────
    sum_result = summary_agent.run(
        question=question,
        results=sql_result.get("results", []),
        columns=sql_result.get("columns", []),
    )

    return AskResponse(
        type="database",
        answer=sum_result["answer"],
        sql_executed=sql_result.get("sql", ""),
        rows_retrieved=len(sql_result.get("results", [])),
        elapsed_seconds=round(time.time() - start_ts, 2),
    )
