"""
app.py
------
Agentic Database RAG — Streamlit entry point.

Agentic Pipeline:
    User Question
        → Question Understanding Agent (classify: general | database)
        → [general] Direct answer
        → [database] SQL Query Agent (RAG schema + generate + execute + retry)
            → Data Evaluation Agent (sufficient? → Summary Agent | insufficient → retry)
                → Summary Agent (RAG knowledge + synthesise)
        → Display final answer + reasoning trace
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.logging_utils import configure_logging

# Configure logging before any imports that use it
configure_logging()

import agents.question_understanding as qu_agent
import agents.sql_query_agent as sql_agent
import agents.data_evaluation_agent as eval_agent
import agents.summary_agent as summary_agent
from db.connection import test_connection

load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic Database RAG | Chop Kong Hin",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — modern dark jewellery-shop aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;600&display=swap');

    :root {
        --gold: #C9A84C;
        --gold-light: #E8C96A;
        --dark-bg: #0E0E0E;
        --card-bg: #1A1A1A;
        --border: #2A2A2A;
        --text-primary: #F5F5F0;
        --text-secondary: #9A9A8A;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--dark-bg);
        color: var(--text-primary);
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1A1100 0%, #0E0E0E 60%, #1A1100 100%);
        border-bottom: 1px solid var(--gold);
        padding: 2rem 2.5rem 1.5rem;
        margin-bottom: 2rem;
    }
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 600;
        color: var(--gold);
        margin: 0;
        letter-spacing: 0.02em;
    }
    .main-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-top: 0.4rem;
        font-weight: 300;
    }

    /* Agent trace cards */
    .agent-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-left: 3px solid var(--gold);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        animation: fadeIn 0.4s ease-in;
    }
    .agent-card.success { border-left-color: #4CAF50; }
    .agent-card.warning { border-left-color: #FF9800; }
    .agent-card.error   { border-left-color: #F44336; }
    .agent-card.info    { border-left-color: var(--gold); }

    .agent-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--gold-light);
        margin-bottom: 0.35rem;
    }
    .agent-content { font-size: 0.88rem; color: var(--text-primary); }

    /* Answer box */
    .final-answer {
        background: linear-gradient(135deg, #1C1500 0%, #1A1A1A 100%);
        border: 1px solid var(--gold);
        border-radius: 12px;
        padding: 1.75rem;
        margin-top: 1.5rem;
        animation: fadeIn 0.6s ease-in;
    }
    .final-answer h4 {
        color: var(--gold);
        font-family: 'Playfair Display', serif;
        margin-bottom: 0.75rem;
    }

    /* Sources section */
    .sources-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1.75rem;
        margin-bottom: 0.85rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }
    .sources-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-secondary);
    }
    /* Policy quote card */
    .policy-quote {
        background: #111;
        border: 1px solid #2A2A2A;
        border-left: 4px solid var(--gold);
        border-radius: 6px;
        padding: 0.85rem 1rem 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        animation: fadeIn 0.4s ease-in;
    }
    .policy-quote-num {
        font-size: 0.65rem;
        font-weight: 700;
        color: var(--gold);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .policy-quote-text {
        font-size: 0.85rem;
        color: #D5D5CC;
        line-height: 1.55;
        white-space: pre-wrap;
    }
    /* DB sources box */
    .db-source-box {
        background: #0D0D0D;
        border: 1px solid #2A2A2A;
        border-top: 3px solid #4CAF50;
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.75rem;
        animation: fadeIn 0.4s ease-in;
    }
    .db-source-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4CAF50;
        margin-bottom: 0.5rem;
    }

    /* Input */
    .stTextArea textarea {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, var(--gold) 0%, #A8882A 100%) !important;
        color: #0E0E0E !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 2rem !important;
        transition: opacity 0.2s ease !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0A0A0A !important;
        border-right: 1px solid var(--border) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--card-bg) !important;
        color: var(--gold-light) !important;
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
    }

    /* Chips */
    .status-chip {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .chip-green  { background: rgba(76,175,80,0.15); color: #4CAF50; border: 1px solid #4CAF50; }
    .chip-red    { background: rgba(244,67,54,0.15);  color: #F44336; border: 1px solid #F44336; }
    .chip-gold   { background: rgba(201,168,76,0.15); color: var(--gold); border: 1px solid var(--gold); }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Hide Streamlit chrome EXCEPT header (required for sidebar toggle) */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache: dict[str, Any] = {}
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💍 Chop Kong Hin")
    st.markdown("<p style='color:#9A9A8A;font-size:0.8rem;'>Agentic Database RAG</p>", unsafe_allow_html=True)
    st.divider()

    # DB connection status
    st.markdown("#### 🔌 Database Connection")
    db_ok = test_connection()
    if db_ok:
        st.markdown('<span class="status-chip chip-green">● Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-chip chip-red">● Disconnected</span>', unsafe_allow_html=True)
        st.caption("Fill in your .env file with correct DB credentials.")

    st.divider()

    # Model info
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    st.markdown(f"#### 🤖 LLM Model")
    st.markdown(f'<span class="status-chip chip-gold">{model_name}</span>', unsafe_allow_html=True)

    st.divider()

    # Cache info
    cache_count = len(st.session_state.answer_cache)
    st.markdown(f"#### 🗄️ Query Cache")
    st.markdown(f"**{cache_count}** cached result(s)")
    if st.button("Clear Cache", use_container_width=True):
        st.session_state.answer_cache = {}
        st.success("Cache cleared.")

    st.divider()

    # Sample questions
    st.markdown("#### 💡 Try These Questions")
    samples = [
        "How many items are currently in stock?",
        "What are the top 5 selling stock types?",
        "Show me all pending bookings",
        "What is your refund policy?",
        "How many customers do we have?",
        "What is the total sales revenue this year?",
        "List all salesmen and their companies",
        "What gold types are available?",
    ]
    for s in samples:
        if st.button(s, key=f"sample_{hashlib.md5(s.encode()).hexdigest()[:6]}", use_container_width=True):
            st.session_state["active_question"] = s
            st.rerun()

    st.divider()
    st.markdown("<p style='color:#555;font-size:0.7rem;text-align:center;'>© 2025 Chop Kong Hin</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">💍 Agentic Database RAG</div>
        <div class="main-subtitle">Chop Kong Hin · Intelligent Business Intelligence Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main input area
# ---------------------------------------------------------------------------
col_input, col_btn = st.columns([5, 1])

# Check if a sample question was clicked
active_q = st.session_state.pop("active_question", None)

with col_input:
    question = st.text_area(
        "Ask me anything about your business data or store policies:",
        height=100,
        placeholder='e.g. "How many gold rings are currently in stock?" or "What is your exchange policy?"',
        key="question_input",
        label_visibility="collapsed",
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("✨ Ask", use_container_width=True)

# If a sample question was clicked, simulate submission
if active_q:
    submit = True
    question = active_q


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
def _cache_key(q: str) -> str:
    return hashlib.sha256(q.strip().lower().encode()).hexdigest()


def run_pipeline(user_question: str) -> dict[str, Any]:
    """Execute the full 4-agent pipeline and return a structured result dict."""
    key = _cache_key(user_question)
    if key in st.session_state.answer_cache:
        return {**st.session_state.answer_cache[key], "cached": True}

    trace: list[dict] = []
    start_ts = time.time()

    # ── Step 1: Question Understanding ──────────────────────────────────────
    trace.append({"agent": "Question Understanding Agent", "status": "running"})
    with st.spinner("🔍 Analysing your question…"):
        qu_result = qu_agent.run(user_question)

    trace[-1]["status"] = "done"
    trace[-1]["output"] = qu_result

    if qu_result["type"] == "general":
        # Direct answer — no DB needed; include policy source chunks for citation
        final = {
            "type": "general",
            "answer": qu_result["answer"],
            "policy_sources": qu_result.get("policy_sources", []),
            "trace": trace,
            "elapsed": round(time.time() - start_ts, 2),
            "cached": False,
        }
        st.session_state.answer_cache[key] = {k: v for k, v in final.items() if k != "cached"}
        return final

    # ── Step 2: SQL Query Agent ─────────────────────────────────────────────
    trace.append({"agent": "SQL Query Agent", "status": "running"})
    feedback: str | None = None
    sql_result: dict[str, Any] = {}

    max_eval_retries = int(os.getenv("MAX_SQL_RETRIES", "3"))
    for eval_attempt in range(max_eval_retries):
        with st.spinner(f"🛢️ Generating SQL query (attempt {eval_attempt + 1})…"):
            sql_result = sql_agent.run(
                qu_result["restructured_question"], previous_feedback=feedback
            )

        trace[-1]["status"] = "done"
        trace[-1]["output"] = sql_result

        if sql_result["error"] and not sql_result["results"]:
            # SQL agent exhausted its internal retries with no results
            break

        # ── Step 3: Data Evaluation Agent ───────────────────────────────────
        trace.append({"agent": "Data Evaluation Agent", "status": "running"})
        with st.spinner("🔎 Evaluating retrieved data…"):
            eval_result = eval_agent.run(
                question=qu_result["restructured_question"],
                sql=sql_result["sql"],
                results=sql_result["results"],
                columns=sql_result["columns"],
            )
        trace[-1]["status"] = "done"
        trace[-1]["output"] = eval_result

        if eval_result["verdict"] == "sufficient":
            break
        else:
            feedback = eval_result["feedback"]
            trace.append({"agent": "SQL Query Agent", "status": "running"})  # retry marker

    # ── Step 4: Summary Agent ────────────────────────────────────────────────
    trace.append({"agent": "Summary Agent", "status": "running"})
    with st.spinner("✍️ Composing your answer…"):
        sum_result = summary_agent.run(
            question=user_question,
            results=sql_result.get("results", []),
            columns=sql_result.get("columns", []),
        )
    trace[-1]["status"] = "done"
    trace[-1]["output"] = sum_result

    final = {
        "type": "database",
        "answer": sum_result["answer"],
        "sql": sql_result.get("sql", ""),
        "results": sql_result.get("results", []),
        "columns": sql_result.get("columns", []),
        "sql_attempts": sql_result.get("attempts", 1),
        "knowledge_context": sum_result.get("knowledge_context_used", ""),
        "trace": trace,
        "elapsed": round(time.time() - start_ts, 2),
        "cached": False,
    }
    st.session_state.answer_cache[key] = {k: v for k, v in final.items() if k != "cached"}
    return final


# ---------------------------------------------------------------------------
# Render pipeline result
# ---------------------------------------------------------------------------
def render_result(result: dict[str, Any]) -> None:
    """Render the full pipeline result in the Streamlit UI."""

    # Cache badge
    if result.get("cached"):
        st.markdown('<span class="status-chip chip-gold">⚡ Cached Result</span>', unsafe_allow_html=True)
        st.markdown("")

    # ── Final Answer ─────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="final-answer">
            <h4>💬 Answer</h4>
            <div>{result["answer"].replace(chr(10), "<br>")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"⏱ {result.get('elapsed', 0)}s")

    # ── 📎 SOURCES & EVIDENCE (always visible, no expander) ──────────────────
    st.markdown(
        '<div class="sources-header">'
        '<span style="font-size:1.1rem;">📎</span>'
        '<span class="sources-title">Sources &amp; Evidence</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if result["type"] == "database":
        # ── Database answer: show SQL + result table ──────────────────────────
        sql = result.get("sql", "")
        rows = result.get("results", [])
        columns = result.get("columns", [])
        attempts = result.get("sql_attempts", 1)

        if sql:
            st.markdown(
                f'<div class="db-source-box">'
                f'<div class="db-source-label">🛢️ SQL Query executed ({attempts} attempt{"s" if attempts > 1 else ""})</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.code(sql, language="sql")

        if rows and columns:
            row_count = len(rows)
            st.markdown(
                f'<div class="db-source-label" style="margin-top:0.5rem;margin-bottom:0.4rem;">'
                f'📊 Database Result — <strong>{row_count} row{"s" if row_count != 1 else ""} returned</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
            df = pd.DataFrame(rows, columns=columns)
            st.dataframe(df, use_container_width=True)
        elif sql:
            st.markdown(
                '<div style="color:#FF9800;font-size:0.82rem;padding:0.5rem 0;">'
                '⚠️ Query returned 0 rows — the answer is based on the best available data.</div>',
                unsafe_allow_html=True,
            )

    else:
        # ── General / policy answer: show quoted policy sentences ─────────────
        policy_sources: list[str] = result.get("policy_sources", [])
        if policy_sources:
            st.markdown(
                '<div style="font-size:0.78rem;color:#9A9A8A;margin-bottom:0.6rem;">'
                'The following passages from the store policy were retrieved and used to formulate the answer:'
                '</div>',
                unsafe_allow_html=True,
            )
            for i, chunk in enumerate(policy_sources, 1):
                # Escape HTML special chars in the raw text
                safe_chunk = (
                    chunk.replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;")
                         .replace('"', "&quot;")
                )
                st.markdown(
                    f'<div class="policy-quote">'
                    f'<div class="policy-quote-num">📄 Source {i}</div>'
                    f'<div class="policy-quote-text">{safe_chunk}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#555;font-size:0.82rem;padding:0.4rem 0;">'
                'ℹ️ This answer was derived from general knowledge; no specific policy passage was matched.</div>',
                unsafe_allow_html=True,
            )

    # ── Reasoning Trace (collapsible) ─────────────────────────────────────────
    with st.expander("🔬 Agent Reasoning Trace", expanded=False):
        for step in result.get("trace", []):
            agent = step.get("agent", "")
            output = step.get("output", {})

            card_class = "info"
            if "SQL" in agent:
                card_class = "warning" if not output.get("results") else "success"
            elif "Summary" in agent:
                card_class = "success"
            elif "Evaluation" in agent:
                verdict = output.get("verdict", "")
                card_class = "success" if verdict == "sufficient" else "warning"

            content_parts = []
            if "type" in output:
                content_parts.append(f"<strong>Classification:</strong> {output['type'].upper()}")
            if output.get("reasoning"):
                content_parts.append(f"<strong>Reasoning:</strong> {output['reasoning']}")
            if output.get("restructured_question"):
                content_parts.append(f"<strong>Restructured Question:</strong> {output['restructured_question']}")
            if output.get("sql"):
                content_parts.append(f"<strong>SQL:</strong> <code style='font-size:0.78rem;'>{output['sql']}</code>")
            if "attempts" in output:
                content_parts.append(f"<strong>SQL Attempts:</strong> {output['attempts']}")
            if output.get("verdict"):
                verdict_icon = "✅" if output['verdict'] == "sufficient" else "⚠️"
                content_parts.append(f"<strong>Verdict:</strong> {verdict_icon} {output['verdict'].upper()}")
            if output.get("feedback"):
                content_parts.append(f"<strong>Feedback:</strong> {output['feedback']}")
            if "results" in output and isinstance(output["results"], list):
                content_parts.append(f"<strong>Rows Retrieved:</strong> {len(output['results'])}")
            if output.get("answer") and agent == "Question Understanding Agent":
                content_parts.append(f"<strong>Direct Answer:</strong> {output['answer']}")

            content_html = "<br>".join(content_parts) if content_parts else "(no output)"
            st.markdown(
                f'<div class="agent-card {card_class}">'
                f'<div class="agent-label">🤖 {agent}</div>'
                f'<div class="agent-content">{content_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if submit and question.strip():
    with st.container():
        try:
            result = run_pipeline(question.strip())
            render_result(result)

            # Add to history
            st.session_state.history.append({
                "question": question.strip(),
                "type": result.get("type", "unknown"),
                "timestamp": time.strftime("%H:%M:%S"),
            })

        except Exception as exc:
            st.error(f"❌ Pipeline error: {exc}")
            st.caption("Check your .env credentials and ensure the database is reachable.")

elif submit:
    st.warning("Please enter a question before clicking Ask.")

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
if st.session_state.history:
    with st.expander("🕑 Question History", expanded=False):
        for i, h in enumerate(reversed(st.session_state.history[-20:]), 1):
            badge = "🗄️" if h["type"] == "database" else "💡"
            st.markdown(
                f"`{h['timestamp']}` {badge} **{h['question']}**",
                unsafe_allow_html=False,
            )

# ---------------------------------------------------------------------------
# Welcome state
# ---------------------------------------------------------------------------
if not submit and not st.session_state.history:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem;color:#555;">
            <div style="font-size:3rem;margin-bottom:1rem;">💍</div>
            <div style="font-size:1.1rem;color:#888;">Welcome to the Chop Kong Hin Intelligence Assistant</div>
            <div style="font-size:0.85rem;color:#555;margin-top:0.5rem;">
                Ask any question about your inventory, sales, bookings, customers, or store policies.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
