"""Streamlit UI for the Autonomous Analyst -- Phase 9 part 2.

A thin front over the existing functions: answer_question(), answer_why(), is_why_question(),
format_why_trace() (agent.py) and maybe_render_chart(), pick_key_why_step() (charts.py). No
reasoning, guardrail, eval, or charting logic lives here -- this file only routes a question to
the right function and renders whatever it returns.

Run: streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import (  # noqa: E402
    DB_PATH,
    LLM_BASE_URL,
    LLM_MODEL,
    answer_question,
    answer_why,
    build_client,
    format_why_trace,
    is_why_question,
)
from charts import maybe_render_chart, pick_key_why_step  # noqa: E402

if not os.path.exists(DB_PATH):
    # Cold start on a fresh checkout (e.g. a Hugging Face Space, which only gets a git clone,
    # not a locally-built warehouse.duckdb -- that file is gitignored). ingest.py's own CSVs
    # are committed and small (~90KB total), so this runs once per Space boot and is cheap.
    import ingest  # noqa: E402

    ingest.main()

EXAMPLE_QUESTIONS = [
    "What is total revenue by state?",
    "What is the monthly revenue trend over time?",
    "Which cities have negative total profit?",
    "What is the average revenue per order?",
    "Which month drove the biggest revenue change and why?",
]

st.set_page_config(page_title="Autonomous Analyst", page_icon="\U0001f4ca", layout="wide")


@st.cache_resource
def get_client():
    return build_client()


def process_question(question: str, history: list[dict]) -> dict:
    """Route the question through the existing agent functions and package whatever they
    return into a plain dict the UI can render. Any unexpected exception is caught here so the
    UI never shows a stack trace -- the app itself never decides anything about the answer."""
    client = get_client()
    try:
        if is_why_question(question):
            result = answer_why(question, history=history, client=client, temperature=0, seed=0)
            key_step = pick_key_why_step(result["steps"]) if result["steps"] else None
            chart_path = maybe_render_chart(key_step["rows"]) if key_step else None
            return {
                "role": "assistant",
                "content": result["answer_text"],
                "is_why": True,
                "trace_text": format_why_trace(result["steps"]) if result["steps"] else None,
                "sql": result.get("sql"),
                "rows": result.get("rows"),
                "chart_path": chart_path,
            }
        result = answer_question(question, history=history, client=client, temperature=0, seed=0)
        chart_path = maybe_render_chart(result["rows"])
        return {
            "role": "assistant",
            "content": result["answer_text"],
            "is_why": False,
            "sql": result.get("sql"),
            "rows": result.get("rows"),
            "chart_path": chart_path,
        }
    except Exception as exc:  # noqa: BLE001 - the UI must never show a raw traceback
        print(f"[app] error answering {question!r}: {exc!r}", file=sys.stderr)
        return {
            "role": "assistant",
            "content": (
                "Something went wrong answering that question. Please try rephrasing it, or "
                "ask something else about the orders / order_items / sales_targets data."
            ),
            "is_why": False,
            "sql": None,
            "rows": None,
            "chart_path": None,
        }


def render_message(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg["role"] != "assistant":
        return
    if msg.get("trace_text"):
        st.markdown("**Reasoning trace**")
        st.code(msg["trace_text"], language=None)
    if msg.get("sql"):
        with st.expander("SQL"):
            st.code(msg["sql"], language="sql")
    if msg.get("rows"):
        st.dataframe(pd.DataFrame(msg["rows"]), width="stretch")
    if msg.get("chart_path") and os.path.exists(msg["chart_path"]):
        # Fixed, bounded width (not "stretch") -- in the wide layout, stretching a chart PNG to
        # the full container upscales it well past its native ~800x450px resolution and makes
        # it look oversized/blurry. Centered in the middle of a 1:3:1 column split so it reads
        # as a bounded figure under the full-width table above it, not a full-viewport image.
        chart_col = st.columns([1, 3, 1])[1]
        chart_col.image(msg["chart_path"], width=700)


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("\U0001f4ca Autonomous Analyst")
st.caption(
    "Ask a question about orders / order_items / sales_targets. Plain questions get a direct "
    "answer; \"why\" questions trigger a bounded root-cause drill-down."
)
st.caption(f"Model: `{LLM_MODEL}` via `{LLM_BASE_URL}`")

question = st.chat_input("Ask a question...")

if not st.session_state.messages:
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(example, width="stretch"):
            question = example

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        st.session_state.messages.append(process_question(question, history))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_message(msg)
