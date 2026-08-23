"""Streamlit UI for the Autonomous Analyst -- portfolio-grade presentation layer.

Presentation only: no reasoning, guardrail, eval, or charting *logic* lives here. This file
routes a question to the existing agent.py / charts.py functions and formats whatever they
return for display -- INR currency formatting, step-card rendering, refusal/error styling --
without altering the underlying data those functions compute or return. Currency formatting in
particular is a pure display-layer transform: it builds a separate formatted copy of text/rows
for rendering and never writes back into the values the agent produced.

Run: streamlit run src/app.py
"""

from __future__ import annotations

import os
import re
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
    is_why_question,
)
from charts import maybe_render_chart, pick_key_why_step  # noqa: E402

if not os.path.exists(DB_PATH):
    # Cold start on a fresh checkout (e.g. Streamlit Community Cloud, which only gets a git
    # clone, not a locally-built warehouse.duckdb -- that file is gitignored). ingest.py's own
    # CSVs are committed and small (~90KB total), so this runs once per deploy and is cheap.
    import ingest  # noqa: E402

    ingest.main()

EXPLORE_QUESTIONS = [
    "What is total revenue by state?",
    "What is the monthly revenue trend over time?",
    "Which cities have negative total profit?",
    "What is the average revenue per order?",
]
DEEP_DIVE_QUESTIONS = [
    "Which month drove the biggest revenue change and why?",
]

# Real, committed 15-question execution-accuracy numbers -- see src/eval/report_ollama.json
# (14/15) and src/eval/report_groq.json (13/15). Update these two strings by hand whenever a new
# eval snapshot is committed; nothing here recomputes them.
EVAL_BADGES = [
    ("\U0001f5a5️", "Local 7B", "93.3%"),
    ("☁️", "Groq gpt-oss-120b", "86.7%"),
]

# Prefixes of agent.py's own fixed refusal/failure strings (destructive-intent guard,
# prompt-injection guard, SQL-retry/step-limit exhaustion) -- used only to pick a notice style
# for an already-final answer_text; never used to decide the answer_text's content itself.
REFUSAL_PREFIXES = (
    "I can only read and analyze the data",
    "I can't follow instructions embedded in a question",
)
FAILURE_PREFIXES = (
    "I couldn't produce a valid query",
    "I wasn't able to reach a final answer",
    "I couldn't complete a drill-down query",
)

st.set_page_config(page_title="Autonomous Analyst", page_icon="\U0001f4ca", layout="wide")

CUSTOM_CSS = """
<style>
.app-tagline {
    color: #94A3B8;
    font-size: 1.02rem;
    margin-top: -0.6rem;
    margin-bottom: 1rem;
}
.badge-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    background: rgba(99, 102, 241, 0.14);
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.35);
}
div[data-testid="stButton"] > button {
    border-radius: 999px !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    background: rgba(99, 102, 241, 0.07) !important;
    color: #C7D2FE !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    transition: background 0.15s ease-in-out, border-color 0.15s ease-in-out, transform 0.1s ease-in-out !important;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(99, 102, 241, 0.24) !important;
    border-color: #818CF8 !important;
    color: #EEF2FF !important;
    transform: translateY(-1px);
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0);
}
[class*="st-key-answer-box"] p,
[class*="st-key-answer-box"] li {
    font-size: 1.12rem !important;
    font-weight: 600 !important;
    line-height: 1.55 !important;
}
img {
    max-width: 100% !important;
    height: auto !important;
}
</style>
"""

# --- display-only helpers --------------------------------------------------------------------
# Everything below formats already-computed values for presentation; none of it feeds back into
# the rows/SQL/answer text the agent returns, and none of it changes how those values are
# computed.

CURRENCY_TOKENS = {"revenue", "profit", "amount", "sales", "cost", "price", "value", "target"}
RATIO_TOKENS = {
    "pct",
    "percent",
    "percentage",
    "margin",
    "growth",
    "attainment",
    "share",
    "rate",
    "ratio",
    "count",
    "quantity",
    "qty",
}
INR_MENTION_RE = re.compile(r"\bINR\s*([\d,]+(?:\.\d+)?)")


def is_currency_column(col: str) -> bool:
    """Heuristic, display-only: does this column represent an INR amount? Token-based (not a
    bare substring check) so e.g. 'target_attainment' (a ratio) isn't caught by 'target'."""
    tokens = set(re.split(r"[_\s]+", col.lower()))
    if tokens & RATIO_TOKENS:
        return False
    return bool(tokens & CURRENCY_TOKENS)


def format_inr(value) -> str:
    try:
        num = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return str(value)
    if num == int(num):
        return f"₹{int(num):,}"
    return f"₹{num:,.2f}"


def format_inr_mentions(text: str | None) -> str:
    """Rewrite 'INR 32,726' / 'INR 431502.0' style mentions in free text as '₹32,726' --
    only ever touches text that already carries an explicit INR marker, so it can't misfire on
    an unrelated number (a percentage, a row count, ...) that happens to appear nearby."""
    if not text:
        return text or ""
    return INR_MENTION_RE.sub(lambda m: format_inr(m.group(1)), text)


def build_display_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in df.columns:
        if is_currency_column(col) and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(format_inr)
    return df


def row_label(row: dict, exclude_col: str | None = None) -> str:
    """Human-readable label out of a finding row dict -- every non-numeric field (category/
    city/month name), skipping the metric column itself. Joins all of them (not just the
    first) because a multi-dimension breakdown -- e.g. a swing narrowed to two specific months
    AND broken down by sub-category -- has more than one identifying field, and dropping one
    would misrepresent which row is actually "the" dominant driver."""
    parts = [str(value) for key, value in row.items() if key != exclude_col and not isinstance(value, (int, float))]
    return " / ".join(parts) if parts else "(row)"


def format_metric(col: str, value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if is_currency_column(col):
        return format_inr(value)
    return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"


def classify_notice(msg: dict) -> str | None:
    if msg.get("is_error"):
        return "error"
    content = msg.get("content") or ""
    if content.startswith(REFUSAL_PREFIXES):
        return "refusal"
    if content.startswith(FAILURE_PREFIXES):
        return "failure"
    return None


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_client():
    return build_client()


def process_question(question: str, history: list[dict]) -> dict:
    """Route the question through the existing agent functions and package whatever they
    return into a plain dict the UI can render. Any unexpected exception is caught here so the
    UI never shows a stack trace -- the app itself never decides anything about the answer."""
    try:
        client = get_client()
        if is_why_question(question):
            result = answer_why(question, history=history, client=client, temperature=0, seed=0)
            key_step = pick_key_why_step(result["steps"]) if result["steps"] else None
            chart_path = maybe_render_chart(key_step["rows"]) if key_step else None
            return {
                "role": "assistant",
                "content": format_inr_mentions(result["answer_text"]),
                "is_why": True,
                "why_steps": result["steps"],
                "sql": result.get("sql"),
                "rows": result.get("rows"),
                "chart_path": chart_path,
            }
        result = answer_question(question, history=history, client=client, temperature=0, seed=0)
        chart_path = maybe_render_chart(result["rows"])
        return {
            "role": "assistant",
            "content": format_inr_mentions(result["answer_text"]),
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
            "is_error": True,
            "sql": None,
            "rows": None,
            "chart_path": None,
        }


def render_why_step_card(step: dict) -> None:
    with st.container(border=True):
        reason = format_inr_mentions(step.get("reason") or "(no reason given)")
        st.markdown(f"**Step {step['step']} — {reason}**")

        if step.get("error"):
            st.error(f"⚠️ {step['error']}")
            return

        if step.get("normalized"):
            st.caption(f"\U0001f527 {step['normalized']}")

        if step.get("sql"):
            with st.expander("View step SQL", expanded=False):
                st.code(step["sql"], language="sql")

        finding = step.get("finding") or {}

        time_series = finding.get("time_series")
        if time_series:
            swing = time_series["biggest_swing"]
            col = time_series["metric_column"]
            st.info(
                f"\U0001f4c8 Biggest swing in **{col}**: **{swing['from_period']}** "
                f"({format_metric(col, swing['from_value'])}) → "
                f"**{swing['to_period']}** ({format_metric(col, swing['to_value'])}) — "
                f"Δ {format_metric(col, swing['delta'])} ({swing['pct_change']}%), "
                f"{swing['swing_share_pct']}% of total movement"
            )

        dominant = finding.get("dominant_driver")
        if dominant:
            col = dominant["column"]
            label = row_label(dominant["extreme_row"], exclude_col=col)
            value = dominant["extreme_row"].get(col)
            st.success(
                f"\U0001f3af Dominant driver in **{col}**: **{label}** — "
                f"{format_metric(col, value)} ({dominant['extreme_share_pct']}% of total magnitude)"
            )
        elif finding.get("per_column"):
            parts = []
            for col, stats in finding["per_column"].items():
                label = row_label(stats["extreme_row"], exclude_col=col)
                value = stats["extreme_row"].get(col)
                share = stats.get("extreme_share_pct")
                share_txt = f", {share}%" if share is not None else ""
                parts.append(f"**{col}**: {label} ({format_metric(col, value)}{share_txt})")
            if parts:
                st.caption("No dominant driver — " + " · ".join(parts))

        if step.get("row_count") is not None:
            st.caption(f"{step['row_count']} rows returned")


def render_message(msg: dict, idx: int) -> None:
    if msg["role"] != "assistant":
        st.markdown(msg["content"])
        return

    notice = classify_notice(msg)
    if notice == "refusal":
        st.warning(f"\U0001f6e1️ {msg['content']}")
        return
    if notice in ("error", "failure"):
        st.error(f"⚠️ {msg['content']}")
        return

    with st.container(key=f"answer-box-{idx}"):
        st.markdown(msg["content"])

    if msg.get("is_why") and msg.get("why_steps"):
        st.markdown("#### \U0001f50d Reasoning trace")
        for step in msg["why_steps"]:
            render_why_step_card(step)

    if msg.get("sql"):
        with st.expander("View SQL"):
            st.code(msg["sql"], language="sql")

    if msg.get("rows"):
        st.dataframe(build_display_df(msg["rows"]), width="stretch", height=340)

    if msg.get("chart_path") and os.path.exists(msg["chart_path"]):
        # Fixed, bounded width (not full container stretch) -- stretching a chart PNG to the
        # full container upscales it well past its native ~800x450px resolution and makes it
        # look blurry. Centered in the middle of a 1:3:1 column split; the global `img { max-
        # width: 100% }` rule above keeps it from ever overflowing a narrower viewport.
        chart_col = st.columns([1, 3, 1])[1]
        chart_col.image(msg["chart_path"], width=700)


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("\U0001f4ca Autonomous Analyst")
st.markdown(
    '<div class="app-tagline">Ask business questions in plain English — get validated SQL, '
    "answers, charts, and honest root-cause analysis.</div>",
    unsafe_allow_html=True,
)

badges_html = "".join(
    f'<span class="badge-pill">{icon} {label}: {value}</span>' for icon, label, value in EVAL_BADGES
)
st.markdown(f'<div class="badge-row">{badges_html}</div>', unsafe_allow_html=True)
st.caption(f"Model in use: `{LLM_MODEL}` via `{LLM_BASE_URL}`")

question = st.chat_input("Ask about revenue, profit, categories, states, months…")

if not st.session_state.messages:
    st.markdown("**Explore**")
    cols = st.columns(len(EXPLORE_QUESTIONS))
    for col, example in zip(cols, EXPLORE_QUESTIONS):
        if col.button(example, key=f"explore-{example}", width="stretch"):
            question = example

    st.markdown("**Deep dive**")
    for example in DEEP_DIVE_QUESTIONS:
        if st.button(f"\U0001f50d {example}", key=f"deepdive-{example}", width="stretch"):
            question = example

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    spinner_text = (
        "Analyzing… running a bounded root-cause drill-down (up to 4 steps)…"
        if is_why_question(question)
        else "Analyzing…"
    )
    with st.spinner(spinner_text):
        st.session_state.messages.append(process_question(question, history))

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        render_message(msg, idx)
