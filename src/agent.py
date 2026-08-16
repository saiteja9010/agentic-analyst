"""Autonomous Analyst: a freeform text-to-SQL agent over the DuckDB warehouse.

Core entry point is answer_question(), used by both the interactive REPL below
and (later) an eval harness. The only tool the model has is run_sql, which is
guarded so it can only ever run a single read-only SELECT/WITH statement.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DB_PATH = os.getenv("DUCKDB_PATH", "data/warehouse.duckdb")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))

SEMANTIC_LAYER_PATH = Path(__file__).parent / "semantic_layer.yaml"

MAX_ROWS = 1000
MAX_TOOL_ROWS_TO_MODEL = 30  # keep tool results small relative to num_ctx
QUERY_TIMEOUT_SECONDS = 10
MAX_SQL_RETRIES = 3
MAX_TOOL_ITERATIONS = 6
PII_COLUMNS = {"customer_name"}

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|PRAGMA|INSTALL|LOAD)\b",
    re.IGNORECASE,
)
SELECT_OR_WITH = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
HAS_LIMIT = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(the\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?(the\s+)?above",
        r"disregard\s+(all\s+)?(the\s+)?(previous|above)\s+instructions",
        r"new\s+instructions\s*:",
        r"reveal\s+your\s+(system\s+)?prompt",
        r"reveal\s+your\s+instructions",
        r"you\s+are\s+now\s+",
        r"act\s+as\s+(a\s+)?(different|new)\s+(ai|assistant|model)",
        r"override\s+(your\s+)?(rules|instructions)",
        r"jailbreak",
        r"developer\s+mode",
        r"change\s+the\s+tool\s+rules",
    ]
]


class SQLGuardError(Exception):
    """Raised when a candidate SQL query fails the run_sql guardrails."""


TOOL_CALL_JSON_RE = re.compile(
    r'\{\s*"name"\s*:\s*"run_sql"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)
SQL_FENCE_RE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_fallback_tool_call(content: str | None) -> str | None:
    """Best-effort extraction of a run_sql call from free-text model output.

    qwen2.5-coder:7b via Ollama advertises tool-calling support but does not
    reliably populate the structured tool_calls field -- it instead emits either a
    bare {"name": "run_sql", "arguments": {...}} blob or a fenced ```sql``` block.
    Parse either form so the agent still works end to end on this model.
    """
    if not content:
        return None
    match = TOOL_CALL_JSON_RE.search(content)
    if match:
        try:
            args = json.loads(match.group(1))
        except json.JSONDecodeError:
            args = None
        if args and args.get("query"):
            return args["query"]
    match = SQL_FENCE_RE.search(content)
    if match:
        return match.group(1).strip()
    return None


def load_semantic_layer_text() -> str:
    return SEMANTIC_LAYER_PATH.read_text()


def detect_prompt_injection(question: str) -> bool:
    return any(pattern.search(question) for pattern in INJECTION_PATTERNS)


def validate_and_prepare_sql(query: str) -> str:
    """Enforce: single statement, SELECT/WITH only, no forbidden keywords, has a LIMIT."""
    q = (query or "").strip()
    if not q:
        raise SQLGuardError("Empty query.")

    body = q[:-1].strip() if q.endswith(";") else q
    if ";" in body:
        raise SQLGuardError("Only a single SQL statement is allowed (no ';'-separated statements).")
    if not SELECT_OR_WITH.match(body):
        raise SQLGuardError("Only SELECT / WITH statements are allowed.")
    if FORBIDDEN_KEYWORDS.search(body):
        raise SQLGuardError(
            "Query contains a forbidden keyword; only read-only SELECT/WITH queries are allowed."
        )
    if not HAS_LIMIT.search(body):
        body = f"{body}\nLIMIT {MAX_ROWS}"
    return body


def run_query_with_timeout(
    con: duckdb.DuckDBPyConnection, sql: str, timeout: int
) -> pd.DataFrame:
    outcome: dict = {}

    def target() -> None:
        try:
            outcome["df"] = con.execute(sql).fetchdf()
        except Exception as exc:  # noqa: BLE001 - surfaced to caller below
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        con.interrupt()
        thread.join(2)
        raise SQLGuardError(f"Query exceeded {timeout}s timeout and was cancelled.")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["df"]


SQL_OPERATOR_UNESCAPES = [
    ("\\u003c", "<"),
    ("\\u003C", "<"),
    ("\\u003e", ">"),
    ("\\u003E", ">"),
    ("\\u0026", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&amp;", "&"),
]


def unescape_sql_operators(query: str) -> str:
    """Undo unicode/HTML escaping of comparison operators the model sometimes
    applies to '<', '>', '&' (e.g. \\u003c, &lt;), which otherwise produces
    syntactically invalid SQL that DuckDB can't parse."""
    for escaped, literal in SQL_OPERATOR_UNESCAPES:
        query = query.replace(escaped, literal)
    return query


def make_run_sql(con: duckdb.DuckDBPyConnection):
    def run_sql(query: str) -> dict:
        query = unescape_sql_operators(query)
        sql = validate_and_prepare_sql(query)
        df = run_query_with_timeout(con, sql, QUERY_TIMEOUT_SECONDS)
        df = df.head(MAX_ROWS)
        for col in PII_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])
        return {"sql": sql, "df": df}

    return run_sql


RUN_SQL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": (
            "Execute a single read-only DuckDB SELECT/WITH query against the warehouse and "
            "return the resulting rows. Use this to answer any factual question about the data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single DuckDB SELECT or WITH statement. No semicolons, no DDL/DML.",
                }
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT_TEMPLATE = """You are the Autonomous Analyst, a text-to-SQL agent over a DuckDB data warehouse.

You MUST call the run_sql tool to query data before answering any factual question. Never
fabricate numbers.

Rules:
- Use DuckDB SQL dialect.
- Use ONLY the tables and columns defined in the semantic layer below. NEVER invent columns or tables.
- Prefer the metric definitions given below over writing ad hoc aggregations when a matching metric exists.
- If the question is ambiguous, state your assumption in the final answer before giving the result.
- customer_name is PII and is stripped from tool results automatically -- never try to work around this.
- If run_sql returns an error, read the error and fix your query. You have a limited number of retries.
- When you have the answer, respond with a concise natural-language answer. The application
  displays the SQL and result table separately, so do not repeat the raw table row by row.

SQL rules (DuckDB dialect):
- Order-level metrics (e.g. average order value) use SUM(amount) / COUNT(DISTINCT order_id).
  NEVER use AVG(amount) over order_items rows -- that averages line items, not orders.
- To filter on an AGGREGATE (e.g. "loss-making" = total profit < 0), GROUP BY first and filter
  with HAVING SUM(...) < 0. NEVER filter with a row-level WHERE on the unaggregated column --
  that keeps any group with at least one matching row, not groups whose total matches.
- For "top N per group", compute a window rank (ROW_NUMBER/RANK) in a CTE, then filter
  WHERE rank = 1 in the outer query. Returning the whole ranked table without that outer filter
  is wrong even if the ranking itself is correct.
- For a "trend over time" question, return ALL periods ordered ascending. NEVER add
  LIMIT 1 (or any LIMIT that cuts off periods) to a trend/time-series query.
- Use the metric definitions in the semantic layer verbatim rather than re-deriving them.
- Return ONLY the columns the question asks for -- the grouping key(s) and the requested
  metric. Do NOT add extra helper columns (e.g. intermediate totals used to compute a ratio)
  unless the question asks for them.

Examples (patterns only -- adapt table/column names to the actual question):

Q: "What is the average revenue per order?"
SQL: SELECT SUM(oi.amount) / NULLIF(COUNT(DISTINCT o.order_id), 0) AS avg_order_value
     FROM orders o JOIN order_items oi ON o.order_id = oi.order_id;

Q: "Which cities have negative total profit?"
SQL: SELECT o.city, SUM(oi.profit) AS total_profit
     FROM orders o JOIN order_items oi ON o.order_id = oi.order_id
     GROUP BY o.city
     HAVING SUM(oi.profit) < 0
     ORDER BY total_profit ASC;

Q: "For each state, which city generated the highest revenue?"
SQL: WITH agg AS (
       SELECT o.state, o.city, SUM(oi.amount) AS revenue
       FROM orders o JOIN order_items oi ON o.order_id = oi.order_id
       GROUP BY o.state, o.city
     ),
     ranked AS (
       SELECT *, ROW_NUMBER() OVER (PARTITION BY state ORDER BY revenue DESC) AS rnk
       FROM agg
     )
     SELECT state, city, revenue FROM ranked WHERE rnk = 1 ORDER BY state;

Q: "Show quantity sold by month over the full year."
SQL: SELECT strftime(o.order_date, '%Y-%m') AS month, SUM(oi.quantity) AS units_sold
     FROM orders o JOIN order_items oi ON o.order_id = oi.order_id
     GROUP BY month
     ORDER BY month;  -- no LIMIT: a trend needs every period

Q: "What is the profit margin for each city?"
SQL: SELECT o.city, SUM(oi.profit) / NULLIF(SUM(oi.amount), 0) AS profit_margin
     FROM orders o JOIN order_items oi ON o.order_id = oi.order_id
     GROUP BY o.city
     ORDER BY profit_margin DESC;
     -- only [city, profit_margin] -- do NOT also select the intermediate
     -- SUM(profit) / SUM(amount) totals, even though they were used to compute it.

=== SEMANTIC LAYER ===
{semantic_layer}
=== END SEMANTIC LAYER ==="""

SYNTHESIS_SYSTEM_PROMPT = """You write the final natural-language answer for the Autonomous \
Analyst. You are given the user's question and the exact result table returned by the SQL \
query that answered it. Summarize ONLY what is in that table -- never use outside knowledge, \
never guess, and never pull numbers or claims from conversation history. Currency values are \
in INR.

Rules for what to state, by shape of the table:
- Trend over time (a period/date column plus a metric, multiple rows ordered by period): the
  JSON below includes "first_row" and "last_row" fields -- these ARE the first and last periods
  in the query's own row order. Use them verbatim (do not re-derive "first"/"last" by scanning
  "rows" yourself, and do not assume calendar order). State their period and value BY NAME AND
  NUMBER, and give the overall direction and magnitude of the change (e.g. "revenue rose from
  INR 12,400 in 2018-04 to INR 18,900 in 2019-03, up about 52%").
- Ranking / top-N: name the top item(s) by their label and state their value(s).
- Single metric (one row, one number): state the number plainly, with INR units if it is a
  currency amount.
- Anything else: pick out the concrete rows/values that matter and state them by name and number.

Hard requirements:
- Every claim must be backed by a concrete number or name taken verbatim from the table. NEVER
  compute or invent a number that is not literally one of the table's cell values -- e.g. if you
  mention a row's growth-percentage cell, do not also invent a "from X to Y" revenue pair for
  that row unless X and Y are themselves literal cell values you can point to. The only
  before/after comparison you may build is the trend first_row-vs-last_row comparison described
  above, because those two values are handed to you explicitly.
- Quote period/date/category labels and numbers EXACTLY as they appear in the table, character
  for character. If a period label in the table is "2018-04", every mention of that period in
  your answer must be written "2018-04" -- CORRECT: "rose from INR 32,726 in 2018-04 to INR
  58,937 in 2019-03". WRONG: "rose from INR 32,726 in April 2018 to INR 58,937 in March 2019"
  (reformatted the label -- never do this, even mid-sentence or in a summary clause).
- Never write vague filler ("fluctuating", "varies", "some", "several", "various", "a mix of")
  unless the very next clause gives the concrete numbers those words refer to.
- Do not repeat the whole table row by row -- summarize it, but every sentence must be grounded
  in an actual value from the table.
- If the table is empty, say so plainly instead of inventing numbers.
- Be concise: a few sentences is enough."""


def build_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def _rows_for_synthesis(df: pd.DataFrame, max_rows: int = 40) -> tuple[list[dict], bool]:
    """Rows to hand to the final-answer synthesis call.

    Trends need the first and last period visible even when the table is large, so a
    truncated table keeps both ends (head + tail) rather than just the head.
    """
    if len(df) <= max_rows:
        return df.to_dict(orient="records"), False
    half = max_rows // 2
    combined = pd.concat([df.head(half), df.tail(max_rows - half)])
    return combined.to_dict(orient="records"), True


def synthesize_final_answer(
    question: str,
    df: pd.DataFrame,
    client: OpenAI,
    temperature: float | None = None,
    seed: int | None = None,
) -> str:
    """Generate the final natural-language answer strictly from the returned table.

    Kept as a separate, single-purpose LLM call (no tools, no conversation history) so the
    model can't fall back on vague phrasing from an earlier drafted response -- it only ever
    sees the question and the actual result rows.
    """
    rows_payload, truncated = _rows_for_synthesis(df) if not df.empty else ([], False)
    first_row = df.iloc[0].to_dict() if not df.empty else None
    last_row = df.iloc[-1].to_dict() if not df.empty else None
    table_json = json.dumps(
        {
            "row_count": len(df),
            "first_row": first_row,
            "last_row": last_row,
            "rows": rows_payload,
            "truncated": truncated,
        },
        default=str,
    )
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Result table (JSON -- the ONLY source of truth for your answer):\n{table_json}"
            ),
        },
    ]

    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed

    response = client.chat.completions.create(model=LLM_MODEL, messages=messages, **kwargs)
    return response.choices[0].message.content or "(no answer produced)"


WHY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bwhy\b",
        r"\bwhat\s+caused\b",
        r"\bwhat(?:'s| is| was)\s+driving\b",
        r"\bdrivers?\s+(?:of|behind)\b",
        r"\broot\s+cause\b",
        r"\bexplain\s+the\s+(?:drop|decline|decrease|fall|increase|rise|surge|jump|spike)\b",
        r"\breasons?\s+(?:for|behind)\b",
    ]
]


def is_why_question(question: str) -> bool:
    """Conservative "why"-detector: only routes to answer_why() on an explicit ask for a
    cause ("why", "what caused", "what's driving", "driver of/behind", "root cause",
    "explain the drop/rise/...", "reason for/behind"). Anything else -- including plain
    factual, ranking, and trend questions -- stays on the normal answer_question() path."""
    return any(pattern.search(question or "") for pattern in WHY_PATTERNS)


# === Root-cause drill-down (answer_why) ===
#
# A separate, bounded mode: does not modify or call into answer_question(). Reuses the same
# guardrailed run_sql tool (read-only, allowlist, LIMIT, timeout, self-correct, PII strip) via
# make_run_sql(), and the same anti-hallucination discipline as synthesize_final_answer() --
# every claim in the final conclusion must trace to a row that actually ran this turn.

MAX_WHY_STEPS = 4
DRIVER_DOMINANCE_PCT = 40.0  # one row explaining >= this % of a breakdown's total magnitude
                             # counts as an isolated driver and stops the loop early.
NEGATIVE_INTENT_RE = re.compile(r"\bloss|declin|drop|fell|decrease|negative\b", re.IGNORECASE)

DRILL_SYSTEM_PROMPT = """You are the Autonomous Analyst's root-cause drill-down assistant. You \
investigate WHY a metric changed or differs across a dimension by running one small SQL \
breakdown query per step. After each step you are given a "finding" that is COMPUTED BY CODE \
from the actual result -- that finding is ground truth; never trust your own reading of a raw \
table over it.

You have at most {max_steps} steps. On EVERY step, respond with ONLY a single JSON object, \
nothing else, of the form:
{{"reason": "<one line: why this sub-query is the next thing to check>", "sql": "<a single DuckDB SELECT/WITH statement>"}}

SQL rules (same as the rest of the Analyst):
- DuckDB dialect. Use ONLY tables/columns in the semantic layer below. Never invent columns.
- Single statement, SELECT/WITH only. No semicolons, no DDL/DML.
- Break the metric down by ONE new dimension per step (e.g. category, city, month) to narrow in
  on what is driving it -- do not repeat a breakdown already listed in "Findings so far" below.
- GROUP BY the dimension and ORDER BY the metric so the extremes are easy to compute.
- Start broad (the metric in the question, broken down by whichever dimension it most directly
  references) and narrow further each step based on the findings so far.

Example (pattern only, adapt table/column/dimension names to the actual question -- this is not
one of the questions you will be asked):
Step 1 reason: "Average order value dropped -- break down by city to see where it's concentrated."
Step 1 SQL: SELECT o.city, SUM(oi.amount) / NULLIF(COUNT(DISTINCT o.order_id), 0) AS avg_order_value
            FROM orders o JOIN order_items oi ON o.order_id = oi.order_id GROUP BY o.city ORDER BY avg_order_value;

=== SEMANTIC LAYER ===
{semantic_layer}
=== END SEMANTIC LAYER ==="""

WHY_SYNTHESIS_SYSTEM_PROMPT = """You write the final root-cause conclusion for the Autonomous \
Analyst's drill-down. You are given the full drill path: each step's reason, SQL, and a \
code-computed "finding" (ground truth, not your own reading of a table). Summarize ONLY from \
these findings -- never use outside knowledge, never invent a cause, never invent a number. \
Currency values are in INR.

This system prompt contains NO example numbers or names anywhere below -- every number and
every label you write MUST be copied character-for-character out of the JSON payload the user
message gives you this turn. If you catch yourself writing a number or a category/city/month
label that you cannot point to inside that JSON, delete it and look again.

Rules:
- If a step's finding has a non-null "dominant_driver": that object has a nested "extreme_row"
  (a dict with the dimension's label and its numeric value) and an "extreme_share_pct". State
  the extreme_row's label, the extreme_row's number, and the extreme_share_pct -- pulled
  directly from that JSON object, nothing else.
- If no step's finding has a "dominant_driver" (it is null on every step), say so honestly:
  from the LAST successful step's finding, pick either "max_row"+"max_share_pct" or
  "min_row"+"min_share_pct" from ONE column in "per_column" (whichever direction matches what
  the question is asking about), state that label/number/percentage, and say plainly that no
  single dimension clearly dominates. Do NOT fabricate a clean single-driver story when the
  data doesn't show one.
- A single claim (one row's label + number + percentage) must ALL come from the SAME finding
  object of the SAME step. Never combine a label from one step's finding with a number or
  percentage from a different step's finding, and never combine a "max_row" label with a
  "min_row" number (or vice versa) within the same column.
- Every number you write must be a value that literally appears in the JSON (e.g. a
  "max_share_pct" or "min_share_pct" field) -- never compute, combine, or round a percentage
  yourself, even from real numbers that ARE in the JSON.
- Quote period/category/city labels exactly as they appear in the JSON, character for character
  (e.g. a label written "2018-07" must stay "2018-07", not become "July 2018").
- Match your wording to the actual sign of the number you cite, regardless of how the question
  was phrased: a positive value is a contributor/gain, never call it a "loss" or a "drop" just
  because the question asked about one. If the question asked for a loss/decline driver and
  every candidate value is positive, say plainly that nothing in the data shows a loss/decline,
  and describe the cited row as the largest (or smallest) contributor instead.
- Be concise: 2-4 sentences."""


def compute_step_finding(df: pd.DataFrame) -> dict:
    """Deterministically compute the salient facts from a drill-down step's result: for each
    numeric column, the row with the largest-magnitude value and its share of the column's
    total absolute magnitude across all rows. This is the "don't trust the 7B to eyeball which
    row moved most" step -- the model only ever sees these code-computed numbers, never the
    raw table itself.

    "dominant_driver" is set when a single row's magnitude accounts for >= DRIVER_DOMINANCE_PCT
    of a breakdown's total magnitude (only evaluated when there's more than one row -- a single
    aggregate row isn't a "driver" of anything).
    """
    if df.empty:
        return {"row_count": 0, "numeric_columns": [], "label_columns": [], "per_column": {}, "dominant_driver": None}

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    label_cols = [c for c in df.columns if c not in numeric_cols]

    per_column: dict = {}
    dominant_driver = None
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        total_abs = float(series.abs().sum())
        max_idx, min_idx = series.idxmax(), series.idxmin()
        max_row = json.loads(df.loc[max_idx].to_json())
        min_row = json.loads(df.loc[min_idx].to_json())
        # Every row-and-percentage pair the model might want to cite is pre-computed here as a
        # literal field, so the synthesis step never has to do its own arithmetic (which it has
        # been observed to get wrong) -- it only ever copies a *_share_pct value straight out.
        # A single-row result has nothing to share against, so leave the pct undefined rather
        # than reporting a trivially-true (and misleading) 100%.
        has_comparison = len(series) > 1
        max_share_pct = round(abs(float(series[max_idx])) / total_abs * 100, 1) if total_abs and has_comparison else None
        min_share_pct = round(abs(float(series[min_idx])) / total_abs * 100, 1) if total_abs and has_comparison else None
        extreme_kind = "max" if abs(series[max_idx]) >= abs(series[min_idx]) else "min"
        extreme_row = max_row if extreme_kind == "max" else min_row
        extreme_share_pct = max_share_pct if extreme_kind == "max" else min_share_pct

        stats = {
            "total": float(series.sum()),
            "max_row": max_row,
            "max_share_pct": max_share_pct,
            "min_row": min_row,
            "min_share_pct": min_share_pct,
            "extreme_row": extreme_row,
            "extreme_kind": extreme_kind,
            "extreme_share_pct": extreme_share_pct,
        }
        per_column[col] = stats
        share_pct = extreme_share_pct

        if dominant_driver is None and len(df) > 1 and share_pct is not None and share_pct >= DRIVER_DOMINANCE_PCT:
            dominant_driver = {"column": col, **stats}

    return {
        "row_count": len(df),
        "numeric_columns": numeric_cols,
        "label_columns": label_cols,
        "per_column": per_column,
        "dominant_driver": dominant_driver,
    }


def parse_drill_step(content: str | None) -> tuple[str | None, str | None]:
    """Parse a drill-down step's {"reason": ..., "sql": ...} JSON out of free-text model
    output, tolerating a fenced ```json block or a bare ```sql block with the reason as
    the preceding prose (mirrors extract_fallback_tool_call's tolerance for this model)."""
    text = (content or "").strip()
    if not text:
        return None, None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("sql"):
            return obj.get("reason"), obj["sql"]
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict) and obj.get("sql"):
                return obj.get("reason"), obj["sql"]
        except json.JSONDecodeError:
            pass

    sql_fence = SQL_FENCE_RE.search(text)
    if sql_fence:
        reason_text = text[: sql_fence.start()].strip() or None
        return reason_text, sql_fence.group(1).strip()

    return text or None, None


def build_drill_user_prompt(question: str, steps: list[dict], history: list[dict] | None = None) -> str:
    lines = []
    if not steps and history:
        lines.append("Recent conversation (context only -- the drill-down itself must stand on its own queries):")
        for turn in history[-4:]:
            lines.append(f"  {turn.get('role')}: {turn.get('content')}")
        lines.append("")
    lines.append(f"Root-cause question: {question}")
    lines.append("")
    if not steps:
        lines.append("This is step 1 of at most {}. Propose the first breakdown query.".format(MAX_WHY_STEPS))
    else:
        lines.append("Findings so far (computed by code from actual query results -- ground truth):")
        for s in steps:
            lines.append(f"- Step {s['step']}: reason={s['reason']!r}")
            lines.append(f"  sql: {s['sql']}")
            if s.get("error"):
                lines.append(f"  error: {s['error']}")
            elif s.get("finding") is not None:
                lines.append(f"  finding: {json.dumps(s['finding'], default=str)}")
        lines.append("")
        if steps[-1].get("error"):
            lines.append(
                f"IMPORTANT: step {steps[-1]['step']}'s SQL FAILED with the error shown above. Read "
                "the error message and fix the actual problem it names (e.g. add a missing JOIN, "
                "correct a column/table name) -- do NOT resubmit that same SQL unchanged."
            )
        lines.append(
            f"Propose step {len(steps) + 1} of at most {MAX_WHY_STEPS}, narrowing further based on "
            "the findings above. Do not repeat a breakdown already listed."
        )
    return "\n".join(lines)


def synthesize_why_conclusion(
    question: str,
    steps: list[dict],
    client: OpenAI,
    temperature: float | None = None,
    seed: int | None = None,
) -> str:
    """Write the final root-cause conclusion strictly from the drill steps' computed
    findings -- same anti-hallucination discipline as synthesize_final_answer()."""
    successful_steps = [s for s in steps if s.get("finding") is not None and not s.get("error")]
    if not successful_steps:
        last_error = next((s["error"] for s in reversed(steps) if s.get("error")), None)
        if last_error:
            return f"I couldn't complete a drill-down query, so I can't identify a driver. Last error: {last_error}"
        return "I couldn't complete a drill-down query, so I can't identify a driver."

    hit_step_limit = len(steps) >= MAX_WHY_STEPS and not any(
        s.get("finding", {}).get("dominant_driver") for s in successful_steps
    )
    payload = {
        "question": question,
        "steps": [
            {
                "step": s["step"],
                "reason": s["reason"],
                "sql": s["sql"],
                "row_count": s.get("row_count"),
                "finding": s.get("finding"),
            }
            for s in successful_steps
        ],
        "hit_step_limit_without_isolating_a_driver": hit_step_limit,
    }

    # Code-computed sign check: the model has repeatedly mislabeled a positive dominant_driver
    # as a "loss"/"decline" when the question used that wording, despite a system-prompt
    # instruction against it -- a deterministic check placed right next to the data it's
    # about is far more reliably followed than a rule buried in the system prompt.
    dominant = next(
        (s["finding"]["dominant_driver"] for s in successful_steps if s.get("finding", {}).get("dominant_driver")),
        None,
    )
    if dominant and NEGATIVE_INTENT_RE.search(question):
        value = dominant["extreme_row"].get(dominant["column"])
        if isinstance(value, (int, float)) and value > 0:
            payload["sign_check"] = (
                "The question asks about a loss/decline, but this dominant_driver's value is "
                "POSITIVE (a gain/contributor, not a loss). State this plainly -- do not call "
                "it a loss."
            )
    messages = [
        {"role": "system", "content": WHY_SYNTHESIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Drill-down data (JSON -- the ONLY source of truth for your answer):\n"
                f"{json.dumps(payload, default=str)}"
            ),
        },
    ]
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    response = client.chat.completions.create(model=LLM_MODEL, messages=messages, **kwargs)
    return response.choices[0].message.content or "(no conclusion produced)"


def answer_why(
    question: str,
    history: list[dict] | None = None,
    client: OpenAI | None = None,
    temperature: float | None = 0,
    seed: int | None = 0,
) -> dict:
    """Bounded root-cause drill-down over the same guardrailed run_sql tool used by
    answer_question(). At most MAX_WHY_STEPS sub-queries, each justified by a one-line reason;
    after every step a code-computed "finding" (never the model's own eyeballing) decides
    whether a dominant driver has been isolated. Stops early on isolation, else after the cap.

    Returns {"answer_text": str, "steps": list[dict], "sql": str | None, "rows": list[dict]}.
    `steps` is the inspectable reasoning trace: each entry has step, reason, sql, row_count,
    finding (or error).
    """
    if detect_prompt_injection(question):
        result = _refusal("refused: prompt-injection pattern detected in question")
        return {"answer_text": result["answer_text"], "steps": [], "sql": None, "rows": []}

    client = client or build_client()
    con = duckdb.connect(DB_PATH, read_only=True)
    run_sql = make_run_sql(con)

    steps: list[dict] = []
    last_sql: str | None = None
    last_df = pd.DataFrame()

    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed

    try:
        semantic_layer = load_semantic_layer_text()
        system_prompt = DRILL_SYSTEM_PROMPT.format(max_steps=MAX_WHY_STEPS, semantic_layer=semantic_layer)

        for step_num in range(1, MAX_WHY_STEPS + 1):
            user_prompt = build_drill_user_prompt(question, steps, history)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = client.chat.completions.create(model=LLM_MODEL, messages=messages, **kwargs)
            content = response.choices[0].message.content
            reason, sql = parse_drill_step(content)

            if not sql:
                steps.append(
                    {
                        "step": step_num,
                        "reason": reason or "(model gave no reason)",
                        "sql": None,
                        "error": "could not parse a SQL query from the model's response",
                        "finding": None,
                    }
                )
                continue  # let the model try again next step, same self-correct spirit as answer_question

            sql = unescape_sql_operators(sql)
            try:
                result = run_sql(sql)
            except (SQLGuardError, duckdb.Error) as exc:
                steps.append(
                    {"step": step_num, "reason": reason, "sql": sql, "error": str(exc), "finding": None}
                )
                continue  # self-correct: the error is fed back into the next step's prompt

            df = result["df"]
            last_sql, last_df = result["sql"], df
            finding = compute_step_finding(df)
            steps.append(
                {
                    "step": step_num,
                    "reason": reason or "(model gave no reason)",
                    "sql": result["sql"],
                    "row_count": len(df),
                    "finding": finding,
                }
            )
            if finding.get("dominant_driver"):
                break

        conclusion = synthesize_why_conclusion(question, steps, client, temperature=temperature, seed=seed)
        return {
            "answer_text": conclusion,
            "steps": steps,
            "sql": last_sql,
            "rows": last_df.to_dict(orient="records"),
        }
    finally:
        con.close()


def format_why_trace(steps: list[dict]) -> str:
    """Render the drill-down reasoning trace: step, reason, SQL, key finding -- the
    inspectable chain behind the final conclusion."""
    lines = []
    for s in steps:
        lines.append(f"Step {s['step']}: {s['reason']}")
        lines.append(f"  SQL: {s['sql']}")
        if s.get("error"):
            lines.append(f"  -> error: {s['error']}")
            continue
        finding = s.get("finding") or {}
        dominant = finding.get("dominant_driver")
        if dominant:
            lines.append(
                f"  -> finding: dominant driver in {dominant['column']!r} "
                f"({dominant['extreme_share_pct']}% of total magnitude): {dominant['extreme_row']}"
            )
        elif finding.get("row_count") == 1:
            per_column = finding.get("per_column", {})
            single = ", ".join(f"{col}={stats['max_row'].get(col)}" for col, stats in per_column.items())
            lines.append(f"  -> finding: single row ({single}) -- no other rows to compare a share against")
        else:
            per_column = finding.get("per_column", {})
            summary = ", ".join(
                f"{col}: extreme={stats['extreme_row']} ({stats['extreme_share_pct']}%)"
                for col, stats in per_column.items()
            )
            lines.append(f"  -> finding: {summary or '(no numeric columns to summarize)'}")
    return "\n".join(lines)


def _refusal(reason: str) -> dict:
    return {
        "answer_text": (
            "I can't follow instructions embedded in a question. Please ask a direct "
            "analytical question about the orders / order_items / sales_targets data."
        ),
        "sql": None,
        "rows": [],
        "trace": [reason],
    }


def answer_question(
    question: str,
    history: list[dict] | None = None,
    client: OpenAI | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> dict:
    """Answer one natural-language question over the warehouse.

    Returns {"answer_text": str, "sql": str | None, "rows": list[dict], "trace": list[str]}.
    `history` is a flat list of prior {"role": "user"|"assistant", "content": str} turns; pass
    None (default) for a stateless single-shot call, e.g. from an eval harness.
    `temperature` and `seed` are passed straight to the LLM call (e.g. temperature=0, seed=0
    for deterministic eval runs); None uses the provider default for each.
    """
    history = history or []

    if detect_prompt_injection(question):
        return _refusal("refused: prompt-injection pattern detected in question")

    client = client or build_client()
    con = duckdb.connect(DB_PATH, read_only=True)
    run_sql = make_run_sql(con)
    trace: list[str] = []

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(semantic_layer=load_semantic_layer_text())
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": question}]

    last_sql: str | None = None
    last_df = pd.DataFrame()
    sql_error_count = 0
    fallback_call_counter = 0
    has_called_tool = False
    nudge_used = False

    completion_kwargs: dict = {"extra_body": {"options": {"num_ctx": NUM_CTX}}}
    if temperature is not None:
        completion_kwargs["temperature"] = temperature
    if seed is not None:
        completion_kwargs["seed"] = seed

    try:
        for _ in range(MAX_TOOL_ITERATIONS):
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=[RUN_SQL_TOOL_SCHEMA],
                tool_choice="auto",
                **completion_kwargs,
            )
            msg = response.choices[0].message

            # Normal path: the model populated structured tool_calls.
            if msg.tool_calls:
                messages.append(msg.model_dump(exclude_none=True))
                tool_call_items = [
                    (tc.id, tc.function.arguments) for tc in msg.tool_calls
                ]
            else:
                # Fallback path: this model doesn't reliably emit structured
                # tool_calls -- try to recover a run_sql call from free text.
                fallback_query = extract_fallback_tool_call(msg.content)
                if fallback_query is None:
                    if not has_called_tool and not nudge_used:
                        nudge_used = True
                        trace.append("nudge: forcing a run_sql call before accepting a toolless answer")
                        messages.append({"role": "assistant", "content": msg.content or ""})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You answered without calling run_sql this turn. Call the "
                                    "run_sql tool now to verify the numbers before giving a "
                                    "final answer."
                                ),
                            }
                        )
                        continue
                    if has_called_tool:
                        answer_text = synthesize_final_answer(
                            question, last_df, client, temperature=temperature, seed=seed
                        )
                        trace.append("synthesis: generated final answer from returned table")
                    else:
                        answer_text = msg.content or "(no answer produced)"
                    return {
                        "answer_text": answer_text,
                        "sql": last_sql,
                        "rows": last_df.to_dict(orient="records"),
                        "trace": trace,
                    }
                fallback_call_counter += 1
                synthetic_id = f"fallback-{fallback_call_counter}"
                synthetic_args = json.dumps({"query": fallback_query})
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": synthetic_id,
                                "type": "function",
                                "function": {"name": "run_sql", "arguments": synthetic_args},
                            }
                        ],
                    }
                )
                trace.append("fallback_extraction: recovered run_sql call from free text")
                tool_call_items = [(synthetic_id, synthetic_args)]

            has_called_tool = True
            for tool_call_id, arguments in tool_call_items:
                query = ""
                try:
                    args = json.loads(arguments or "{}")
                    query = args.get("query", "")
                    trace.append(f"tool_call: run_sql({query!r})")
                    result = run_sql(query)
                    last_sql = result["sql"]
                    last_df = result["df"]
                    preview = last_df.head(MAX_TOOL_ROWS_TO_MODEL).to_dict(orient="records")
                    tool_content = json.dumps(
                        {
                            "sql": last_sql,
                            "row_count": len(last_df),
                            "rows": preview,
                            "truncated": len(last_df) > MAX_TOOL_ROWS_TO_MODEL,
                        },
                        default=str,
                    )
                except (SQLGuardError, duckdb.Error, json.JSONDecodeError) as exc:
                    sql_error_count += 1
                    trace.append(f"sql_error ({sql_error_count}/{MAX_SQL_RETRIES}): {exc}")
                    if sql_error_count >= MAX_SQL_RETRIES:
                        return {
                            "answer_text": (
                                f"I couldn't produce a valid query after {MAX_SQL_RETRIES} "
                                f"attempts. Last error: {exc}"
                            ),
                            "sql": last_sql,
                            "rows": [],
                            "trace": trace,
                        }
                    tool_content = json.dumps({"error": str(exc)})

                messages.append(
                    {"role": "tool", "tool_call_id": tool_call_id, "content": tool_content}
                )

        return {
            "answer_text": "I wasn't able to reach a final answer within the step limit.",
            "sql": last_sql,
            "rows": last_df.to_dict(orient="records"),
            "trace": trace,
        }
    finally:
        con.close()


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    return pd.DataFrame(rows).to_string(index=False)


def repl() -> None:
    print("Autonomous Analyst -- ask a question about orders / order_items / sales_targets.")
    print("Type 'exit' or 'quit' to leave.\n")
    client = build_client()
    history: list[dict] = []

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        if is_why_question(question):
            result = answer_why(question, history=history, client=client)
            if result["steps"]:
                print("\nREASONING TRACE:")
                print(format_why_trace(result["steps"]))
            print(f"\n{result['answer_text']}\n")
        else:
            result = answer_question(question, history=history, client=client)

            print(f"\n{result['answer_text']}\n")
            if result["sql"]:
                print("SQL:")
                print(result["sql"])
                print()
            if result["rows"]:
                print(format_table(result["rows"]))
                print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer_text"]})


if __name__ == "__main__":
    repl()
