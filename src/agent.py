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


def build_client() -> OpenAI:
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


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
                    return {
                        "answer_text": msg.content or "(no answer produced)",
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
