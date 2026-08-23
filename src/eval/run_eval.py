"""Execution-accuracy eval harness for the Autonomous Analyst agent.

For every question in questions.yaml: run the agent single-shot (no conversation
history, temperature=0, seed=0), run the gold SQL directly against the warehouse, and
compare the two result sets. Also does a light heuristic check for "SQL passed
but the natural-language answer doesn't actually mention the key number/name" --
a synthesis problem distinct from a SQL correctness problem.

Usage: python src/eval/run_eval.py (or: python -m src.eval.run_eval from repo root)

Every run overwrites report.json, which is gitignored (a local scratch artifact of whatever
provider you last ran against -- not committed, not authoritative). The two checked-in,
provider-labelled snapshots are report_ollama.json (qwen2.5-coder:7b, local, 14/15) and
report_groq.json (openai/gpt-oss-120b via Groq, 13/15) -- copy report.json to the matching
name after a run if you want to update one of those.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import DB_PATH, answer_question, build_client  # noqa: E402

QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"
REPORT_PATH = Path(__file__).parent / "report.json"

MAX_DIFF_ROWS_SHOWN = 10
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def load_questions() -> list[dict]:
    data = yaml.safe_load(QUESTIONS_PATH.read_text())
    questions = []
    for difficulty in ("easy", "medium", "hard"):
        for item in data.get(difficulty, []):
            questions.append({"difficulty": difficulty, "q": item["q"], "sql": item["sql"]})
    return questions


def normalize_scalar(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value).strip()


def df_to_row_set(df: pd.DataFrame) -> set[tuple]:
    rows = set()
    for _, row in df.iterrows():
        values = [normalize_scalar(v) for v in row.tolist()]
        # ignore column order: canonicalize each row by sorting its normalized values
        values.sort(key=lambda v: (v is None, isinstance(v, str), str(v)))
        rows.add(tuple(values))
    return rows


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for match in NUMBER_RE.findall(text or ""):
        cleaned = match.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def number_mentioned(value: float, text: str) -> bool:
    tol = max(0.5, abs(value) * 0.01)
    return any(abs(n - value) <= tol for n in extract_numbers(text))


def name_mentioned(value: str, text: str) -> bool:
    value = value.strip()
    if not value:
        return False
    return value.lower() in (text or "").lower()


def pick_key_value(gold_df: pd.DataFrame):
    """Pick one representative "key" cell from the gold result to check the
    answer_text against: prefer a string (name) over a number, from row 0."""
    if gold_df.empty:
        return None
    row = gold_df.iloc[0]
    for value in row.tolist():
        if isinstance(value, str):
            return ("name", value)
    for value in row.tolist():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not pd.isna(value):
            return ("number", float(value))
    return None


def run() -> dict:
    questions = load_questions()
    gold_con = duckdb.connect(DB_PATH, read_only=True)
    client = build_client()

    results = []
    for item in questions:
        difficulty, q, gold_sql = item["difficulty"], item["q"], item["sql"]

        gold_df = gold_con.execute(gold_sql).fetchdf()
        gold_set = df_to_row_set(gold_df)

        agent_result = answer_question(q, history=None, client=client, temperature=0, seed=0)
        agent_df = pd.DataFrame(agent_result["rows"])
        agent_set = df_to_row_set(agent_df)

        execution_pass = gold_set == agent_set

        key = pick_key_value(gold_df)
        key_mentioned = None
        synthesis_suspect = False
        if key is not None:
            kind, value = key
            key_mentioned = (
                name_mentioned(value, agent_result["answer_text"])
                if kind == "name"
                else number_mentioned(value, agent_result["answer_text"])
            )
            synthesis_suspect = execution_pass and not key_mentioned

        gold_only = sorted(gold_set - agent_set, key=str)[:MAX_DIFF_ROWS_SHOWN]
        agent_only = sorted(agent_set - gold_set, key=str)[:MAX_DIFF_ROWS_SHOWN]

        results.append(
            {
                "difficulty": difficulty,
                "question": q,
                "gold_sql": gold_sql.strip(),
                "agent_sql": agent_result["sql"],
                "execution_pass": execution_pass,
                "gold_row_count": len(gold_df),
                "agent_row_count": len(agent_df),
                "gold_only_rows": [list(r) for r in gold_only],
                "agent_only_rows": [list(r) for r in agent_only],
                "answer_text": agent_result["answer_text"],
                "key_value": key[1] if key else None,
                "key_value_mentioned": key_mentioned,
                "synthesis_suspect": synthesis_suspect,
                "trace": agent_result["trace"],
            }
        )

    gold_con.close()

    total = len(results)
    passed = sum(r["execution_pass"] for r in results)
    by_difficulty = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"] == difficulty]
        subset_passed = sum(r["execution_pass"] for r in subset)
        by_difficulty[difficulty] = {
            "passed": subset_passed,
            "total": len(subset),
            "pct": round(100 * subset_passed / len(subset), 1) if subset else None,
        }

    suspects = [r for r in results if r["synthesis_suspect"]]

    return {
        "overall": {"passed": passed, "total": total, "pct": round(100 * passed / total, 1)},
        "by_difficulty": by_difficulty,
        "synthesis_suspect_count": len(suspects),
        "results": results,
    }


def print_report(report: dict) -> None:
    print("=" * 78)
    for r in report["results"]:
        status = "PASS" if r["execution_pass"] else "FAIL"
        print(f"[{r['difficulty']:6}] {status}  {r['question']}")
        print(f"  agent_sql: {r['agent_sql']}")
        if not r["execution_pass"]:
            print(f"  gold_only_rows ({r['gold_row_count']} gold rows total): {r['gold_only_rows']}")
            print(f"  agent_only_rows ({r['agent_row_count']} agent rows total): {r['agent_only_rows']}")
        print(f"  answer_text: {r['answer_text']!r}")
        if r["synthesis_suspect"]:
            print(f"  ⚠ SYNTHESIS SUSPECT: key value {r['key_value']!r} not found in answer_text")
        print("-" * 78)

    print()
    print(f"OVERALL ACCURACY: {report['overall']['passed']}/{report['overall']['total']} "
          f"({report['overall']['pct']}%)")
    print()
    print("Per-difficulty breakdown:")
    diff_df = pd.DataFrame(
        [
            {"difficulty": d, "passed": v["passed"], "total": v["total"], "pct": v["pct"]}
            for d, v in report["by_difficulty"].items()
        ]
    )
    print(diff_df.to_string(index=False))
    print()

    suspects = [r for r in report["results"] if r["synthesis_suspect"]]
    print(f"SQL-pass-but-synthesis-suspect: {len(suspects)}")
    for r in suspects:
        print(f"  - [{r['difficulty']}] {r['question']!r}: expected {r['key_value']!r} "
              f"not mentioned in: {r['answer_text']!r}")


def main() -> None:
    report = run()
    print_report(report)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved machine-readable report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
