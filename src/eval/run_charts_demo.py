"""Manual-inspection demo for Phase 7 part 2 (charts): runs three scenarios chosen to exercise
each chart_spec branch -- a plain dimension+metric bar, a time-series line, and answer_why()'s
key-step charting -- and prints the saved PNG paths plus the exact rows each chart was built
from, so the output can be checked against the chart image by eye.

Not an eval (no pass/fail, no gold data) -- charts.py is a presentation layer over already-
computed results and doesn't touch agent reasoning, guardrails, or eval logic.

Usage: python src/eval/run_charts_demo.py (or: python -m src.eval.run_charts_demo from repo root)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import answer_question, answer_why, build_client  # noqa: E402
from charts import maybe_render_chart, pick_key_why_step  # noqa: E402


def run_single_shot(label: str, question: str, client) -> None:
    print("=" * 78)
    print(f"[{label}] Q: {question}")
    result = answer_question(question, history=None, client=client, temperature=0, seed=0)
    print(f"answer_text: {result['answer_text']}")
    print(f"rows ({len(result['rows'])}): {json.dumps(result['rows'], default=str)}")
    chart_path = maybe_render_chart(result["rows"])
    print(f"chart: {chart_path}")
    print()


def run_why(label: str, question: str, client) -> None:
    print("=" * 78)
    print(f"[{label}] Q: {question}")
    result = answer_why(question, history=None, client=client, temperature=0, seed=0)
    print(f"answer_text: {result['answer_text']}")
    key_step = pick_key_why_step(result["steps"])
    if key_step is None:
        print("chart: None (no successful step to chart)")
        print()
        return
    print(f"key step: {key_step['step']} ({key_step['reason']!r})")
    print(f"key step rows ({len(key_step['rows'])}): {json.dumps(key_step['rows'], default=str)}")
    chart_path = maybe_render_chart(key_step["rows"])
    print(f"chart: {chart_path}")
    print()


def main() -> None:
    client = build_client()
    run_single_shot("bar expected", "What is total revenue by state?", client)
    run_single_shot("line expected", "What is the monthly revenue trend over time?", client)
    run_why("why-question, driver-step expected", "Which month drove the biggest revenue change and why?", client)


if __name__ == "__main__":
    main()
