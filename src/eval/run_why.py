"""Manual-inspection runner for answer_why() (Phase 7, part 1: root-cause drill-down).

Not an execution-accuracy eval like run_eval.py -- these are open-ended "why" questions
with no single gold SQL to diff against. This just runs each question in why_questions.yaml
through the bounded drill-down loop, temperature=0 seed=0, and prints the full reasoning
trace (step, reason, SQL, code-computed finding) plus the final conclusion, so every claim
in the conclusion can be checked against a real query result by eye.

Usage: python src/eval/run_why.py (or: python -m src.eval.run_why from repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import answer_why, build_client, format_why_trace  # noqa: E402

QUESTIONS_PATH = Path(__file__).parent / "why_questions.yaml"


def main() -> None:
    data = yaml.safe_load(QUESTIONS_PATH.read_text())
    client = build_client()

    for item in data.get("questions", []):
        q = item["q"]
        print("=" * 78)
        print(f"Q: {q}")
        print("-" * 78)
        result = answer_why(q, history=None, client=client, temperature=0, seed=0)
        print("REASONING TRACE:")
        print(format_why_trace(result["steps"]))
        print()
        print("CONCLUSION:")
        print(result["answer_text"])
        print()


if __name__ == "__main__":
    main()
