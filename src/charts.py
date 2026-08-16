"""Phase 7 part 2: chart rendering, a presentation layer on top of already-computed results.

Does not run SQL, does not call the LLM, and is not imported by anything in the reasoning path
(answer_question, answer_why, or the eval harnesses) -- it is only ever called AFTER a result
table already exists, from the REPL. A chart is built strictly from the rows already returned;
if the shape doesn't fit a simple bar/line chart, no chart is produced rather than forcing one.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd

from agent import PERIOD_RE

CHARTS_DIR = os.getenv("CHARTS_DIR", "charts")
MAX_CHART_ROWS = 30
MIN_CHART_ROWS = 2


def build_chart_spec(rows: list[dict]) -> dict | None:
    """Inspect already-returned rows and decide whether a simple chart makes sense.

    Only handles the one shape a bar/line chart can honestly represent: exactly one *varying*
    label (non-numeric) column and one numeric column. A label column that's constant across
    every row (e.g. a month/category already pinned by a WHERE filter, echoed in every row) is
    dropped before that check -- it's not a real dimension to plot, just fixed context. Anything
    that still doesn't reduce to one label + one metric -- no rows, a single row/value, more than
    ~30 rows, multiple genuinely varying dimensions, multiple metrics -- returns None rather than
    forcing a misleading chart. Returns a plain dict describing what to plot; rendering is a
    separate step so the decision can be unit-tested without touching matplotlib.
    """
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if not (MIN_CHART_ROWS <= len(df) <= MAX_CHART_ROWS):
        return None

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    label_cols = [c for c in df.columns if c not in numeric_cols]
    varying_label_cols = [c for c in label_cols if df[c].nunique(dropna=False) > 1]
    if len(numeric_cols) != 1 or len(varying_label_cols) != 1:
        return None  # nothing sensible to plot -- e.g. a multi-dimension breakdown
    label_cols = varying_label_cols

    label_col, metric_col = label_cols[0], numeric_cols[0]
    plot_df = df[[label_col, metric_col]].dropna(subset=[metric_col])
    if len(plot_df) < MIN_CHART_ROWS:
        return None

    is_time_series = plot_df[label_col].astype(str).map(lambda v: bool(PERIOD_RE.match(v))).all()
    if is_time_series:
        plot_df = plot_df.sort_values(label_col)
        chart_type = "line"
    else:
        chart_type = "bar"

    return {
        "type": chart_type,
        "x_label": label_col,
        "y_label": metric_col,
        "x_values": [str(v) for v in plot_df[label_col].tolist()],
        "y_values": [float(v) for v in plot_df[metric_col].tolist()],
        "title": f"{metric_col} by {label_col}",
    }


def render_chart(spec: dict, out_path: str) -> str | None:
    """Render a chart_spec to out_path with matplotlib (Agg backend, headless). Any failure
    here (missing display backend, bad path, matplotlib not installed, ...) is caught and
    logged -- charting is decoration, never allowed to break a text answer."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        if spec["type"] == "line":
            ax.plot(spec["x_values"], spec["y_values"], marker="o")
        else:
            ax.bar(spec["x_values"], spec["y_values"])
        ax.set_xlabel(spec["x_label"])
        ax.set_ylabel(spec["y_label"])
        ax.set_title(spec["title"])
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except Exception as exc:  # noqa: BLE001 - charting must never break the answer
        print(f"[charts] chart rendering failed, continuing text-only: {exc}", file=sys.stderr)
        return None


def maybe_render_chart(rows: list[dict], out_dir: str = CHARTS_DIR) -> str | None:
    """End-to-end: build a chart_spec from already-returned rows and render it, or return None
    if nothing sensible can be charted or rendering fails. Defensive by construction -- never
    raises, so a caller can call this unconditionally after any result without extra try/except."""
    try:
        spec = build_chart_spec(rows)
        if spec is None:
            return None
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        out_path = os.path.join(out_dir, f"{timestamp}.png")
        return render_chart(spec, out_path)
    except Exception as exc:  # noqa: BLE001 - charting must never break the answer
        print(f"[charts] chart build failed, continuing text-only: {exc}", file=sys.stderr)
        return None


def pick_key_why_step(steps: list[dict]) -> dict | None:
    """Pick the drill-down step whose breakdown the final conclusion is actually based on, so
    answer_why()'s chart matches what the text says rather than just whatever ran last:
    - a step with an isolated "dominant_driver" (the conclusion leads with it), else
    - a step with a "time_series" swing (the conclusion leads with that instead), else
    - the last successful (>= 2 row) step, as a fallback.
    Mirrors the same precedence synthesize_why_conclusion() uses when writing the text."""
    for s in steps:
        if (s.get("finding") or {}).get("dominant_driver"):
            return s
    for s in steps:
        if (s.get("finding") or {}).get("time_series"):
            return s
    for s in reversed(steps):
        if s.get("finding") is not None and not s.get("error"):
            return s
    return None
