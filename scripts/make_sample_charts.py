"""Render the sample charts shown in the README from the demo dataset.

Runs the agent's own tools (no LLM required) and writes deterministic file names to
docs/images/charts/. Usage:

    python scripts/make_sample_charts.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.analysis import coerce_types  # noqa: E402
from backend.app.charts import ChartRenderer  # noqa: E402
from backend.app.demo import build_demo_dataframe  # noqa: E402
from backend.app.tools import ToolContext, build_tools  # noqa: E402

OUT = ROOT / "docs" / "images" / "charts"


def main() -> None:
    tmp = ROOT / "artifacts" / "_readme_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    OUT.mkdir(parents=True, exist_ok=True)

    df = coerce_types(build_demo_dataframe())
    ctx = ToolContext(df=df, dataset_name="demo_ecommerce_orders.csv", renderer=ChartRenderer(tmp))
    tools = {t.name: t for t in build_tools(ctx)}

    plan = [
        ("histogram-revenue", "plot_chart", {"kind": "histogram", "x": "revenue"}),
        ("correlation-heatmap", "correlation_analysis", {}),
        (
            "bar-revenue-by-category",
            "group_aggregate",
            {"group_by": "category", "metric": "revenue", "aggregation": "sum"},
        ),
        (
            "scatter-price-vs-revenue",
            "plot_chart",
            {"kind": "scatter", "x": "unit_price", "y": "revenue", "hue": "category"},
        ),
        (
            "line-monthly-revenue",
            "time_series_trend",
            {"date_column": "order_date", "metric": "revenue", "freq": "M"},
        ),
        (
            "box-satisfaction-by-segment",
            "plot_chart",
            {"kind": "box", "x": "satisfaction_score", "hue": "customer_segment"},
        ),
        ("model-revenue", "train_baseline_model", {"target": "revenue"}),
        ("model-returned", "train_baseline_model", {"target": "returned"}),
    ]

    for name, tool, args in plan:
        before = len(ctx.charts)
        tools[tool].invoke(args)
        produced = ctx.charts[before:]
        for chart in produced:
            suffix = "" if len(produced) == 1 else f"-{chart.kind}"
            dest = OUT / f"{name}{suffix}.png"
            shutil.copyfile(chart.path, dest)
            print(f"✓ {dest.relative_to(ROOT)}")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
