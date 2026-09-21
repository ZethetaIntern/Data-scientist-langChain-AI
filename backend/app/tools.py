"""LangChain tools that give the agent hands-on access to the active dataset.

The tools close over a :class:`ToolContext` so each request gets tools bound to the
dataset being analysed. Every tool returns *text* (Markdown) that the LLM can reason
about; charts are rendered to PNG and recorded on the context so the API can return
them alongside the answer.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import BaseTool, tool

from . import analysis as an
from .charts import ChartArtifact, ChartRenderer
from .modeling import ModelingError, train_baseline


@dataclass
class ToolContext:
    df: pd.DataFrame
    dataset_name: str
    renderer: ChartRenderer
    charts: list[ChartArtifact] = field(default_factory=list)
    enable_code_tool: bool = False

    def add_chart(self, chart: ChartArtifact) -> str:
        self.charts.append(chart)
        return f"\n\n![{chart.title}]({chart.url})"


def _split_cols(value: str | None) -> list[str]:
    if not value:
        return []
    return [c.strip() for c in str(value).replace(";", ",").split(",") if c.strip()]


def _resolve(ctx: ToolContext, name: str | None, *, required: bool = True) -> str | None:
    col = an.find_column(ctx.df, name)
    if col is None and required:
        raise ValueError(
            f"Column '{name}' not found. Available columns: {', '.join(ctx.df.columns)}"
        )
    return col


def build_tools(ctx: ToolContext) -> list[BaseTool]:
    df = ctx.df

    @tool
    def dataset_overview() -> str:
        """Get the shape, column types, missing values and a preview of the dataset. Call this first when you know nothing about the data."""
        p = an.profile_dataframe(df)
        cols = pd.DataFrame(p["column_profiles"])[
            ["name", "kind", "dtype", "missing_pct", "unique", "sample"]
        ]
        cols["sample"] = cols["sample"].apply(lambda v: ", ".join(map(str, v)))
        lines = [
            f"**Dataset:** {ctx.dataset_name}",
            f"- Rows: {p['rows']:,}  ·  Columns: {p['columns']}  ·  Memory: {p['memory_mb']} MB",
            f"- Completeness: {p['completeness_pct']}%  ·  Duplicate rows: {p['duplicate_rows']:,}  ·  Quality score: {p['quality_score']}/100",
            f"- Numeric: {', '.join(p['numeric_columns']) or '—'}",
            f"- Categorical: {', '.join(p['categorical_columns']) or '—'}",
            f"- Datetime: {', '.join(p['datetime_columns']) or '—'}",
            "",
            "**Columns**",
            an.frame_to_markdown(cols, max_rows=60),
            "",
            "**First 5 rows**",
            an.frame_to_markdown(df.head(5)),
        ]
        return "\n".join(lines)

    @tool
    def describe_columns(columns: str = "") -> str:
        """Summary statistics (mean, std, quartiles, skew for numeric; unique/top for categorical). `columns` is an optional comma-separated list; empty means all columns."""
        wanted = [_resolve(ctx, c) for c in _split_cols(columns)] or None
        return an.frame_to_markdown(an.describe(df, wanted), max_rows=80)

    @tool
    def missing_values_report() -> str:
        """List columns that contain missing values with counts and percentages, plus a recommendation."""
        rep = an.missing_report(df)
        if rep.empty:
            return "No missing values in any column."
        advice = []
        for _, r in rep.iterrows():
            pct = r["missing_pct"]
            if pct > 40:
                advice.append(
                    f"- `{r['column']}`: {pct}% missing → consider dropping or flagging the column"
                )
            elif pct > 5:
                advice.append(
                    f"- `{r['column']}`: {pct}% missing → impute (median/mode) and add a missing indicator"
                )
            else:
                advice.append(
                    f"- `{r['column']}`: {pct}% missing → simple imputation or row removal is fine"
                )
        return an.frame_to_markdown(rep) + "\n\n**Suggested handling**\n" + "\n".join(advice)

    @tool
    def value_counts(column: str, top_n: int = 10) -> str:
        """Frequency table of the most common values in a (categorical) column, with a bar chart."""
        col = _resolve(ctx, column)
        vc = df[col].value_counts(dropna=True).head(int(top_n))
        frame = pd.DataFrame(
            {
                col: vc.index.astype(str),
                "count": vc.values,
                "share_pct": (100 * vc.values / len(df)).round(2),
            }
        )
        text = an.frame_to_markdown(frame)
        if len(frame) > 1:
            text += ctx.add_chart(
                ctx.renderer.bar(frame, col, "count", f"Most common values of {col}")
            )
        return text

    @tool
    def correlation_analysis(method: str = "pearson", top_n: int = 10) -> str:
        """Correlation matrix of numeric columns (pearson or spearman) with the strongest pairs and a heatmap chart."""
        method = method if method in ("pearson", "spearman", "kendall") else "pearson"
        corr, pairs = an.correlation_pairs(df, method=method, top_n=int(top_n))
        if corr.empty:
            return "Fewer than two numeric columns with variance — correlation is not defined."
        text = f"**Strongest {method} correlations**\n" + an.frame_to_markdown(pairs)
        text += ctx.add_chart(ctx.renderer.heatmap(corr, f"{method.title()} correlation matrix"))
        return text

    @tool
    def detect_outliers(columns: str = "") -> str:
        """Count outliers per numeric column using the 1.5×IQR rule. `columns` optional comma-separated list."""
        wanted = [_resolve(ctx, c) for c in _split_cols(columns)] or None
        rep = an.iqr_outliers(df, wanted)
        if rep.empty:
            return "No numeric columns to check."
        text = an.frame_to_markdown(rep)
        worst = rep.iloc[0]
        if worst["outliers"] > 0:
            text += ctx.add_chart(ctx.renderer.box(df, worst["column"]))
        return text

    @tool
    def group_aggregate(group_by: str, metric: str, aggregation: str = "mean") -> str:
        """Aggregate a numeric metric per category (e.g. mean revenue by region). aggregation: mean|sum|count|median|min|max. Returns a table and a bar chart."""
        g = _resolve(ctx, group_by)
        m = _resolve(ctx, metric)
        frame = an.group_aggregate(df, g, m, aggregation)
        value_col = [c for c in frame.columns if c not in (g, "n")][0]
        text = an.frame_to_markdown(frame)
        text += ctx.add_chart(ctx.renderer.bar(frame, g, value_col, f"{aggregation} of {m} by {g}"))
        return text

    @tool
    def plot_chart(kind: str, x: str, y: str = "", hue: str = "") -> str:
        """Render a chart. kind: histogram|scatter|box|bar|line. x = main column, y = second column (scatter/line/bar), hue = optional grouping column (scatter/box)."""
        kind = kind.lower().strip()
        xc = _resolve(ctx, x)
        yc = _resolve(ctx, y, required=False) if y else None
        hc = _resolve(ctx, hue, required=False) if hue else None
        if kind in ("histogram", "hist", "distribution"):
            chart = ctx.renderer.histogram(df[xc], xc)
            s = pd.to_numeric(df[xc], errors="coerce").dropna()
            skew = s.skew()
            shape = (
                "right-skewed (long tail of high values)"
                if skew > 0.5
                else "left-skewed (long tail of low values)"
                if skew < -0.5
                else "roughly symmetric"
            )
            summary = (
                f"**{xc}** — {len(s):,} values, {shape}\n"
                f"- Mean {s.mean():,.3f} · Median {s.median():,.3f} · Std {s.std():,.3f}\n"
                f"- Range [{s.min():,.3f}, {s.max():,.3f}] · IQR [{s.quantile(0.25):,.3f}, {s.quantile(0.75):,.3f}] · Skew {skew:,.3f}"
            )
        elif kind == "scatter":
            if not yc:
                raise ValueError("scatter needs both x and y")
            chart = ctx.renderer.scatter(df, xc, yc, hc)
            r = pd.to_numeric(df[xc], errors="coerce").corr(pd.to_numeric(df[yc], errors="coerce"))
            summary = f"Pearson r({xc}, {yc}) = {r:.3f}"
        elif kind == "box":
            chart = ctx.renderer.box(df, xc, hc or yc)
            summary = f"Box plot of {xc}" + (f" grouped by {hc or yc}" if (hc or yc) else "")
        elif kind == "bar":
            if yc:
                frame = an.group_aggregate(df, xc, yc, "mean")
                chart = ctx.renderer.bar(frame, xc, f"mean_{yc}", f"mean {yc} by {xc}")
                summary = an.frame_to_markdown(frame)
            else:
                vc = df[xc].value_counts().head(15)
                frame = pd.DataFrame({xc: vc.index.astype(str), "count": vc.values})
                chart = ctx.renderer.bar(frame, xc, "count", f"Count by {xc}")
                summary = an.frame_to_markdown(frame)
        elif kind == "line":
            if not yc:
                raise ValueError("line needs x (date column) and y (metric)")
            series, label = an.time_series(df, xc, yc)
            chart = ctx.renderer.line(series, "date", series.columns[1], f"{yc} per {label}")
            table = series.tail(12).copy()
            table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
            summary = an.frame_to_markdown(table)
        else:
            raise ValueError("kind must be one of histogram, scatter, box, bar, line")
        return summary + ctx.add_chart(chart)

    @tool
    def time_series_trend(
        date_column: str, metric: str, aggregation: str = "sum", freq: str = "auto"
    ) -> str:
        """Aggregate a metric over time and report growth, peak and trough with a line chart. aggregation: sum|mean|count|median. freq: auto|D|W|M|Y (day/week/month/year buckets)."""
        dc = _resolve(ctx, date_column)
        mc = _resolve(ctx, metric)
        freq_map = {
            "auto": "auto",
            "d": "D",
            "day": "D",
            "daily": "D",
            "w": "W",
            "week": "W",
            "weekly": "W",
            "m": "MS",
            "ms": "MS",
            "month": "MS",
            "monthly": "MS",
            "y": "YS",
            "ys": "YS",
            "year": "YS",
            "yearly": "YS",
        }
        series, label = an.time_series(
            df, dc, mc, freq=freq_map.get(str(freq).lower(), "auto"), aggregation=aggregation
        )
        vcol = series.columns[1]
        if series[vcol].dropna().empty:
            return "No data points after aggregation."
        t = an.trend_summary(series, vcol)
        direction = (
            "upward"
            if t["slope_per_period"] > 0
            else "downward"
            if t["slope_per_period"] < 0
            else "flat"
        )
        text = (
            f"**{aggregation} of {mc} per {label}** — {t['periods']} periods, overall {direction} trend\n"
            f"- First half average: {t['first_half_mean']:,.2f} → second half average: {t['second_half_mean']:,.2f}"
            + (f" ({t['change_pct']:+.1f}%)\n" if t["change_pct"] is not None else "\n")
            + f"- Peak: {t['peak_value']:,.2f} ({pd.Timestamp(t['peak_date']).date()})  ·  Trough: {t['trough_value']:,.2f} ({pd.Timestamp(t['trough_date']).date()})\n"
            f"- Mean per {label}: {t['mean']:,.2f}  ·  Std: {t['std']:,.2f}  ·  Slope: {t['slope_per_period']:+,.2f} per {label}\n"
            f"- Note: the first and last {label} may be partial periods.\n\n"
        )
        table = series.tail(12).copy()
        table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
        text += an.frame_to_markdown(table)
        text += ctx.add_chart(
            ctx.renderer.line(series, "date", vcol, f"{aggregation} of {mc} per {label}")
        )
        return text

    @tool
    def train_baseline_model(target: str, task: str = "auto", features: str = "") -> str:
        """Train a random-forest baseline to predict `target` (task: auto|regression|classification) on a held-out split. Reports metrics vs a naive baseline and permutation feature importance with charts."""
        tc = _resolve(ctx, target)
        feats = [_resolve(ctx, f) for f in _split_cols(features)] or None
        try:
            res = train_baseline(df, tc, task=task, features=feats)
        except ModelingError as exc:
            return f"Could not train a model: {exc}"
        metric_rows = [
            {"metric": k, "random_forest": v, "naive_baseline": res.baseline_metrics.get(k)}
            for k, v in res.metrics.items()
        ]
        best = res.importance.iloc[0]
        text = (
            f"**{res.task.title()} baseline for `{res.target}`** — random forest trained on {res.n_train:,} rows, evaluated on {res.n_test:,} held-out rows.\n\n"
            f"- Features used ({len(res.features)}): {', '.join(res.features)}\n"
            f"- Most influential feature: `{best['feature']}` (permutation importance {best['importance']:.3f})\n\n"
            + an.frame_to_markdown(pd.DataFrame(metric_rows))
            + "\n\n**Permutation importance (held-out, top 8)**\n"
            + an.frame_to_markdown(res.importance.head(8))
        )
        if res.notes:
            text += "\n\nNotes: " + "; ".join(res.notes)
        text += ctx.add_chart(
            ctx.renderer.importance(res.importance, f"Feature importance for {res.target}")
        )
        if res.task == "regression" and res.y_test is not None:
            text += ctx.add_chart(
                ctx.renderer.actual_vs_predicted(
                    res.y_test, res.y_pred, f"Actual vs predicted {res.target}"
                )
            )
        elif res.confusion is not None:
            text += ctx.add_chart(
                ctx.renderer.confusion(
                    res.confusion, res.classes, f"Confusion matrix for {res.target}"
                )
            )
        text += "\n\n_Caveat: a strong hold-out score does not prove causality or future performance; check for leakage and representativeness._"
        return text

    tools: list[BaseTool] = [
        dataset_overview,
        describe_columns,
        missing_values_report,
        value_counts,
        correlation_analysis,
        detect_outliers,
        group_aggregate,
        plot_chart,
        time_series_trend,
        train_baseline_model,
    ]

    if ctx.enable_code_tool:

        @tool
        def run_pandas_expression(expression: str) -> str:
            """Evaluate ONE restricted pandas expression against the dataframe `df` (pandas as `pd`, numpy as `np`). No imports, assignments, attribute names starting with '_' or file access. Example: df.groupby('region')['revenue'].sum().sort_values()"""
            result = safe_eval_pandas(expression, df)
            if isinstance(result, pd.DataFrame):
                return an.frame_to_markdown(
                    result.reset_index() if not isinstance(result.index, pd.RangeIndex) else result,
                    max_rows=30,
                )
            if isinstance(result, pd.Series):
                return an.frame_to_markdown(result.reset_index(), max_rows=30)
            return str(result)

        tools.append(run_pandas_expression)

    return tools


# ── restricted pandas evaluation ─────────────────────────────────────────────
_ALLOWED_NODES = (
    ast.Expression,
    ast.Attribute,
    ast.Subscript,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Slice,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.keyword,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Invert,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Lambda,
    ast.arguments,
    ast.arg,
    ast.IfExp,
    ast.ListComp,
    ast.comprehension,
    ast.Starred,
)
_BLOCKED_ATTRS = {
    "to_pickle",
    "to_csv",
    "to_parquet",
    "to_sql",
    "to_hdf",
    "to_excel",
    "to_json",
    "to_clipboard",
    "read_csv",
    "read_pickle",
    "eval",
    "query",
    "apply",
    "pipe",
    "agg",
    "aggregate",
    "transform",
    "applymap",
    "map",
    "load",
    "save",
    "system",
    "open",
}
_ALLOWED_NAMES = {
    "df",
    "pd",
    "np",
    "True",
    "False",
    "None",
    "len",
    "min",
    "max",
    "sum",
    "abs",
    "round",
    "sorted",
    "list",
    "dict",
    "str",
    "int",
    "float",
    "range",
}


def safe_eval_pandas(expression: str, df: pd.DataFrame):
    expression = expression.strip().strip("`")
    if len(expression) > 600:
        raise ValueError("Expression too long.")
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in _BLOCKED_ATTRS
        ):
            raise ValueError(f"Attribute '{node.attr}' is not allowed")
        if (
            isinstance(node, ast.Name)
            and node.id not in _ALLOWED_NAMES
            and not (isinstance(node.ctx, ast.Load) and node.id in {"x", "row", "v", "s", "g"})
        ):
            raise ValueError(f"Name '{node.id}' is not allowed")
    safe_globals = {
        "__builtins__": {},
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "str": str,
        "int": int,
        "float": float,
        "range": range,
    }
    return eval(compile(tree, "<agent>", "eval"), safe_globals, {})  # noqa: S307 - AST allow-listed above


def artifact_path_is_safe(artifact_dir: Path, name: str) -> Path | None:
    candidate = (artifact_dir / name).resolve()
    if candidate.parent != artifact_dir.resolve() or candidate.suffix != ".png":
        return None
    return candidate if candidate.exists() else None
