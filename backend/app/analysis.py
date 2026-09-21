"""Deterministic pandas helpers shared by the agent tools and the API.

Every number the agent reports is produced here (or in :mod:`modeling`), never by the LLM.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

MISSING_TOKENS = ["", " ", "null", "NULL", "NaN", "nan", "N/A", "n/a", "None"]


class DatasetError(ValueError):
    """Raised when an uploaded file cannot be turned into a usable dataframe."""


# ── loading ─────────────────────────────────────────────────────────────────
def _sniff_separator(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_csv(raw: bytes, *, max_rows: int, max_columns: int) -> pd.DataFrame:
    if not raw or not raw.strip():
        raise DatasetError("The file is empty.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError("Only UTF-8 encoded CSV files are supported.") from exc

    sep = _sniff_separator(text[:20_000])
    header = next(
        csv.reader(io.StringIO(text.splitlines()[0] if text.splitlines() else ""), delimiter=sep),
        [],
    )
    header = [h.strip() or f"column_{i}" for i, h in enumerate(header)]
    if len(set(header)) != len(header):
        raise DatasetError("Column names must be unique.")
    if len(header) > max_columns:
        raise DatasetError(f"Too many columns ({len(header)}); the limit is {max_columns}.")
    if any(len(h) > 100 for h in header):
        raise DatasetError("Column names must be at most 100 characters.")
    # Identifier-like columns are read as text so values such as 001 keep their leading zeros.
    id_like = {h: str for h in header if _is_id_name(h)}
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=sep,
            na_values=MISSING_TOKENS,
            keep_default_na=False,
            skipinitialspace=True,
            dtype=id_like or None,
            engine="python",
        )
    except Exception as exc:  # pragma: no cover - pandas raises many subclasses
        raise DatasetError(f"Could not parse CSV: {exc}") from exc

    if df.shape[0] == 0:
        raise DatasetError("The CSV has a header but no data rows.")
    if df.shape[1] < 2:
        raise DatasetError("The CSV needs at least two columns.")
    if df.shape[0] > max_rows:
        raise DatasetError(f"Too many rows ({df.shape[0]:,}); the limit is {max_rows:,}.")

    df.columns = (
        header[: df.shape[1]]
        if len(header) >= df.shape[1]
        else [str(c).strip() for c in df.columns]
    )
    df = df.dropna(axis=1, how="all")
    return coerce_types(df)


def _is_id_name(name: str) -> bool:
    n = name.strip().lower()
    return (
        n == "id"
        or n.endswith("_id")
        or n.endswith(" id")
        or (n.endswith("id") and len(n) <= 12)
        or n in {"sku", "zip", "zipcode", "postal_code", "phone"}
    )


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Infer numeric / boolean / datetime columns from object columns."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        stripped = s.astype("string").str.strip()
        lowered = stripped.str.lower()
        non_null = lowered.dropna()
        if non_null.empty:
            continue
        # booleans
        if (
            set(non_null.unique()) <= {"true", "false", "yes", "no", "0", "1"}
            and len(non_null.unique()) <= 2
        ):
            out[col] = lowered.map(
                {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}
            ).astype("boolean")
            continue
        # numbers (allow thousands separators)
        numeric = pd.to_numeric(stripped.str.replace(",", "", regex=False), errors="coerce")
        if numeric.notna().sum() >= 0.95 * non_null.size and not _is_id_name(col):
            out[col] = numeric
            continue
        # dates
        if _looks_like_date(col, non_null):
            parsed = pd.to_datetime(stripped, errors="coerce", utc=False, format="mixed")
            if parsed.notna().sum() >= 0.9 * non_null.size:
                out[col] = parsed
    return out


def _looks_like_date(name: str, values: pd.Series) -> bool:
    hint = any(k in name.lower() for k in ("date", "time", "day", "month", "year", "timestamp"))
    sample = values.head(50)
    has_sep = sample.str.contains(r"[-/:]").mean() > 0.8
    has_digit = sample.str.contains(r"\d").mean() > 0.9
    return (hint or has_sep) and has_digit and has_sep


# ── column typing ───────────────────────────────────────────────────────────
def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    nunique = s.nunique(dropna=True)
    if nunique <= max(20, int(0.05 * len(s))) and nunique < len(s) * 0.5:
        return "categorical"
    if nunique >= 0.9 * s.notna().sum():
        return "identifier"
    return "text"


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if column_kind(df[c]) == "numeric"]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if column_kind(df[c]) in ("categorical", "boolean")]


def datetime_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if column_kind(df[c]) == "datetime"]


def default_metric(df: pd.DataFrame) -> str | None:
    """The numeric column a business user most likely cares about (revenue-like names first)."""
    nums = numeric_columns(df)
    for pref in ("revenue", "sales", "amount", "total", "price", "value", "score", "count"):
        for c in nums:
            if pref in c.lower():
                return c
    return nums[0] if nums else None


def find_column(df: pd.DataFrame, name: str | None) -> str | None:
    """Fuzzy column lookup (case / punctuation insensitive)."""
    if not name:
        return None
    name = name.strip().strip("'\"`")
    if name in df.columns:
        return name

    def norm(x: str) -> str:
        return "".join(ch for ch in x.lower() if ch.isalnum())

    target = norm(name)
    if not target:
        return None
    for c in df.columns:
        if norm(c) == target:
            return c
    for c in df.columns:
        if target in norm(c) or norm(c) in target:
            return c
    return None


# ── profiling ───────────────────────────────────────────────────────────────
@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str
    missing: int
    missing_pct: float
    unique: int
    sample: list


def profile_dataframe(df: pd.DataFrame) -> dict:
    n = len(df)
    columns = []
    for c in df.columns:
        s = df[c]
        missing = int(s.isna().sum())
        sample_vals = s.dropna().astype(str).unique()[:3].tolist()
        columns.append(
            {
                "name": c,
                "dtype": str(s.dtype),
                "kind": column_kind(s),
                "missing": missing,
                "missing_pct": round(100 * missing / n, 2) if n else 0.0,
                "unique": int(s.nunique(dropna=True)),
                "sample": sample_vals,
            }
        )
    total_cells = int(df.size) or 1
    missing_cells = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    completeness = 100 * (1 - missing_cells / total_cells)
    uniqueness = 100 * (1 - duplicates / n) if n else 100.0
    return {
        "rows": int(n),
        "columns": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1_048_576, 3),
        "missing_cells": missing_cells,
        "completeness_pct": round(completeness, 2),
        "duplicate_rows": duplicates,
        "quality_score": round(0.8 * completeness + 0.2 * uniqueness, 1),
        "column_profiles": columns,
        "numeric_columns": numeric_columns(df),
        "categorical_columns": categorical_columns(df),
        "datetime_columns": datetime_columns(df),
    }


def describe(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or list(df.columns)
    rows = []
    for c in cols:
        s = df[c]
        kind = column_kind(s)
        row: dict = {"column": c, "kind": kind, "missing": int(s.isna().sum())}
        if kind == "numeric":
            desc = s.describe()
            row.update(
                {
                    "mean": _r(desc.get("mean")),
                    "std": _r(desc.get("std")),
                    "min": _r(desc.get("min")),
                    "25%": _r(desc.get("25%")),
                    "median": _r(desc.get("50%")),
                    "75%": _r(desc.get("75%")),
                    "max": _r(desc.get("max")),
                    "skew": _r(s.skew()) if s.notna().sum() > 2 else None,
                }
            )
        elif kind == "datetime":
            row.update({"min": str(s.min()), "max": str(s.max()), "unique": int(s.nunique())})
        else:
            vc = s.value_counts(dropna=True)
            row.update(
                {
                    "unique": int(s.nunique(dropna=True)),
                    "top": str(vc.index[0]) if len(vc) else None,
                    "top_freq": int(vc.iloc[0]) if len(vc) else None,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df) or 1
    miss = df.isna().sum()
    rep = pd.DataFrame(
        {
            "column": miss.index,
            "missing": miss.values,
            "missing_pct": (100 * miss.values / n).round(2),
        }
    )
    return rep[rep["missing"] > 0].sort_values("missing", ascending=False).reset_index(drop=True)


def correlation_pairs(
    df: pd.DataFrame, method: str = "pearson", top_n: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in numeric_columns(df) if df[c].nunique(dropna=True) > 1]
    if len(cols) < 2:
        return pd.DataFrame(), pd.DataFrame()
    corr = df[cols].corr(method=method, min_periods=10)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append({"feature_a": a, "feature_b": b, "correlation": round(float(v), 3)})
    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df["abs"] = pairs_df["correlation"].abs()
        pairs_df = (
            pairs_df.sort_values("abs", ascending=False)
            .drop(columns="abs")
            .head(top_n)
            .reset_index(drop=True)
        )
    return corr, pairs_df


def iqr_outliers(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or numeric_columns(df)
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            rows.append(
                {
                    "column": c,
                    "outliers": 0,
                    "outlier_pct": 0.0,
                    "lower_bound": _r(q1),
                    "upper_bound": _r(q3),
                    "note": "zero IQR",
                }
            )
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        rows.append(
            {
                "column": c,
                "outliers": int(mask.sum()),
                "outlier_pct": round(100 * mask.mean(), 2),
                "lower_bound": _r(lo),
                "upper_bound": _r(hi),
                "note": "",
            }
        )
    return (
        pd.DataFrame(rows).sort_values("outliers", ascending=False).reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def group_aggregate(
    df: pd.DataFrame, group_by: str, metric: str, aggregation: str = "mean", top_n: int = 15
) -> pd.DataFrame:
    agg = aggregation.lower()
    if agg not in {"mean", "sum", "count", "median", "min", "max"}:
        raise ValueError("aggregation must be one of mean, sum, count, median, min, max")
    grouped = df.groupby(group_by, dropna=True, observed=True)[metric]
    result = getattr(grouped, agg)()
    counts = grouped.size()
    out = pd.DataFrame(
        {group_by: result.index.astype(str), f"{agg}_{metric}": result.values, "n": counts.values}
    )
    return out.sort_values(f"{agg}_{metric}", ascending=False).head(top_n).reset_index(drop=True)


def time_series(
    df: pd.DataFrame, date_col: str, metric: str, freq: str = "auto", aggregation: str = "sum"
) -> tuple[pd.DataFrame, str]:
    dates = pd.to_datetime(df[date_col], errors="coerce")
    frame = pd.DataFrame(
        {"date": dates, "value": pd.to_numeric(df[metric], errors="coerce")}
    ).dropna()
    if frame.empty:
        raise ValueError("No overlapping non-missing dates and values.")
    span_days = max((frame["date"].max() - frame["date"].min()).days, 1)
    if freq == "auto":
        freq = (
            "D"
            if span_days <= 62
            else "W"
            if span_days <= 400
            else "MS"
            if span_days <= 365 * 6
            else "YS"
        )
    agg = aggregation if aggregation in {"sum", "mean", "count", "median"} else "sum"
    series = frame.set_index("date")["value"].resample(freq).agg(agg)
    label = {"D": "day", "W": "week", "MS": "month", "M": "month", "YS": "year"}.get(freq, freq)
    out = series.reset_index().rename(columns={"value": f"{agg}_{metric}"})
    if freq == "W":  # label weeks by their first day instead of the period end
        out["date"] = out["date"] - pd.Timedelta(days=6)
    return out, label


def trend_summary(series: pd.DataFrame, value_col: str) -> dict:
    """Robust trend statistics: compares first vs second half and fits a linear slope."""
    vals = series[value_col].astype(float).fillna(0)
    n = len(vals)
    if n < 2:
        return {"periods": n}
    half = n // 2
    first, second = vals.iloc[:half].mean(), vals.iloc[half:].mean()
    change = (second - first) / abs(first) * 100 if first else float("nan")
    slope = float(np.polyfit(np.arange(n), vals.values, 1)[0])
    peak_i, trough_i = int(vals.idxmax()), int(vals.idxmin())
    return {
        "periods": n,
        "first_half_mean": round(float(first), 2),
        "second_half_mean": round(float(second), 2),
        "change_pct": round(float(change), 1) if pd.notna(change) else None,
        "slope_per_period": round(slope, 3),
        "peak_value": round(float(vals.max()), 2),
        "peak_date": series.loc[peak_i, "date"],
        "trough_value": round(float(vals.min()), 2),
        "trough_date": series.loc[trough_i, "date"],
        "mean": round(float(vals.mean()), 2),
        "std": round(float(vals.std()), 2),
    }


# ── formatting helpers ──────────────────────────────────────────────────────
def _r(v, nd: int = 4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return v


def frame_to_markdown(df: pd.DataFrame, max_rows: int = 25) -> str:
    """Render a small dataframe as a GitHub-flavoured Markdown table (no tabulate needed)."""
    if df is None or df.empty:
        return "_(no rows)_"
    show = df.head(max_rows)
    cols = [str(c) for c in show.columns]

    def fmt(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float):
            if v == 0:
                return "0"
            return (
                f"{v:,.4g}"
                if abs(v) < 1e-3 or abs(v) >= 1e6
                else f"{v:,.3f}".rstrip("0").rstrip(".")
            )
        if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
            return f"{int(v):,}"
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_… {len(df) - max_rows} more rows not shown_")
    return "\n".join(lines)
