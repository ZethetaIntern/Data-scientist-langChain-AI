from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app import analysis as an
from backend.app.demo import build_demo_dataframe


def test_demo_dataset_is_deterministic():
    a, b = build_demo_dataframe(), build_demo_dataframe()
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (1500, 14)
    assert a["revenue"].min() >= 1.0


def test_load_csv_infers_types_and_separator():
    raw = b"id;date;amount;flag;city\n001;2024-01-05;1,200.5;yes;Paris\n002;2024-02-10;300;no;Rome\n003;2024-03-15;;yes;Oslo\n"
    df = an.load_csv(raw, max_rows=1000, max_columns=50)
    assert list(df.columns) == ["id", "date", "amount", "flag", "city"]
    assert df["id"].tolist() == ["001", "002", "003"], "ID-like columns keep leading zeros"
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["amount"].tolist()[:2] == [1200.5, 300.0]
    assert pd.api.types.is_bool_dtype(df["flag"])
    assert df["amount"].isna().sum() == 1


@pytest.mark.parametrize(
    "raw, message",
    [
        (b"", "empty"),
        (b"a,b\n", "no data rows"),
        (b"only_one\n1\n2\n", "at least two columns"),
        (b"a,a\n1,2\n", "unique"),
        ("a,b\n1,2\n".encode("utf-16"), "UTF-8"),
    ],
)
def test_load_csv_rejects_bad_input(raw, message):
    with pytest.raises(an.DatasetError) as exc:
        an.load_csv(raw, max_rows=1000, max_columns=50)
    assert message in str(exc.value)


def test_load_csv_enforces_limits():
    raw = b"a,b\n" + b"\n".join(b"1,2" for _ in range(20))
    with pytest.raises(an.DatasetError, match="Too many rows"):
        an.load_csv(raw, max_rows=10, max_columns=50)


def test_profile_and_column_kinds():
    df = an.coerce_types(build_demo_dataframe())
    p = an.profile_dataframe(df)
    assert p["rows"] == 1500 and p["columns"] == 14
    kinds = {c["name"]: c["kind"] for c in p["column_profiles"]}
    assert kinds["order_id"] == "identifier"
    assert kinds["order_date"] == "datetime"
    assert kinds["region"] == "categorical"
    assert kinds["revenue"] == "numeric"
    assert kinds["returned"] == "boolean"
    assert 0 < p["completeness_pct"] < 100
    assert p["quality_score"] > 95


def test_find_column_is_fuzzy():
    df = pd.DataFrame({"Unit Price": [1], "customer_age": [2]})
    assert an.find_column(df, "unit_price") == "Unit Price"
    assert an.find_column(df, "Customer Age") == "customer_age"
    assert an.find_column(df, "nothing") is None


def test_correlations_and_outliers():
    df = build_demo_dataframe()
    corr, pairs = an.correlation_pairs(df, top_n=3)
    assert corr.shape[0] == corr.shape[1] >= 5
    assert pairs.iloc[0]["feature_a"] == "unit_price" and pairs.iloc[0]["feature_b"] == "revenue"
    out = an.iqr_outliers(df, ["revenue", "discount"])
    assert set(out["column"]) == {"revenue", "discount"}
    assert out.loc[out["column"] == "revenue", "outliers"].iloc[0] > 0


def test_group_aggregate_and_time_series():
    df = an.coerce_types(build_demo_dataframe())
    g = an.group_aggregate(df, "category", "revenue", "sum")
    assert g.iloc[0]["category"] == "Electronics"
    assert g["n"].sum() == 1500
    series, label = an.time_series(df, "order_date", "revenue", freq="MS")
    assert label == "month" and len(series) == 12
    t = an.trend_summary(series, series.columns[1])
    assert t["periods"] == 12 and t["peak_value"] >= t["trough_value"]
    with pytest.raises(ValueError):
        an.group_aggregate(df, "category", "revenue", "bogus")


def test_frame_to_markdown_escapes_pipes_and_truncates():
    df = pd.DataFrame({"a|b": ["x|y", np.nan], "n": [1.0, 2.5]})
    md = an.frame_to_markdown(df, max_rows=1)
    assert "x\\|y" in md
    assert "1 more rows" in md
