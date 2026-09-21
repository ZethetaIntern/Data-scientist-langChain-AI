"""A reproducible, explicitly *synthetic* e-commerce dataset used for the demo.

The generator is deterministic (seed 42) so screenshots, tests and the README stay in
sync. Revenue is derived from units, unit price and discount plus a little noise, which
means a baseline model can learn it well — that illustrates the workflow, it does not
forecast a real business.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEMO_NAME = "demo_ecommerce_orders.csv"

REGIONS = ["North", "South", "East", "West", "Central"]
CATEGORIES = {
    "Electronics": (120.0, 45.0),
    "Home & Kitchen": (55.0, 20.0),
    "Clothing": (35.0, 12.0),
    "Sports": (48.0, 18.0),
    "Books": (18.0, 6.0),
    "Beauty": (27.0, 9.0),
}
SEGMENTS = ["New", "Returning", "Loyal", "VIP"]
CHANNELS = ["Web", "Mobile App", "Marketplace", "Retail Partner"]


def build_demo_dataframe(n_rows: int = 1_500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2024-01-01")
    day_offsets = rng.integers(0, 365, size=n_rows)
    # Seasonal ramp: more orders in Q4 and a mild weekend bump
    order_date = start + pd.to_timedelta(day_offsets, unit="D")

    category = rng.choice(list(CATEGORIES), size=n_rows, p=[0.22, 0.2, 0.24, 0.12, 0.12, 0.1])
    region = rng.choice(REGIONS, size=n_rows, p=[0.24, 0.2, 0.22, 0.2, 0.14])
    segment = rng.choice(SEGMENTS, size=n_rows, p=[0.35, 0.35, 0.2, 0.1])
    channel = rng.choice(CHANNELS, size=n_rows, p=[0.4, 0.35, 0.15, 0.1])

    means = np.array([CATEGORIES[c][0] for c in category])
    sds = np.array([CATEGORIES[c][1] for c in category])
    unit_price = np.clip(rng.normal(means, sds), 4.0, None).round(2)

    units = rng.poisson(lam=2.2, size=n_rows) + 1
    seg_bonus = np.select([segment == "Loyal", segment == "VIP"], [0.4, 0.9], default=0.0)
    units = np.clip(np.round(units + seg_bonus).astype(int), 1, 12)

    discount = np.where(
        rng.random(n_rows) < 0.55, 0.0, rng.choice([0.05, 0.1, 0.15, 0.2, 0.3], size=n_rows)
    )
    month = pd.DatetimeIndex(order_date).month
    discount = np.where(month == 11, np.maximum(discount, 0.15), discount)  # Black-Friday season

    gross = units * unit_price
    revenue = (gross * (1 - discount) + rng.normal(0, 3.0, size=n_rows)).round(2)
    revenue = np.clip(revenue, 1.0, None)

    customer_age = np.clip(rng.normal(38, 12, size=n_rows), 18, 80).round(0)
    shipping_days = np.clip(rng.normal(4.2, 1.6, size=n_rows), 1, 14).round(0)
    shipping_days = np.where(
        channel == "Retail Partner", np.maximum(shipping_days - 1, 1), shipping_days
    )

    # Satisfaction drops with slow shipping and rises for loyal / VIP customers
    satisfaction = (
        4.3
        - 0.12 * (shipping_days - 4)
        + np.select([segment == "Loyal", segment == "VIP"], [0.25, 0.4], default=0.0)
        + rng.normal(0, 0.45, size=n_rows)
    )
    satisfaction = np.clip(satisfaction, 1, 5).round(1)

    return_prob = (
        0.04
        + 0.10 * (category == "Clothing")
        + 0.02 * (discount >= 0.2)
        + 0.03 * (satisfaction < 3)
    )
    returned = rng.random(n_rows) < return_prob

    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{100000 + i}" for i in range(n_rows)],
            "order_date": order_date.strftime("%Y-%m-%d"),
            "region": region,
            "channel": channel,
            "category": category,
            "customer_segment": segment,
            "customer_age": customer_age,
            "units": units,
            "unit_price": unit_price,
            "discount": discount,
            "revenue": revenue,
            "shipping_days": shipping_days,
            "satisfaction_score": satisfaction,
            "returned": returned,
        }
    )

    # A pinch of realistic missingness
    df.loc[rng.random(n_rows) < 0.03, "satisfaction_score"] = np.nan
    df.loc[rng.random(n_rows) < 0.02, "customer_age"] = np.nan

    return df.sort_values("order_date", kind="stable").reset_index(drop=True)
