"""Baseline modelling with leakage-aware preprocessing.

* preprocessing (imputation, scaling, one-hot) is fitted on the training split only
* the target and columns that are exact copies of it are excluded from the features
* metrics are reported on a held-out split and compared against a naive baseline
* feature importance is *permutation* importance computed on the held-out split
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .analysis import column_kind

MIN_ROWS = 60
MAX_FEATURES = 30
MAX_CLASSES = 20
SAMPLE_LIMIT = 10_000
SEED = 42


class ModelingError(ValueError):
    pass


@dataclass
class ModelResult:
    task: str
    target: str
    features: list[str]
    n_train: int
    n_test: int
    metrics: dict
    baseline_metrics: dict
    importance: pd.DataFrame
    classes: list[str] = field(default_factory=list)
    confusion: np.ndarray | None = None
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def infer_task(y: pd.Series) -> str:
    kind = column_kind(y)
    if kind in ("boolean", "categorical", "text"):
        return "classification"
    if kind == "numeric":
        nunique = y.nunique(dropna=True)
        if nunique <= 10 and pd.api.types.is_integer_dtype(y.dropna()):
            return "classification"
        return "regression"
    raise ModelingError(f"Column '{y.name}' ({kind}) cannot be used as a target.")


def _candidate_features(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    features, notes = [], []
    y = df[target]
    for c in df.columns:
        if c == target:
            continue
        kind = column_kind(df[c])
        if kind in ("identifier", "text", "datetime"):
            notes.append(f"skipped '{c}' ({kind})")
            continue
        if df[c].nunique(dropna=True) <= 1:
            notes.append(f"skipped '{c}' (constant)")
            continue
        if kind == "numeric" and pd.api.types.is_numeric_dtype(y):
            aligned = pd.concat([df[c], y], axis=1).dropna()
            if len(aligned) > 10 and np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1]):
                notes.append(f"skipped '{c}' (exact copy of target)")
                continue
        features.append(c)
    return features[:MAX_FEATURES], notes


def train_baseline(
    df: pd.DataFrame, target: str, task: str = "auto", features: list[str] | None = None
) -> ModelResult:
    if target not in df.columns:
        raise ModelingError(f"Unknown target column '{target}'.")
    data = df.drop_duplicates()
    data = data[data[target].notna()]
    if len(data) < MIN_ROWS:
        raise ModelingError(
            f"Need at least {MIN_ROWS} rows with a non-missing target (have {len(data)})."
        )

    notes: list[str] = []
    if len(data) > SAMPLE_LIMIT:
        data = data.sample(SAMPLE_LIMIT, random_state=SEED)
        notes.append(f"trained on a reproducible sample of {SAMPLE_LIMIT:,} rows")

    y = data[target]
    resolved_task = infer_task(y) if task == "auto" else task
    if resolved_task not in ("regression", "classification"):
        raise ModelingError("task must be 'auto', 'regression' or 'classification'")

    auto_features, skip_notes = _candidate_features(data, target)
    notes.extend(skip_notes)
    if features:
        unknown = [f for f in features if f not in data.columns]
        if unknown:
            raise ModelingError(f"Unknown feature columns: {unknown}")
        chosen = [f for f in features if f != target][:MAX_FEATURES]
    else:
        chosen = auto_features
    if not chosen:
        raise ModelingError("No usable feature columns were found.")

    X = data[chosen]
    if resolved_task == "classification":
        y = y.astype(str)
        counts = y.value_counts()
        if len(counts) > MAX_CLASSES:
            raise ModelingError(f"Too many classes ({len(counts)}); the limit is {MAX_CLASSES}.")
        if len(counts) < 2:
            raise ModelingError("The target has a single class.")
        rare = counts[counts < 5]
        if not rare.empty:
            keep = ~y.isin(rare.index)
            X, y = X[keep], y[keep]
            notes.append(f"dropped {len(rare)} rare class(es) with < 5 examples")
        stratify = y
    else:
        y = pd.to_numeric(y, errors="coerce")
        keep = y.notna()
        X, y = X[keep], y[keep]
        if y.nunique() < 2:
            raise ModelingError("The target is constant.")
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=stratify
    )

    numeric = [c for c in chosen if column_kind(data[c]) == "numeric"]
    categorical = [c for c in chosen if c not in numeric]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    for c in categorical:
        X_train[c] = X_train[c].astype(str).where(X_train[c].notna(), np.nan)
        X_test[c] = X_test[c].astype(str).where(X_test[c].notna(), np.nan)

    if resolved_task == "regression":
        est = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=SEED, n_jobs=1)
    else:
        est = RandomForestClassifier(
            n_estimators=120, max_depth=12, random_state=SEED, n_jobs=1, class_weight="balanced"
        )

    model = Pipeline([("pre", pre), ("model", est)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if resolved_task == "regression":
        naive = np.full(len(y_test), float(y_train.mean()))
        metrics = {
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse": round(float(root_mean_squared_error(y_test, y_pred)), 4),
        }
        baseline = {
            "r2": round(float(r2_score(y_test, naive)), 4),
            "mae": round(float(mean_absolute_error(y_test, naive)), 4),
            "rmse": round(float(root_mean_squared_error(y_test, naive)), 4),
        }
        scoring = "r2"
        classes: list[str] = []
        cm = None
    else:
        majority = y_train.mode().iloc[0]
        naive = np.full(len(y_test), majority, dtype=object)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        }
        baseline = {
            "accuracy": round(float(accuracy_score(y_test, naive)), 4),
            "f1_macro": round(float(f1_score(y_test, naive, average="macro", zero_division=0)), 4),
        }
        scoring = "f1_macro"
        classes = sorted(y.unique().tolist())
        cm = confusion_matrix(y_test, y_pred, labels=classes)

    perm_rows = min(len(X_test), 300)
    perm = permutation_importance(
        model,
        X_test.iloc[:perm_rows],
        y_test.iloc[:perm_rows],
        n_repeats=3,
        random_state=SEED,
        scoring=scoring,
        n_jobs=1,
    )
    importance = (
        pd.DataFrame({"feature": chosen, "importance": np.round(perm.importances_mean, 4)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return ModelResult(
        task=resolved_task,
        target=target,
        features=chosen,
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        metrics=metrics,
        baseline_metrics=baseline,
        importance=importance,
        classes=[str(c) for c in classes],
        confusion=cm,
        y_test=np.asarray(y_test) if resolved_task == "regression" else None,
        y_pred=np.asarray(y_pred) if resolved_task == "regression" else None,
        notes=notes,
    )
