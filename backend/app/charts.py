"""Matplotlib chart rendering.

Charts are written as PNG files under the artifact directory and served by the API at
``/api/artifacts/<file>``. The tools return the relative URL so the frontend (and the LLM)
can reference them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#4f8cff", "#ff8a4c", "#3ddc97", "#c77dff", "#ffd166", "#ef476f", "#06d6a0", "#8ecae6"]

plt.rcParams.update(
    {
        "figure.facecolor": "#0f1420",
        "axes.facecolor": "#151b2b",
        "axes.edgecolor": "#2a3550",
        "axes.labelcolor": "#dfe6f5",
        "axes.titlecolor": "#ffffff",
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#243050",
        "grid.alpha": 0.6,
        "xtick.color": "#aab4cc",
        "ytick.color": "#aab4cc",
        "text.color": "#dfe6f5",
        "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#0f1420",
    }
)


@dataclass
class ChartArtifact:
    id: str
    title: str
    kind: str
    url: str
    path: str

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "kind": self.kind, "url": self.url}


class ChartRenderer:
    def __init__(self, artifact_dir: Path, url_prefix: str = "/api/artifacts") -> None:
        self.dir = Path(artifact_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.url_prefix = url_prefix.rstrip("/")

    # ── plumbing ─────────────────────────────────────────────────────────
    def _save(self, fig, title: str, kind: str, name: str | None = None) -> ChartArtifact:
        chart_id = name or f"{kind}-{uuid.uuid4().hex[:10]}"
        path = self.dir / f"{chart_id}.png"
        fig.savefig(path)
        plt.close(fig)
        return ChartArtifact(
            id=chart_id,
            title=title,
            kind=kind,
            url=f"{self.url_prefix}/{path.name}",
            path=str(path),
        )

    # ── chart types ──────────────────────────────────────────────────────
    def histogram(
        self, s: pd.Series, column: str, bins: int = 30, name: str | None = None
    ) -> ChartArtifact:
        data = pd.to_numeric(s, errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(
            data,
            bins=min(bins, max(10, int(np.sqrt(len(data))))),
            color=PALETTE[0],
            alpha=0.9,
            edgecolor="#0f1420",
        )
        mean, median = data.mean(), data.median()
        ax.axvline(
            mean, color=PALETTE[1], linestyle="--", linewidth=1.5, label=f"mean = {mean:,.2f}"
        )
        ax.axvline(
            median, color=PALETTE[2], linestyle=":", linewidth=1.5, label=f"median = {median:,.2f}"
        )
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        ax.legend(frameon=False)
        return self._save(fig, f"Distribution of {column}", "histogram", name)

    def bar(
        self, frame: pd.DataFrame, x: str, y: str, title: str, name: str | None = None
    ) -> ChartArtifact:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = frame[x].astype(str).tolist()
        values = frame[y].astype(float).tolist()
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(values))]
        bars = ax.bar(labels, values, color=colors, edgecolor="#0f1420")
        for b, v in zip(bars, values, strict=False):
            ax.annotate(
                f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if max(len(lbl) for lbl in labels) > 8 or len(labels) > 8:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        return self._save(fig, title, "bar", name)

    def scatter(
        self, df: pd.DataFrame, x: str, y: str, hue: str | None = None, name: str | None = None
    ) -> ChartArtifact:
        fig, ax = plt.subplots(figsize=(8, 5))
        frame = df[[x, y] + ([hue] if hue else [])].dropna()
        if len(frame) > 3000:
            frame = frame.sample(3000, random_state=42)
        if hue:
            for i, (level, part) in enumerate(frame.groupby(hue, observed=True)):
                ax.scatter(
                    part[x],
                    part[y],
                    s=14,
                    alpha=0.7,
                    label=str(level),
                    color=PALETTE[i % len(PALETTE)],
                )
            ax.legend(title=hue, frameon=False, fontsize=8)
        else:
            ax.scatter(frame[x], frame[y], s=14, alpha=0.7, color=PALETTE[0])
        xs, ys = pd.to_numeric(frame[x], errors="coerce"), pd.to_numeric(frame[y], errors="coerce")
        ok = xs.notna() & ys.notna()
        if ok.sum() > 2 and xs[ok].nunique() > 1:
            slope, intercept = np.polyfit(xs[ok], ys[ok], 1)
            grid = np.linspace(xs[ok].min(), xs[ok].max(), 50)
            ax.plot(
                grid,
                slope * grid + intercept,
                color=PALETTE[1],
                linewidth=1.5,
                linestyle="--",
                label="trend",
            )
        title = f"{y} vs {x}" + (f" by {hue}" if hue else "")
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return self._save(fig, title, "scatter", name)

    def line(
        self, frame: pd.DataFrame, x: str, y: str, title: str, name: str | None = None
    ) -> ChartArtifact:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(frame[x], frame[y], color=PALETTE[0], linewidth=2, marker="o", markersize=3)
        ax.fill_between(frame[x], frame[y], alpha=0.12, color=PALETTE[0])
        if len(frame) >= 6:
            rolling = (
                frame[y].rolling(window=max(3, len(frame) // 8), min_periods=1, center=True).mean()
            )
            ax.plot(
                frame[x],
                rolling,
                color=PALETTE[1],
                linewidth=1.5,
                linestyle="--",
                label="rolling mean",
            )
            ax.legend(frameon=False)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.autofmt_xdate()
        return self._save(fig, title, "line", name)

    def box(
        self, df: pd.DataFrame, column: str, by: str | None = None, name: str | None = None
    ) -> ChartArtifact:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if by:
            groups = [
                (str(k), pd.to_numeric(g[column], errors="coerce").dropna())
                for k, g in df.groupby(by, observed=True)
            ]
            groups = [(k, g) for k, g in groups if len(g)][:12]
            bp = ax.boxplot(
                [g for _, g in groups], tick_labels=[k for k, _ in groups], patch_artist=True
            )
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(PALETTE[i % len(PALETTE)])
                patch.set_alpha(0.8)
            title = f"{column} by {by}"
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        else:
            bp = ax.boxplot(
                pd.to_numeric(df[column], errors="coerce").dropna(),
                tick_labels=[column],
                patch_artist=True,
            )
            bp["boxes"][0].set_facecolor(PALETTE[0])
            title = f"Box plot of {column}"
        for key in ("whiskers", "caps", "medians"):
            for artist in bp[key]:
                artist.set_color("#dfe6f5")
        ax.set_title(title)
        return self._save(fig, title, "box", name)

    def heatmap(
        self, corr: pd.DataFrame, title: str = "Correlation matrix", name: str | None = None
    ) -> ChartArtifact:
        n = len(corr.columns)
        size = min(max(5, 0.7 * n + 2), 12)
        fig, ax = plt.subplots(figsize=(size + 1, size))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        ax.grid(False)
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                if not np.isnan(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#dfe6f5" if abs(v) > 0.55 else "#0f1420",
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        return self._save(fig, title, "heatmap", name)

    def importance(self, frame: pd.DataFrame, title: str, name: str | None = None) -> ChartArtifact:
        frame = frame.sort_values("importance", ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(frame) + 1.5)))
        ax.barh(frame["feature"], frame["importance"], color=PALETTE[2], edgecolor="#0f1420")
        ax.set_title(title)
        ax.set_xlabel("permutation importance (score drop)")
        return self._save(fig, title, "importance", name)

    def confusion(
        self, matrix: np.ndarray, labels: list[str], title: str, name: str | None = None
    ) -> ChartArtifact:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.grid(False)
        thresh = matrix.max() / 2 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="#dfe6f5" if matrix[i, j] > thresh else "#0f1420",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        return self._save(fig, title, "confusion", name)

    def actual_vs_predicted(
        self, y_true, y_pred, title: str, name: str | None = None
    ) -> ChartArtifact:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        if len(y_true) > 2000:
            idx = np.random.default_rng(42).choice(len(y_true), 2000, replace=False)
            y_true, y_pred = y_true[idx], y_pred[idx]
        ax.scatter(y_true, y_pred, s=12, alpha=0.6, color=PALETTE[0])
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        ax.plot(
            [lo, hi],
            [lo, hi],
            color=PALETTE[1],
            linestyle="--",
            linewidth=1.5,
            label="perfect prediction",
        )
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
        ax.set_title(title)
        ax.legend(frameon=False)
        return self._save(fig, title, "actual_vs_predicted", name)
