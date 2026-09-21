"""Tests for the tools, the heuristic brain and the LangChain brain (with a fake LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from backend.app import analysis as an
from backend.app.agent import DataScientistAgent, HeuristicBrain
from backend.app.charts import ChartRenderer
from backend.app.config import Settings
from backend.app.demo import build_demo_dataframe
from backend.app.llm import LLMInfo, build_chat_model
from backend.app.modeling import ModelingError, train_baseline
from backend.app.store import Dataset
from backend.app.tools import ToolContext, build_tools, safe_eval_pandas


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return an.coerce_types(build_demo_dataframe())


@pytest.fixture
def ctx(df: pd.DataFrame, tmp_path: Path) -> ToolContext:
    return ToolContext(
        df=df, dataset_name="demo.csv", renderer=ChartRenderer(tmp_path), enable_code_tool=True
    )


def _tool(ctx: ToolContext, name: str):
    return {t.name: t for t in build_tools(ctx)}[name]


class ScriptedToolModel(BaseChatModel):
    """A fake chat model that replays scripted AI messages (with tool calls) and accepts tool binding."""

    script: list[AIMessage]
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self

    def _next(self) -> AIMessage:
        msg = self.script[self.cursor]
        self.cursor += 1
        return msg

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        msg = self._next()
        chunk = AIMessageChunk(
            content=msg.content,
            tool_call_chunks=[
                {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
                for i, tc in enumerate(msg.tool_calls)
            ],
        )
        yield ChatGenerationChunk(message=chunk)


# ── tools ───────────────────────────────────────────────────────────────────
def test_tool_names_and_descriptions(ctx: ToolContext):
    tools = build_tools(ctx)
    names = {t.name for t in tools}
    assert {
        "dataset_overview",
        "correlation_analysis",
        "train_baseline_model",
        "plot_chart",
        "run_pandas_expression",
    } <= names
    assert all(t.description for t in tools), "every tool needs a docstring for the LLM"


def test_overview_and_describe(ctx: ToolContext):
    text = _tool(ctx, "dataset_overview").invoke({})
    assert "Rows: 1,500" in text and "order_date" in text
    desc = _tool(ctx, "describe_columns").invoke({"columns": "Revenue, region"})
    assert "revenue" in desc and "region" in desc and "median" in desc


def test_chart_tools_register_artifacts(ctx: ToolContext):
    _tool(ctx, "plot_chart").invoke({"kind": "histogram", "x": "revenue"})
    _tool(ctx, "plot_chart").invoke(
        {"kind": "scatter", "x": "unit_price", "y": "revenue", "hue": "category"}
    )
    _tool(ctx, "group_aggregate").invoke(
        {"group_by": "region", "metric": "revenue", "aggregation": "mean"}
    )
    _tool(ctx, "correlation_analysis").invoke({})
    _tool(ctx, "time_series_trend").invoke(
        {"date_column": "order_date", "metric": "revenue", "freq": "M"}
    )
    assert [c.kind for c in ctx.charts] == ["histogram", "scatter", "bar", "heatmap", "line"]
    for c in ctx.charts:
        assert Path(c.path).exists() and c.url.startswith("/api/artifacts/")


def test_unknown_column_gives_helpful_error(ctx: ToolContext):
    with pytest.raises(ValueError, match="Available columns"):
        _tool(ctx, "plot_chart").invoke({"kind": "histogram", "x": "does_not_exist"})


def test_restricted_pandas_expression(ctx: ToolContext, df: pd.DataFrame):
    out = _tool(ctx, "run_pandas_expression").invoke(
        {"expression": "df.groupby('region')['revenue'].sum().sort_values(ascending=False)"}
    )
    assert "North" in out
    for bad in [
        "__import__('os')",
        "df.to_csv('x.csv')",
        "open('/etc/passwd')",
        "df.eval('1')",
        "[x for x in ().__class__.__mro__]",
    ]:
        with pytest.raises(ValueError):
            safe_eval_pandas(bad, df)


# ── modelling ────────────────────────────────────────────────────────────────
def test_regression_baseline_beats_naive(df: pd.DataFrame):
    res = train_baseline(df, "revenue")
    assert res.task == "regression"
    assert res.metrics["r2"] > 0.9 > res.baseline_metrics["r2"]
    assert res.importance.iloc[0]["feature"] in {"unit_price", "units"}
    assert "revenue" not in res.features and "order_id" not in res.features


def test_classification_baseline(df: pd.DataFrame):
    res = train_baseline(df, "returned")
    assert res.task == "classification" and res.classes == ["False", "True"]
    assert res.confusion is not None and res.confusion.sum() == res.n_test


def test_modeling_guardrails(df: pd.DataFrame):
    with pytest.raises(ModelingError, match="Unknown target"):
        train_baseline(df, "nope")
    with pytest.raises(ModelingError, match="at least 60"):
        train_baseline(df.head(30), "revenue")
    with pytest.raises(ModelingError, match="cannot be used as a target"):
        train_baseline(df, "order_id")


# ── heuristic brain ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "question, expected_tool",
    [
        ("Give me an overview of the dataset", "dataset_overview"),
        ("which columns have missing values?", "missing_values_report"),
        ("what correlates most strongly?", "correlation_analysis"),
        ("show the distribution of revenue", "plot_chart"),
        ("average revenue by region", "group_aggregate"),
        ("total revenue per category", "group_aggregate"),
        ("monthly trend of revenue", "time_series_trend"),
        ("predict revenue", "train_baseline_model"),
        ("what drives satisfaction_score?", "train_baseline_model"),
        ("are there outliers?", "detect_outliers"),
        ("most common channel", "value_counts"),
        ("revenue vs unit_price", "plot_chart"),
        ("describe customer_age", "describe_columns"),
    ],
)
def test_heuristic_routing(ctx: ToolContext, question: str, expected_tool: str):
    reply = HeuristicBrain().run(question, ctx, build_tools(ctx), history=[])
    assert [s.tool for s in reply.steps][0] == expected_tool
    assert reply.answer and "Error" not in reply.answer
    assert reply.mode == "heuristic"


def test_heuristic_ignores_column_names_when_reading_aggregation(tmp_path: Path):
    """A column literally called 'Total' must not turn 'average Total by City' into a sum."""
    frame = pd.DataFrame(
        {
            "City": ["A", "B", "A", "B"] * 20,
            "Total": [10.0, 20.0, 30.0, 40.0] * 20,
            "Qty": [1, 2, 3, 4] * 20,
        }
    )
    c = ToolContext(df=frame, dataset_name="t.csv", renderer=ChartRenderer(tmp_path))
    reply = HeuristicBrain().run("average Total by City", c, build_tools(c), history=[])
    assert reply.steps[0].tool == "group_aggregate"
    assert reply.steps[0].input["aggregation"] == "mean"
    reply = HeuristicBrain().run("total Qty per City", c, build_tools(c), history=[])
    assert reply.steps[0].input == {"group_by": "City", "metric": "Qty", "aggregation": "sum"}


def test_heuristic_greeting_needs_no_tools(ctx: ToolContext):
    reply = HeuristicBrain().run("hello", ctx, build_tools(ctx), history=[])
    assert reply.steps == [] and "Ask me" in reply.answer


def test_images_are_stripped_from_answer_but_returned_as_charts(ctx: ToolContext):
    reply = HeuristicBrain().run("average revenue by region", ctx, build_tools(ctx), history=[])
    assert "![" not in reply.answer
    assert len(reply.charts) == 1 and reply.charts[0]["kind"] == "bar"


# ── provider factory ─────────────────────────────────────────────────────────
def test_provider_auto_detection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    s = Settings(llm_provider="auto", artifact_dir=tmp_path)
    llm, info = build_chat_model(s)
    assert llm is None and info.mode == "heuristic"

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    s = Settings(llm_provider="auto", artifact_dir=tmp_path)
    assert s.resolved_provider() == "groq" and s.resolved_model() == "llama-3.3-70b-versatile"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = Settings(llm_provider="auto", artifact_dir=tmp_path)
    llm, info = build_chat_model(s)
    assert info.provider == "openai" and info.model == "gpt-4o-mini" and llm is not None

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_chat_model(Settings(llm_provider="anthropic", artifact_dir=tmp_path))


# ── LangChain brain with a fake LLM ──────────────────────────────────────────
def test_langchain_agent_executes_tool_calls(df: pd.DataFrame, tmp_path: Path):
    """Drive the real AgentExecutor with a scripted model: one tool call, then a final answer."""
    scripted = ScriptedToolModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "group_aggregate",
                        "args": {"group_by": "region", "metric": "revenue", "aggregation": "mean"},
                    }
                ],
            ),
            AIMessage(
                content="East has the highest average revenue. ![chart](/api/artifacts/x.png)"
            ),
        ]
    )
    settings = Settings(llm_provider="openai", artifact_dir=tmp_path)
    agent = DataScientistAgent(scripted, LLMInfo("fake", "fake-model", "llm", "test"), settings)
    dataset = Dataset(id="t", name="demo.csv", df=df, source="demo")
    reply = agent.ask("average revenue by region", dataset, history=[])
    assert reply.mode == "llm"
    assert [s.tool for s in reply.steps] == ["group_aggregate"]
    assert reply.answer == "East has the highest average revenue."
    assert len(reply.charts) == 1 and reply.charts[0]["kind"] == "bar"


def test_llm_failure_falls_back_to_heuristic(df: pd.DataFrame, tmp_path: Path):
    class ExplodingModel(ScriptedToolModel):
        def _generate(self, *a, **k):  # noqa: D401
            raise ConnectionError("provider down")

        def _stream(self, *a, **k):  # noqa: D401
            raise ConnectionError("provider down")

    settings = Settings(llm_provider="openai", artifact_dir=tmp_path)
    agent = DataScientistAgent(
        ExplodingModel(script=[]), LLMInfo("fake", "fake", "llm", "test"), settings
    )
    dataset = Dataset(id="t", name="demo.csv", df=df, source="demo")
    reply = agent.ask("which columns have missing values?", dataset, history=[])
    assert reply.mode == "heuristic-fallback"
    assert "ConnectionError" in reply.answer and "satisfaction_score" in reply.answer
