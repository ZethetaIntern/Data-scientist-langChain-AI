"""The Data Scientist agent.

Two interchangeable "brains" share the same tools and produce the same reply shape:

* :class:`LangChainBrain` — a LangChain tool-calling agent (OpenAI / Anthropic / Groq / Ollama).
  The LLM plans which tools to call, observes their output and writes the final insight.
* :class:`HeuristicBrain` — a deterministic intent router used when no LLM is configured.
  It maps the question to the same tools with keyword rules, so the app (and the tests)
  work fully offline. Numbers are identical in both modes because the tools compute them.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from . import analysis as an
from .charts import ChartRenderer
from .config import Settings
from .llm import LLMInfo
from .store import ChatTurn, Dataset
from .tools import ToolContext, build_tools

log = logging.getLogger(__name__)

IMAGE_MD = re.compile(r"!\[[^\]]*\]\((/api/artifacts/[^)]+)\)")

SYSTEM_PROMPT = """You are an expert data scientist assistant analysing a tabular dataset called "{dataset_name}".

Ground rules:
- NEVER invent numbers. Every statistic, chart or model result must come from a tool call.
- Start with `dataset_overview` if you do not yet know the columns. Column names are case-insensitive.
- Prefer one or two well-chosen tool calls over many. Do not repeat a tool with identical arguments.
- Charts: tools return Markdown images; keep them in your answer exactly as returned.
- Answer like a senior analyst: lead with the insight, then evidence (numbers, short tables), then caveats and one or two concrete next steps.
- Use concise Markdown with headings, bullets and tables. Never mention internal tool names to the user.
- If the request is impossible with the available columns, say so and suggest what is possible.

Dataset columns: {columns}
"""


@dataclass
class AgentStep:
    tool: str
    input: dict
    output: str
    duration_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "tool": self.tool,
            "input": self.input,
            "output": self.output[:1500],
            "duration_ms": self.duration_ms,
        }


@dataclass
class AgentReply:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    mode: str = "heuristic"
    elapsed_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "steps": [s.as_dict() for s in self.steps],
            "charts": self.charts,
            "suggestions": self.suggestions,
            "mode": self.mode,
            "elapsed_ms": self.elapsed_ms,
        }


def _history_messages(history: list[ChatTurn], limit: int = 12) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    for turn in history[-limit:]:
        text = IMAGE_MD.sub("", turn.content).strip()
        msgs.append(
            HumanMessage(content=text) if turn.role == "user" else AIMessage(content=text[:4000])
        )
    return msgs


def _strip_images(text: str) -> str:
    return IMAGE_MD.sub("", text).strip()


# ── LangChain brain ────────────────────────────────────────────────────────
class LangChainBrain:
    def __init__(self, llm: BaseChatModel, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    def run(
        self, question: str, ctx: ToolContext, tools: list[BaseTool], history: list[ChatTurn]
    ) -> AgentReply:
        from langchain.agents import AgentExecutor, create_tool_calling_agent

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=self.settings.max_iterations,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            verbose=False,
        )
        started = time.perf_counter()
        result = executor.invoke(
            {
                "input": question,
                "chat_history": _history_messages(history),
                "dataset_name": ctx.dataset_name,
                "columns": ", ".join(ctx.df.columns),
            }
        )
        steps = []
        for action, observation in result.get("intermediate_steps", []):
            steps.append(
                AgentStep(
                    tool=action.tool,
                    input=dict(action.tool_input)
                    if isinstance(action.tool_input, dict)
                    else {"input": action.tool_input},
                    output=str(observation),
                )
            )
        output = result.get("output", "")
        if isinstance(output, list):  # Anthropic returns content blocks
            output = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in output
            )
        return AgentReply(
            answer=_strip_images(str(output)),
            steps=steps,
            charts=[c.as_dict() for c in ctx.charts],
            mode="llm",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )


# ── Heuristic brain ────────────────────────────────────────────────────────
class HeuristicBrain:
    """Keyword-driven planner. Good enough to make the workspace useful without an API key."""

    GREETING = re.compile(r"^\s*(hi|hello|hey|help|what can you do|yo)\b", re.I)
    AGG_WORDS = {"total", "sum", "count", "average", "mean", "median", "max", "min"}

    def run(
        self, question: str, ctx: ToolContext, tools: list[BaseTool], history: list[ChatTurn]
    ) -> AgentReply:
        started = time.perf_counter()
        by_name = {t.name: t for t in tools}
        steps: list[AgentStep] = []
        q = question.strip()
        ql = q.lower()
        df = ctx.df
        mentioned = self._mentioned_columns(df, q)
        # A column literally named like an aggregation word ("Total", "Count") is ambiguous:
        # if another numeric column is mentioned too, read the word as the aggregation instead.
        numeric_cols = an.numeric_columns(df)
        ambiguous = [c for c in mentioned if c.lower() in self.AGG_WORDS]
        for c in ambiguous:
            if any(o != c and o in numeric_cols for o in mentioned):
                mentioned.remove(c)
        intent = self._without_columns(ql, mentioned)  # keywords only, column names removed
        numeric = [c for c in mentioned if c in numeric_cols]
        categorical = [c for c in mentioned if c in an.categorical_columns(df)]
        dates = an.datetime_columns(df) or [c for c in df.columns if "date" in c.lower()]

        def call(name: str, **kwargs) -> str:
            t0 = time.perf_counter()
            try:
                out = by_name[name].invoke(kwargs)
            except Exception as exc:  # tools raise ValueError for bad columns
                out = f"Error: {exc}"
            steps.append(
                AgentStep(
                    tool=name,
                    input=kwargs,
                    output=str(out),
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                )
            )
            return str(out)

        if self.GREETING.match(ql) and len(ql) < 40:
            answer = self._help_text(df)
        elif re.search(r"\bmissing|null|nan\b|incomplete|empty values", ql):
            answer = "## Missing values\n\n" + call("missing_values_report")
        elif re.search(r"correlat|relationship|related to|heatmap|associat", ql):
            method = "spearman" if "spearman" in ql or "rank" in ql else "pearson"
            answer = (
                "## Correlation analysis\n\n"
                + call("correlation_analysis", method=method, top_n=10)
                + self._corr_insight(df, method)
            )
        elif re.search(r"outlier|anomal|extreme value", ql):
            answer = "## Outlier check (1.5×IQR rule)\n\n" + call(
                "detect_outliers", columns=",".join(numeric)
            )
        elif re.search(
            r"predict|forecast|model|train|classif|regress|drive|which features|what features|factors|important feature|feature importance|what influences|what affects|explain",
            ql,
        ):
            target = self._pick_target(df, q, mentioned)
            if target is None:
                answer = 'I can train a baseline model, but I need to know which column to predict. Try: *"predict revenue"* or *"which features drive satisfaction_score?"*'
            else:
                task = (
                    "classification"
                    if "classif" in ql
                    else "regression"
                    if "regress" in ql
                    else "auto"
                )
                feats = [c for c in mentioned if c != target]
                answer = f"## Baseline model for `{target}`\n\n" + call(
                    "train_baseline_model", target=target, task=task, features=",".join(feats)
                )
        elif (
            re.search(
                r"trend|over time|time series|per month|monthly|weekly|daily|season|growth|by date|by month|timeline",
                ql,
            )
            and dates
        ):
            metric = numeric[0] if numeric else self._default_metric(df)
            agg = self._aggregation(intent, default="sum")
            freq = (
                "M"
                if re.search(r"month", ql)
                else "W"
                if re.search(r"week", ql)
                else "D"
                if re.search(r"daily|per day", ql)
                else "Y"
                if re.search(r"year|annual", ql)
                else "auto"
            )
            answer = f"## Trend of `{metric}` over time\n\n" + call(
                "time_series_trend", date_column=dates[0], metric=metric, aggregation=agg, freq=freq
            )
        elif re.search(r"distribution|histogram|spread|skew", ql) and (
            numeric or an.numeric_columns(df)
        ):
            col = numeric[0] if numeric else self._default_metric(df)
            answer = f"## Distribution of `{col}`\n\n" + call("plot_chart", kind="histogram", x=col)
        elif re.search(r"scatter|\bvs\.?\b|versus|against", ql) and len(numeric) >= 2:
            hue = categorical[0] if categorical else ""
            answer = f"## `{numeric[1]}` vs `{numeric[0]}`\n\n" + call(
                "plot_chart", kind="scatter", x=numeric[0], y=numeric[1], hue=hue
            )
        elif re.search(r"box ?plot", ql) and numeric:
            answer = f"## Box plot of `{numeric[0]}`\n\n" + call(
                "plot_chart", kind="box", x=numeric[0], hue=categorical[0] if categorical else ""
            )
        elif (
            categorical
            and (numeric or re.search(r"\bby\b|\bper\b|across|compare|breakdown|segment|group", ql))
            and (numeric or re.search(r"revenue|sales|amount|price|score|value", ql))
        ):
            metric = numeric[0] if numeric else self._default_metric(df)
            agg = self._aggregation(intent, default="mean")
            answer = (
                f"## {agg.title()} `{metric}` by `{categorical[0]}`\n\n"
                + call("group_aggregate", group_by=categorical[0], metric=metric, aggregation=agg)
                + self._group_insight(df, categorical[0], metric, agg)
            )
        elif categorical and re.search(
            r"most common|top|frequent|count|how many|popular|share|proportion|percentage|values",
            ql,
        ):
            answer = f"## Most common `{categorical[0]}` values\n\n" + call(
                "value_counts", column=categorical[0], top_n=10
            )
        elif re.search(
            r"describe|statistic|summary of|summar|mean|average|median|std|deviation|min|max|range",
            ql,
        ):
            answer = "## Summary statistics\n\n" + call(
                "describe_columns", columns=",".join(mentioned)
            )
        elif categorical and not numeric:
            answer = f"## Most common `{categorical[0]}` values\n\n" + call(
                "value_counts", column=categorical[0], top_n=10
            )
        elif numeric:
            answer = f"## Distribution of `{numeric[0]}`\n\n" + call(
                "plot_chart", kind="histogram", x=numeric[0]
            )
        elif (
            re.search(
                r"overview|shape|column|what.*(in|about).*(data|dataset|file)|explore|eda|analy|insight|start|look",
                ql,
            )
            or not steps
        ):
            answer = "## Dataset overview\n\n" + call("dataset_overview")
            if re.search(r"full|complete|eda|explore|analy|insight", ql):
                answer += "\n\n## Missing values\n\n" + call("missing_values_report")
                answer += "\n\n## Correlations\n\n" + call(
                    "correlation_analysis", method="pearson", top_n=8
                )
        else:  # pragma: no cover - defensive
            answer = "## Dataset overview\n\n" + call("dataset_overview")

        return AgentReply(
            answer=_strip_images(answer),
            steps=steps,
            charts=[c.as_dict() for c in ctx.charts],
            mode="heuristic",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _mentioned_columns(df: pd.DataFrame, q: str) -> list[str]:
        ql = q.lower()
        found: list[tuple[int, str]] = []
        for c in df.columns:
            variants = {c.lower(), c.lower().replace("_", " "), c.lower().replace("_", "")}
            best = None
            for v in variants:
                if not v:
                    continue
                m = re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?:s|es)?(?![a-z0-9])", ql)
                if m and (best is None or m.start() < best):
                    best = m.start()
            if best is not None:
                found.append((best, c))
        found.sort()
        return [c for _, c in found]

    @staticmethod
    def _without_columns(ql: str, columns: list[str]) -> str:
        """Remove column names from the question so e.g. a column called 'Total' is not read as 'sum'."""
        text = ql
        for c in sorted(columns, key=len, reverse=True):
            for v in {c.lower(), c.lower().replace("_", " "), c.lower().replace("_", "")}:
                if v:
                    text = re.sub(
                        r"(?<![a-z0-9])" + re.escape(v) + r"(?:s|es)?(?![a-z0-9])", " ", text
                    )
        return text

    @staticmethod
    def _aggregation(intent: str, default: str = "mean") -> str:
        if re.search(r"average|mean|avg", intent):
            return "mean"
        if re.search(r"median", intent):
            return "median"
        if re.search(r"total|\bsum\b|overall", intent):
            return "sum"
        if re.search(r"how many|number of|\bcount", intent):
            return "count"
        if re.search(r"highest|maximum|\bmax\b|largest", intent):
            return "max"
        if re.search(r"lowest|minimum|\bmin\b|smallest", intent):
            return "min"
        return default

    @staticmethod
    def _default_metric(df: pd.DataFrame) -> str:
        return an.default_metric(df) or df.columns[0]

    def _pick_target(self, df: pd.DataFrame, q: str, mentioned: list[str]) -> str | None:
        m = re.search(
            r"(?:predict|forecast|classify|model|drive[s]?|influence[s]?|affect[s]?|explain[s]?)\s+(?:the\s+)?([a-zA-Z_][\w ]{0,40})",
            q,
            re.I,
        )
        if m:
            col = an.find_column(
                df, m.group(1).strip().split(" using ")[0].split(" from ")[0].split(" with ")[0]
            )
            if col:
                return col
        usable = [
            c for c in mentioned if an.column_kind(df[c]) in ("numeric", "categorical", "boolean")
        ]
        if usable:
            return usable[0]
        return None

    @staticmethod
    def _corr_insight(df: pd.DataFrame, method: str) -> str:
        _, pairs = an.correlation_pairs(df, method=method, top_n=3)
        if pairs.empty:
            return ""
        lines = ["\n\n**Reading the numbers**"]
        for _, r in pairs.iterrows():
            v = r["correlation"]
            strength = (
                "very strong"
                if abs(v) > 0.8
                else "strong"
                if abs(v) > 0.6
                else "moderate"
                if abs(v) > 0.3
                else "weak"
            )
            direction = "positive" if v > 0 else "negative"
            lines.append(
                f"- `{r['feature_a']}` and `{r['feature_b']}`: {strength} {direction} relationship (r = {v:+.2f})."
            )
        lines.append(
            "- Correlation is not causation — confirm with domain knowledge or an experiment."
        )
        return "\n".join(lines)

    @staticmethod
    def _group_insight(df: pd.DataFrame, group: str, metric: str, agg: str) -> str:
        try:
            frame = an.group_aggregate(df, group, metric, agg)
        except Exception:
            return ""
        if len(frame) < 2:
            return ""
        vcol = [c for c in frame.columns if c not in (group, "n")][0]
        top, bottom = frame.iloc[0], frame.iloc[-1]
        gap = (top[vcol] - bottom[vcol]) / abs(bottom[vcol]) * 100 if bottom[vcol] else float("nan")
        return (
            f"\n\n**Key takeaway:** `{top[group]}` leads with {top[vcol]:,.2f} while `{bottom[group]}` is lowest at {bottom[vcol]:,.2f}"
            + (f" — a {gap:,.0f}% gap." if pd.notna(gap) else ".")
        )

    @staticmethod
    def _help_text(df: pd.DataFrame) -> str:
        nums = an.numeric_columns(df)
        cats = an.categorical_columns(df)
        ex_num = HeuristicBrain._default_metric(df) if nums else "value"
        ex_cat = cats[0] if cats else "category"
        return (
            "## Hi! I'm your data-science assistant\n\n"
            "Ask me questions about the loaded dataset in plain English. For example:\n\n"
            f"- *Give me an overview of the data*\n- *Which columns have missing values?*\n- *Show the distribution of {ex_num}*\n"
            f"- *Average {ex_num} by {ex_cat}*\n- *What correlates with {ex_num}?*\n- *Predict {ex_num}*\n"
        )


def follow_up_questions(df: pd.DataFrame, question: str, limit: int = 3) -> list[str]:
    """Generic, dataset-aware follow-up prompts shown as chips under an answer."""
    ql = question.lower()
    nums = an.numeric_columns(df)
    cats = an.categorical_columns(df)
    dates = an.datetime_columns(df)
    metric = an.default_metric(df)
    other = next((c for c in reversed(nums) if c != metric), None)
    ideas: list[str] = []
    if "correlat" not in ql and len(nums) >= 2:
        ideas.append(f"What correlates with {other or metric}?")
    if not re.search(r"predict|model|drive", ql) and metric:
        ideas.append(f"Predict {metric} — which features matter?")
    if cats and metric and not re.search(r"\bby\b|\bper\b", ql):
        ideas.append(f"Average {metric} by {cats[0]}")
    if dates and metric and not re.search(r"trend|month|week|over time", ql):
        ideas.append(f"Monthly trend of {metric}")
    if "missing" not in ql:
        ideas.append("Which columns have missing values?")
    if "outlier" not in ql:
        ideas.append("Are there outliers I should worry about?")
    return ideas[:limit]


# ── facade ────────────────────────────────────────────────────────────────
class DataScientistAgent:
    def __init__(self, llm: BaseChatModel | None, info: LLMInfo, settings: Settings) -> None:
        self.info = info
        self.settings = settings
        self.renderer = ChartRenderer(settings.artifact_dir)
        self.brain = LangChainBrain(llm, settings) if llm is not None else HeuristicBrain()
        self.fallback = HeuristicBrain()

    def ask(self, question: str, dataset: Dataset, history: list[ChatTurn]) -> AgentReply:
        ctx = ToolContext(
            df=dataset.df,
            dataset_name=dataset.name,
            renderer=self.renderer,
            enable_code_tool=self.settings.enable_code_tool,
        )
        tools = build_tools(ctx)
        try:
            return self.brain.run(question, ctx, tools, history)
        except Exception as exc:
            if isinstance(self.brain, HeuristicBrain):
                raise
            log.exception("LLM agent failed; falling back to heuristic analyst")
            ctx.charts.clear()
            reply = self.fallback.run(question, ctx, tools, history)
            reply.answer = (
                f"> ⚠️ The LLM provider returned an error (`{type(exc).__name__}`), so this answer was produced by the offline analyst.\n\n"
                + reply.answer
            )
            reply.mode = "heuristic-fallback"
            return reply
