"""Pydantic request / response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnProfileOut(BaseModel):
    name: str
    dtype: str
    kind: str
    missing: int
    missing_pct: float
    unique: int
    sample: list[str]


class DatasetSummary(BaseModel):
    id: str
    name: str
    source: str
    rows: int
    columns: int
    created_at: float


class DatasetDetail(DatasetSummary):
    memory_mb: float
    missing_cells: int
    completeness_pct: float
    duplicate_rows: int
    quality_score: float
    column_profiles: list[ColumnProfileOut]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    preview: list[dict]
    suggested_questions: list[str]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


class ChartOut(BaseModel):
    id: str
    title: str
    kind: str
    url: str


class StepOut(BaseModel):
    tool: str
    input: dict
    output: str
    duration_ms: int


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    steps: list[StepOut]
    charts: list[ChartOut]
    suggestions: list[str] = []
    mode: str
    elapsed_ms: int


class TurnOut(BaseModel):
    role: str
    content: str
    charts: list[ChartOut]
    steps: list[StepOut]
    suggestions: list[str] = []
    created_at: float


class HealthOut(BaseModel):
    status: str
    version: str
    llm: dict
    code_tool_enabled: bool
