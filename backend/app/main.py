"""FastAPI application: REST API + static hosting of the built React frontend."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __doc__ as _pkg_doc  # noqa: F401
from .agent import DataScientistAgent
from .analysis import DatasetError, default_metric, load_csv, profile_dataframe
from .config import REPO_ROOT, Settings, get_settings
from .llm import build_chat_model
from .schemas import ChatRequest, ChatResponse, DatasetDetail, DatasetSummary, HealthOut, TurnOut
from .store import ChatTurn, Dataset, DatasetStore
from .tools import artifact_path_is_safe

VERSION = "0.1.0"
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ds_agent")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = DatasetStore(max_datasets=settings.max_datasets)
    llm, info = build_chat_model(settings)
    agent = DataScientistAgent(llm, info, settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.ensure_demo()
        log.info("LLM mode: %s (%s)", info.mode, info.detail)
        yield

    app = FastAPI(
        title="Data Scientist LangChain AI",
        version=VERSION,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.agent = agent
    app.state.settings = settings

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ── helpers ─────────────────────────────────────────────────────────
    def get_dataset(dataset_id: str) -> Dataset:
        ds = store.get(dataset_id)
        if ds is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found (uploads expire when the server restarts).",
            )
        return ds

    def summary(ds: Dataset) -> DatasetSummary:
        return DatasetSummary(
            id=ds.id,
            name=ds.name,
            source=ds.source,
            rows=int(len(ds.df)),
            columns=int(ds.df.shape[1]),
            created_at=ds.created_at,
        )

    def suggested_questions(ds: Dataset, profile: dict) -> list[str]:
        nums, cats, dates = (
            profile["numeric_columns"],
            profile["categorical_columns"],
            profile["datetime_columns"],
        )
        metric = default_metric(ds.df)
        other = next((c for c in reversed(nums) if c != metric), None)
        qs = ["Give me an overview of this dataset", "Which columns have missing values?"]
        if metric:
            qs.append(f"Show the distribution of {metric}")
        if len(nums) >= 2:
            qs.append("What correlates most strongly?")
        if metric and cats:
            qs.append(f"Average {metric} by {cats[0]}")
        if dates and metric:
            qs.append(f"Monthly trend of {metric}")
        if metric:
            qs.append(f"Predict {metric} — which features matter?")
        if other:
            qs.append(f"What drives {other}?")
        qs.append("Are there outliers I should worry about?")
        return qs[:8]

    # ── routes ──────────────────────────────────────────────────────────
    @app.get("/api/health", response_model=HealthOut)
    def health() -> HealthOut:
        return HealthOut(
            status="ok",
            version=VERSION,
            llm=info.as_dict(),
            code_tool_enabled=settings.enable_code_tool,
        )

    @app.get("/api/datasets", response_model=list[DatasetSummary])
    def list_datasets() -> list[DatasetSummary]:
        return [summary(ds) for ds in store.list()]

    @app.post("/api/datasets", response_model=DatasetDetail, status_code=201)
    async def upload_dataset(file: UploadFile = File(...)) -> DatasetDetail:
        name = (file.filename or "upload.csv").strip()
        if not name.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Please upload a .csv file.")
        raw = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(raw) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB."
            )
        try:
            df = await run_in_threadpool(
                load_csv, raw, max_rows=settings.max_rows, max_columns=settings.max_columns
            )
        except DatasetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ds = store.add(name, df)
        log.info("uploaded %s (%s rows × %s cols) as %s", name, *df.shape, ds.id)
        return detail(ds.id)

    @app.get("/api/datasets/{dataset_id}", response_model=DatasetDetail)
    def detail(dataset_id: str) -> DatasetDetail:
        ds = get_dataset(dataset_id)
        profile = profile_dataframe(ds.df)
        preview = json.loads(ds.df.head(8).to_json(orient="records", date_format="iso"))
        return DatasetDetail(
            **summary(ds).model_dump(),
            **{k: v for k, v in profile.items() if k not in ("rows", "columns")},
            preview=preview,
            suggested_questions=suggested_questions(ds, profile),
        )

    @app.delete("/api/datasets/{dataset_id}", status_code=204)
    def delete_dataset(dataset_id: str) -> None:
        if dataset_id == "demo":
            raise HTTPException(status_code=400, detail="The demo dataset cannot be deleted.")
        if not store.delete(dataset_id):
            raise HTTPException(status_code=404, detail="Dataset not found.")

    @app.get("/api/datasets/{dataset_id}/rows")
    def rows(
        dataset_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)
    ) -> dict:
        ds = get_dataset(dataset_id)
        chunk = ds.df.iloc[offset : offset + limit].replace({np.nan: None})
        return {
            "total": int(len(ds.df)),
            "offset": offset,
            "rows": json.loads(chunk.to_json(orient="records", date_format="iso")),
        }

    @app.post("/api/datasets/{dataset_id}/chat", response_model=ChatResponse)
    async def chat(dataset_id: str, body: ChatRequest) -> ChatResponse:
        ds = get_dataset(dataset_id)
        session_id = body.session_id or uuid.uuid4().hex[:12]
        history = ds.history(session_id)
        try:
            reply = await run_in_threadpool(agent.ask, body.message, ds, list(history))
        except Exception as exc:
            log.exception("agent failure")
            raise HTTPException(status_code=500, detail=f"The agent failed: {exc}") from exc
        history.append(ChatTurn(role="user", content=body.message))
        history.append(
            ChatTurn(
                role="assistant",
                content=reply.answer,
                charts=reply.charts,
                steps=[s.as_dict() for s in reply.steps],
            )
        )
        return ChatResponse(session_id=session_id, **reply.as_dict())

    @app.get("/api/datasets/{dataset_id}/history", response_model=list[TurnOut])
    def history(dataset_id: str, session_id: str = Query(..., max_length=64)) -> list[TurnOut]:
        ds = get_dataset(dataset_id)
        return [
            TurnOut(
                role=t.role,
                content=t.content,
                charts=t.charts,
                steps=t.steps,
                created_at=t.created_at,
            )
            for t in ds.history(session_id)
        ]

    @app.get("/api/datasets/{dataset_id}/report", response_class=PlainTextResponse)
    def report(dataset_id: str, session_id: str = Query(..., max_length=64)) -> PlainTextResponse:
        ds = get_dataset(dataset_id)
        profile = profile_dataframe(ds.df)
        lines = [
            f"# Analysis report — {ds.name}",
            "",
            f"*Generated by Data Scientist LangChain AI ({info.mode} mode{', ' + info.model if info.model else ''}) on {pd.Timestamp.utcnow():%Y-%m-%d %H:%M UTC}*",
            "",
            f"- Rows: {profile['rows']:,}  ·  Columns: {profile['columns']}  ·  Completeness: {profile['completeness_pct']}%  ·  Quality score: {profile['quality_score']}/100",
            "",
        ]
        for turn in ds.history(session_id):
            if turn.role == "user":
                lines += [f"## ❓ {turn.content}", ""]
            else:
                lines += [turn.content, ""]
                for c in turn.charts:
                    lines.append(f"![{c['title']}]({c['url']})")
                lines.append("")
        text = "\n".join(lines)
        return PlainTextResponse(
            text,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="report-{ds.id}.md"'},
        )

    @app.get("/api/artifacts/{name}")
    def artifact(name: str) -> FileResponse:
        path = artifact_path_is_safe(settings.artifact_dir, name)
        if path is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(
            path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"}
        )

    # ── frontend (built assets) ─────────────────────────────────────────
    if FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str, request: Request) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404)
            candidate = FRONTEND_DIST / full_path
            if (
                full_path
                and candidate.is_file()
                and candidate.resolve().is_relative_to(FRONTEND_DIST.resolve())
            ):
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")
    else:

        @app.get("/", include_in_schema=False)
        def root() -> dict:
            return {
                "message": "Data Scientist LangChain AI API is running. Build the frontend (npm run build in ./frontend) to serve the UI here.",
                "docs": "/api/docs",
            }

    return app


app = create_app()


def run() -> None:  # pragma: no cover - convenience entry point
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover
    run()


__all__ = ["app", "create_app", "Path"]
