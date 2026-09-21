from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    artifacts = tmp_path_factory.mktemp("artifacts")
    return Settings(llm_provider="heuristic", artifact_dir=artifacts, cors_origins="")


@pytest.fixture(scope="session")
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
