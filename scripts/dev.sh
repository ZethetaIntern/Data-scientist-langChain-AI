#!/usr/bin/env bash
# Start the FastAPI backend (hot reload, :8000) and the Vite dev server (:5173) together.
# Vite proxies /api/* to the backend, so open http://localhost:5173 in the browser.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi

$PY -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir backend &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

cd frontend && npm run dev
