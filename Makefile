.PHONY: help install install-dev build start dev test lint format assets docker

PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

help:
	@echo "make install      create .venv, install Python deps and frontend packages"
	@echo "make build        build the React frontend into frontend/dist"
	@echo "make start        serve API + built frontend on http://localhost:8000"
	@echo "make dev          hot-reload dev servers (FastAPI :8000 + Vite :5173)"
	@echo "make test         run backend tests (no API key needed)"
	@echo "make lint         ruff + TypeScript type-check"
	@echo "make assets       regenerate README screenshots / diagrams / sample charts"
	@echo "make docker       build the container image"

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(BIN)/python
	$(BIN)/pip install -r requirements-dev.txt
	cd frontend && npm install

build:
	cd frontend && npm run build

start: build
	$(BIN)/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

dev:
	./scripts/dev.sh

test:
	$(BIN)/python -m pytest

lint:
	$(BIN)/ruff check backend scripts
	$(BIN)/ruff format --check backend scripts
	cd frontend && npm run typecheck

format:
	$(BIN)/ruff format backend scripts
	$(BIN)/ruff check --fix backend scripts

assets:
	$(BIN)/python scripts/make_sample_charts.py
	cd scripts && npm install && npm run assets

docker:
	docker build -t data-scientist-langchain-ai .
