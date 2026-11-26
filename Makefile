.PHONY: install lint format-check typecheck test check up down notebook

install:
	uv sync --extra feature --extra dev

lint:
	uv run ruff check src tests feature_repo

format-check:
	uv run ruff format --check src tests feature_repo

typecheck:
	uv run mypy src tests feature_repo

test:
	uv run pytest

check: lint format-check typecheck test

up:
	docker compose up --build -d

down:
	docker compose down -v

notebook:
	docker compose up --build notebook
