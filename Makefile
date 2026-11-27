.PHONY: install lint format-check typecheck test check fetch-data up down notebook feast-ui train

install:
	uv sync --extra feature --extra train --extra dev

lint:
	uv run ruff check src tests feature_repo

format-check:
	uv run ruff format --check src tests feature_repo

typecheck:
	uv run mypy src tests feature_repo

test:
	uv run pytest

check: lint format-check typecheck test

fetch-data:
	uv run python -m banking_fti.features.csv_loader --raw-dir data/raw

up:
	docker compose up --build -d

down:
	docker compose down -v

notebook:
	docker compose up --build notebook

feast-ui:
	docker compose up --build feast-ui

train:
	docker compose up --build train
