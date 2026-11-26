.PHONY: install lint format-check typecheck test check

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests

format-check:
	uv run ruff format --check src tests

typecheck:
	uv run mypy src tests

test:
	uv run pytest

check: lint format-check typecheck test
