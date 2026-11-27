FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY feature_repo ./feature_repo

RUN uv sync --frozen --no-dev --extra feature

CMD ["python", "-m", "banking_fti.features.pipeline", "--repo-path", "/app/feature_repo", "--data-dir", "/app/feature_repo/data"]
