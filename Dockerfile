# Етап 1: збираємо середовище з uv.lock і будуємо індекс з невеликого корпусу.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Спершу лише залежності: цей шар кешується, поки не змінився uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Потім сам код.
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Індекс будуємо під час збірки образу: у репозиторії його немає (*.bin у .gitignore).
COPY deploy_corpus ./corpus
RUN .venv/bin/findex index corpus --out index.bin --executor serial

# Етап 2: легкий образ без uv і без збірки.
FROM python:3.14-slim-bookworm AS runtime
RUN useradd --system --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/index.bin ./index.bin
COPY --from=builder /app/corpus ./corpus
ENV PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH=/app/index.bin \
    CORPUS_DIR=/app/corpus \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000

# PORT дає платформа (Render, Railway); локально — 8000. exec: щоб SIGTERM дійшов до uvicorn
# і lifespan встиг виконати завершення.
CMD ["sh", "-c", "exec findex serve --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
