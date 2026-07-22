# syntax=docker/dockerfile:1.7
# Образ голосового бота (воркер LiveKit).
FROM python:3.12-slim

# Системные зависимости: ffmpeg для аудио, build-essential для сборки колёс.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Менеджер зависимостей проекта (тянется один раз, слой кэшируется навсегда).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# 1) Зависимости отдельным слоем: кэш живёт, пока не менялись pyproject.toml / uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# 2) Код проекта: правка бьёт только по этому слою и ниже, зависимости не переставляются.
COPY src ./src
RUN uv sync --frozen

# 3) Веса моделей (turn-detector) в образ, чтобы не тянуть на первом звонке.
#    Кэш-маунт переживает пересборки, cp кладёт файлы внутрь образа.
ENV HF_HOME=/opt/hf
RUN --mount=type=cache,target=/opt/hf-cache \
    HF_HOME=/opt/hf-cache python -m voice_bot.agent.main download-files \
    && mkdir -p /opt/hf && cp -a /opt/hf-cache/. /opt/hf/

# По умолчанию — продакшн-режим воркера.
CMD ["python", "-m", "voice_bot.agent.main", "start"]