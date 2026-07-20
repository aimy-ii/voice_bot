# Образ голосового бота (воркер LiveKit).
FROM python:3.12-slim

# Системные зависимости: ffmpeg для аудио, build-essential для сборки колёс.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ставим зависимости отдельным слоем — кэшируется, пока не менялся pyproject.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Заранее скачиваем веса моделей (VAD, turn detector), чтобы не тянуть в
# рантайме на первом звонке. Не требует секретов.
RUN python -m voice_bot.agent.main download-files || true

# По умолчанию — продакшн-режим воркера.
CMD ["python", "-m", "voice_bot.agent.main", "start"]
