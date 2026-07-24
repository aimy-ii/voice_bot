.DEFAULT_GOAL := help

help:  ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

install:  ## Установить зависимости (uv sync)
	uv sync

download-files:  ## Скачать веса моделей (turn-detector и т.п.), разово
	uv run python -m voice_bot.agent.main download-files

console:  ## Локальный тест агента в терминале (микрофон)
	uv run python -m voice_bot.agent.main console

dev:  ## Запустить воркер против LiveKit-сервера (dev)
	uv run python -m voice_bot.agent.main dev

start:  ## Запустить воркер против LiveKit-сервера (prod)
	uv run python -m voice_bot.agent.main start

token:  ## JWT-токен для LiveKit Playground (stdout)
	uv run python -m voice_bot.agent.token

lint:  ## Проверить код линтером ruff
	uv run ruff check src tests

format:  ## Отформатировать код и починить импорты
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:  ## Прогнать тесты
	uv run pytest

check:  ## Всё сразу: формат + линт + тесты
	uv run ruff format src tests
	uv run ruff check src tests
	uv run pytest

up:  ## Поднять LiveKit-сервер + бота в docker
	docker compose up --build

play:  ## Подсказка: URL локального веб-клиента агента
	@echo "Playground: http://localhost:3000"

down:  ## Остановить контейнеры
	docker compose down

logs:  ## Логи docker-стека (follow)
	docker compose logs -f

.PHONY: help install download-files console dev start token lint format test check up play down logs
