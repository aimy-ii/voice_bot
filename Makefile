.DEFAULT_GOAL := help
PY := python

help:  ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install:  ## Установить зависимости (dev)
	$(PY) -m pip install -e ".[dev]"

download-files:  ## Скачать веса моделей (turn-detector и т.п.), разово
	uv run python -m voice_bot.agent.main download-files

console:  ## Локальный тест агента в терминале (микрофон)
	$(PY) -m voice_bot.agent.main console

dev:  ## Запустить воркер против LiveKit-сервера
	$(PY) -m voice_bot.agent.main dev

lint:  ## Проверить код линтером ruff
	ruff check src tests

format:  ## Отформатировать код и починить импорты
	ruff format src tests
	ruff check --fix src tests

test:  ## Прогнать тесты
	pytest

up:  ## Поднять LiveKit-сервер + бота в docker
	docker compose up --build

down:  ## Остановить контейнеры
	docker compose down

.PHONY: help install download-files console dev lint format test up down
