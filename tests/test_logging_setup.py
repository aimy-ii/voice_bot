"""Тесты настройки уровней логирования без добавления обработчиков."""

import logging

from voice_bot.logging_setup import setup_logging


def test_setup_logging_sets_voice_bot_level() -> None:
    """После вызова уровень логгера voice_bot соответствует переданному."""
    setup_logging(level="DEBUG", noisy_level="WARNING")

    assert logging.getLogger("voice_bot").level == logging.DEBUG


def test_setup_logging_sets_noisy_loggers() -> None:
    """Шумные логгеры получают noisy_level."""
    setup_logging(level="INFO", noisy_level="ERROR")

    assert logging.getLogger("livekit").level == logging.ERROR
    assert logging.getLogger("livekit.agents").level == logging.ERROR
    assert logging.getLogger("aiohttp").level == logging.ERROR
    assert logging.getLogger("httpx").level == logging.ERROR
    assert logging.getLogger("openai").level == logging.ERROR


def test_setup_logging_does_not_add_root_handlers() -> None:
    """Функция не добавляет обработчиков на корневой логгер — защита от дублей."""
    root = logging.getLogger()
    before = len(root.handlers)

    setup_logging(level="INFO", noisy_level="WARNING")

    assert len(root.handlers) == before
