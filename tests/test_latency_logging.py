"""Тесты логирования текста и e2e-задержки ответа ассистента."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from voice_bot.agent.main import _assistant_text, _log_assistant_latency


def test_log_assistant_latency_ignores_item_without_role(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Служебный объект без ``role`` не роняет обработчик и ничего не логирует."""
    item = object()  # нет атрибута role

    with caplog.at_level("INFO", logger="voice_bot"):
        _log_assistant_latency(item)

    assert not any("Бот" in r.message or "Задержка ответа" in r.message for r in caplog.records)


def test_log_assistant_latency_logs_assistant_text_and_e2e(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Реплика ассистента с текстом и ``e2e_latency`` попадает в лог."""
    item = SimpleNamespace(
        role="assistant",
        text_content="Здравствуйте",
        metrics={"e2e_latency": 1.25},
    )

    with caplog.at_level("INFO", logger="voice_bot"):
        _log_assistant_latency(item)

    assert any("Бот (e2e=1.25с): Здравствуйте" in r.message for r in caplog.records)


def test_log_assistant_latency_logs_text_without_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Текст opening_line без метрик тоже логируется."""
    item = SimpleNamespace(role="assistant", text_content="Добрый день", metrics={})

    with caplog.at_level("INFO", logger="voice_bot"):
        _log_assistant_latency(item)

    assert any("Бот: Добрый день" in r.message for r in caplog.records)


def test_log_assistant_latency_skips_user_role(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Реплика пользователя не логируется как ответ бота."""
    item = SimpleNamespace(
        role="user",
        text_content="Привет",
        metrics={"e2e_latency": 0.5},
    )

    with caplog.at_level("INFO", logger="voice_bot"):
        _log_assistant_latency(item)

    assert not any("Бот" in r.message or "Задержка ответа" in r.message for r in caplog.records)


def test_log_assistant_latency_handles_non_dict_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Несловарный ``metrics`` не вызывает ошибку; текст всё равно логируется."""
    item = SimpleNamespace(
        role="assistant",
        text_content="Ок",
        metrics=MagicMock(),
    )

    with caplog.at_level("INFO", logger="voice_bot"):
        _log_assistant_latency(item)

    assert any("Бот: Ок" in r.message for r in caplog.records)


def test_assistant_text_falls_back_to_content_list() -> None:
    """Если нет ``text_content``, текст собирается из ``content``."""
    item = SimpleNamespace(content=["Часть 1", "часть 2"])

    assert _assistant_text(item) == "Часть 1 часть 2"
