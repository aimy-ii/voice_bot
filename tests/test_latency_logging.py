"""Тесты безопасного логирования e2e-задержки ответа ассистента."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from voice_bot.agent.main import _log_assistant_latency


def test_log_assistant_latency_ignores_item_without_role(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Служебный объект без ``role`` не роняет обработчик и ничего не логирует."""
    item = object()  # нет атрибута role

    with caplog.at_level("INFO", logger="voice_bot"):
        _log_assistant_latency(item)

    assert not any("Задержка ответа" in r.message for r in caplog.records)


def test_log_assistant_latency_logs_assistant_e2e(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Реплика ассистента с ``metrics.e2e_latency`` попадает в лог."""
    item = SimpleNamespace(role="assistant", metrics={"e2e_latency": 1.25})

    with caplog.at_level("INFO", logger="voice_bot"):
        _log_assistant_latency(item)

    assert any("Задержка ответа: e2e=1.25с" in r.message for r in caplog.records)


def test_log_assistant_latency_skips_user_role(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Реплика пользователя с метрикой не логируется как задержка ответа."""
    item = SimpleNamespace(role="user", metrics={"e2e_latency": 0.5})

    with caplog.at_level("INFO", logger="voice_bot"):
        _log_assistant_latency(item)

    assert not any("Задержка ответа" in r.message for r in caplog.records)


def test_log_assistant_latency_handles_non_dict_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Несловарный ``metrics`` не вызывает ошибку и не логирует."""
    item = SimpleNamespace(role="assistant", metrics=MagicMock())

    with caplog.at_level("INFO", logger="voice_bot"):
        _log_assistant_latency(item)

    assert not any("Задержка ответа" in r.message for r in caplog.records)
