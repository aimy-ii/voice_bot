"""Тест чтения настроек из переменных окружения."""

import pytest

from voice_bot.config import Settings


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Настройки читают ключи из окружения по ожидаемым именам."""
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-123")

    settings = Settings()  # type: ignore[call-arg]

    assert settings.livekit_url == "ws://localhost:7880"
    assert settings.elevenlabs_voice_id == "voice-123"
    assert settings.language == "ru"
