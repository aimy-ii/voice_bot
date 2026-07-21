"""Тест чтения настроек из переменных окружения."""

import pytest

from voice_bot.config import Settings


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Задать обязательные переменные окружения для Settings."""
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-123")


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Настройки читают ключи из окружения по ожидаемым именам."""
    _set_required_env(monkeypatch)

    settings = Settings()  # type: ignore[call-arg]

    assert settings.livekit_url == "ws://localhost:7880"
    assert settings.elevenlabs_voice_id == "voice-123"
    assert settings.language == "ru"


def test_elevenlabs_voice_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без ELEVENLABS_STABILITY/SIMILARITY/STYLE — дефолты для ровного тона."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("ELEVENLABS_STABILITY", raising=False)
    monkeypatch.delenv("ELEVENLABS_SIMILARITY", raising=False)
    monkeypatch.delenv("ELEVENLABS_STYLE", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.elevenlabs_stability == 0.7
    assert settings.elevenlabs_similarity == 0.8
    assert settings.elevenlabs_style == 0.0


def test_elevenlabs_voice_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ELEVENLABS_STABILITY/SIMILARITY/STYLE читаются из окружения."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_STABILITY", "0.55")
    monkeypatch.setenv("ELEVENLABS_SIMILARITY", "0.9")
    monkeypatch.setenv("ELEVENLABS_STYLE", "0.1")

    settings = Settings()  # type: ignore[call-arg]

    assert settings.elevenlabs_stability == 0.55
    assert settings.elevenlabs_similarity == 0.9
    assert settings.elevenlabs_style == 0.1


def test_elevenlabs_model_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без ELEVENLABS_MODEL — стабильная multilingual_v2."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("ELEVENLABS_MODEL", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.tts_model == "eleven_multilingual_v2"


def test_elevenlabs_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ELEVENLABS_MODEL читается из окружения."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.tts_model == "eleven_turbo_v2_5"


def test_proxy_url_none_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без PROXY_HOST свойство proxy_url возвращает None.

    Пустая строка перекрывает значение из ``.env`` (если файл есть локально).
    """
    _set_required_env(monkeypatch)
    monkeypatch.setenv("PROXY_HOST", "")

    settings = Settings()  # type: ignore[call-arg]

    assert settings.proxy_url is None


def test_proxy_url_with_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """При всех PROXY_* собирается socks5h://user:pass@host:port."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("PROXY_HOST", "proxy.example")
    monkeypatch.setenv("PROXY_PORT", "1080")
    monkeypatch.setenv("PROXY_USER", "user")
    monkeypatch.setenv("PROXY_PASS", "pass")

    settings = Settings()  # type: ignore[call-arg]

    assert settings.proxy_url == "socks5h://user:pass@proxy.example:1080"


def test_proxy_url_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без user/pass собирается socks5h://host:port."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("PROXY_HOST", "proxy.example")
    monkeypatch.setenv("PROXY_PORT", "1080")
    monkeypatch.setenv("PROXY_USER", "")
    monkeypatch.setenv("PROXY_PASS", "")

    settings = Settings()  # type: ignore[call-arg]

    assert settings.proxy_url == "socks5h://proxy.example:1080"


def test_proxy_fields_none_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без PROXY_HOST свойство proxy_fields возвращает None."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("PROXY_HOST", "")

    settings = Settings()  # type: ignore[call-arg]

    assert settings.proxy_fields is None


def test_proxy_fields_with_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """При всех PROXY_* — host/port/rdns и username/password."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("PROXY_HOST", "proxy.example")
    monkeypatch.setenv("PROXY_PORT", "1080")
    monkeypatch.setenv("PROXY_USER", "user")
    monkeypatch.setenv("PROXY_PASS", "pass")

    settings = Settings()  # type: ignore[call-arg]

    assert settings.proxy_fields == {
        "host": "proxy.example",
        "port": 1080,
        "rdns": True,
        "username": "user",
        "password": "pass",
    }


def test_proxy_fields_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без user/pass — только host/port/rdns."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("PROXY_HOST", "proxy.example")
    monkeypatch.setenv("PROXY_PORT", "1080")
    monkeypatch.setenv("PROXY_USER", "")
    monkeypatch.setenv("PROXY_PASS", "")

    settings = Settings()  # type: ignore[call-arg]

    assert settings.proxy_fields == {
        "host": "proxy.example",
        "port": 1080,
        "rdns": True,
    }


def test_background_audio_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без BG_* — фон включён, тихий эмбиент и средняя громкость thinking."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("BG_ENABLED", raising=False)
    monkeypatch.delenv("BG_AMBIENT_VOLUME", raising=False)
    monkeypatch.delenv("BG_THINKING_VOLUME", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.bg_enabled is True
    assert settings.bg_ambient_volume == 0.15
    assert settings.bg_thinking_volume == 0.6


def test_background_audio_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """BG_ENABLED / громкости читаются из окружения; false парсится в bool."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("BG_ENABLED", "false")
    monkeypatch.setenv("BG_AMBIENT_VOLUME", "0.12")
    monkeypatch.setenv("BG_THINKING_VOLUME", "0.45")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.bg_enabled is False
    assert settings.bg_ambient_volume == 0.12
    assert settings.bg_thinking_volume == 0.45
