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
    monkeypatch.setenv("IS_PROXY", "true")
    monkeypatch.setenv("PROXY_HOST", "proxy.example")
    monkeypatch.setenv("PROXY_PORT", "1080")
    monkeypatch.setenv("PROXY_USER", "user")
    monkeypatch.setenv("PROXY_PASS", "pass")

    settings = Settings()  # type: ignore[call-arg]

    assert settings.proxy_url == "socks5h://user:pass@proxy.example:1080"


def test_proxy_url_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без user/pass собирается socks5h://host:port."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("IS_PROXY", "true")
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
    monkeypatch.setenv("IS_PROXY", "true")
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
    monkeypatch.setenv("IS_PROXY", "true")
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


def test_proxy_disabled_by_is_proxy_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """При IS_PROXY=false заполненные PROXY_* игнорируются."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("IS_PROXY", "false")
    monkeypatch.setenv("PROXY_HOST", "proxy.example")
    monkeypatch.setenv("PROXY_PORT", "1080")
    monkeypatch.setenv("PROXY_USER", "user")
    monkeypatch.setenv("PROXY_PASS", "pass")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.proxy_url is None
    assert settings.proxy_fields is None


def test_proxy_disabled_when_is_proxy_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без IS_PROXY дефолт false — заполненные PROXY_* тоже игнорируются."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("IS_PROXY", raising=False)
    monkeypatch.setenv("PROXY_HOST", "proxy.example")
    monkeypatch.setenv("PROXY_PORT", "1080")
    monkeypatch.setenv("PROXY_USER", "user")
    monkeypatch.setenv("PROXY_PASS", "pass")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.is_proxy is False
    assert settings.proxy_url is None
    assert settings.proxy_fields is None


def test_proxy_empty_port_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустая строка в PROXY_PORT не роняет старт и даёт None в свойствах."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("IS_PROXY", "true")
    monkeypatch.setenv("PROXY_HOST", "proxy.example")
    monkeypatch.setenv("PROXY_PORT", "")
    monkeypatch.setenv("PROXY_USER", "user")
    monkeypatch.setenv("PROXY_PASS", "pass")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.proxy_port is None
    assert settings.proxy_url is None
    assert settings.proxy_fields is None


def test_proxy_whitespace_port_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Строка из пробелов в PROXY_PORT ведёт себя как пустая."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("IS_PROXY", "true")
    monkeypatch.setenv("PROXY_HOST", "proxy.example")
    monkeypatch.setenv("PROXY_PORT", "  ")
    monkeypatch.setenv("PROXY_USER", "user")
    monkeypatch.setenv("PROXY_PASS", "pass")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.proxy_port is None
    assert settings.proxy_url is None
    assert settings.proxy_fields is None


def test_background_audio_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без BG_* — фон включён, тихий эмбиент и средняя громкость thinking."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("BG_ENABLED", raising=False)
    monkeypatch.delenv("BG_AMBIENT_VOLUME", raising=False)
    monkeypatch.delenv("BG_THINKING_VOLUME", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.bg_enabled is True
    assert settings.bg_ambient_volume == 0.4
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


def test_agent_auto_accept_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без AGENT_AUTO_ACCEPT — автоподхват комнат включён."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("AGENT_AUTO_ACCEPT", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.agent_auto_accept is True


def test_agent_auto_accept_false_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """AGENT_AUTO_ACCEPT=false парсится в bool False."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("AGENT_AUTO_ACCEPT", "false")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.agent_auto_accept is False


def test_llm_provider_defaults_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без VOICE_BOT_LLM_PROVIDER — openai (старое поведение по умолчанию)."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("VOICE_BOT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("VOICE_BOT_AGENT_URL", raising=False)
    monkeypatch.delenv("VOICE_BOT_AGENT_GRAPH", raising=False)
    monkeypatch.delenv("VOICE_BOT_AGENT_TIMEOUT", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.llm_provider == "openai"
    assert settings.agent_url == "http://172.17.0.1:8127"
    assert settings.agent_graph == "vector_agent"
    assert settings.agent_timeout == 30.0


def test_agent_partial_defaults_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без VOICE_BOT_AGENT_PARTIAL_* — предподготовка выключена, URL пустой."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("VOICE_BOT_AGENT_PARTIAL_ENABLED", raising=False)
    monkeypatch.delenv("VOICE_BOT_AGENT_PARTIAL_URL", raising=False)
    monkeypatch.delenv("VOICE_BOT_AGENT_PARTIAL_GRAPH", raising=False)
    monkeypatch.delenv("VOICE_BOT_AGENT_PARTIAL_TIMEOUT", raising=False)
    monkeypatch.delenv("VOICE_BOT_AGENT_MAIN_MULTITASK", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.agent_partial_enabled is False
    assert settings.agent_partial_url == ""
    assert settings.agent_partial_graph == "vector_checker"
    assert settings.agent_partial_timeout == 5.0
    assert settings.agent_main_multitask == ""


def test_agent_partial_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Параметры второй точки входа читаются из окружения."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_ENABLED", "true")
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_URL", "http://agent.test:8127")
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_GRAPH", "my_partial")
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_TIMEOUT", "3.5")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.agent_partial_enabled is True
    assert settings.agent_partial_url == "http://agent.test:8127"
    assert settings.agent_partial_graph == "my_partial"
    assert settings.agent_partial_timeout == 3.5


def test_agent_partial_enabled_ok_with_url_and_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """enabled=true с явным URL и дефолтным именем графа — валидатор проходит."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_ENABLED", "true")
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_URL", "http://vector-agent-api:8000")
    monkeypatch.delenv("VOICE_BOT_AGENT_PARTIAL_GRAPH", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.agent_partial_enabled is True
    assert settings.agent_partial_url == "http://vector-agent-api:8000"
    assert settings.agent_partial_graph == "vector_checker"


def test_agent_partial_requires_url_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустой PARTIAL_URL при включённом флаге — ошибка на настройках."""
    from pydantic import ValidationError

    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_ENABLED", "true")
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_URL", "")

    with pytest.raises(ValidationError, match="VOICE_BOT_AGENT_PARTIAL_URL"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_agent_partial_requires_url_when_enabled_and_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enabled=true без PARTIAL_URL в окружении (дефолт пустой) — ошибка."""
    from pydantic import ValidationError

    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_ENABLED", "true")
    monkeypatch.delenv("VOICE_BOT_AGENT_PARTIAL_URL", raising=False)

    with pytest.raises(ValidationError, match="VOICE_BOT_AGENT_PARTIAL_URL"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_agent_partial_requires_graph_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустой PARTIAL_GRAPH при включённом флаге — ошибка на настройках."""
    from pydantic import ValidationError

    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_ENABLED", "true")
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_URL", "http://agent.test:8127")
    monkeypatch.setenv("VOICE_BOT_AGENT_PARTIAL_GRAPH", "")

    with pytest.raises(ValidationError, match="VOICE_BOT_AGENT_PARTIAL_GRAPH"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_empty_required_secret_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустой OPENAI_API_KEY (VAR=) — понятная ошибка, не тихий старт."""
    from pydantic import ValidationError

    _set_required_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_stt_service_url_required_when_provider_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пустой STT_SERVICE_URL при provider=service — ошибка на настройках."""
    from pydantic import ValidationError

    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_STT_PROVIDER", "service")
    monkeypatch.setenv("VOICE_BOT_STT_SERVICE_URL", "")

    with pytest.raises(ValidationError, match="VOICE_BOT_STT_SERVICE_URL"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_agent_graph_required_when_provider_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустой AGENT_GRAPH при provider=agent — ошибка на настройках."""
    from pydantic import ValidationError

    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_LLM_PROVIDER", "agent")
    monkeypatch.setenv("VOICE_BOT_AGENT_URL", "http://172.17.0.1:8127")
    monkeypatch.setenv("VOICE_BOT_AGENT_GRAPH", "")

    with pytest.raises(ValidationError, match="VOICE_BOT_AGENT_GRAPH"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_agent_main_multitask_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """VOICE_BOT_AGENT_MAIN_MULTITASK читается как точка расширения."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_AGENT_MAIN_MULTITASK", "interrupt")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.agent_main_multitask == "interrupt"


def test_llm_provider_agent_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """VOICE_BOT_LLM_PROVIDER=agent и параметры графа читаются из окружения."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_LLM_PROVIDER", "agent")
    monkeypatch.setenv("VOICE_BOT_AGENT_URL", "http://172.17.0.1:8127")
    monkeypatch.setenv("VOICE_BOT_AGENT_GRAPH", "vector_agent")
    monkeypatch.setenv("VOICE_BOT_AGENT_TIMEOUT", "45")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.llm_provider == "agent"
    assert settings.agent_url == "http://172.17.0.1:8127"
    assert settings.agent_graph == "vector_agent"
    assert settings.agent_timeout == 45.0


def test_agent_url_required_when_provider_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустой VOICE_BOT_AGENT_URL при provider=agent — ошибка на этапе настроек."""
    from pydantic import ValidationError

    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_LLM_PROVIDER", "agent")
    monkeypatch.setenv("VOICE_BOT_AGENT_URL", "")

    with pytest.raises(ValidationError, match="VOICE_BOT_AGENT_URL"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_silence_and_continuation_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Дефолты тишины и лимита продолжений без переменных окружения."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("VOICE_BOT_SILENCE_TIMEOUT", raising=False)
    monkeypatch.delenv("VOICE_BOT_SILENCE_GOODBYE", raising=False)
    monkeypatch.delenv("VOICE_BOT_SILENCE_ATTEMPTS", raising=False)
    monkeypatch.delenv("VOICE_BOT_MAX_CONTINUATIONS", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.silence_timeout == 6.0
    assert settings.max_continuations == 3
    assert settings.silence_attempts == 2
    assert (
        settings.silence_goodbye == "Видимо, что-то со связью. Попробуйте, пожалуйста, перезвонить."
    )


def test_tts_provider_defaults_to_elevenlabs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без VOICE_BOT_TTS_PROVIDER — elevenlabs (старое поведение по умолчанию)."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("VOICE_BOT_TTS_PROVIDER", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.tts_provider == "elevenlabs"


def test_tts_provider_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Недопустимое значение VOICE_BOT_TTS_PROVIDER роняет валидацию."""
    from pydantic import ValidationError

    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_TTS_PROVIDER", "yandex")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_tts_provider_openai_allows_empty_elevenlabs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При openai пустые/отсутствующие ключи ElevenLabs не роняют настройки."""
    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_TTS_PROVIDER", "openai")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.tts_provider == "openai"

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.tts_provider == "openai"
    assert settings.elevenlabs_api_key == ""
    assert settings.elevenlabs_voice_id == ""


def test_tts_provider_elevenlabs_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При elevenlabs пустой ELEVENLABS_API_KEY — ошибка с именем переменной."""
    from pydantic import ValidationError

    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_TTS_PROVIDER", "elevenlabs")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")

    with pytest.raises(ValidationError, match="ELEVENLABS_API_KEY"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_tts_provider_elevenlabs_requires_voice_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При elevenlabs пустой ELEVENLABS_VOICE_ID — ошибка с именем переменной."""
    from pydantic import ValidationError

    _set_required_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_TTS_PROVIDER", "elevenlabs")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "")

    with pytest.raises(ValidationError, match="ELEVENLABS_VOICE_ID"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_openai_tts_field_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Дефолты полей синтеза OpenAI без переменных окружения."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("OPENAI_TTS_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_TTS_VOICE", raising=False)
    monkeypatch.delenv("OPENAI_TTS_INSTRUCTIONS", raising=False)
    monkeypatch.delenv("OPENAI_TTS_SPEED", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.openai_tts_model == "gpt-4o-mini-tts"
    assert settings.openai_tts_voice == "shimmer"
    assert settings.openai_tts_instructions == ""
    assert settings.openai_tts_speed == 1.0


def test_elevenlabs_text_normalization_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без ELEVENLABS_TEXT_NORMALIZATION — штатная нормализация включена (on)."""
    _set_required_env(monkeypatch)
    monkeypatch.delenv("ELEVENLABS_TEXT_NORMALIZATION", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.elevenlabs_text_normalization == "on"


def test_elevenlabs_text_normalization_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ELEVENLABS_TEXT_NORMALIZATION читается из окружения (auto / off)."""
    _set_required_env(monkeypatch)

    monkeypatch.setenv("ELEVENLABS_TEXT_NORMALIZATION", "auto")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.elevenlabs_text_normalization == "auto"

    monkeypatch.setenv("ELEVENLABS_TEXT_NORMALIZATION", "off")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.elevenlabs_text_normalization == "off"


def test_elevenlabs_text_normalization_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Недопустимое значение ELEVENLABS_TEXT_NORMALIZATION роняет валидацию."""
    from pydantic import ValidationError

    _set_required_env(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_TEXT_NORMALIZATION", "yes")

    with pytest.raises(ValidationError, match="ELEVENLABS_TEXT_NORMALIZATION"):
        Settings(_env_file=None)  # type: ignore[call-arg]
