"""Тесты сборки голосовой сессии (без сети и без реальных ключей)."""

from unittest.mock import MagicMock
from uuid import UUID

import pytest
from livekit.plugins import langchain

from voice_bot.agent import session as session_module
from voice_bot.config import Settings


def _set_session_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Обязательные переменные для сборки сессии в офлайн-тестах."""
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-123")
    monkeypatch.setenv("PROXY_HOST", "")
    monkeypatch.delenv("VOICE_BOT_STT_PROVIDER", raising=False)
    monkeypatch.delenv("VOICE_BOT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("VOICE_BOT_AGENT_URL", raising=False)
    monkeypatch.delenv("VOICE_BOT_AGENT_GRAPH", raising=False)
    monkeypatch.delenv("VOICE_BOT_AGENT_TIMEOUT", raising=False)


def _patch_common_providers(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Подменить STT/TTS/VAD/сессию заглушками; вернуть моки для ассертов."""
    stt_mock = MagicMock(name="STT")
    llm_mock = MagicMock(name="LLM")
    tts_mock = MagicMock(name="TTS")
    vad_instance = MagicMock(name="VAD")
    vad_load_mock = MagicMock(name="VAD.load", return_value=vad_instance)
    turn_model_mock = MagicMock(name="MultilingualModel")
    session_ctor = MagicMock(name="AgentSession", return_value=MagicMock())

    monkeypatch.setattr(session_module.openai, "STT", stt_mock)
    monkeypatch.setattr(session_module.openai, "LLM", llm_mock)
    monkeypatch.setattr(session_module.elevenlabs, "TTS", tts_mock)
    monkeypatch.setattr(session_module.silero.VAD, "load", vad_load_mock)
    monkeypatch.setattr(session_module, "MultilingualModel", turn_model_mock)
    monkeypatch.setattr(session_module, "AgentSession", session_ctor)

    return {
        "stt": stt_mock,
        "llm": llm_mock,
        "tts": tts_mock,
        "vad_load": vad_load_mock,
        "vad": vad_instance,
        "turn_model": turn_model_mock,
        "session_ctor": session_ctor,
    }


def test_build_session_passes_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ключи API явно прокидываются в STT/LLM/TTS из настроек проекта."""
    _set_session_env(monkeypatch)
    monkeypatch.delenv("ELEVENLABS_STABILITY", raising=False)
    monkeypatch.delenv("ELEVENLABS_SIMILARITY", raising=False)
    monkeypatch.delenv("ELEVENLABS_STYLE", raising=False)
    monkeypatch.delenv("ELEVENLABS_MODEL", raising=False)

    mocks = _patch_common_providers(monkeypatch)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    result = session_module.build_session(settings)

    mocks["stt"].assert_called_once()
    assert mocks["stt"].call_args.kwargs["api_key"] == "sk-test-openai"
    mocks["llm"].assert_called_once()
    assert mocks["llm"].call_args.kwargs["api_key"] == "sk-test-openai"
    mocks["tts"].assert_called_once()
    assert mocks["tts"].call_args.kwargs["api_key"] == "el-test-key"
    assert mocks["tts"].call_args.kwargs["model"] == "eleven_multilingual_v2"
    voice_settings = mocks["tts"].call_args.kwargs["voice_settings"]
    assert voice_settings.stability == 0.7
    assert voice_settings.similarity_boost == 0.8
    assert voice_settings.style == 0.0
    assert voice_settings.speed == 1.0
    mocks["vad_load"].assert_called_once_with()
    mocks["turn_model"].assert_called_once_with()
    mocks["session_ctor"].assert_called_once()
    turn_handling = mocks["session_ctor"].call_args.kwargs["turn_handling"]
    assert turn_handling["turn_detection"] is mocks["turn_model"].return_value
    assert result is mocks["session_ctor"].return_value


def test_build_session_uses_openai_stt_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """По умолчанию (openai) вызывается openai.STT, build_service_stt — нет."""
    _set_session_env(monkeypatch)

    mocks = _patch_common_providers(monkeypatch)
    build_service_stt_mock = MagicMock(name="build_service_stt")
    monkeypatch.setattr(session_module, "build_service_stt", build_service_stt_mock)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    session_module.build_session(settings)

    mocks["stt"].assert_called_once()
    build_service_stt_mock.assert_not_called()
    assert mocks["session_ctor"].call_args.kwargs["stt"] is mocks["stt"].return_value
    assert mocks["session_ctor"].call_args.kwargs["vad"] is mocks["vad"]


def test_build_session_uses_service_stt_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При VOICE_BOT_STT_PROVIDER=service — build_service_stt с тем же VAD."""
    _set_session_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_STT_PROVIDER", "service")
    monkeypatch.setenv("VOICE_BOT_STT_SERVICE_URL", "http://172.17.0.1:8137")

    mocks = _patch_common_providers(monkeypatch)
    service_stt = MagicMock(name="service_stt")
    build_service_stt_mock = MagicMock(name="build_service_stt", return_value=service_stt)
    transcription_client = MagicMock(name="transcription_client")
    build_client_mock = MagicMock(
        name="build_transcription_client", return_value=transcription_client
    )

    monkeypatch.setattr(session_module, "build_service_stt", build_service_stt_mock)
    monkeypatch.setattr(session_module, "build_transcription_client", build_client_mock)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    session_module.build_session(settings)

    mocks["stt"].assert_not_called()
    build_client_mock.assert_called_once_with(
        base_url="http://172.17.0.1:8137",
        timeout=15.0,
    )
    build_service_stt_mock.assert_called_once()
    assert build_service_stt_mock.call_args.kwargs["client"] is transcription_client
    assert build_service_stt_mock.call_args.kwargs["vad"] is mocks["vad"]
    assert build_service_stt_mock.call_args.kwargs["language"] == "ru"
    assert mocks["session_ctor"].call_args.kwargs["stt"] is service_stt
    assert mocks["session_ctor"].call_args.kwargs["vad"] is mocks["vad"]


def test_build_session_uses_openai_llm_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """По умолчанию (openai) в сессии стоит openai.LLM — старый путь цел."""
    _set_session_env(monkeypatch)
    mocks = _patch_common_providers(monkeypatch)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    session_module.build_session(settings)

    mocks["llm"].assert_called_once()
    assert mocks["session_ctor"].call_args.kwargs["llm"] is mocks["llm"].return_value
    assert settings.llm_provider == "openai"


def test_build_session_uses_agent_llm_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """При llm_provider=agent в сессии стоит LLMAdapter с нужными параметрами."""
    _set_session_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_LLM_PROVIDER", "agent")
    monkeypatch.setenv("VOICE_BOT_AGENT_URL", "http://agent.test:8127")
    monkeypatch.setenv("VOICE_BOT_AGENT_GRAPH", "my_custom_graph")

    mocks = _patch_common_providers(monkeypatch)
    remote_graph = MagicMock(name="RemoteGraphInstance")
    remote_ctor = MagicMock(name="RemoteGraph", return_value=remote_graph)
    http_client = MagicMock(name="AsyncClient")
    async_client_ctor = MagicMock(name="httpx.AsyncClient", return_value=http_client)
    monkeypatch.setattr(session_module, "RemoteGraph", remote_ctor)
    monkeypatch.setattr(session_module.httpx, "AsyncClient", async_client_ctor)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    session_module.build_session(settings, room_name="room-alpha")

    mocks["llm"].assert_not_called()
    llm = mocks["session_ctor"].call_args.kwargs["llm"]
    assert isinstance(llm, langchain.LLMAdapter)
    assert isinstance(llm._stream_mode, str)
    assert llm._stream_mode == "custom"
    assert llm._subgraphs is False
    # В config лежит преобразованный UUID, а не сырое имя комнаты.
    thread_id = llm._config["configurable"]["thread_id"]
    assert thread_id == session_module.thread_id_for_room("room-alpha")
    assert thread_id != "room-alpha"
    assert UUID(thread_id)  # валидный UUID для LangGraph Server
    assert llm._graph is remote_graph

    remote_ctor.assert_called_once()
    assert remote_ctor.call_args.args[0] == "my_custom_graph"
    async_client_ctor.assert_called_once()
    assert async_client_ctor.call_args.kwargs["base_url"] == "http://agent.test:8127"
    assert async_client_ctor.call_args.kwargs["trust_env"] is False


def test_build_session_agent_thread_id_per_room(monkeypatch: pytest.MonkeyPatch) -> None:
    """Две сессии с разными комнатами получают разные UUID thread_id."""
    _set_session_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_LLM_PROVIDER", "agent")

    mocks = _patch_common_providers(monkeypatch)
    remote_ctor = MagicMock(name="RemoteGraph", return_value=MagicMock())
    monkeypatch.setattr(session_module, "RemoteGraph", remote_ctor)
    monkeypatch.setattr(
        session_module.httpx,
        "AsyncClient",
        MagicMock(return_value=MagicMock()),
    )

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    session_module.build_session(settings, room_name="room-one")
    session_module.build_session(settings, room_name="room-two")

    llm_one = mocks["session_ctor"].call_args_list[0].kwargs["llm"]
    llm_two = mocks["session_ctor"].call_args_list[1].kwargs["llm"]
    tid_one = llm_one._config["configurable"]["thread_id"]
    tid_two = llm_two._config["configurable"]["thread_id"]

    assert tid_one == session_module.thread_id_for_room("room-one")
    assert tid_two == session_module.thread_id_for_room("room-two")
    assert tid_one != "room-one"
    assert tid_two != "room-two"
    assert tid_one != tid_two
    assert UUID(tid_one)
    assert UUID(tid_two)


def test_thread_id_for_room_returns_valid_uuid() -> None:
    """Результат — валидный UUID (штатный парсер, не длина строки)."""
    thread_id = session_module.thread_id_for_room("voice_assistant_room_5285")
    parsed = UUID(thread_id)
    assert str(parsed) == thread_id


def test_thread_id_for_room_is_deterministic() -> None:
    """Одна комната — всегда один и тот же thread_id."""
    room = "voice_assistant_room_5285"
    assert session_module.thread_id_for_room(room) == session_module.thread_id_for_room(room)


def test_thread_id_for_room_differs_by_room() -> None:
    """Разные имена комнат дают разные thread_id."""
    assert session_module.thread_id_for_room("room-a") != session_module.thread_id_for_room(
        "room-b"
    )


def test_thread_id_for_room_rejects_empty() -> None:
    """Пустое или пробельное имя комнаты — понятная ошибка."""
    with pytest.raises(ValueError, match="room_name"):
        session_module.thread_id_for_room("")
    with pytest.raises(ValueError, match="room_name"):
        session_module.thread_id_for_room("   ")


def test_build_session_agent_requires_room_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """При llm_provider=agent без room_name — понятная ошибка."""
    _set_session_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_LLM_PROVIDER", "agent")
    _patch_common_providers(monkeypatch)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="room_name"):
        session_module.build_session(settings)


def test_build_session_agent_rejects_empty_room_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пустое имя комнаты при agent — ошибка на сборке сессии."""
    _set_session_env(monkeypatch)
    monkeypatch.setenv("VOICE_BOT_LLM_PROVIDER", "agent")
    _patch_common_providers(monkeypatch)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="room_name"):
        session_module.build_session(settings, room_name="")
    with pytest.raises(ValueError, match="room_name"):
        session_module.build_session(settings, room_name="   ")


def test_build_session_openai_does_not_create_llm_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Путь через OpenAI не создаёт LLMAdapter — старое поведение цело."""
    _set_session_env(monkeypatch)
    mocks = _patch_common_providers(monkeypatch)
    adapter_ctor = MagicMock(name="LLMAdapter")
    monkeypatch.setattr(session_module.langchain, "LLMAdapter", adapter_ctor)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    session_module.build_session(settings, room_name="room-ignored")

    mocks["llm"].assert_called_once()
    adapter_ctor.assert_not_called()
    assert mocks["session_ctor"].call_args.kwargs["llm"] is mocks["llm"].return_value
