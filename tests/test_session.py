"""Тесты сборки голосовой сессии (без сети и без реальных ключей)."""

from unittest.mock import MagicMock

import pytest

from voice_bot.agent import session as session_module
from voice_bot.config import Settings


def test_build_session_passes_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ключи API явно прокидываются в STT/LLM/TTS из настроек проекта."""
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-123")
    # Перекрыть локальный .env: без прокси клиенты не создаются; дефолты тона.
    monkeypatch.setenv("PROXY_HOST", "")
    monkeypatch.delenv("VOICE_BOT_STT_PROVIDER", raising=False)
    monkeypatch.delenv("ELEVENLABS_STABILITY", raising=False)
    monkeypatch.delenv("ELEVENLABS_SIMILARITY", raising=False)
    monkeypatch.delenv("ELEVENLABS_STYLE", raising=False)
    monkeypatch.delenv("ELEVENLABS_MODEL", raising=False)

    stt_mock = MagicMock(name="STT")
    llm_mock = MagicMock(name="LLM")
    tts_mock = MagicMock(name="TTS")
    vad_load_mock = MagicMock(name="VAD.load")
    turn_model_mock = MagicMock(name="MultilingualModel")
    session_ctor = MagicMock(name="AgentSession", return_value=MagicMock())

    monkeypatch.setattr(session_module.openai, "STT", stt_mock)
    monkeypatch.setattr(session_module.openai, "LLM", llm_mock)
    monkeypatch.setattr(session_module.elevenlabs, "TTS", tts_mock)
    monkeypatch.setattr(session_module.silero.VAD, "load", vad_load_mock)
    monkeypatch.setattr(session_module, "MultilingualModel", turn_model_mock)
    monkeypatch.setattr(session_module, "AgentSession", session_ctor)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    result = session_module.build_session(settings)

    stt_mock.assert_called_once()
    assert stt_mock.call_args.kwargs["api_key"] == "sk-test-openai"
    llm_mock.assert_called_once()
    assert llm_mock.call_args.kwargs["api_key"] == "sk-test-openai"
    tts_mock.assert_called_once()
    assert tts_mock.call_args.kwargs["api_key"] == "el-test-key"
    assert tts_mock.call_args.kwargs["model"] == "eleven_multilingual_v2"
    voice_settings = tts_mock.call_args.kwargs["voice_settings"]
    assert voice_settings.stability == 0.7
    assert voice_settings.similarity_boost == 0.8
    assert voice_settings.style == 0.0
    assert voice_settings.speed == 1.0
    vad_load_mock.assert_called_once_with()
    turn_model_mock.assert_called_once_with()
    session_ctor.assert_called_once()
    turn_handling = session_ctor.call_args.kwargs["turn_handling"]
    assert turn_handling["turn_detection"] is turn_model_mock.return_value
    assert result is session_ctor.return_value


def test_build_session_uses_openai_stt_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """По умолчанию (openai) вызывается openai.STT, build_service_stt — нет."""
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-123")
    monkeypatch.setenv("PROXY_HOST", "")
    monkeypatch.delenv("VOICE_BOT_STT_PROVIDER", raising=False)

    stt_mock = MagicMock(name="STT")
    llm_mock = MagicMock(name="LLM")
    tts_mock = MagicMock(name="TTS")
    vad_instance = MagicMock(name="VAD")
    vad_load_mock = MagicMock(name="VAD.load", return_value=vad_instance)
    turn_model_mock = MagicMock(name="MultilingualModel")
    session_ctor = MagicMock(name="AgentSession", return_value=MagicMock())
    build_service_stt_mock = MagicMock(name="build_service_stt")

    monkeypatch.setattr(session_module.openai, "STT", stt_mock)
    monkeypatch.setattr(session_module.openai, "LLM", llm_mock)
    monkeypatch.setattr(session_module.elevenlabs, "TTS", tts_mock)
    monkeypatch.setattr(session_module.silero.VAD, "load", vad_load_mock)
    monkeypatch.setattr(session_module, "MultilingualModel", turn_model_mock)
    monkeypatch.setattr(session_module, "AgentSession", session_ctor)
    monkeypatch.setattr(session_module, "build_service_stt", build_service_stt_mock)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    session_module.build_session(settings)

    stt_mock.assert_called_once()
    build_service_stt_mock.assert_not_called()
    assert session_ctor.call_args.kwargs["stt"] is stt_mock.return_value
    assert session_ctor.call_args.kwargs["vad"] is vad_instance


def test_build_session_uses_service_stt_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При VOICE_BOT_STT_PROVIDER=service — build_service_stt с тем же VAD."""
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice-123")
    monkeypatch.setenv("PROXY_HOST", "")
    monkeypatch.setenv("VOICE_BOT_STT_PROVIDER", "service")
    monkeypatch.setenv("VOICE_BOT_STT_SERVICE_URL", "http://172.17.0.1:8137")

    stt_mock = MagicMock(name="STT")
    llm_mock = MagicMock(name="LLM")
    tts_mock = MagicMock(name="TTS")
    vad_instance = MagicMock(name="VAD")
    vad_load_mock = MagicMock(name="VAD.load", return_value=vad_instance)
    turn_model_mock = MagicMock(name="MultilingualModel")
    session_ctor = MagicMock(name="AgentSession", return_value=MagicMock())
    service_stt = MagicMock(name="service_stt")
    build_service_stt_mock = MagicMock(name="build_service_stt", return_value=service_stt)
    transcription_client = MagicMock(name="transcription_client")
    build_client_mock = MagicMock(
        name="build_transcription_client", return_value=transcription_client
    )

    monkeypatch.setattr(session_module.openai, "STT", stt_mock)
    monkeypatch.setattr(session_module.openai, "LLM", llm_mock)
    monkeypatch.setattr(session_module.elevenlabs, "TTS", tts_mock)
    monkeypatch.setattr(session_module.silero.VAD, "load", vad_load_mock)
    monkeypatch.setattr(session_module, "MultilingualModel", turn_model_mock)
    monkeypatch.setattr(session_module, "AgentSession", session_ctor)
    monkeypatch.setattr(session_module, "build_service_stt", build_service_stt_mock)
    monkeypatch.setattr(session_module, "build_transcription_client", build_client_mock)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    session_module.build_session(settings)

    stt_mock.assert_not_called()
    build_client_mock.assert_called_once_with(
        base_url="http://172.17.0.1:8137",
        timeout=15.0,
    )
    build_service_stt_mock.assert_called_once()
    assert build_service_stt_mock.call_args.kwargs["client"] is transcription_client
    assert build_service_stt_mock.call_args.kwargs["vad"] is vad_instance
    assert build_service_stt_mock.call_args.kwargs["language"] == "ru"
    assert session_ctor.call_args.kwargs["stt"] is service_stt
    assert session_ctor.call_args.kwargs["vad"] is vad_instance
