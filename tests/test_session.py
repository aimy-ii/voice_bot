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
    # Перекрыть локальный .env: без прокси клиенты не создаются.
    monkeypatch.setenv("PROXY_HOST", "")

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

    settings = Settings()  # type: ignore[call-arg]
    result = session_module.build_session(settings)

    stt_mock.assert_called_once()
    assert stt_mock.call_args.kwargs["api_key"] == "sk-test-openai"
    llm_mock.assert_called_once()
    assert llm_mock.call_args.kwargs["api_key"] == "sk-test-openai"
    tts_mock.assert_called_once()
    assert tts_mock.call_args.kwargs["api_key"] == "el-test-key"
    vad_load_mock.assert_called_once_with()
    turn_model_mock.assert_called_once_with()
    session_ctor.assert_called_once()
    turn_handling = session_ctor.call_args.kwargs["turn_handling"]
    assert turn_handling["turn_detection"] is turn_model_mock.return_value
    assert result is session_ctor.return_value
