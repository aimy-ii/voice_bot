"""Офлайн-тесты подключения фонового звука (без реального аудио-движка)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_bot.agent.main import _start_background_audio
from voice_bot.config import Settings


def _settings(**overrides: object) -> Settings:
    """Минимальные Settings для теста хелпера фона."""
    base = {
        "LIVEKIT_URL": "ws://localhost:7880",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "secret",
        "OPENAI_API_KEY": "sk-test",
        "ELEVENLABS_API_KEY": "el-test",
        "ELEVENLABS_VOICE_ID": "voice-123",
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_start_background_audio_skipped_when_disabled() -> None:
    """При BG_ENABLED=false плеер не создаётся."""
    settings = _settings(BG_ENABLED="false")
    room = MagicMock()
    session = MagicMock()

    with patch("voice_bot.agent.main.BackgroundAudioPlayer") as player_cls:
        result = await _start_background_audio(room=room, session=session, settings=settings)

    assert result is None
    player_cls.assert_not_called()


@pytest.mark.asyncio
async def test_start_background_audio_starts_player() -> None:
    """При включённом фоне плеер стартует с room и agent_session."""
    settings = _settings(BG_AMBIENT_VOLUME="0.1", BG_THINKING_VOLUME="0.5")
    room = MagicMock()
    session = MagicMock()
    player = SimpleNamespace(start=AsyncMock())

    with patch("voice_bot.agent.main.BackgroundAudioPlayer", return_value=player) as player_cls:
        result = await _start_background_audio(room=room, session=session, settings=settings)

    assert result is player
    player_cls.assert_called_once()
    kwargs = player_cls.call_args.kwargs
    assert kwargs["ambient_sound"].volume == 0.1
    assert kwargs["thinking_sound"].volume == 0.5
    player.start.assert_awaited_once_with(room=room, agent_session=session)
