"""Офлайн-тесты старта звонка: кто произносит вступление (мозг vs сценарий)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_bot.agent import main as main_module
from voice_bot.config import Settings


def _settings(**overrides: object) -> Settings:
    """Минимальные Settings для теста ``entrypoint``."""
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


def _scenario() -> SimpleNamespace:
    """Лёгкий сценарий с opening_line и id для лога."""
    return SimpleNamespace(id="vector_ru", opening_line="Привет из сценария бота")


def _ctx() -> SimpleNamespace:
    """Контекст задания с комнатой."""
    return SimpleNamespace(room=SimpleNamespace(name="room-test-1"))


@pytest.mark.asyncio
async def test_entrypoint_agent_skips_opening_line_and_starts_graph() -> None:
    """При llm_provider=agent opening_line не говорится — зовётся первый ход графа."""
    settings = _settings(VOICE_BOT_LLM_PROVIDER="agent")
    scenario = _scenario()
    session = MagicMock()
    session.start = AsyncMock()
    session.say = AsyncMock()
    session.generate_reply = AsyncMock()

    with (
        patch.object(main_module, "get_settings", return_value=settings),
        patch.object(main_module, "load_scenario", return_value=scenario) as load_mock,
        patch.object(main_module, "build_session", return_value=session),
        patch.object(main_module, "_attach_latency_logging"),
        patch.object(main_module, "_attach_partial_transcript_sender"),
        patch.object(main_module, "_start_background_audio", new_callable=AsyncMock),
        patch.object(main_module, "ScriptAgent"),
        patch.object(main_module, "build_system_prompt") as prompt_mock,
    ):
        await main_module.entrypoint(_ctx())  # type: ignore[arg-type]

    load_mock.assert_called_once_with(settings.scenario)
    prompt_mock.assert_not_called()
    session.say.assert_not_called()
    session.generate_reply.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_entrypoint_openai_says_opening_line_literally() -> None:
    """При llm_provider=openai вступление произносится дословно из сценария."""
    settings = _settings(VOICE_BOT_LLM_PROVIDER="openai")
    scenario = _scenario()
    session = MagicMock()
    session.start = AsyncMock()
    session.say = AsyncMock()
    session.generate_reply = AsyncMock()

    with (
        patch.object(main_module, "get_settings", return_value=settings),
        patch.object(main_module, "load_scenario", return_value=scenario),
        patch.object(main_module, "build_session", return_value=session),
        patch.object(main_module, "_attach_latency_logging"),
        patch.object(main_module, "_attach_partial_transcript_sender"),
        patch.object(main_module, "_start_background_audio", new_callable=AsyncMock),
        patch.object(main_module, "ScriptAgent"),
        patch.object(main_module, "build_system_prompt", return_value="prompt") as prompt_mock,
    ):
        await main_module.entrypoint(_ctx())  # type: ignore[arg-type]

    prompt_mock.assert_called_once_with(scenario)
    session.say.assert_awaited_once_with(
        scenario.opening_line,
        allow_interruptions=False,
    )
    session.generate_reply.assert_not_called()
