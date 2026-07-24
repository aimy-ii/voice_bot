"""Офлайн-тесты фоновой отправки накопленного STT на вторую точку входа."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit.agents import (
    AgentStateChangedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)

from voice_bot.agent import main as main_module
from voice_bot.agent import session as session_module
from voice_bot.agent.session import PartialTranscriptSender, thread_id_for_room
from voice_bot.config import Settings


def _settings(**overrides: object) -> Settings:
    """Минимальные Settings для тестов partial-отправки."""
    base: dict[str, object] = {
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


def _make_sender(*, create: AsyncMock | None = None) -> tuple[PartialTranscriptSender, AsyncMock]:
    """Собрать отправитель с заглушкой ``runs.create``."""
    runs_create = create or AsyncMock(return_value={"run_id": "r1"})
    client = MagicMock()
    client.runs.create = runs_create
    sender = PartialTranscriptSender(
        client=client,
        graph="vector_partial",
        thread_id=thread_id_for_room("room-partial"),
    )
    return sender, runs_create


async def _drain(sender: PartialTranscriptSender) -> None:
    """Дождаться фоновых задач отправки."""
    if sender._tasks:
        await asyncio.gather(*list(sender._tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_partial_disabled_attach_does_nothing() -> None:
    """При выключенном флаге отправитель не создаётся и сессия не трогается."""
    settings = _settings(VOICE_BOT_AGENT_PARTIAL_ENABLED="false")
    session = MagicMock()

    with patch.object(main_module, "build_partial_transcript_sender") as build_mock:
        main_module._attach_partial_transcript_sender(
            session, settings=settings, room_name="room-a"
        )

    build_mock.assert_not_called()
    session.on.assert_not_called()


@pytest.mark.asyncio
async def test_partial_sends_accumulated_text_not_delta() -> None:
    """Уходит накопленный текст целиком, а не приращение к прошлому куску."""
    sender, runs_create = _make_sender()

    sender.on_transcript(UserInputTranscribedEvent(transcript="привет", is_final=False))
    await _drain(sender)
    sender.on_transcript(UserInputTranscribedEvent(transcript="привет как", is_final=False))
    await _drain(sender)
    sender.on_transcript(UserInputTranscribedEvent(transcript="привет как дела", is_final=False))
    await _drain(sender)

    assert runs_create.await_count == 3
    sent = [call.kwargs["input"]["messages"][0]["content"] for call in runs_create.await_args_list]
    assert sent == ["привет", "привет как", "привет как дела"]
    # Не дельта: последний вызов — полная фраза, не « дела».
    assert sent[-1] == "привет как дела"
    assert sent[-1] != " дела"


@pytest.mark.asyncio
async def test_partial_combines_final_segments_with_interim() -> None:
    """После финального сегмента interim добавляется к накопленному, не заменяет."""
    sender, runs_create = _make_sender()

    sender.on_transcript(UserInputTranscribedEvent(transcript="меня зовут", is_final=True))
    await _drain(sender)
    sender.on_transcript(UserInputTranscribedEvent(transcript="Иван", is_final=False))
    await _drain(sender)

    sent = [call.kwargs["input"]["messages"][0]["content"] for call in runs_create.await_args_list]
    assert sent == ["меня зовут", "меня зовут Иван"]


@pytest.mark.asyncio
async def test_partial_send_error_does_not_block_or_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ошибка/таймаут отправки не роняет ход и не блокирует обработчик."""
    slow = asyncio.Event()

    async def _hang(**_kwargs: object) -> None:
        await slow.wait()
        raise TimeoutError("agent partial timeout")

    sender, _ = _make_sender(create=AsyncMock(side_effect=_hang))

    # Обработчик возвращается сразу, не дожидаясь HTTP.
    sender.on_transcript(UserInputTranscribedEvent(transcript="раз", is_final=False))
    assert len(sender._tasks) == 1

    # Вторая реплика тоже принимается, пока первая «висит».
    sender.on_transcript(UserInputTranscribedEvent(transcript="раз два", is_final=False))
    assert len(sender._tasks) == 2

    slow.set()
    await _drain(sender)

    with caplog.at_level("WARNING", logger="voice_bot.partial"):
        # Дочищаем done-callback логирование.
        await asyncio.sleep(0)

    assert any(
        "partial" in r.message.lower() or "TimeoutError" in r.message for r in caplog.records
    )


@pytest.mark.asyncio
async def test_partial_thread_id_matches_main_turn() -> None:
    """``thread_id`` отправки совпадает с ``thread_id`` основного хода."""
    room = "voice_assistant_room_42"
    expected = thread_id_for_room(room)
    settings = _settings(
        VOICE_BOT_AGENT_PARTIAL_ENABLED="true",
        VOICE_BOT_AGENT_PARTIAL_URL="http://agent.test:8127",
        VOICE_BOT_AGENT_PARTIAL_GRAPH="vector_partial",
    )

    with patch.object(session_module.httpx, "AsyncClient", return_value=MagicMock()):
        sender = session_module.build_partial_transcript_sender(settings=settings, room_name=room)

    assert sender.thread_id == expected

    # Тот же UUID, что кладётся в LLMAdapter при llm_provider=agent.
    runs_create = AsyncMock(return_value={})
    sender._client.runs.create = runs_create
    sender.on_transcript(UserInputTranscribedEvent(transcript="ок", is_final=False))
    await _drain(sender)

    assert runs_create.await_args.kwargs["thread_id"] == expected
    assert runs_create.await_args.kwargs["assistant_id"] == "vector_partial"


@pytest.mark.asyncio
async def test_partial_stops_after_main_turn_starts() -> None:
    """После перехода агента в thinking отправка кусков прекращается."""
    sender, runs_create = _make_sender()

    sender.on_transcript(UserInputTranscribedEvent(transcript="пока говорю", is_final=False))
    await _drain(sender)
    assert runs_create.await_count == 1

    sender.on_agent_state(AgentStateChangedEvent(old_state="listening", new_state="thinking"))
    sender.on_transcript(UserInputTranscribedEvent(transcript="уже поздно", is_final=False))
    await _drain(sender)
    assert runs_create.await_count == 1

    # Новая реплика клиента снова включает отправку.
    sender.on_user_state(UserStateChangedEvent(old_state="listening", new_state="speaking"))
    sender.on_transcript(UserInputTranscribedEvent(transcript="новая фраза", is_final=False))
    await _drain(sender)
    assert runs_create.await_count == 2
    assert runs_create.await_args.kwargs["input"]["messages"][0]["content"] == "новая фраза"


def test_attach_partial_when_enabled_subscribes() -> None:
    """При включённом флаге отправитель вешается на события сессии."""
    settings = _settings(
        VOICE_BOT_AGENT_PARTIAL_ENABLED="true",
        VOICE_BOT_AGENT_PARTIAL_URL="http://agent.test:8127",
        VOICE_BOT_AGENT_PARTIAL_GRAPH="vector_partial",
    )
    session = MagicMock()
    fake_sender = MagicMock()
    fake_sender.thread_id = thread_id_for_room("room-x")

    with patch.object(
        main_module, "build_partial_transcript_sender", return_value=fake_sender
    ) as build_mock:
        main_module._attach_partial_transcript_sender(
            session, settings=settings, room_name="room-x"
        )

    build_mock.assert_called_once()
    assert build_mock.call_args.kwargs["room_name"] == "room-x"
    fake_sender.attach.assert_called_once_with(session)
