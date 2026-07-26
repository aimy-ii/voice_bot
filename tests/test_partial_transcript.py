"""Офлайн-тесты фоновой отправки накопленного STT на вторую точку входа."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit.agents import UserInputTranscribedEvent, UserStateChangedEvent

from voice_bot.agent import main as main_module
from voice_bot.agent import session as session_module
from voice_bot.agent.session import (
    PartialTranscriptSender,
    live_thread_id_for_room,
    thread_id_for_room,
)
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
    room = "room-partial"
    sender = PartialTranscriptSender(
        client=client,
        graph="vector_checker",
        thread_id=live_thread_id_for_room(room),
        call_id=thread_id_for_room(room),
    )
    return sender, runs_create


async def _drain(sender: PartialTranscriptSender) -> None:
    """Дождаться фоновых задач отправки."""
    if sender._tasks:
        await asyncio.gather(*list(sender._tasks), return_exceptions=True)


def test_live_thread_id_for_room_returns_valid_uuid() -> None:
    """Лайв-тред — валидный UUID, принимаемый LangGraph Server."""
    live_id = live_thread_id_for_room("voice_assistant_room_5285")
    parsed = uuid.UUID(live_id)
    assert str(parsed) == live_id


def test_live_thread_id_differs_from_main_thread() -> None:
    """Лайв-тред и основной тред одной комнаты не совпадают."""
    room = "voice_assistant_room_42"
    assert live_thread_id_for_room(room) != thread_id_for_room(room)


def test_live_thread_id_is_deterministic_and_differs_by_room() -> None:
    """Одна комната — один лайв-тред; разные комнаты — разные."""
    room = "room-stable"
    assert live_thread_id_for_room(room) == live_thread_id_for_room(room)
    assert live_thread_id_for_room("room-a") != live_thread_id_for_room("room-b")


def test_live_thread_id_rejects_empty() -> None:
    """Пустое имя комнаты для лайв-треда — ValueError."""
    with pytest.raises(ValueError):
        live_thread_id_for_room("")
    with pytest.raises(ValueError):
        live_thread_id_for_room("   ")


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
async def test_partial_send_payload_graph_and_field() -> None:
    """``_send`` уходит на ``vector_checker`` с текстом в ``partial_reply``."""
    sender, runs_create = _make_sender()

    sender.on_transcript(UserInputTranscribedEvent(transcript="проверка", is_final=False))
    await _drain(sender)

    assert runs_create.await_count == 1
    kwargs = runs_create.await_args.kwargs
    assert kwargs["assistant_id"] == "vector_checker"
    payload = kwargs["input"]
    assert payload["partial_reply"] == "проверка"
    assert "partial_utterance_id" in payload
    assert payload["partial_is_final"] is False
    assert kwargs["multitask_strategy"] == "interrupt"
    assert kwargs["if_not_exists"] == "create"
    assert kwargs["config"] == {"configurable": {"call_id": sender.call_id}}


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
    sent = [call.kwargs["input"]["partial_reply"] for call in runs_create.await_args_list]
    assert sent == ["привет", "привет как", "привет как дела"]
    # Не дельта: последний вызов — полная фраза, не « дела».
    assert sent[-1] == "привет как дела"
    assert sent[-1] != " дела"
    # Каждый служебный run прерывает предыдущий.
    assert all(c.kwargs["multitask_strategy"] == "interrupt" for c in runs_create.await_args_list)


@pytest.mark.asyncio
async def test_partial_combines_final_segments_with_interim() -> None:
    """После финального сегмента interim добавляется к накопленному, не заменяет."""
    sender, runs_create = _make_sender()

    sender.on_transcript(UserInputTranscribedEvent(transcript="меня зовут", is_final=True))
    await _drain(sender)
    sender.on_transcript(UserInputTranscribedEvent(transcript="Иван", is_final=False))
    await _drain(sender)

    sent = [call.kwargs["input"]["partial_reply"] for call in runs_create.await_args_list]
    assert sent == ["меня зовут", "меня зовут Иван"]


@pytest.mark.asyncio
async def test_partial_sends_on_final_transcript() -> None:
    """``is_final=True`` — отправка происходит."""
    sender, runs_create = _make_sender()

    sender.on_transcript(UserInputTranscribedEvent(transcript="финальная реплика", is_final=True))
    await _drain(sender)

    assert runs_create.await_count == 1
    assert runs_create.await_args.kwargs["input"]["partial_reply"] == "финальная реплика"


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
async def test_partial_uses_live_thread_and_call_id() -> None:
    """Служебный run — на лайв-треде; ``call_id`` равен треду основного хода."""
    room = "voice_assistant_room_42"
    main_thread = thread_id_for_room(room)
    live_thread = live_thread_id_for_room(room)
    settings = _settings(
        VOICE_BOT_AGENT_PARTIAL_ENABLED="true",
        VOICE_BOT_AGENT_PARTIAL_URL="http://agent.test:8127",
        VOICE_BOT_AGENT_PARTIAL_GRAPH="vector_checker",
    )

    with patch.object(session_module.httpx, "AsyncClient", return_value=MagicMock()):
        sender = session_module.build_partial_transcript_sender(settings=settings, room_name=room)

    assert sender.thread_id == live_thread
    assert sender.call_id == main_thread
    assert sender.thread_id != sender.call_id

    runs_create = AsyncMock(return_value={})
    sender._client.runs.create = runs_create
    sender.on_transcript(UserInputTranscribedEvent(transcript="ок", is_final=False))
    await _drain(sender)

    kwargs = runs_create.await_args.kwargs
    assert kwargs["thread_id"] == live_thread
    assert kwargs["thread_id"] != main_thread
    assert kwargs["config"] == {"configurable": {"call_id": main_thread}}
    assert kwargs["assistant_id"] == "vector_checker"
    assert kwargs["input"]["partial_reply"] == "ок"
    assert "partial_utterance_id" in kwargs["input"]
    assert kwargs["input"]["partial_is_final"] is False
    assert kwargs["multitask_strategy"] == "interrupt"


@pytest.mark.asyncio
async def test_partial_keeps_sending_while_agent_thinking() -> None:
    """Переход агента в thinking не останавливает лайв-отправку."""
    sender, runs_create = _make_sender()

    sender.on_transcript(UserInputTranscribedEvent(transcript="пока говорю", is_final=False))
    await _drain(sender)
    assert runs_create.await_count == 1

    # Подписки и обработчика agent_state больше нет — thinking не глушит канал.
    assert not hasattr(sender, "on_agent_state")
    sender.on_transcript(UserInputTranscribedEvent(transcript="уже думает", is_final=False))
    await _drain(sender)
    assert runs_create.await_count == 2
    assert runs_create.await_args.kwargs["input"]["partial_reply"] == "уже думает"

    # Новая реплика клиента по-прежнему сбрасывает буфер.
    sender.on_user_state(UserStateChangedEvent(old_state="listening", new_state="speaking"))
    sender.on_transcript(UserInputTranscribedEvent(transcript="новая фраза", is_final=False))
    await _drain(sender)
    assert runs_create.await_count == 3
    assert runs_create.await_args.kwargs["input"]["partial_reply"] == "новая фраза"


@pytest.mark.asyncio
async def test_partial_payload_has_utterance_id_and_is_final() -> None:
    """В полезной нагрузке есть ``partial_utterance_id`` и ``partial_is_final``."""
    sender, runs_create = _make_sender()

    sender.on_transcript(UserInputTranscribedEvent(transcript="привет", is_final=False))
    await _drain(sender)

    payload = runs_create.await_args.kwargs["input"]
    assert isinstance(payload["partial_utterance_id"], str)
    assert payload["partial_utterance_id"]
    uuid.UUID(payload["partial_utterance_id"])
    assert payload["partial_is_final"] is False


@pytest.mark.asyncio
async def test_partial_utterance_id_stable_within_utterance() -> None:
    """Внутри одной реплики идентификатор не меняется."""
    sender, runs_create = _make_sender()

    sender.on_transcript(UserInputTranscribedEvent(transcript="Впервые.", is_final=False))
    await _drain(sender)
    sender.on_transcript(UserInputTranscribedEvent(transcript="Впервые. да", is_final=False))
    await _drain(sender)

    ids = [c.kwargs["input"]["partial_utterance_id"] for c in runs_create.await_args_list]
    assert len(ids) == 2
    assert ids[0] == ids[1]


@pytest.mark.asyncio
async def test_partial_utterance_id_changes_on_new_utterance() -> None:
    """На новой реплике (сброс в ``on_user_state``) идентификатор другой."""
    sender, runs_create = _make_sender()

    sender.on_transcript(UserInputTranscribedEvent(transcript="Впервые.", is_final=True))
    await _drain(sender)
    first_id = runs_create.await_args.kwargs["input"]["partial_utterance_id"]

    sender.on_user_state(UserStateChangedEvent(old_state="listening", new_state="speaking"))
    sender.on_transcript(UserInputTranscribedEvent(transcript="Автомат.", is_final=True))
    await _drain(sender)
    second_id = runs_create.await_args.kwargs["input"]["partial_utterance_id"]

    assert first_id != second_id
    assert runs_create.await_count == 2


@pytest.mark.asyncio
async def test_partial_is_final_flag_matches_transcript() -> None:
    """Финальный текст — ``partial_is_final=True``, промежуточный — ``False``."""
    sender, runs_create = _make_sender()

    sender.on_transcript(UserInputTranscribedEvent(transcript="промежуток", is_final=False))
    await _drain(sender)
    assert runs_create.await_args.kwargs["input"]["partial_is_final"] is False

    sender.on_transcript(UserInputTranscribedEvent(transcript="финал", is_final=True))
    await _drain(sender)
    assert runs_create.await_args.kwargs["input"]["partial_is_final"] is True


def test_attach_partial_when_enabled_subscribes() -> None:
    """При включённом флаге отправитель вешается на события сессии."""
    settings = _settings(
        VOICE_BOT_AGENT_PARTIAL_ENABLED="true",
        VOICE_BOT_AGENT_PARTIAL_URL="http://agent.test:8127",
        VOICE_BOT_AGENT_PARTIAL_GRAPH="vector_checker",
    )
    session = MagicMock()
    fake_sender = MagicMock()
    fake_sender.thread_id = live_thread_id_for_room("room-x")
    fake_sender.call_id = thread_id_for_room("room-x")

    with patch.object(
        main_module, "build_partial_transcript_sender", return_value=fake_sender
    ) as build_mock:
        main_module._attach_partial_transcript_sender(
            session, settings=settings, room_name="room-x"
        )

    build_mock.assert_called_once()
    assert build_mock.call_args.kwargs["room_name"] == "room-x"
    fake_sender.attach.assert_called_once_with(session)
