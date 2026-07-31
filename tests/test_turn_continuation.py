"""Офлайн-тесты продолжения речи бота и окликов при user_state=away."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_bot.agent import main as main_module
from voice_bot.agent import session as session_module
from voice_bot.config import Settings


def _settings(**overrides: object) -> Settings:
    """Минимальные Settings для тестов продолжения и тишины."""
    base: dict[str, object] = {
        "LIVEKIT_URL": "ws://localhost:7880",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "secret",
        "OPENAI_API_KEY": "sk-test",
        "ELEVENLABS_API_KEY": "el-test",
        "ELEVENLABS_VOICE_ID": "voice-123",
        "VOICE_BOT_SILENCE_TIMEOUT": 0,
        "VOICE_BOT_MAX_CONTINUATIONS": 3,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[call-arg]


def _fake_session() -> MagicMock:
    """Сессия с ``say`` / ``generate_reply`` и LLM-конфигом под turn_kind."""
    session = MagicMock()
    session.say = MagicMock(
        side_effect=lambda *_a, **_k: SimpleNamespace(wait_for_playout=AsyncMock())
    )
    session.generate_reply = AsyncMock()
    session.llm = SimpleNamespace(
        config={"configurable": {"thread_id": "tid", "turn_kind": "client"}}
    )
    # set_turn_kind читает _config
    session.llm._config = session.llm.config  # type: ignore[attr-defined]
    session.on = MagicMock(return_value=lambda fn: fn)
    session.agent_state = "listening"
    session.user_state = "listening"
    return session


def _controller(
    *,
    session: MagicMock | None = None,
    settings: Settings | None = None,
    lg_client: object | None = None,
    thread_id: str | None = "thread-1",
) -> main_module.CallTurnController:
    """Собрать контроллер с фейковой сессией и нулевым таймаутом тишины."""
    sess = session or _fake_session()
    cfg = settings or _settings()
    ctx = SimpleNamespace(delete_room=MagicMock())
    return main_module.CallTurnController(
        session=sess,
        ctx=ctx,  # type: ignore[arg-type]
        settings=cfg,
        thread_id=thread_id,
        lg_client=lg_client,
    )


# --- expects_continuation -------------------------------------------------


@pytest.mark.asyncio
async def test_expects_continuation_true_when_flag_set() -> None:
    """Флаг expect_continuation=True в state → True."""
    client = MagicMock()
    client.threads.get_state = AsyncMock(return_value={"values": {"expect_continuation": True}})

    assert await session_module.expects_continuation(client, "tid") is True
    client.threads.get_state.assert_awaited_once_with("tid")


@pytest.mark.asyncio
async def test_expects_continuation_false_without_flag() -> None:
    """Нет ключа в state → False."""
    client = MagicMock()
    client.threads.get_state = AsyncMock(return_value={"values": {"messages": []}})

    assert await session_module.expects_continuation(client, "tid") is False


@pytest.mark.asyncio
async def test_expects_continuation_false_on_client_error() -> None:
    """Ошибка клиента → False, без исключения наружу."""
    client = MagicMock()
    client.threads.get_state = AsyncMock(side_effect=RuntimeError("boom"))

    assert await session_module.expects_continuation(client, "tid") is False


@pytest.mark.asyncio
async def test_expects_continuation_false_on_read_timeout(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Таймаут чтения state → False и отличимая строка в логе."""

    async def _slow_get_state(_thread_id: str) -> dict[str, object]:
        await asyncio.sleep(10)
        return {"values": {"expect_continuation": True}}

    client = MagicMock()
    client.threads.get_state = AsyncMock(side_effect=_slow_get_state)

    with (
        patch.object(session_module, "CONTINUATION_READ_TIMEOUT", 0.05),
        caplog.at_level(logging.INFO, logger="voice_bot.partial"),
    ):
        assert await session_module.expects_continuation(client, "tid") is False

    assert any("таймаут чтения" in r.message for r in caplog.records)
    assert not any("flag=False" in r.message for r in caplog.records)
    assert not any("ошибка чтения" in r.message for r in caplog.records)


# --- is_conversation_ended ------------------------------------------------


@pytest.mark.asyncio
async def test_is_conversation_ended_true_when_flag_set() -> None:
    """Флаг conversation_ended=True в state → True."""
    client = MagicMock()
    client.threads.get_state = AsyncMock(return_value={"values": {"conversation_ended": True}})

    assert await session_module.is_conversation_ended(client, "tid") is True
    client.threads.get_state.assert_awaited_once_with("tid")


@pytest.mark.asyncio
async def test_is_conversation_ended_false_without_flag() -> None:
    """Нет ключа в state → False."""
    client = MagicMock()
    client.threads.get_state = AsyncMock(return_value={"values": {"messages": []}})

    assert await session_module.is_conversation_ended(client, "tid") is False


@pytest.mark.asyncio
async def test_is_conversation_ended_false_on_client_error() -> None:
    """Ошибка клиента → False, без исключения наружу."""
    client = MagicMock()
    client.threads.get_state = AsyncMock(side_effect=RuntimeError("boom"))

    assert await session_module.is_conversation_ended(client, "tid") is False


# --- продолжение после реплики бота --------------------------------------


@pytest.mark.asyncio
async def test_finished_with_flag_starts_continuation() -> None:
    """Флаг стоит → generate_reply с continuation, затем turn_kind=client."""
    lg = MagicMock()
    controller = _controller(lg_client=lg)
    session = controller._session
    seen: list[str] = []

    async def _capture() -> None:
        seen.append(session.llm._config["configurable"]["turn_kind"])

    session.generate_reply = AsyncMock(side_effect=_capture)

    with (
        patch.object(main_module, "is_conversation_ended", AsyncMock(return_value=False)),
        patch.object(main_module, "expects_continuation", AsyncMock(return_value=True)),
    ):
        await controller.on_agent_finished_speaking()

    session.generate_reply.assert_awaited_once_with()
    assert seen == ["continuation"]
    assert session.llm._config["configurable"]["turn_kind"] == "client"
    assert controller.continuation_count == 1


@pytest.mark.asyncio
async def test_finished_without_flag_skips_continuation() -> None:
    """Флага нет → generate_reply не вызван, turn_kind=client."""
    controller = _controller(lg_client=MagicMock())
    session = controller._session

    with (
        patch.object(main_module, "is_conversation_ended", AsyncMock(return_value=False)),
        patch.object(main_module, "expects_continuation", AsyncMock(return_value=False)),
    ):
        await controller.on_agent_finished_speaking()

    session.generate_reply.assert_not_called()
    assert session.llm._config["configurable"]["turn_kind"] == "client"
    assert controller.continuation_count == 0


@pytest.mark.asyncio
async def test_finished_conversation_ended_hangs_up_without_goodbye(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Признак завершения → delete_room, silence_goodbye не произносится."""
    settings = _settings(VOICE_BOT_SILENCE_GOODBYE="до связи из настроек")
    controller = _controller(settings=settings, lg_client=MagicMock())
    session = controller._session
    ctx = controller._ctx
    session.current_speech = SimpleNamespace(wait_for_playout=AsyncMock())

    with (
        patch.object(main_module, "is_conversation_ended", AsyncMock(return_value=True)),
        patch.object(main_module, "expects_continuation", AsyncMock(return_value=False)) as cont,
        caplog.at_level(logging.INFO, logger="voice_bot"),
    ):
        await controller.on_agent_finished_speaking()

    session.say.assert_not_called()
    session.generate_reply.assert_not_called()
    cont.assert_not_awaited()
    session.current_speech.wait_for_playout.assert_awaited_once_with()
    ctx.delete_room.assert_called_once_with()
    assert any("признак conversation_ended от мозга" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_finished_ended_and_continuation_prefers_hangup() -> None:
    """Оба флага → продолжение не запускается, комната закрывается."""
    controller = _controller(lg_client=MagicMock())
    session = controller._session
    ctx = controller._ctx

    with (
        patch.object(main_module, "is_conversation_ended", AsyncMock(return_value=True)),
        patch.object(main_module, "expects_continuation", AsyncMock(return_value=True)) as cont,
    ):
        await controller.on_agent_finished_speaking()

    session.generate_reply.assert_not_called()
    session.say.assert_not_called()
    cont.assert_not_awaited()
    ctx.delete_room.assert_called_once_with()
    assert controller.continuation_count == 0
    assert controller._ending is True


@pytest.mark.asyncio
async def test_away_after_conversation_ended_skips_silence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Признак завершения стоит — away не запускает оклик."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=10.0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_GOODBYE="до связи из настроек",
    )
    controller = _controller(settings=settings, lg_client=MagicMock())
    session = controller._session
    session.agent_state = "listening"

    with (
        patch.object(main_module, "is_conversation_ended", AsyncMock(return_value=True)),
        patch.object(main_module, "expects_continuation", AsyncMock(return_value=False)),
    ):
        await controller.on_agent_finished_speaking()

    with caplog.at_level(logging.INFO, logger="voice_bot"):
        controller.on_user_away()

    assert controller.away_task is None
    assert controller.silence_attempts == 0
    assert controller._silence_deferred is False
    session.generate_reply.assert_not_called()
    session.say.assert_not_called()
    assert any("away проигнорирован: звонок завершается" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_deferred_silence_cancelled_when_conversation_ended() -> None:
    """Отложенная отметка молчания снимается при завершении по признаку мозга."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=10.0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
    )
    controller = _controller(settings=settings, lg_client=MagicMock())
    session = controller._session

    session.agent_state = "speaking"
    controller.on_user_away()
    assert controller._silence_deferred is True

    with (
        patch.object(main_module, "is_conversation_ended", AsyncMock(return_value=True)),
        patch.object(main_module, "expects_continuation", AsyncMock(return_value=False)),
    ):
        await controller.on_agent_finished_speaking()

    assert controller._ending is True
    assert controller._silence_deferred is False

    session.agent_state = "listening"
    controller.on_agent_listening()
    assert controller._listen_away_task is None
    assert controller.away_task is None
    session.generate_reply.assert_not_called()


@pytest.mark.asyncio
async def test_finished_flag_error_does_not_propagate() -> None:
    """Чтение флага упало → generate_reply не вызван, исключение не летит."""
    controller = _controller(lg_client=MagicMock())
    session = controller._session

    with (
        patch.object(main_module, "is_conversation_ended", AsyncMock(return_value=False)),
        patch.object(
            main_module,
            "expects_continuation",
            AsyncMock(side_effect=RuntimeError("boom")),
        ),
    ):
        await controller.on_agent_finished_speaking()

    session.generate_reply.assert_not_called()
    assert controller.continuation_count == 0
    assert session.llm._config["configurable"]["turn_kind"] == "client"


@pytest.mark.asyncio
async def test_finished_flag_timeout_skips_continuation() -> None:
    """Таймаут чтения флага → generate_reply не вызван."""
    controller = _controller(lg_client=MagicMock())
    session = controller._session

    with (
        patch.object(main_module, "is_conversation_ended", AsyncMock(return_value=False)),
        patch.object(main_module, "expects_continuation", AsyncMock(return_value=False)),
    ):
        await controller.on_agent_finished_speaking()

    session.generate_reply.assert_not_called()
    assert controller.continuation_count == 0


@pytest.mark.asyncio
async def test_max_continuations_stops_monologue() -> None:
    """После max_continuations ход отдаётся клиенту даже при стоящем флаге."""
    settings = _settings(VOICE_BOT_MAX_CONTINUATIONS=2)
    controller = _controller(settings=settings, lg_client=MagicMock())
    controller.continuation_count = 2
    session = controller._session

    with (
        patch.object(main_module, "is_conversation_ended", AsyncMock(return_value=False)),
        patch.object(main_module, "expects_continuation", AsyncMock(return_value=True)) as expect,
    ):
        await controller.on_agent_finished_speaking()

    expect.assert_awaited_once()
    session.generate_reply.assert_not_called()
    assert controller.continuation_count == 0
    assert session.llm._config["configurable"]["turn_kind"] == "client"


# --- away / оклики --------------------------------------------------------


@pytest.mark.asyncio
async def test_away_while_thinking_does_not_start_silence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Away при thinking — оклик не запускается, ставится отложенная отметка."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=10.0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
    )
    controller = _controller(settings=settings)
    session = controller._session
    session.agent_state = "thinking"
    controller.silence_attempts = 0

    with caplog.at_level(logging.INFO, logger="voice_bot"):
        controller.on_user_away()

    assert controller.away_task is None
    assert controller.silence_attempts == 0
    assert controller._silence_deferred is True
    session.generate_reply.assert_not_called()
    assert any(
        "away проигнорирован" in r.message and "thinking" in r.message for r in caplog.records
    )


@pytest.mark.asyncio
async def test_away_while_speaking_does_not_start_silence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Away при speaking — оклик не запускается, ставится отложенная отметка."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=10.0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
    )
    controller = _controller(settings=settings)
    session = controller._session
    session.agent_state = "speaking"
    controller.silence_attempts = 1

    with caplog.at_level(logging.INFO, logger="voice_bot"):
        controller.on_user_away()

    assert controller.away_task is None
    assert controller.silence_attempts == 1
    assert controller._silence_deferred is True
    session.generate_reply.assert_not_called()
    assert any(
        "away проигнорирован" in r.message and "speaking" in r.message for r in caplog.records
    )


@pytest.mark.asyncio
async def test_away_starts_silence_turn() -> None:
    """Away при listening → generate_reply с turn_kind=silence; say не вызывается."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=10.0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_GOODBYE="пока",
    )
    controller = _controller(settings=settings)
    session = controller._session
    session.agent_state = "listening"

    seen: list[str] = []

    async def _capture() -> None:
        seen.append(session.llm._config["configurable"]["turn_kind"])

    session.generate_reply = AsyncMock(side_effect=_capture)

    controller.on_user_away()
    task = controller.away_task
    assert task is not None
    assert controller._silence_deferred is False

    for _ in range(20):
        await asyncio.sleep(0)
        if session.generate_reply.await_count >= 1:
            break

    assert session.generate_reply.await_count == 1
    assert seen == ["silence"]
    session.say.assert_not_called()

    controller.on_user_present()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_deferred_silence_after_speaking_away(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Away при speaking → listening, человек молчит — оклик после таймаута от перехода."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=0.05,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_GOODBYE="пока",
    )
    controller = _controller(settings=settings)
    session = controller._session

    session.agent_state = "speaking"
    controller.on_user_away()
    assert controller.away_task is None
    assert controller._silence_deferred is True
    assert controller.silence_attempts == 0
    session.generate_reply.assert_not_called()

    session.agent_state = "listening"
    controller.on_agent_listening()
    wait_task = controller._listen_away_task
    assert wait_task is not None
    assert controller.away_task is None

    with caplog.at_level(logging.INFO, logger="voice_bot"):
        await wait_task

    task = controller.away_task
    assert task is not None

    for _ in range(20):
        await asyncio.sleep(0)
        if session.generate_reply.await_count >= 1:
            break

    assert session.generate_reply.await_count >= 1
    assert controller.silence_attempts >= 1
    assert any("оклик по отложенной отметке" in r.message for r in caplog.records)

    controller.on_user_present()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_deferred_silence_cancelled_when_user_speaks_before_timeout() -> None:
    """Away при speaking → listening, человек заговорил до таймаута — оклика нет."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=0.5,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_GOODBYE="пока",
    )
    controller = _controller(settings=settings)
    session = controller._session

    session.agent_state = "speaking"
    controller.on_user_away()
    assert controller._silence_deferred is True

    session.agent_state = "listening"
    controller.on_agent_listening()
    wait_task = controller._listen_away_task
    assert wait_task is not None

    controller.on_user_present()
    assert controller._silence_deferred is False
    assert controller._listen_away_task is None
    assert controller.silence_attempts == 0
    assert controller.away_task is None

    await asyncio.gather(wait_task, return_exceptions=True)
    await asyncio.sleep(0.6)

    session.generate_reply.assert_not_called()
    assert controller.silence_attempts == 0


@pytest.mark.asyncio
async def test_deferred_silence_mark_does_not_accumulate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Два проигнорированных away подряд → один оклик после listening."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_GOODBYE="пока",
    )
    controller = _controller(settings=settings)
    session = controller._session
    started = asyncio.Event()

    async def _block() -> None:
        started.set()
        await asyncio.Event().wait()

    session.generate_reply = AsyncMock(side_effect=_block)

    session.agent_state = "speaking"
    controller.on_user_away()
    controller.on_user_away()
    assert controller._silence_deferred is True
    assert controller.away_task is None

    session.agent_state = "listening"
    with caplog.at_level(logging.INFO, logger="voice_bot"):
        controller.on_agent_listening()
        wait_task = controller._listen_away_task
        assert wait_task is not None
        await wait_task
        await started.wait()

    task = controller.away_task
    assert task is not None
    deferred_logs = [r for r in caplog.records if "оклик по отложенной отметке" in r.message]
    assert len(deferred_logs) == 1
    assert session.generate_reply.await_count == 1

    controller.on_user_present()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_away_silence_turns_then_goodbye_and_delete_room() -> None:
    """Две попытки silence, затем прощание и delete_room."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_GOODBYE="до связи",
    )
    controller = _controller(settings=settings)
    session = controller._session
    ctx = controller._ctx

    turn_kinds: list[str] = []

    async def _track_reply() -> None:
        turn_kinds.append(session.llm._config["configurable"]["turn_kind"])

    session.generate_reply = AsyncMock(side_effect=_track_reply)

    controller.on_user_away()
    task = controller.away_task
    assert task is not None
    await task

    assert session.generate_reply.await_count == 2
    assert turn_kinds == ["silence", "silence"]
    session.say.assert_called_once_with("до связи")
    ctx.delete_room.assert_called_once_with()
    assert session.llm._config["configurable"]["turn_kind"] == "client"
    assert controller.silence_attempts == 2


@pytest.mark.asyncio
async def test_silence_turn_resets_turn_kind_to_client() -> None:
    """После хода по молчанию turn_kind возвращается в client."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=10.0,
        VOICE_BOT_SILENCE_ATTEMPTS=1,
    )
    controller = _controller(settings=settings)
    session = controller._session

    seen: list[str] = []

    async def _capture() -> None:
        seen.append(session.llm._config["configurable"]["turn_kind"])

    session.generate_reply = AsyncMock(side_effect=_capture)

    controller.on_user_away()
    task = controller.away_task
    assert task is not None

    for _ in range(20):
        await asyncio.sleep(0)
        if session.generate_reply.await_count >= 1:
            break

    assert seen == ["silence"]
    assert session.llm._config["configurable"]["turn_kind"] == "client"

    controller.on_user_present()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_silence_cancel_during_generate_resets_turn_kind_to_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Отмена во время generate_reply по молчанию → turn_kind=client."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=10.0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
    )
    controller = _controller(settings=settings)
    session = controller._session
    started = asyncio.Event()

    async def _block() -> None:
        started.set()
        await asyncio.Event().wait()

    session.generate_reply = AsyncMock(side_effect=_block)

    with caplog.at_level(logging.INFO, logger="voice_bot"):
        controller.on_user_away()
        task = controller.away_task
        assert task is not None

        await started.wait()
        assert session.llm._config["configurable"]["turn_kind"] == "silence"

        controller.on_user_present()
        await asyncio.gather(task, return_exceptions=True)

    assert session.llm._config["configurable"]["turn_kind"] == "client"
    assert any("turn_kind возвращён в client после отмены" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_silence_normal_cycle_leaves_turn_kind_client() -> None:
    """Нормальное завершение цикла окликов → turn_kind=client."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_GOODBYE="до связи",
    )
    controller = _controller(settings=settings)
    session = controller._session

    controller.on_user_away()
    task = controller.away_task
    assert task is not None
    await task

    assert session.generate_reply.await_count == 2
    assert session.llm._config["configurable"]["turn_kind"] == "client"


@pytest.mark.asyncio
async def test_continuation_resets_turn_kind_to_client_after_generate() -> None:
    """После запуска продолжения turn_kind возвращается в client."""
    controller = _controller(lg_client=MagicMock())
    session = controller._session
    seen: list[str] = []

    async def _capture() -> None:
        seen.append(session.llm._config["configurable"]["turn_kind"])

    session.generate_reply = AsyncMock(side_effect=_capture)

    with (
        patch.object(main_module, "is_conversation_ended", AsyncMock(return_value=False)),
        patch.object(main_module, "expects_continuation", AsyncMock(return_value=True)),
    ):
        await controller.on_agent_finished_speaking()

    assert seen == ["continuation"]
    assert session.llm._config["configurable"]["turn_kind"] == "client"


@pytest.mark.asyncio
async def test_continuation_cancel_during_generate_resets_turn_kind_to_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Отмена во время generate_reply продолжения → turn_kind=client."""
    controller = _controller(lg_client=MagicMock())
    session = controller._session
    started = asyncio.Event()

    async def _block() -> None:
        started.set()
        await asyncio.Event().wait()

    session.generate_reply = AsyncMock(side_effect=_block)

    with (
        patch.object(main_module, "is_conversation_ended", AsyncMock(return_value=False)),
        patch.object(main_module, "expects_continuation", AsyncMock(return_value=True)),
        caplog.at_level(logging.INFO, logger="voice_bot"),
    ):
        task = asyncio.create_task(controller.on_agent_finished_speaking())
        await started.wait()
        assert session.llm._config["configurable"]["turn_kind"] == "continuation"
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert session.llm._config["configurable"]["turn_kind"] == "client"
    assert any("turn_kind возвращён в client после отмены" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_user_present_cancels_away_prompts() -> None:
    """Клиент заговорил между попытками — цикл прерван, счётчик обнулён."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=0.5,
        VOICE_BOT_SILENCE_ATTEMPTS=3,
        VOICE_BOT_SILENCE_GOODBYE="пока",
    )
    controller = _controller(settings=settings)
    session = controller._session

    controller.on_user_away()
    task = controller.away_task
    assert task is not None

    # Дать первому ходу silence завершиться и уйти в sleep между попытками.
    for _ in range(20):
        await asyncio.sleep(0)
        if session.generate_reply.await_count >= 1:
            break
    assert session.generate_reply.await_count == 1

    controller.on_user_present()
    assert controller.silence_attempts == 0
    assert controller.away_task is None

    await asyncio.gather(task, return_exceptions=True)

    await asyncio.sleep(0.6)
    assert session.generate_reply.await_count == 1
    session.say.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_away_does_not_start_second_task() -> None:
    """Повторный away при уже идущей задаче не запускает вторую."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=1.0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
    )
    controller = _controller(settings=settings)

    controller.on_user_away()
    first = controller.away_task
    assert first is not None
    assert not first.done()

    controller.on_user_away()
    assert controller.away_task is first

    controller.on_user_present()
    await asyncio.gather(first, return_exceptions=True)
    assert first.done()
