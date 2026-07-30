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


# --- продолжение после реплики бота --------------------------------------


@pytest.mark.asyncio
async def test_finished_with_flag_starts_continuation() -> None:
    """Флаг стоит → generate_reply, turn_kind=continuation."""
    lg = MagicMock()
    controller = _controller(lg_client=lg)
    session = controller._session

    with patch.object(main_module, "expects_continuation", AsyncMock(return_value=True)):
        await controller.on_agent_finished_speaking()

    session.generate_reply.assert_awaited_once_with()
    assert session.llm._config["configurable"]["turn_kind"] == "continuation"
    assert controller.continuation_count == 1


@pytest.mark.asyncio
async def test_finished_without_flag_skips_continuation() -> None:
    """Флага нет → generate_reply не вызван, turn_kind=client."""
    controller = _controller(lg_client=MagicMock())
    session = controller._session

    with patch.object(main_module, "expects_continuation", AsyncMock(return_value=False)):
        await controller.on_agent_finished_speaking()

    session.generate_reply.assert_not_called()
    assert session.llm._config["configurable"]["turn_kind"] == "client"
    assert controller.continuation_count == 0


@pytest.mark.asyncio
async def test_finished_flag_error_does_not_propagate() -> None:
    """Чтение флага упало → generate_reply не вызван, исключение не летит."""
    controller = _controller(lg_client=MagicMock())
    session = controller._session

    with patch.object(
        main_module,
        "expects_continuation",
        AsyncMock(side_effect=RuntimeError("boom")),
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

    with patch.object(main_module, "expects_continuation", AsyncMock(return_value=False)):
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

    with patch.object(main_module, "expects_continuation", AsyncMock(return_value=True)) as expect:
        await controller.on_agent_finished_speaking()

    expect.assert_awaited_once()
    session.generate_reply.assert_not_called()
    assert controller.continuation_count == 0
    assert session.llm._config["configurable"]["turn_kind"] == "client"


# --- away / оклики --------------------------------------------------------


@pytest.mark.asyncio
async def test_away_three_prompts_then_delete_room() -> None:
    """Away → три фразы; после последней — delete_room, без generate_reply."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_PROMPTS=["раз", "два", "три"],
    )
    controller = _controller(settings=settings, lg_client=None, thread_id=None)
    session = controller._session
    ctx = controller._ctx

    controller.on_user_away()
    task = controller.away_task
    assert task is not None
    await task

    assert session.say.call_count == 3
    assert [c.args[0] for c in session.say.call_args_list] == ["раз", "два", "три"]
    ctx.delete_room.assert_called_once_with()
    session.generate_reply.assert_not_called()


@pytest.mark.asyncio
async def test_user_present_cancels_away_prompts() -> None:
    """Клиент заговорил во время окликов — задача снята, следующая фраза не звучит."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=0.5,
        VOICE_BOT_SILENCE_PROMPTS=["раз", "два", "три"],
    )
    controller = _controller(settings=settings, lg_client=None, thread_id=None)
    session = controller._session

    controller.on_user_away()
    task = controller.away_task
    assert task is not None

    # Дать первой фразе прозвучать и уйти в sleep между окликами.
    for _ in range(20):
        await asyncio.sleep(0)
        if session.say.call_count >= 1:
            break
    assert session.say.call_count == 1

    controller.on_user_present()
    assert controller.silence_attempts == 0
    assert controller.away_task is None

    await asyncio.gather(task, return_exceptions=True)

    await asyncio.sleep(0.6)
    assert session.say.call_count == 1


@pytest.mark.asyncio
async def test_duplicate_away_does_not_start_second_task() -> None:
    """Повторный away при уже идущей задаче не запускает вторую."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=1.0,
        VOICE_BOT_SILENCE_PROMPTS=["раз", "два"],
    )
    controller = _controller(settings=settings, lg_client=None, thread_id=None)

    controller.on_user_away()
    first = controller.away_task
    assert first is not None
    assert not first.done()

    controller.on_user_away()
    assert controller.away_task is first

    controller.on_user_present()
    await asyncio.gather(first, return_exceptions=True)
    assert first.done()
