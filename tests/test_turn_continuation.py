"""Офлайн-тесты продолжения речи бота и реакции на тишину клиента."""

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
    client.threads.get_state = AsyncMock(side_effect=TimeoutError("boom"))

    assert await session_module.expects_continuation(client, "tid") is False


# --- продолжение после реплики бота --------------------------------------


@pytest.mark.asyncio
async def test_finished_with_flag_starts_continuation() -> None:
    """Флаг стоит → generate_reply, таймер не взведён, turn_kind=continuation."""
    lg = MagicMock()
    lg.threads.get_state = AsyncMock(return_value={"values": {"expect_continuation": True}})
    controller = _controller(lg_client=lg)
    session = controller._session

    with patch.object(main_module, "expects_continuation", AsyncMock(return_value=True)):
        await controller.on_agent_finished_speaking()

    session.generate_reply.assert_awaited_once_with()
    assert controller.silence_task is None
    assert session.llm._config["configurable"]["turn_kind"] == "continuation"
    assert controller.continuation_count == 1


@pytest.mark.asyncio
async def test_finished_without_flag_arms_silence_timer() -> None:
    """Флага нет → generate_reply не вызван, таймер взведён."""
    controller = _controller(lg_client=MagicMock())
    session = controller._session

    with patch.object(main_module, "expects_continuation", AsyncMock(return_value=False)):
        await controller.on_agent_finished_speaking()

    session.generate_reply.assert_not_called()
    assert controller.silence_task is not None
    assert not controller.silence_task.done()
    controller.cancel_silence_timer()


@pytest.mark.asyncio
async def test_max_continuations_stops_monologue() -> None:
    """После max_continuations ход отдаётся клиенту даже при стоящем флаге."""
    settings = _settings(VOICE_BOT_MAX_CONTINUATIONS=2)
    controller = _controller(settings=settings, lg_client=MagicMock())
    controller.continuation_count = 2
    session = controller._session

    with patch.object(main_module, "expects_continuation", AsyncMock(return_value=True)) as expect:
        await controller.on_agent_finished_speaking()

    expect.assert_not_awaited()
    session.generate_reply.assert_not_called()
    assert controller.continuation_count == 0
    assert controller.silence_task is not None
    controller.cancel_silence_timer()


# --- тишина ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_silence_three_prompts_then_delete_room() -> None:
    """Три фразы на тишину; после третьей — delete_room, без generate_reply."""
    settings = _settings(
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_PROMPTS=["раз", "два", "три"],
    )
    controller = _controller(settings=settings, lg_client=None, thread_id=None)
    session = controller._session
    ctx = controller._ctx

    await controller.on_agent_finished_speaking()
    task = controller.silence_task
    assert task is not None
    await task

    assert session.say.call_count == 3
    assert [c.args[0] for c in session.say.call_args_list] == ["раз", "два", "три"]
    ctx.delete_room.assert_called_once_with()
    session.generate_reply.assert_not_called()


@pytest.mark.asyncio
async def test_user_speaking_cancels_timer_and_continuation() -> None:
    """Клиент заговорил — таймер снят, попытки обнулены, продолжение отменено."""
    controller = _controller(lg_client=MagicMock())
    session = controller._session
    controller.arm_silence_timer()
    controller.silence_attempts = 2
    controller.continuation_count = 1
    assert controller.silence_task is not None

    controller.on_user_started_speaking()

    assert controller.silence_task is None
    assert controller.silence_attempts == 0
    assert controller._abort_continuation is True
    assert session.llm._config["configurable"]["turn_kind"] == "client"

    with patch.object(main_module, "expects_continuation", AsyncMock(return_value=True)):
        await controller.on_agent_finished_speaking()

    session.generate_reply.assert_not_called()
    assert controller.continuation_count == 0
    assert controller.silence_task is None
