"""Офлайн-тесты механики умных пауз в CallTurnController."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_bot.agent import main as main_module
from voice_bot.config import Settings


def _settings(**overrides: object) -> Settings:
    """Минимальные Settings для тестов пауз."""
    base: dict[str, object] = {
        "LIVEKIT_URL": "ws://localhost:7880",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "secret",
        "OPENAI_API_KEY": "sk-test",
        "ELEVENLABS_API_KEY": "el-test",
        "ELEVENLABS_VOICE_ID": "voice-123",
        "VOICE_BOT_SILENCE_TIMEOUT": 0.05,
        "VOICE_BOT_SILENCE_ATTEMPTS": 2,
        "VOICE_BOT_SILENCE_GOODBYE": "до связи",
        "VOICE_BOT_SILENCE_SMART_PAUSES": False,
        "VOICE_BOT_SILENCE_PAUSE_QUESTION": 0.04,
        "VOICE_BOT_SILENCE_PAUSE_STATEMENT": 0.02,
        "VOICE_BOT_SILENCE_LINK_CHECK": "Алло, меня слышно?",
        "VOICE_BOT_SILENCE_LINK_CHECK_PAUSE": 0.01,
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
    session.llm._config = session.llm.config  # type: ignore[attr-defined]
    session.on = MagicMock(return_value=lambda fn: fn)
    session.agent_state = "listening"
    session.user_state = "listening"
    return session


def _controller(
    *,
    session: MagicMock | None = None,
    settings: Settings | None = None,
) -> main_module.CallTurnController:
    """Собрать контроллер с фейковой сессией."""
    sess = session or _fake_session()
    cfg = settings or _settings()
    ctx = SimpleNamespace(delete_room=MagicMock())
    return main_module.CallTurnController(
        session=sess,
        ctx=ctx,  # type: ignore[arg-type]
        settings=cfg,
    )


def test_pick_pause_toggle_off_is_fixed() -> None:
    """Тумблер выключен — всегда silence_timeout, причина «фиксированная»."""
    settings = _settings(
        VOICE_BOT_SILENCE_SMART_PAUSES=False,
        VOICE_BOT_SILENCE_TIMEOUT=6.0,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=4.0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=1.2,
    )
    controller = _controller(settings=settings)
    history = [{"type": "ai", "content": "Вам удобно?"}]

    with patch.object(main_module, "chat_history_snapshot", return_value=history):
        pause, reason = controller._pick_pause()

    assert pause == 6.0
    assert reason == "фиксированная"


def test_pick_pause_toggle_on_question() -> None:
    """Тумблер включён, реплика с вопросом → silence_pause_question."""
    settings = _settings(
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=4.0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=1.2,
    )
    controller = _controller(settings=settings)
    history = [{"type": "ai", "content": "Как вас зовут?"}]

    with patch.object(main_module, "chat_history_snapshot", return_value=history):
        pause, reason = controller._pick_pause()

    assert pause == 4.0
    assert reason == "вопрос"


def test_pick_pause_toggle_on_statement() -> None:
    """Тумблер включён, реплика без вопроса → silence_pause_statement."""
    settings = _settings(
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=4.0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=1.2,
    )
    controller = _controller(settings=settings)
    history = [{"type": "ai", "content": "Хорошо, записала."}]

    with patch.object(main_module, "chat_history_snapshot", return_value=history):
        pause, reason = controller._pick_pause()

    assert pause == 1.2
    assert reason == "без вопроса"


@pytest.mark.asyncio
async def test_link_check_once_before_goodbye() -> None:
    """При включённом тумблере: проверка связи, затем прощание."""
    settings = _settings(
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_LINK_CHECK="Алло, меня слышно?",
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
        VOICE_BOT_SILENCE_GOODBYE="до связи",
        VOICE_BOT_SILENCE_PAUSE_QUESTION=0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
    )
    controller = _controller(settings=settings)
    session = controller._session
    said: list[str] = []

    def _track_say(text: str, *_a: object, **_k: object) -> SimpleNamespace:
        said.append(text)
        return SimpleNamespace(wait_for_playout=AsyncMock())

    session.say = MagicMock(side_effect=_track_say)

    controller.on_user_away()
    task = controller.away_task
    assert task is not None
    await task

    assert said == ["Алло, меня слышно?", "до связи"]
    controller._ctx.delete_room.assert_called_once_with()


@pytest.mark.asyncio
async def test_no_link_check_when_toggle_off() -> None:
    """При выключенном тумблере в say уходит только прощание."""
    settings = _settings(
        VOICE_BOT_SILENCE_SMART_PAUSES=False,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_GOODBYE="до связи",
        VOICE_BOT_SILENCE_LINK_CHECK="Алло, меня слышно?",
    )
    controller = _controller(settings=settings)
    session = controller._session
    said: list[str] = []

    def _track_say(text: str, *_a: object, **_k: object) -> SimpleNamespace:
        said.append(text)
        return SimpleNamespace(wait_for_playout=AsyncMock())

    session.say = MagicMock(side_effect=_track_say)

    controller.on_user_away()
    task = controller.away_task
    assert task is not None
    await task

    assert said == ["до связи"]
    assert "Алло, меня слышно?" not in said


@pytest.mark.asyncio
async def test_user_present_resets_attempts_and_link_checked() -> None:
    """Реплика человека обнуляет счётчик попыток и признак проверки связи."""
    controller = _controller()
    controller.silence_attempts = 2
    controller._link_checked = True

    controller.on_user_present()

    assert controller.silence_attempts == 0
    assert controller._link_checked is False


@pytest.mark.asyncio
async def test_reschedule_silence_wait_cancels_previous() -> None:
    """Два подряд _schedule_silence_wait оставляют одну живую задачу."""
    settings = _settings(VOICE_BOT_SILENCE_TIMEOUT=10.0)
    controller = _controller(settings=settings)

    controller._schedule_silence_wait()
    first = controller._silence_wait_task
    assert first is not None
    assert not first.done()

    controller._schedule_silence_wait()
    second = controller._silence_wait_task
    assert second is not None
    assert second is not first
    await asyncio.sleep(0)
    assert first.cancelled() or first.done()
    assert not second.done()

    controller._cancel_silence_wait()
    await asyncio.gather(first, second, return_exceptions=True)
