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
        "VOICE_BOT_SILENCE_LINK_CHECK_SECOND": "Алло?",
        "VOICE_BOT_SILENCE_LINK_CHECK_SECOND_PAUSE": 0.01,
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
    """При включённом тумблере: две попытки проверки связи, затем прощание."""
    settings = _settings(
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_LINK_CHECK="Алло, меня слышно?",
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND="Алло?",
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND_PAUSE=0,
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

    assert said == ["Алло, меня слышно?", "Алло?", "до связи"]
    controller._ctx.delete_room.assert_called_once_with()


@pytest.mark.asyncio
async def test_link_check_second_empty_skips_second_attempt() -> None:
    """Пустая вторая фраза выключает вторую попытку: первая и прощание."""
    settings = _settings(
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_LINK_CHECK="Алло, меня слышно?",
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND="",
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND_PAUSE=0,
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
    assert session.say.call_count == 2


@pytest.mark.asyncio
async def test_user_present_between_link_check_attempts_skips_goodbye() -> None:
    """Реплика человека между попытками проверки связи — прощание не произносится."""
    settings = _settings(
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_LINK_CHECK="Алло, меня слышно?",
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0.5,
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND="Алло?",
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND_PAUSE=0,
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

    for _ in range(50):
        await asyncio.sleep(0)
        if said == ["Алло, меня слышно?"]:
            break
    assert said == ["Алло, меня слышно?"]

    controller.on_user_present()
    await asyncio.gather(task, return_exceptions=True)

    assert "до связи" not in said
    assert "Алло?" not in said
    controller._ctx.delete_room.assert_not_called()


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


@pytest.mark.asyncio
async def test_modes_off_always_sends_silence_never_pull() -> None:
    """Тумблер выключен — цикл попыток, в set_turn_kind только silence, никогда pull."""
    settings = _settings(
        VOICE_BOT_SILENCE_MODES=False,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
    )
    controller = _controller(settings=settings)
    session = controller._session

    turn_kinds: list[str] = []

    async def _track_generate() -> None:
        turn_kinds.append(session.llm._config["configurable"]["turn_kind"])

    session.generate_reply = AsyncMock(side_effect=_track_generate)

    controller.on_user_away()
    task = controller.away_task
    assert task is not None
    await task

    assert turn_kinds == ["silence", "silence"]
    assert "pull" not in turn_kinds
    assert session.generate_reply.await_count == 2


@pytest.mark.asyncio
async def test_modes_on_statement_sends_pull_then_link_check() -> None:
    """Тумблер включён, реплика без вопроса — pull один раз, затем проверка связи и прощание."""
    settings = _settings(
        VOICE_BOT_SILENCE_MODES=True,
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
        VOICE_BOT_SILENCE_LINK_CHECK="Алло, меня слышно?",
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND="Алло?",
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND_PAUSE=0,
        VOICE_BOT_SILENCE_GOODBYE="до связи",
    )
    controller = _controller(settings=settings)
    session = controller._session
    history = [{"type": "ai", "content": "Хорошо, записала."}]

    turn_kinds: list[str] = []
    said: list[str] = []

    async def _track_generate() -> None:
        turn_kinds.append(session.llm._config["configurable"]["turn_kind"])

    def _track_say(text: str, *_a: object, **_k: object) -> SimpleNamespace:
        said.append(text)
        return SimpleNamespace(wait_for_playout=AsyncMock())

    session.generate_reply = AsyncMock(side_effect=_track_generate)
    session.say = MagicMock(side_effect=_track_say)

    with patch.object(main_module, "chat_history_snapshot", return_value=history):
        controller.on_user_away()
        task = controller.away_task
        assert task is not None
        await task

    assert turn_kinds == ["pull"]
    assert said == ["Алло, меня слышно?", "Алло?", "до связи"]
    assert session.generate_reply.await_count == 1


@pytest.mark.asyncio
async def test_modes_on_question_skips_generate_goes_to_link_check() -> None:
    """Тумблер включён, реплика с вопросом — generate_reply не вызывается, сразу проверка связи."""
    settings = _settings(
        VOICE_BOT_SILENCE_MODES=True,
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
        VOICE_BOT_SILENCE_LINK_CHECK="Алло, меня слышно?",
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND="Алло?",
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND_PAUSE=0,
        VOICE_BOT_SILENCE_GOODBYE="до связи",
    )
    controller = _controller(settings=settings)
    session = controller._session
    history = [{"type": "ai", "content": "Как вас зовут?"}]
    said: list[str] = []

    def _track_say(text: str, *_a: object, **_k: object) -> SimpleNamespace:
        said.append(text)
        return SimpleNamespace(wait_for_playout=AsyncMock())

    session.say = MagicMock(side_effect=_track_say)

    with patch.object(main_module, "chat_history_snapshot", return_value=history):
        controller.on_user_away()
        task = controller.away_task
        assert task is not None
        await task

    session.generate_reply.assert_not_awaited()
    assert said == ["Алло, меня слышно?", "Алло?", "до связи"]


@pytest.mark.asyncio
async def test_modes_on_question_skips_silence_pause_question() -> None:
    """Реплика с вопросом — между решением и проверкой связи нет silence_pause_question."""
    pause_question = 0.04
    settings = _settings(
        VOICE_BOT_SILENCE_MODES=True,
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=pause_question,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
        VOICE_BOT_SILENCE_LINK_CHECK="Алло, меня слышно?",
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND="",
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND_PAUSE=0,
        VOICE_BOT_SILENCE_GOODBYE="до связи",
    )
    controller = _controller(settings=settings)
    history = [{"type": "ai", "content": "Как вас зовут?"}]
    sleep_args: list[float] = []
    real_sleep = asyncio.sleep

    async def _track_sleep(delay: float, *args: object, **kwargs: object) -> None:
        sleep_args.append(delay)
        await real_sleep(0)

    with (
        patch.object(main_module, "chat_history_snapshot", return_value=history),
        patch.object(main_module.asyncio, "sleep", side_effect=_track_sleep),
    ):
        controller.on_user_away()
        task = controller.away_task
        assert task is not None
        await task

    assert pause_question not in sleep_args


@pytest.mark.asyncio
async def test_modes_on_statement_sleeps_silence_pause_question_once() -> None:
    """Реплика без вопроса — silence_pause_question ровно один раз после pull."""
    pause_question = 0.04
    settings = _settings(
        VOICE_BOT_SILENCE_MODES=True,
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=pause_question,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
        VOICE_BOT_SILENCE_LINK_CHECK="Алло, меня слышно?",
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND="",
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND_PAUSE=0,
        VOICE_BOT_SILENCE_GOODBYE="до связи",
    )
    controller = _controller(settings=settings)
    session = controller._session
    history = [{"type": "ai", "content": "Хорошо, записала."}]
    sleep_args: list[float] = []
    pull_done = False
    real_sleep = asyncio.sleep

    async def _track_generate() -> None:
        nonlocal pull_done
        pull_done = True

    async def _track_sleep(delay: float, *args: object, **kwargs: object) -> None:
        sleep_args.append(delay)
        if delay == pause_question:
            assert pull_done is True
        await real_sleep(0)

    session.generate_reply = AsyncMock(side_effect=_track_generate)

    with (
        patch.object(main_module, "chat_history_snapshot", return_value=history),
        patch.object(main_module.asyncio, "sleep", side_effect=_track_sleep),
    ):
        controller.on_user_away()
        task = controller.away_task
        assert task is not None
        await task

    assert sleep_args.count(pause_question) == 1
    assert session.generate_reply.await_count == 1


@pytest.mark.asyncio
async def test_modes_on_generate_reply_at_most_once() -> None:
    """При включённом silence_modes generate_reply вызывается не более одного раза."""
    settings = _settings(
        VOICE_BOT_SILENCE_MODES=True,
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
    )
    controller = _controller(settings=settings)
    session = controller._session
    history = [{"type": "ai", "content": "Записала."}]

    with patch.object(main_module, "chat_history_snapshot", return_value=history):
        controller.on_user_away()
        task = controller.away_task
        assert task is not None
        await task

    assert session.generate_reply.await_count <= 1


@pytest.mark.asyncio
async def test_ending_skips_silence_ladder() -> None:
    """При выставленном признаке завершения звонка лестница не запускается."""
    settings = _settings(
        VOICE_BOT_SILENCE_MODES=True,
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0.05,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
    )
    controller = _controller(settings=settings)
    session = controller._session

    # Отсчёт стартует до признака завершения, затем флаг блокирует лестницу.
    controller._schedule_silence_wait()
    wait_task = controller._silence_wait_task
    assert wait_task is not None
    controller._ending = True
    await wait_task

    assert controller.away_task is None
    session.generate_reply.assert_not_awaited()
    session.say.assert_not_called()


@pytest.mark.asyncio
async def test_ending_before_ladder_skips_all_steps() -> None:
    """Признак до запуска лестницы — ни generate_reply, ни say."""
    settings = _settings(
        VOICE_BOT_SILENCE_MODES=True,
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND_PAUSE=0,
    )
    controller = _controller(settings=settings)
    session = controller._session
    history = [{"type": "ai", "content": "Хорошо, записала."}]

    controller._ending = True
    with patch.object(main_module, "chat_history_snapshot", return_value=history):
        await controller._away_prompts()

    session.generate_reply.assert_not_awaited()
    session.say.assert_not_called()
    controller._ctx.delete_room.assert_not_called()


@pytest.mark.asyncio
async def test_ending_after_pull_skips_link_check_and_goodbye() -> None:
    """Признак после pull — проверка связи и прощание по тишине не произносятся."""
    settings = _settings(
        VOICE_BOT_SILENCE_MODES=True,
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
        VOICE_BOT_SILENCE_LINK_CHECK="Алло, меня слышно?",
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND="Алло?",
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND_PAUSE=0,
        VOICE_BOT_SILENCE_GOODBYE="до связи",
    )
    controller = _controller(settings=settings)
    session = controller._session
    history = [{"type": "ai", "content": "Хорошо, записала."}]
    said: list[str] = []

    async def _track_generate() -> None:
        controller._ending = True

    def _track_say(text: str, *_a: object, **_k: object) -> SimpleNamespace:
        said.append(text)
        return SimpleNamespace(wait_for_playout=AsyncMock())

    session.generate_reply = AsyncMock(side_effect=_track_generate)
    session.say = MagicMock(side_effect=_track_say)

    with patch.object(main_module, "chat_history_snapshot", return_value=history):
        controller.on_user_away()
        task = controller.away_task
        assert task is not None
        await task

    assert session.generate_reply.await_count == 1
    assert said == []
    controller._ctx.delete_room.assert_not_called()


@pytest.mark.asyncio
async def test_ending_between_link_checks_skips_second_and_goodbye() -> None:
    """Признак между фразами проверки связи — вторая и прощание не звучат."""
    settings = _settings(
        VOICE_BOT_SILENCE_MODES=True,
        VOICE_BOT_SILENCE_SMART_PAUSES=True,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
        VOICE_BOT_SILENCE_LINK_CHECK="Алло, меня слышно?",
        VOICE_BOT_SILENCE_LINK_CHECK_PAUSE=0,
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND="Алло?",
        VOICE_BOT_SILENCE_LINK_CHECK_SECOND_PAUSE=0,
        VOICE_BOT_SILENCE_GOODBYE="до связи",
    )
    controller = _controller(settings=settings)
    session = controller._session
    history = [{"type": "ai", "content": "Как вас зовут?"}]
    said: list[str] = []

    def _track_say(text: str, *_a: object, **_k: object) -> SimpleNamespace:
        said.append(text)
        if text == "Алло, меня слышно?":
            controller._ending = True
        return SimpleNamespace(wait_for_playout=AsyncMock())

    session.say = MagicMock(side_effect=_track_say)

    with patch.object(main_module, "chat_history_snapshot", return_value=history):
        controller.on_user_away()
        task = controller.away_task
        assert task is not None
        await task

    session.generate_reply.assert_not_awaited()
    assert said == ["Алло, меня слышно?"]
    controller._ctx.delete_room.assert_not_called()


@pytest.mark.asyncio
async def test_ending_modes_off_skips_silence_attempts() -> None:
    """Тумблер выключен, признак выставлен — цикл попыток не делает ходов."""
    settings = _settings(
        VOICE_BOT_SILENCE_MODES=False,
        VOICE_BOT_SILENCE_TIMEOUT=0,
        VOICE_BOT_SILENCE_ATTEMPTS=2,
        VOICE_BOT_SILENCE_PAUSE_QUESTION=0,
        VOICE_BOT_SILENCE_PAUSE_STATEMENT=0,
    )
    controller = _controller(settings=settings)
    session = controller._session

    controller._ending = True
    await controller._away_prompts()

    session.generate_reply.assert_not_awaited()
    session.say.assert_not_called()
    controller._ctx.delete_room.assert_not_called()
