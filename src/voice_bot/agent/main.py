"""Точка входа голосового бота (воркер LiveKit).

Запуск:

    python -m voice_bot.agent.main console   # локальный тест в терминале (микрофон)
    python -m voice_bot.agent.main dev        # подключиться к LiveKit-серверу
    python -m voice_bot.agent.main start       # продакшн-режим

Воркер подключается к LiveKit-серверу (адрес в ``LIVEKIT_URL``) и, когда
начинается звонок, ведёт разговор по сценарию.
"""

import asyncio
import inspect
import logging
import os
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    AgentStateChangedEvent,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    ConversationItemAddedEvent,
    UserStateChangedEvent,
)

from voice_bot.agent.script_agent import ScriptAgent
from voice_bot.agent.session import (
    build_agent_langgraph_client,
    build_partial_transcript_sender,
    build_session,
    expects_continuation,
    set_turn_kind,
    thread_id_for_room,
)
from voice_bot.config import Settings, get_settings
from voice_bot.logging_setup import setup_logging
from voice_bot.scenario.loader import load_scenario
from voice_bot.scenario.prompt import build_system_prompt

# .env загружаем в окружение ДО чтения настроек и инициализации плагинов:
# плагины OpenAI/ElevenLabs читают ключи напрямую из переменных окружения.
load_dotenv()
setup_logging(
    os.getenv("VOICE_BOT_LOG_LEVEL", "INFO"),
    os.getenv("VOICE_BOT_NOISY_LOG_LEVEL", "WARNING"),
)
logger = logging.getLogger("voice_bot")

# Имя и режим подхвата читаем без полной валидации настроек, чтобы служебные
# команды CLI (download-files, --help) работали и без секретов в окружении.
_AGENT_NAME = os.getenv("VOICE_BOT_AGENT_NAME", "voice-bot")
_AGENT_AUTO_ACCEPT = os.getenv("AGENT_AUTO_ACCEPT", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

server = AgentServer()


def _rtc_session_agent_name(*, auto_accept: bool, agent_name: str) -> str:
    """Имя для ``rtc_session``: пустая строка = автоподхват комнат (API 1.6.6).

    При непустом ``agent_name`` воркер ждёт явный dispatch; при ``""`` —
    принимает комнаты автоматически (удобно для локальных тестов).

    Args:
        auto_accept: включать ли автоподхват комнат.
        agent_name: имя агента для явного dispatch.

    Returns:
        Пустая строка при автоподхвате, иначе ``agent_name``.
    """
    return "" if auto_accept else agent_name


class CallTurnController:
    """Продолжение собственной речи бота и оклики при ``user_state=away``.

    Тишину детектит LiveKit (``user_away_timeout``): таймер идёт, пока оба
    в ``listening``. Оклик запускаем только когда бот слушает; если away
    пришёл во время thinking/speaking — игнорируем и при возврате бота
    в listening ждём ``silence_timeout``, затем стартуем оклики.
    """

    def __init__(
        self,
        *,
        session: AgentSession,
        ctx: agents.JobContext,
        settings: Settings,
        thread_id: str | None = None,
        lg_client: object | None = None,
    ) -> None:
        """Сохранить зависимости звонка и обнулить счётчики.

        Args:
            session: голосовая сессия текущего звонка.
            ctx: контекст задания LiveKit (нужен ``delete_room``).
            settings: таймаут тишины, лимит попыток и продолжений.
            thread_id: UUID треда основного графа; ``None`` — без продолжений.
            lg_client: клиент LangGraph для чтения ``expect_continuation``.
        """
        self._session = session
        self._ctx = ctx
        self._settings = settings
        self._thread_id = thread_id
        self._lg_client = lg_client

        self.continuation_count: int = 0
        self.silence_attempts: int = 0
        self.away_task: asyncio.Task[None] | None = None
        self._listen_away_task: asyncio.Task[None] | None = None
        self._finished_tasks: set[asyncio.Task[None]] = set()

    def attach(self) -> None:
        """Подписать обработчики на конец речи бота и смену состояния клиента."""

        @self._session.on("agent_state_changed")
        def _on_agent(ev: AgentStateChangedEvent) -> None:
            if ev.old_state == "speaking":
                logger.info("[turn] бот договорил: agent_state_changed speaking→%s", ev.new_state)
                self._schedule_finished()
            if ev.new_state == "listening":
                self.on_agent_listening()
            else:
                self._cancel_listen_away_wait()

        @self._session.on("user_state_changed")
        def _on_user(ev: UserStateChangedEvent) -> None:
            if ev.new_state == "away":
                self.on_user_away()
            else:
                self.on_user_present()

        logger.info(
            "[turn] обработчики подключены: silence_timeout=%s max_continuations=%s",
            self._settings.silence_timeout,
            self._settings.max_continuations,
        )

    def _agent_state(self) -> str:
        """Текущее состояние агента сессии (для тестов — атрибут заглушки).

        Returns:
            Строка состояния; при отсутствии атрибута — ``listening``.
        """
        state = getattr(self._session, "agent_state", "listening")
        return state if isinstance(state, str) else "listening"

    def _user_state(self) -> str:
        """Текущее состояние клиента сессии (для тестов — атрибут заглушки).

        Returns:
            Строка состояния; при отсутствии атрибута — ``listening``.
        """
        state = getattr(self._session, "user_state", "listening")
        return state if isinstance(state, str) else "listening"

    def on_user_away(self) -> None:
        """Клиент пропал (``away``): запустить оклики, если бот слушает.

        Пока бот думает или говорит, тишины нет: событие игнорируется без
        запуска задачи и без изменения счётчика попыток.

        Returns:
            None.
        """
        agent_state = self._agent_state()
        if agent_state != "listening":
            logger.info(
                "[turn] away проигнорирован: бот не слушает (agent_state=%s)",
                agent_state,
            )
            return

        self._cancel_listen_away_wait()
        if self.away_task is not None and not self.away_task.done():
            logger.info("[turn] пользователь ушёл в away: оклики уже идут, повтор пропущен")
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("Нет event loop для окликов на тишину")
            return

        logger.info("[turn] пользователь ушёл в away")
        self.away_task = loop.create_task(self._away_prompts(), name="away-prompts")

    def on_user_present(self) -> None:
        """Клиент снова на связи: отменить оклики и обнулить счётчик попыток.

        Returns:
            None.
        """
        self._cancel_listen_away_wait()
        task = self.away_task
        self.away_task = None
        self.silence_attempts = 0
        if task is None or task.done():
            return
        logger.info("[turn] пользователь вернулся: оклики отменены")
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if task is not current:
            task.cancel()

    def on_agent_listening(self) -> None:
        """Бот перешёл в ожидание ответа: при уже ``away`` — отсчёт молчания.

        LiveKit не перевыставляет ``away``, если клиент уже away. Поэтому
        после конца речи бота сами ждём ``silence_timeout`` и запускаем оклики.

        Returns:
            None.
        """
        if self._user_state() != "away":
            return
        if self.away_task is not None and not self.away_task.done():
            return
        if self._listen_away_task is not None and not self._listen_away_task.done():
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("Нет event loop для отсчёта тишины после речи бота")
            return

        logger.info(
            "[turn] бот слушает, клиент уже away: отсчёт тишины %.1fс",
            self._settings.silence_timeout,
        )
        self._listen_away_task = loop.create_task(
            self._away_after_listen_timeout(),
            name="listen-away-wait",
        )

    def _cancel_listen_away_wait(self) -> None:
        """Отменить отложенный старт окликов после перехода в listening.

        Returns:
            None.
        """
        task = self._listen_away_task
        self._listen_away_task = None
        if task is None or task.done():
            return
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if task is not current:
            task.cancel()

    async def _away_after_listen_timeout(self) -> None:
        """После ``silence_timeout`` с момента listening запустить оклики.

        Returns:
            None.
        """
        try:
            await asyncio.sleep(self._settings.silence_timeout)
        except asyncio.CancelledError:
            return

        if self._agent_state() != "listening":
            return
        if self._user_state() != "away":
            return
        self.on_user_away()

    def _schedule_finished(self) -> None:
        """Поставить обработку «бот договорил» в фон.

        Returns:
            None.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("Нет event loop для обработки конца реплики бота")
            return
        task = loop.create_task(self.on_agent_finished_speaking(), name="agent-finished")
        self._finished_tasks.add(task)
        task.add_done_callback(self._finished_tasks.discard)

    async def on_agent_finished_speaking(self) -> None:
        """После реплики бота: прочитать флаг продолжения и при необходимости запустить ход.

        Ранних выходов до чтения флага нет. Ошибка или таймаут чтения —
        считаем, что флага нет. Лимит ``max_continuations`` проверяется
        вместе с флагом.

        Returns:
            None.
        """
        flag = False
        try:
            if self._lg_client is not None and self._thread_id:
                flag = await expects_continuation(
                    self._lg_client,  # type: ignore[arg-type]
                    self._thread_id,
                )
        except Exception as exc:
            logger.info("[turn] продолжение: ошибка чтения флага: %s", exc)
            flag = False

        logger.info("[turn] продолжение: флаг прочитан, expect_continuation=%s", flag)

        if flag and self.continuation_count < self._settings.max_continuations:
            self.continuation_count += 1
            set_turn_kind(self._session.llm, "continuation")
            logger.info("[turn] продолжение: запущено #%s", self.continuation_count)
            cancelled = False
            try:
                await self._session.generate_reply()
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                set_turn_kind(self._session.llm, "client")
                if cancelled:
                    logger.info("[turn] продолжение: turn_kind возвращён в client после отмены")
            return

        if flag:
            logger.info(
                "[turn] продолжение: лимит исчерпан (count=%s max=%s)",
                self.continuation_count,
                self._settings.max_continuations,
            )
        self.continuation_count = 0
        set_turn_kind(self._session.llm, "client")

    async def _away_prompts(self) -> None:
        """Вернуть человека ходами ``silence``; после лимита — прощание и конец звонка.

        Пока попыток меньше ``silence_attempts``: выставить ``turn_kind=silence``,
        запустить обычный ход через ``generate_reply``, затем в ``finally``
        вернуть ``turn_kind=client`` (и при отмене тоже). Между попытками
        ждём ``silence_timeout``. Когда попытки исчерпаны — произнести
        ``silence_goodbye`` и завершить звонок. Отмена снаружи (клиент
        заговорил) прерывает цикл через ``CancelledError``.

        Returns:
            None.
        """
        try:
            max_attempts = self._settings.silence_attempts
            while self.silence_attempts < max_attempts:
                self.silence_attempts += 1
                logger.info(
                    "[turn] тишина: попытка #%s, ход silence",
                    self.silence_attempts,
                )
                set_turn_kind(self._session.llm, "silence")
                cancelled = False
                try:
                    await self._session.generate_reply()
                except asyncio.CancelledError:
                    cancelled = True
                    raise
                finally:
                    set_turn_kind(self._session.llm, "client")
                    if cancelled:
                        logger.info("[turn] тишина: turn_kind возвращён в client после отмены")

                await asyncio.sleep(self._settings.silence_timeout)

            goodbye = self._settings.silence_goodbye
            logger.info(
                "[turn] тишина: звонок завершается (исчерпаны попытки), фраза=%r",
                goodbye,
            )
            handle = self._session.say(goodbye)
            wait = getattr(handle, "wait_for_playout", None)
            if callable(wait):
                result = wait()
                if inspect.isawaitable(result):
                    await result
            self._ctx.delete_room()
        except asyncio.CancelledError:
            return


@server.rtc_session(
    agent_name=_rtc_session_agent_name(auto_accept=_AGENT_AUTO_ACCEPT, agent_name=_AGENT_NAME)
)
async def entrypoint(ctx: agents.JobContext) -> None:
    """Обработать один звонок: собрать сессию и начать разговор по сценарию.

    При ``llm_provider=agent`` первая фраза появляется не мгновенно: её
    генерирует модель (примерно 1.5–2 с после подключения). Это осознанный
    размен на единственную точку правды — вступление идёт из шага графа
    «Приветствие», а не из ``opening_line`` сценария бота.

    Args:
        ctx: контекст задания LiveKit (комната, участники).
    """
    settings = get_settings()  # здесь уже нужны все ключи — но и звонок реальный
    scenario = load_scenario(settings.scenario)
    logger.info("Старт звонка. Сценарий=%s llm=%s", scenario.id, settings.llm_provider)

    # room.name — thread_id для удалённого графа; при openai можно и без него.
    session = build_session(settings, room_name=ctx.room.name)
    _attach_latency_logging(session)
    _attach_partial_transcript_sender(session, settings=settings, room_name=ctx.room.name)
    _attach_turn_controller(session, ctx=ctx, settings=settings, room_name=ctx.room.name)

    # При llm_provider=agent промпт собирает сам граф; инструкции бота приедут
    # системным сообщением и подерутся с графским промптом. При openai —
    # сценарный промпт как раньше.
    instructions = "" if settings.llm_provider == "agent" else build_system_prompt(scenario)
    await session.start(room=ctx.room, agent=ScriptAgent(instructions))
    await _start_background_audio(room=ctx.room, session=session, settings=settings)

    # При работе с мозгом вступление — обычный первый шаг скрипта, поэтому
    # граф зовётся сразу на пустой истории. Дословная opening_line остаётся
    # только для старого пути llm_provider=openai.
    # Тишину после реплик детектит LiveKit (user_away_timeout).
    if settings.llm_provider == "agent":
        set_turn_kind(session.llm, "client")
        await session.generate_reply()
    else:
        await session.say(scenario.opening_line, allow_interruptions=False)


def _attach_turn_controller(
    session: AgentSession,
    *,
    ctx: agents.JobContext,
    settings: Settings,
    room_name: str,
) -> CallTurnController:
    """Подключить продолжение речи и реакцию на тишину к сессии звонка.

    Args:
        session: голосовая сессия текущего звонка.
        ctx: контекст задания (завершение комнаты после прощальной фразы).
        settings: таймаут тишины, лимит попыток и продолжений.
        room_name: имя комнаты LiveKit для ``thread_id`` основного графа.

    Returns:
        Контроллер ходов (уже подписан на события сессии).
    """
    lg_client = None
    thread_id = None
    if settings.llm_provider == "agent":
        lg_client = build_agent_langgraph_client(settings=settings)
        thread_id = thread_id_for_room(room_name)

    controller = CallTurnController(
        session=session,
        ctx=ctx,
        settings=settings,
        thread_id=thread_id,
        lg_client=lg_client,
    )
    controller.attach()
    return controller


def _assistant_text(item: object) -> str:
    """Достать текст реплики ассистента из элемента разговора.

    Args:
        item: элемент из ``conversation_item_added`` (обычно ``ChatMessage``).

    Returns:
        Текст без крайних пробелов или пустая строка, если текста нет.
    """
    text = getattr(item, "text_content", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    content = getattr(item, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = [part for part in content if isinstance(part, str) and part.strip()]
        if parts:
            return " ".join(parts).strip()
    return ""


def _log_assistant_latency(item: object) -> None:
    """Залогировать текст ответа бота и e2e-задержку, если есть.

    Через ``conversation_item_added`` приходят и служебные объекты без ``role``
    (например, ``AgentHandoff``) — их безопасно игнорируем.

    Args:
        item: элемент разговора из события ``conversation_item_added``.
    """
    role = getattr(item, "role", None)
    if role != "assistant":
        return

    text = _assistant_text(item)
    metrics = getattr(item, "metrics", None) or {}
    latency = metrics.get("e2e_latency") if isinstance(metrics, dict) else None

    if text and latency is not None:
        logger.info("Бот (e2e=%.2fс): %s", latency, text)
    elif text:
        logger.info("Бот: %s", text)
    elif latency is not None:
        logger.info("Задержка ответа: e2e=%.2fс", latency)


def _attach_latency_logging(session: AgentSession) -> None:
    """Логировать текст и задержку каждого ответа бота.

    Args:
        session: голосовая сессия, к которой вешается обработчик.
    """

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        _log_assistant_latency(ev.item)


def _partial_url_host(url: str) -> str:
    """Host[:port] из URL без query/фрагмента и токенов.

    Args:
        url: полный URL второй точки входа (может содержать секреты в query).

    Returns:
        Только ``netloc`` (например ``172.17.0.1:8127``), иначе исходная строка.
    """
    return urlparse(url).netloc or url


def _attach_partial_transcript_sender(
    session: AgentSession,
    *,
    settings: Settings,
    room_name: str,
) -> None:
    """Подключить фоновую отправку промежуточного STT, если флаг включён.

    При выключенном ``VOICE_BOT_AGENT_PARTIAL_ENABLED`` ничего не делает —
    поведение звонка как раньше. Хук сессии не трогает конвейер распознавания.

    Args:
        session: голосовая сессия текущего звонка.
        settings: настройки с флагом и адресом второй точки входа.
        room_name: имя комнаты LiveKit для того же ``thread_id``, что у хода.
    """
    enabled = settings.agent_partial_enabled
    logger.info(
        "[live] прогрев: enabled=%s, граф=%s, url=%s",
        str(enabled).lower(),
        settings.agent_partial_graph,
        _partial_url_host(settings.agent_partial_url),
    )
    if not enabled:
        logger.info("[live] фоновая отправка ВЫКЛЮЧЕНА")
        logger.info(
            "[live] сендер не подписан: живой режим выключен "
            "(VOICE_BOT_AGENT_PARTIAL_ENABLED=false)"
        )
        return

    sender = build_partial_transcript_sender(settings=settings, room_name=room_name)
    sender.attach(session)


async def _start_background_audio(
    *,
    room: rtc.Room,
    session: AgentSession,
    settings: Settings,
) -> BackgroundAudioPlayer | None:
    """Подключить тихий офисный эмбиент и звук клавиатуры в паузах thinking.

    Использует встроенные клипы LiveKit Agents (``BuiltinAudioClip``).
    При ``BG_ENABLED=false`` плеер не создаётся — звонок идёт без фона.

    Args:
        room: комната LiveKit текущего звонка.
        session: голосовая сессия агента.
        settings: настройки громкости и флага включения фона.

    Returns:
        Плеер фона или ``None``, если фон отключён.
    """
    if not settings.bg_enabled:
        logger.info("Фоновый звук отключён (BG_ENABLED=false)")
        return None

    player = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(
            BuiltinAudioClip.OFFICE_AMBIENCE,
            volume=settings.bg_ambient_volume,
        ),
        thinking_sound=AudioConfig(
            BuiltinAudioClip.KEYBOARD_TYPING,
            volume=settings.bg_thinking_volume,
        ),
    )
    await player.start(room=room, agent_session=session)
    logger.info(
        "Фоновый звук подключён: ambient=%.2f thinking=%.2f",
        settings.bg_ambient_volume,
        settings.bg_thinking_volume,
    )
    return player


def _probe_agent_service() -> None:
    """Разово проверить доступность агентского сервиса при старте воркера.

    Читает провайдер и URL из окружения без полной валидации настроек
    (как ``_AGENT_NAME``): служебные команды CLI не должны требовать секретов.
    Звонки не блокирует — только пишет результат в лог.
    """
    provider = os.getenv("VOICE_BOT_LLM_PROVIDER", "openai").strip().lower()
    if provider != "agent":
        return
    base_url = os.getenv("VOICE_BOT_AGENT_URL", "http://172.17.0.1:8127").rstrip("/")
    probe_url = f"{base_url}/ok"
    try:
        # trust_env=False: SOCKS5 из окружения воркера не должен перехватывать
        # локальный трафик к графу на том же хосте.
        response = httpx.get(probe_url, timeout=5.0, trust_env=False)
        logger.info("Агентский сервис %s → HTTP %s", probe_url, response.status_code)
    except Exception as exc:
        logger.warning("Агентский сервис недоступен (%s): %s", probe_url, exc)


if __name__ == "__main__":
    _probe_agent_service()
    agents.cli.run_app(server)
