"""Сборка голосовой сессии из готовых плагинов.

Здесь мы соединяем четыре кубика:

- STT (OpenAI или свой сервис) — распознаёт речь клиента (звук → текст);
- LLM (OpenAI или удалённый граф LangGraph) — формулирует ответ (текст → текст);
- TTS (ElevenLabs) — озвучивает ответ голосом компании (текст → голос);
- VAD (Silero) + turn detection — ловят, когда клиент начал и закончил
  говорить.

Каждый кубик — сменный: чтобы поменять провайдера, достаточно заменить одну
строку в настройках, остальной код не меняется. Позже, когда важна будет
минимальная задержка, STT и LLM можно объединить в OpenAI Realtime (режим
«только текст»), не трогая TTS и остальную логику.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import aiohttp
import httpx
import openai as openai_sdk
from aiohttp_socks import ProxyConnector, ProxyType
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk.client import LangGraphClient
from livekit.agents import (
    AgentSession,
    TurnHandlingOptions,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit.agents import llm as llm_module
from livekit.agents import stt as stt_module
from livekit.plugins import elevenlabs, langchain, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from voice_bot.config import Settings
from voice_bot.stt import build_service_stt, build_transcription_client

logger = logging.getLogger("voice_bot.partial")

#: Собственное пространство имён UUID5 для ``thread_id`` из имени комнаты.
#:
#: Не используем стандартные ``uuid.NAMESPACE_*``: треды голосового бота
#: не должны пересекаться с чужими UUID5 от тех же строк в других системах.
#: Константу нельзя менять после выката — смена пространства имён даст
#: новые ``thread_id`` для тех же комнат, и все идущие треды на LangGraph
#: Server осиротеют (история диалога не подхватится при переподключении).
_ROOM_THREAD_NAMESPACE = uuid.UUID("7c3e9f2a-4b8d-4e1c-9f06-2a5d8c1b3e47")


def thread_id_for_room(room_name: str) -> str:
    """Собрать валидный ``thread_id`` LangGraph из имени комнаты LiveKit.

    LangGraph Server принимает только UUID. Имя комнаты (например,
    ``voice_assistant_room_5285``) — произвольная строка, поэтому строим
    детерминированный ``uuid5``: одна комната → всегда один тред, разные
    комнаты → разные треды, переподключение воркера продолжает разговор.

    Args:
        room_name: непустое имя комнаты LiveKit.

    Returns:
        Строка UUID для ``configurable.thread_id``.

    Raises:
        ValueError: если ``room_name`` пуст или из одних пробелов.
    """
    if not room_name or not room_name.strip():
        raise ValueError(
            "room_name обязателен для thread_id: пустое имя комнаты "
            "нельзя отобразить в UUID треда LangGraph"
        )
    return str(uuid.uuid5(_ROOM_THREAD_NAMESPACE, room_name))


def live_thread_id_for_room(room_name: str) -> str:
    """Собрать UUID лайв-треда ``vector_checker`` из имени комнаты LiveKit.

    Лайв-канал и канал генератора не должны делить один тред LangGraph:
    Server сериализует запуски на треде, и служебный run выталкивает
    основной ход. Здесь — отдельный детерминированный ``uuid5`` по
    ``"{room_name}#live"`` в том же ``_ROOM_THREAD_NAMESPACE``: одна
    комната → один лайв-тред, отличный от основного ``thread_id_for_room``.

    Args:
        room_name: непустое имя комнаты LiveKit.

    Returns:
        Строка UUID для лайв-треда ``runs.create``.

    Raises:
        ValueError: если ``room_name`` пуст или из одних пробелов.
    """
    if not room_name or not room_name.strip():
        raise ValueError(
            "room_name обязателен для live_thread_id: пустое имя комнаты "
            "нельзя отобразить в UUID треда LangGraph"
        )
    return str(uuid.uuid5(_ROOM_THREAD_NAMESPACE, f"{room_name}#live"))


def build_agent_langgraph_client(*, settings: Settings) -> LangGraphClient:
    """Собрать LangGraph-клиент к основному графу с ``trust_env=False``.

    Тот же способ, что у ``_build_agent_llm``: SOCKS5 из окружения воркера
    не должен перехватывать локальный трафик к мозгу на том же хосте.

    Args:
        settings: настройки с URL и таймаутом основного графа.

    Returns:
        Готовый ``LangGraphClient`` (например, для чтения state треда).
    """
    http_client = httpx.AsyncClient(
        base_url=settings.agent_url,
        timeout=httpx.Timeout(
            connect=5.0,
            read=settings.agent_timeout,
            write=settings.agent_timeout,
            pool=5.0,
        ),
        trust_env=False,
    )
    return LangGraphClient(http_client)


async def expects_continuation(client: LangGraphClient, thread_id: str) -> bool:
    """Обещал ли мозг продолжить реплику на следующем ходу.

    Args:
        client: клиент LangGraph, тот же, что и для основного графа.
        thread_id: идентификатор треда звонка.

    Returns:
        True — надо запустить следующий ход без реплики клиента.
        Любая ошибка чтения — False: молчание безопаснее лишнего хода.
    """
    try:
        state = await client.threads.get_state(thread_id)
    except Exception as exc:
        logger.warning(
            "Не удалось прочитать expect_continuation (thread_id=%s): %s",
            thread_id,
            exc,
        )
        return False

    values = state.get("values") if isinstance(state, dict) else getattr(state, "values", None)
    if not isinstance(values, dict):
        logger.info(
            "expect_continuation отсутствует в state (thread_id=%s): values=%s",
            thread_id,
            type(values).__name__,
        )
        return False

    flag = values.get("expect_continuation")
    if flag is None:
        logger.info("expect_continuation нет в state (thread_id=%s)", thread_id)
        return False
    return bool(flag)


def set_turn_kind(llm: object, turn_kind: str) -> None:
    """Записать ``turn_kind`` в ``configurable`` адаптера LLM (рядом с thread_id).

    Args:
        llm: плагин сессии (ожидается ``LLMAdapter`` с ``_config``).
        turn_kind: ``client`` для обычного хода или ``continuation`` для
            продолжения без реплики клиента.
    """
    config = getattr(llm, "_config", None)
    if not isinstance(config, dict):
        return
    configurable = config.setdefault("configurable", {})
    if not isinstance(configurable, dict):
        return
    configurable["turn_kind"] = turn_kind


class PartialTranscriptSender:
    """Фоновая отправка накопленного STT на служебный граф ``vector_checker``.

    Слушает ``user_input_transcribed`` сессии LiveKit: туда уже приходит
    и промежуточный, и финальный текст, без вмешательства в конвейер STT.
    Каждый раз уходит **весь** накопленный текст реплики в поле
    ``partial_reply``, не дельта. Вместе с текстом — ``partial_utterance_id``
    (свой на реплику) и ``partial_is_final``. HTTP ставится через
    ``asyncio.create_task`` — обработчик хода не ждёт ответа; ошибки и
    таймауты только в лог. Служебные run идут на отдельном лайв-треде со
    ``multitask_strategy="interrupt"``; скрипт в кеше мозга ключуется
    через ``call_id`` (UUID основного хода), без общей очереди запусков.
    """

    def __init__(
        self,
        *,
        client: LangGraphClient,
        graph: str,
        thread_id: str,
        call_id: str,
    ) -> None:
        """Сохранить клиент и идентификаторы для фоновых ``runs.create``.

        Args:
            client: LangGraph SDK-клиент ко второй точке входа.
            graph: имя графа/ассистента предподготовки.
            thread_id: UUID лайв-треда (не тред основного хода).
            call_id: UUID звонка / основного хода; ключ скрипта в кеше.
        """
        self._client = client
        self._graph = graph
        self._thread_id = thread_id
        self._call_id = call_id
        self._sending = True
        self._committed = ""
        self._last_sent = ""
        # Стабилен внутри реплики; новый UUID только при сбросе в on_user_state.
        self._utterance_id = str(uuid.uuid4())
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def thread_id(self) -> str:
        """Лайв-``thread_id``, с которым уходят фоновые run."""
        return self._thread_id

    @property
    def call_id(self) -> str:
        """Идентификатор звонка (тред основного хода) для кеша скрипта."""
        return self._call_id

    def attach(self, session: AgentSession) -> None:
        """Подписать обработчики на события сессии (не блокирует STT).

        Args:
            session: голосовая сессия текущего звонка.
        """

        @session.on("user_input_transcribed")
        def _on_transcript(ev: UserInputTranscribedEvent) -> None:
            self.on_transcript(ev)

        @session.on("user_state_changed")
        def _on_user_state(ev: UserStateChangedEvent) -> None:
            self.on_user_state(ev)

    def on_transcript(self, ev: UserInputTranscribedEvent) -> None:
        """Обновить накопленный текст и поставить неблокирующую отправку.

        Args:
            ev: промежуточный (``is_final=False``) или сегментный финал STT.
        """
        if not self._sending:
            return
        chunk = (ev.transcript or "").strip()
        if not chunk:
            return

        if ev.is_final:
            self._committed = f"{self._committed} {chunk}".strip()
            payload = self._committed
        else:
            # Interim в LiveKit — только незавершённый хвост после финалов.
            payload = f"{self._committed} {chunk}".strip() if self._committed else chunk

        if not payload or payload == self._last_sent:
            return
        self._last_sent = payload
        self._schedule_send(
            payload,
            utterance_id=self._utterance_id,
            is_final=ev.is_final,
        )

    def on_user_state(self, ev: UserStateChangedEvent) -> None:
        """Сбросить буфер и снова разрешить отправку на новой реплике.

        Args:
            ev: смена состояния пользователя.
        """
        if ev.new_state == "speaking":
            self._sending = True
            self._committed = ""
            self._last_sent = ""
            self._utterance_id = str(uuid.uuid4())

    def _schedule_send(self, text: str, *, utterance_id: str, is_final: bool) -> None:
        """Поставить отправку в фон; исключения гасятся в done-callback.

        Args:
            text: накопленный текст реплики целиком.
            utterance_id: идентификатор реплики на момент постановки.
            is_final: ``True``, если кусок — финальный распознанный текст.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "Нет event loop для отправки partial-текста (thread_id=%s)",
                self._thread_id,
            )
            return

        task = loop.create_task(
            self._send(text, utterance_id=utterance_id, is_final=is_final),
            name="agent-partial-transcript",
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_send_done)

    def _on_send_done(self, task: asyncio.Task[Any]) -> None:
        """Убрать задачу из набора и залогировать необработанную ошибку.

        Args:
            task: завершившаяся фоновая отправка.
        """
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "Ошибка фоновой отправки partial-текста (thread_id=%s): %s",
                self._thread_id,
                exc,
            )

    async def _send(self, text: str, *, utterance_id: str, is_final: bool) -> None:
        """Поставить фоновый run на ``vector_checker``; ответ не читаем.

        Граф читает накопленный текст из ``partial_reply``. Новый служебный
        проход идёт со ``multitask_strategy="interrupt"``: незавершённый
        предыдущий на лайв-треде отменяется; канал генератора не затрагивается.
        ``call_id`` в ``configurable`` указывает на скрипт того же звонка.
        ``partial_utterance_id`` стабилен внутри реплики; мозг сбрасывает
        точку отсчёта прироста при смене идентификатора.

        Args:
            text: накопленный распознанный текст целиком.
            utterance_id: идентификатор реплики на момент постановки отправки.
            is_final: ``True``, если это финальный распознанный текст реплики.
        """
        preview = text[:40]
        logger.info(
            "[live] отправка в %s: %d симв., «%s» "
            "(live_thread=%s, call_id=%s, utterance_id=%s, is_final=%s)",
            self._graph,
            len(text),
            preview,
            self._thread_id,
            self._call_id,
            utterance_id,
            is_final,
        )
        try:
            await self._client.runs.create(
                thread_id=self._thread_id,
                assistant_id=self._graph,
                input={
                    "partial_reply": text,
                    "partial_utterance_id": utterance_id,
                    "partial_is_final": is_final,
                },
                config={"configurable": {"call_id": self._call_id}},
                if_not_exists="create",
                # Свежий кусок важнее очереди устаревших гипотез на лайв-треде.
                multitask_strategy="interrupt",
            )
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            reason = getattr(response, "reason_phrase", None) or getattr(response, "reason", None)
            if status is not None:
                suffix = reason if reason else type(exc).__name__
                detail = f"код={status}, {suffix}"
                logger.info("[live] отправка в %s: ошибка (%s)", self._graph, detail)
            else:
                logger.info(
                    "[live] отправка в %s: ошибка (%s: %s)",
                    self._graph,
                    type(exc).__name__,
                    str(exc)[:120],
                )
            logger.warning(
                "Не удалось отправить partial-текст на %s (thread_id=%s): %s",
                self._graph,
                self._thread_id,
                exc,
            )
            return
        logger.info("[live] отправка в %s: успех", self._graph)


def build_partial_transcript_sender(
    *,
    settings: Settings,
    room_name: str,
) -> PartialTranscriptSender:
    """Собрать отправителя промежуточного текста для комнаты.

    HTTP-клиент с ``trust_env=False``: как у основного графа, SOCKS5 из
    окружения воркера не должен перехватывать локальный трафик к агенту.
    Лайв-тред — ``live_thread_id_for_room``; ``call_id`` — UUID основного
    хода той же комнаты, чтобы оба канала делили скрипт в кеше.

    Args:
        settings: настройки с URL/именем второй точки входа и таймаутом.
        room_name: имя комнаты LiveKit; из него собираются лайв-тред и
            ``call_id``.

    Returns:
        Готовый ``PartialTranscriptSender`` (ещё не подписан на сессию).

    Raises:
        ValueError: если ``room_name`` пуст (как у ``thread_id_for_room``).
    """
    call_id = thread_id_for_room(room_name)
    live_thread_id = live_thread_id_for_room(room_name)
    http_client = httpx.AsyncClient(
        base_url=settings.agent_partial_url,
        timeout=httpx.Timeout(
            connect=5.0,
            read=settings.agent_partial_timeout,
            write=settings.agent_partial_timeout,
            pool=5.0,
        ),
        trust_env=False,
    )
    client = LangGraphClient(http_client)
    return PartialTranscriptSender(
        client=client,
        graph=settings.agent_partial_graph,
        thread_id=live_thread_id,
        call_id=call_id,
    )


def _build_openai_client(*, api_key: str, proxy_url: str) -> openai_sdk.AsyncClient:
    """Собрать OpenAI AsyncClient с httpx-транспортом через SOCKS5.

    Клиент живёт до завершения процесса воркера (один воркер = один процесс);
    явный lifecycle не требуется.

    Args:
        api_key: ключ OpenAI API.
        proxy_url: SOCKS5-URL прокси (``socks5h://...``).

    Returns:
        Настроенный асинхронный клиент OpenAI.
    """
    http_client = httpx.AsyncClient(
        proxy=proxy_url,
        timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=50),
    )
    return openai_sdk.AsyncClient(
        api_key=api_key,
        http_client=http_client,
        max_retries=0,
    )


def _build_elevenlabs_session(*, proxy_fields: dict[str, object]) -> aiohttp.ClientSession:
    """Собрать aiohttp-сессию с SOCKS5-коннектором для ElevenLabs.

    Использует ``ProxyConnector(proxy_type=SOCKS5, host=..., port=..., rdns=True)``,
    а не URL со схемой socks5h — python_socks её не понимает.

    Сессия живёт до завершения процесса воркера (один воркер = один процесс);
    явный lifecycle не требуется.

    Args:
        proxy_fields: поля SOCKS5 из ``Settings.proxy_fields``.

    Returns:
        aiohttp-сессия с прокси-коннектором.
    """
    connector = ProxyConnector(proxy_type=ProxyType.SOCKS5, **proxy_fields)
    return aiohttp.ClientSession(connector=connector)


def _build_stt(
    *,
    settings: Settings,
    stt_kwargs: dict[str, object],
    vad: silero.VAD,
) -> stt_module.STT:
    """Выбрать провайдера распознавания речи по настройкам.

    ``openai`` — облачная модель через прокси (поведение по умолчанию, как
    было). ``service`` — локальный транскрибатор в контуре: наружу ничего не
    уходит, прокси не нужен, минуты не тарифицируются.

    Args:
        settings: настройки приложения.
        stt_kwargs: параметры облачного плагина OpenAI.
        vad: детектор речи сессии; по нему режутся реплики для сервиса.

    Returns:
        Готовый плагин STT для ``AgentSession``.
    """
    if settings.stt_provider == "service":
        client = build_transcription_client(
            base_url=settings.stt_service_url,
            timeout=settings.stt_service_timeout,
        )
        return build_service_stt(client=client, vad=vad, language=settings.language)
    return openai.STT(**stt_kwargs)


def _build_agent_llm(*, settings: Settings, room_name: str) -> llm_module.LLM:
    """Собрать «мозг» на удалённом графе LangGraph.

    Граф сам ведёт разговор, ходит в справочник и собирает промпт. Плагин
    LiveKit озвучивает только поток ``stream_mode="custom"`` (строка или
    словарь с ``content``); служебные события без ``content`` отбрасываются.

    ``thread_id`` фиксируется в конструкторе ``LLMAdapter``, поэтому адаптер
    создаётся на каждый звонок. Тот же UUID (``thread_id_for_room``) использует
    служебный ``PartialTranscriptSender`` / ``vector_checker``.

    ``VOICE_BOT_AGENT_MAIN_MULTITASK`` зарезервирован как точка расширения:
    ``LLMAdapter`` плагина LiveKit не принимает и не пробрасывает
    ``multitask_strategy`` в ``RemoteGraph.astream`` / ``runs.stream``.
    Пока плагин не доработан, старт основного хода **не гарантирует**
    отмену идущего служебного ``vector_checker`` (возможен ``enqueue``).

    HTTP-клиент к графу — с ``trust_env=False``: в окружении воркера задан
    SOCKS5-прокси для ElevenLabs/OpenAI, а агентский сервис живёт на том
    же хосте и в прокси заворачивать нельзя.

    Args:
        settings: настройки с URL, именем графа и таймаутом.
        room_name: имя комнаты LiveKit; из него собирается ``thread_id``.

    Returns:
        ``LLMAdapter`` над ``RemoteGraph``, готовый для ``AgentSession``.
    """
    # Свой клиент: get_client из langgraph_sdk создаёт httpx с trust_env=True
    # (по умолчанию). Даже при явном transport прокси из окружения — ловушка
    # при смене версии SDK; отключаем чтение env явно.
    http_client = httpx.AsyncClient(
        base_url=settings.agent_url,
        timeout=httpx.Timeout(
            connect=5.0,
            read=settings.agent_timeout,
            write=settings.agent_timeout,
            pool=5.0,
        ),
        trust_env=False,
    )
    graph = RemoteGraph(
        settings.agent_graph,
        client=LangGraphClient(http_client),
    )
    return langchain.LLMAdapter(
        graph=graph,
        config={
            "configurable": {
                "thread_id": thread_id_for_room(room_name),
                "turn_kind": "client",
            }
        },
        stream_mode="custom",
        subgraphs=False,
    )


def _build_llm(
    *,
    settings: Settings,
    llm_kwargs: dict[str, object],
    room_name: str | None,
) -> llm_module.LLM:
    """Выбрать провайдера «мозга» по настройкам.

    ``openai`` — облачная модель (поведение по умолчанию). ``agent`` —
    удалённый граф LangGraph; без ``room_name`` собирать нельзя: иначе
    ``thread_id`` будет общим и разговоры перемешаются.

    Args:
        settings: настройки приложения.
        llm_kwargs: параметры облачного плагина OpenAI.
        room_name: имя комнаты LiveKit (обязателен при провайдере ``agent``).

    Returns:
        Готовый плагин LLM для ``AgentSession``.

    Raises:
        ValueError: если провайдер ``agent``, а ``room_name`` не передан.
    """
    if settings.llm_provider == "agent":
        if not room_name:
            raise ValueError(
                "room_name обязателен при VOICE_BOT_LLM_PROVIDER=agent: "
                "без него thread_id будет общим на все звонки"
            )
        return _build_agent_llm(settings=settings, room_name=room_name)
    return openai.LLM(**llm_kwargs)


def build_session(settings: Settings, *, room_name: str | None = None) -> AgentSession:
    """Создать :class:`AgentSession` с провайдерами из настроек.

    Ключи API передаются в плагины явно из ``settings``, чтобы не зависеть
    от внутренних имён переменных окружения провайдеров (например,
    ``ELEVEN_API_KEY`` у ElevenLabs vs ``ELEVENLABS_API_KEY`` в проекте).

    Если задан прокси, внешний трафик OpenAI (``proxy_url`` / httpx) и
    ElevenLabs (``proxy_fields`` / aiohttp_socks) идёт через SOCKS5;
    иначе клиенты работают напрямую. Трафик к агентскому графу в прокси
    не заворачивается.

    Args:
        settings: настройки приложения (модели, язык, идентификатор голоса, ключи).
        room_name: имя комнаты LiveKit; из него через ``thread_id_for_room``
            собирается ``thread_id`` при ``llm_provider=agent``.
            Для ``openai`` можно не передавать.

    Returns:
        Готовая к запуску голосовая сессия.
    """
    stt_kwargs: dict[str, object] = {
        "model": settings.stt_model,
        "language": settings.language,
        "api_key": settings.openai_api_key,
    }
    llm_kwargs: dict[str, object] = {
        "model": settings.llm_model,
        "api_key": settings.openai_api_key,
    }
    tts_kwargs: dict[str, object] = {
        "voice_id": settings.elevenlabs_voice_id,
        "model": settings.tts_model,
        "api_key": settings.elevenlabs_api_key,
        # Ровный тон между фразами: стабильность/похожесть из настроек, без style.
        "voice_settings": elevenlabs.VoiceSettings(
            stability=settings.elevenlabs_stability,
            similarity_boost=settings.elevenlabs_similarity,
            style=settings.elevenlabs_style,
            speed=1.0,
        ),
    }

    proxy_url = settings.proxy_url
    proxy_fields = settings.proxy_fields
    if proxy_url is not None and proxy_fields is not None:
        openai_client = _build_openai_client(
            api_key=settings.openai_api_key,
            proxy_url=proxy_url,
        )
        stt_kwargs["client"] = openai_client
        llm_kwargs["client"] = openai_client
        tts_kwargs["http_session"] = _build_elevenlabs_session(proxy_fields=proxy_fields)

    # VAD один на сессию: его же переиспользует StreamAdapter сервиса
    # распознавания, чтобы не держать в памяти вторую копию модели Silero.
    vad = silero.VAD.load()

    return AgentSession(
        # Речь → текст. Провайдер переключается в настройках, без правок кода.
        stt=_build_stt(settings=settings, stt_kwargs=stt_kwargs, vad=vad),
        # «Мозг»: OpenAI или удалённый граф — переключатель в настройках.
        llm=_build_llm(settings=settings, llm_kwargs=llm_kwargs, room_name=room_name),
        # Текст → голос. voice_id — тот самый записанный голос компании.
        tts=elevenlabs.TTS(**tts_kwargs),
        # Слышит границы речи (начал/закончил говорить).
        vad=vad,
        # Определяет, что реплика клиента завершена (мультиязычная модель).
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
    )
