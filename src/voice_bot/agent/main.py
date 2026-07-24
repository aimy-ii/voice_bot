"""Точка входа голосового бота (воркер LiveKit).

Запуск:

    python -m voice_bot.agent.main console   # локальный тест в терминале (микрофон)
    python -m voice_bot.agent.main dev        # подключиться к LiveKit-серверу
    python -m voice_bot.agent.main start       # продакшн-режим

Воркер подключается к LiveKit-серверу (адрес в ``LIVEKIT_URL``) и, когда
начинается звонок, ведёт разговор по сценарию.
"""

import logging
import os

import httpx
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    ConversationItemAddedEvent,
)

from voice_bot.agent.script_agent import ScriptAgent
from voice_bot.agent.session import build_partial_transcript_sender, build_session
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


@server.rtc_session(
    agent_name=_rtc_session_agent_name(auto_accept=_AGENT_AUTO_ACCEPT, agent_name=_AGENT_NAME)
)
async def entrypoint(ctx: agents.JobContext) -> None:
    """Обработать один звонок: собрать сессию и начать разговор по сценарию.

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

    # При llm_provider=agent промпт собирает сам граф; инструкции бота приедут
    # системным сообщением и подерутся с графским промптом. При openai —
    # сценарный промпт как раньше.
    instructions = "" if settings.llm_provider == "agent" else build_system_prompt(scenario)
    await session.start(room=ctx.room, agent=ScriptAgent(instructions))
    await _start_background_audio(room=ctx.room, session=session, settings=settings)

    # Приветствие произносим дословно из сценария: первая фраза всегда
    # одинаковая и предсказуемая. Не generate_reply — иначе граф вызовется
    # на пустой истории и заговорит вместо opening_line.
    await session.say(scenario.opening_line, allow_interruptions=False)


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
    if not settings.agent_partial_enabled:
        return

    sender = build_partial_transcript_sender(settings=settings, room_name=room_name)
    sender.attach(session)
    logger.info(
        "Partial STT → %s/%s (thread_id=%s)",
        settings.agent_partial_url.rstrip("/"),
        settings.agent_partial_graph,
        sender.thread_id,
    )


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
