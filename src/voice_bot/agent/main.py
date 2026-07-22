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
from voice_bot.agent.session import build_session
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
    """
    return "" if auto_accept else agent_name


@server.rtc_session(
    agent_name=_rtc_session_agent_name(auto_accept=_AGENT_AUTO_ACCEPT, agent_name=_AGENT_NAME)
)
async def entrypoint(ctx: agents.JobContext) -> None:
    """Обработать один звонок: собрать сессию и начать разговор по сценарию."""
    settings = get_settings()  # здесь уже нужны все ключи — но и звонок реальный
    scenario = load_scenario(settings.scenario)
    logger.info("Старт звонка. Сценарий=%s", scenario.id)

    session = build_session(settings)
    _attach_latency_logging(session)

    await session.start(room=ctx.room, agent=ScriptAgent(build_system_prompt(scenario)))
    await _start_background_audio(room=ctx.room, session=session, settings=settings)

    # Приветствие произносим дословно из сценария: первая фраза всегда
    # одинаковая и предсказуемая, её не отдаём на волю модели.
    await session.say(scenario.opening_line, allow_interruptions=False)


def _log_assistant_latency(item: object) -> None:
    """Залогировать e2e-задержку ответа ассистента, если метрика есть.

    Через ``conversation_item_added`` приходят и служебные объекты без ``role``
    (например, ``AgentHandoff``) — их безопасно игнорируем.
    """
    role = getattr(item, "role", None)
    metrics = getattr(item, "metrics", None) or {}
    latency = metrics.get("e2e_latency") if isinstance(metrics, dict) else None
    if role == "assistant" and latency is not None:
        logger.info("Задержка ответа: e2e=%.2fс", latency)


def _attach_latency_logging(session: AgentSession) -> None:
    """Логировать задержку каждого ответа бота — для контроля скорости."""

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        _log_assistant_latency(ev.item)


async def _start_background_audio(
    *,
    room: rtc.Room,
    session: AgentSession,
    settings: Settings,
) -> BackgroundAudioPlayer | None:
    """Подключить тихий офисный эмбиент и звук клавиатуры в паузах thinking.

    Использует встроенные клипы LiveKit Agents (``BuiltinAudioClip``).
    При ``BG_ENABLED=false`` плеер не создаётся — звонок идёт без фона.
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


if __name__ == "__main__":
    agents.cli.run_app(server)
