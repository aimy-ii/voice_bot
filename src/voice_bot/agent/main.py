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
from livekit import agents
from livekit.agents import AgentServer, AgentSession, ConversationItemAddedEvent

from voice_bot.agent.script_agent import ScriptAgent
from voice_bot.agent.session import build_session
from voice_bot.config import get_settings
from voice_bot.logging_setup import setup_logging
from voice_bot.scenario.loader import load_scenario
from voice_bot.scenario.prompt import build_system_prompt

# .env загружаем в окружение ДО чтения настроек и инициализации плагинов:
# плагины OpenAI/ElevenLabs читают ключи напрямую из переменных окружения.
load_dotenv()
setup_logging(os.getenv("VOICE_BOT_LOG_LEVEL", "INFO"))
logger = logging.getLogger("voice_bot")

# Имя агента читаем без полной валидации настроек, чтобы служебные команды CLI
# (download-files, --help) работали и без секретов в окружении.
_AGENT_NAME = os.getenv("VOICE_BOT_AGENT_NAME", "voice-bot")

server = AgentServer()


@server.rtc_session(agent_name=_AGENT_NAME)
async def entrypoint(ctx: agents.JobContext) -> None:
    """Обработать один звонок: собрать сессию и начать разговор по сценарию."""
    settings = get_settings()  # здесь уже нужны все ключи — но и звонок реальный
    scenario = load_scenario(settings.scenario)
    logger.info("Старт звонка. Сценарий=%s", scenario.id)

    session = build_session(settings)
    _attach_latency_logging(session)

    await session.start(room=ctx.room, agent=ScriptAgent(build_system_prompt(scenario)))

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


if __name__ == "__main__":
    agents.cli.run_app(server)
