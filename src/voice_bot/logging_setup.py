"""Настройка уровней логирования для бота.

Обработчики вешает LiveKit CLI (``agents.cli.run_app``), причём делает это
ПОСЛЕ импорта модулей и не очищает уже существующие. Поэтому свой обработчик
мы не добавляем — иначе каждая запись печатается дважды. Здесь только уровни.
"""

import logging

#: Библиотеки, которые в DEBUG заваливают поток служебными событиями.
_NOISY_LOGGERS = (
    "livekit",
    "livekit.agents",
    "aiohttp",
    "httpx",
    "httpcore",
    "openai",
    "urllib3",
    "asyncio",
    "numba",
)


def setup_logging(level: str = "INFO", noisy_level: str = "WARNING") -> None:
    """Выставить уровни логирования, не трогая обработчики.

    Args:
        level: уровень для логгеров проекта (``voice_bot`` и вложенные).
        noisy_level: уровень для шумных библиотек — их служебные события
            в поток не пускаем.
    """
    logging.getLogger("voice_bot").setLevel(level.upper())
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level.upper())
