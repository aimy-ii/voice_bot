"""Настройка логирования для бота."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Настроить корневой логгер: единый формат, вывод в stdout.

    Формат компактный и читаемый в контейнере — одна строка на событие.

    Args:
        level: уровень логирования (``INFO``, ``DEBUG`` и т.д.).
    """
    logging.basicConfig(
        level=level.upper(),
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
