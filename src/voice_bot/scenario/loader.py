"""Загрузка сценария из JSON-файла."""

import json
from pathlib import Path

from .models import Scenario

_DATA_DIR = Path(__file__).parent / "data"


def load_scenario(name: str) -> Scenario:
    """Загрузить сценарий по имени файла (без ``.json``) из папки ``data``.

    Args:
        name: идентификатор сценария, например ``"vector_ru"``.

    Returns:
        Разобранный и провалидированный объект :class:`Scenario`.

    Raises:
        FileNotFoundError: если файла сценария нет.
    """
    path = _DATA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Сценарий не найден: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Scenario.model_validate(raw)
