"""Правила пауз при молчании собеседника.

Чистые функции без сессии, звука и провайдеров: по снимку истории разговора
решают, сколько ждать ответа человека, прежде чем подать голос. Вынесены
отдельно, чтобы тестироваться офлайн без LiveKit и без ключей.
"""

from __future__ import annotations


def last_agent_text(history: list[dict[str, str]]) -> str:
    """Найти текст последней реплики бота в снимке истории.

    Args:
        history: снимок истории вида ``{"type": "human"|"ai", "content": "..."}``
            в хронологическом порядке.

    Returns:
        Текст последней реплики бота без пробелов по краям; пустая строка,
        если бот ещё ничего не говорил.
    """
    for item in reversed(history):
        if item.get("type") == "ai":
            return (item.get("content") or "").strip()
    return ""


def has_question(text: str) -> bool:
    """Проверить, есть ли в реплике знак вопроса.

    Знак ищется по всей строке, а не только в последнем символе: реплика может
    заканчиваться уточнением после вопроса.

    Args:
        text: текст реплики.

    Returns:
        ``True``, если знак вопроса встречается хотя бы один раз.
    """
    return "?" in text


def pick_pause(
    history: list[dict[str, str]],
    *,
    pause_question: float,
    pause_statement: float,
) -> tuple[float, str]:
    """Выбрать длину паузы по последней реплике бота.

    Args:
        history: снимок истории разговора.
        pause_question: пауза, если бот закончил вопросом.
        pause_statement: пауза, если вопроса не было.

    Returns:
        Пара «длина паузы в секундах, причина выбора для лога». Если реплик
        бота в истории нет, считается, что вопроса не было.
    """
    if has_question(last_agent_text(history)):
        return pause_question, "вопрос"
    return pause_statement, "без вопроса"
