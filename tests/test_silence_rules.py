"""Офлайн-тесты чистых правил выбора паузы при молчании."""

from voice_bot.agent.silence import has_question, last_agent_text, pick_pause


def test_last_agent_text_takes_latest_ai_before_human() -> None:
    """Берёт последнюю реплику бота, даже если после неё есть реплики человека."""
    history = [
        {"type": "ai", "content": "Первая"},
        {"type": "human", "content": "Да"},
        {"type": "ai", "content": "Вторая реплика"},
        {"type": "human", "content": "Нет"},
    ]
    assert last_agent_text(history) == "Вторая реплика"


def test_last_agent_text_empty_without_ai() -> None:
    """Без реплик бота — пустая строка."""
    history = [
        {"type": "human", "content": "Алло"},
        {"type": "human", "content": "Есть кто?"},
    ]
    assert last_agent_text(history) == ""
    assert last_agent_text([]) == ""


def test_has_question_finds_mark_in_middle() -> None:
    """Знак вопроса находится в середине реплики, не только в конце."""
    assert has_question("Как вас зовут? Уточните, пожалуйста.") is True
    assert has_question("Здравствуйте.") is False


def test_pick_pause_question() -> None:
    """Реплика с вопросом → pause_question и причина «вопрос»."""
    history = [{"type": "ai", "content": "Вам удобно сейчас?"}]
    pause, reason = pick_pause(history, pause_question=4.0, pause_statement=1.2)
    assert pause == 4.0
    assert reason == "вопрос"


def test_pick_pause_statement() -> None:
    """Реплика без вопроса → pause_statement и причина «без вопроса»."""
    history = [{"type": "ai", "content": "Хорошо, записала."}]
    pause, reason = pick_pause(history, pause_question=4.0, pause_statement=1.2)
    assert pause == 1.2
    assert reason == "без вопроса"


def test_pick_pause_empty_history_is_statement() -> None:
    """Пустая история → пауза без вопроса."""
    pause, reason = pick_pause([], pause_question=4.0, pause_statement=1.2)
    assert pause == 1.2
    assert reason == "без вопроса"
