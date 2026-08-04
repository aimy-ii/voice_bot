"""Офлайн-тесты чистых правил выбора паузы при молчании."""

from voice_bot.agent.silence import has_question, last_agent_text, next_counts, pick_pause


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


def test_next_counts_question_increments_and_clears_statements() -> None:
    """Реплика с вопросом увеличивает счёт вопросов и обнуляет счёт без вопроса."""
    assert next_counts((0, 2), "Вам удобно?") == (1, 0)
    assert next_counts((1, 0), "Как вас зовут?") == (2, 0)


def test_next_counts_statement_increments_and_clears_questions() -> None:
    """Реплика без вопроса увеличивает второй счёт и обнуляет первый."""
    assert next_counts((2, 0), "Хорошо, записала.") == (0, 1)
    assert next_counts((0, 1), "Поняла.") == (0, 2)


def test_next_counts_alternation_resets_question_streak() -> None:
    """Чередование вопрос — не вопрос — вопрос даёт единицу по вопросам."""
    counts = (0, 0)
    counts = next_counts(counts, "Как вас зовут?")
    counts = next_counts(counts, "Хорошо, записала.")
    counts = next_counts(counts, "Вам удобно?")
    assert counts == (1, 0)


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
