"""Тесты загрузки сценария и сборки промпта (без сети и без ключей)."""

from voice_bot.scenario.loader import load_scenario
from voice_bot.scenario.prompt import build_system_prompt


def test_scenario_loads() -> None:
    """Сценарий vector_ru загружается и содержит шаги."""
    scenario = load_scenario("vector_ru")
    assert scenario.id == "vector_ru"
    assert scenario.persona.agent_name
    assert len(scenario.steps) >= 1


def test_prompt_mentions_persona() -> None:
    """В системном промпте есть имя агента и правило про русский язык."""
    scenario = load_scenario("vector_ru")
    prompt = build_system_prompt(scenario)
    assert scenario.persona.agent_name in prompt
    assert "русск" in prompt.lower()
