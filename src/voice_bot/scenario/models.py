"""Модель сценария разговора.

Сценарий — это ДАННЫЕ (JSON), а не код: персона, приветствие, шаги.
Так сценарий можно менять, не трогая логику бота. На этом этапе модель
намеренно простая. Позже сюда добавятся ветвления, обязательные дословные
блоки и правила перехода между шагами.
"""

from pydantic import BaseModel, Field


class Persona(BaseModel):
    """Кого отыгрывает бот: имя, компания, роль и тон речи."""

    agent_name: str
    company: str
    role: str
    tone: str


class Step(BaseModel):
    """Один шаг скрипта: какую цель закрыть и как её достичь."""

    id: str
    goal: str
    prompt: str


class Scenario(BaseModel):
    """Полный сценарий разговора."""

    id: str
    persona: Persona
    opening_line: str
    rules: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
