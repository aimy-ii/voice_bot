"""Конфигурация приложения.

Все настройки читаются из переменных окружения (или файла ``.env``).
Секреты (ключи API) в код НЕ пишутся — здесь только имена переменных.
Реальные значения приходят из окружения при запуске контейнера.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Типизированные настройки бота.

    Поля с алиасами в ВЕРХНЕМ регистре соответствуют именам переменных,
    которые мы запрашивали у команды (``LIVEKIT_*``, ``OPENAI_*``,
    ``ELEVENLABS_*``). Остальные — с префиксом ``VOICE_BOT_`` и разумными
    значениями по умолчанию, их можно не задавать.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LiveKit: транспорт (труба для звука) ---
    livekit_url: str = Field(alias="LIVEKIT_URL")
    livekit_api_key: str = Field(alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(alias="LIVEKIT_API_SECRET")

    # --- OpenAI: распознавание речи (STT) и «мозг» (LLM) ---
    openai_api_key: str = Field(alias="OPENAI_API_KEY")

    # --- ElevenLabs: синтез голоса (TTS) ---
    elevenlabs_api_key: str = Field(alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(alias="ELEVENLABS_VOICE_ID")

    # --- Необязательные настройки (со значениями по умолчанию) ---
    language: str = Field(default="ru", alias="VOICE_BOT_LANGUAGE")
    stt_model: str = Field(default="gpt-4o-mini-transcribe", alias="VOICE_BOT_STT_MODEL")
    llm_model: str = Field(default="gpt-4.1-mini", alias="VOICE_BOT_LLM_MODEL")
    tts_model: str = Field(default="eleven_flash_v2_5", alias="VOICE_BOT_TTS_MODEL")
    agent_name: str = Field(default="voice-bot", alias="VOICE_BOT_AGENT_NAME")
    scenario: str = Field(default="vector_ru", alias="VOICE_BOT_SCENARIO")
    log_level: str = Field(default="INFO", alias="VOICE_BOT_LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    """Вернуть единственный экземпляр настроек (кэшируется на процесс).

    Настройки читаются лениво — только при первом вызове. Это важно: пока
    функцию не вызвали, отсутствие ключей в окружении не мешает работе
    служебных команд (например, скачиванию весов моделей при сборке образа).
    """
    return Settings()  # type: ignore[call-arg]
