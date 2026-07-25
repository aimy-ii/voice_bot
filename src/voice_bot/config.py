"""Конфигурация приложения.

Все настройки читаются из переменных окружения (или файла ``.env``).
Секреты (ключи API) в код НЕ пишутся — здесь только имена переменных.
Реальные значения приходят из окружения при запуске контейнера.
"""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, model_validator
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
    # multilingual_v2 — стабильнее по тону на клонированном голосе, чем flash.
    tts_model: str = Field(default="eleven_multilingual_v2", alias="ELEVENLABS_MODEL")
    # Ровный деловой тон: высокая стабильность, без лишней экспрессии.
    elevenlabs_stability: float = Field(default=0.7, alias="ELEVENLABS_STABILITY")
    elevenlabs_similarity: float = Field(default=0.8, alias="ELEVENLABS_SIMILARITY")
    elevenlabs_style: float = Field(default=0.0, alias="ELEVENLABS_STYLE")

    # --- SOCKS5-прокси (опционально; обход региональных блокировок) ---
    proxy_host: str | None = Field(default=None, alias="PROXY_HOST")
    proxy_port: int | None = Field(default=None, alias="PROXY_PORT")
    proxy_user: str | None = Field(default=None, alias="PROXY_USER")
    proxy_pass: str | None = Field(default=None, alias="PROXY_PASS")

    # --- Необязательные настройки (со значениями по умолчанию) ---
    language: str = Field(default="ru", alias="VOICE_BOT_LANGUAGE")
    stt_model: str = Field(default="gpt-4o-mini-transcribe", alias="VOICE_BOT_STT_MODEL")
    llm_model: str = Field(default="gpt-4.1-mini", alias="VOICE_BOT_LLM_MODEL")

    # --- Распознавание речи: облачный OpenAI или свой сервис ---
    # "openai" — как было; "service" — локальный транскрибатор по HTTP.
    stt_provider: Literal["openai", "service"] = Field(
        default="openai", alias="VOICE_BOT_STT_PROVIDER"
    )
    # Сервис распознавания поднят отдельным стеком на том же сервере.
    stt_service_url: str = Field(
        default="http://172.17.0.1:8137", alias="VOICE_BOT_STT_SERVICE_URL"
    )
    # Сколько ждать ответа сервиса на одну реплику, секунды.
    stt_service_timeout: float = Field(default=15.0, alias="VOICE_BOT_STT_SERVICE_TIMEOUT")

    # --- «Мозг»: облачный OpenAI или удалённый граф LangGraph ---
    # "openai" — как было (поведение по умолчанию); "agent" — граф на LangGraph Server.
    llm_provider: Literal["openai", "agent"] = Field(
        default="openai", alias="VOICE_BOT_LLM_PROVIDER"
    )
    # Базовый URL агентского сервиса (отдельный стек на том же хосте).
    agent_url: str = Field(default="http://172.17.0.1:8127", alias="VOICE_BOT_AGENT_URL")
    # Имя графа из langgraph.json агентского сервиса.
    agent_graph: str = Field(default="vector_agent", alias="VOICE_BOT_AGENT_GRAPH")
    # Таймаут HTTP до графа, секунды (стриминговое соединение на весь ход).
    agent_timeout: float = Field(default=30.0, alias="VOICE_BOT_AGENT_TIMEOUT")

    # --- Живой режим: предподготовка по промежуточному STT (vector_checker) ---
    # Выключено по умолчанию (безопасно). На стенде — окружением:
    # VOICE_BOT_AGENT_PARTIAL_ENABLED=true (чекер + контекстер, пока клиент говорит).
    agent_partial_enabled: bool = Field(default=False, alias="VOICE_BOT_AGENT_PARTIAL_ENABLED")
    # URL LangGraph-сервера агента (вторая точка входа; может совпадать с AGENT_URL).
    agent_partial_url: str = Field(
        default="http://172.17.0.1:8127", alias="VOICE_BOT_AGENT_PARTIAL_URL"
    )
    # Имя служебного графа из langgraph.json агента.
    agent_partial_graph: str = Field(
        default="vector_checker", alias="VOICE_BOT_AGENT_PARTIAL_GRAPH"
    )
    # Таймаут HTTP на постановку фонового run (не ждём завершения графа).
    agent_partial_timeout: float = Field(default=5.0, alias="VOICE_BOT_AGENT_PARTIAL_TIMEOUT")
    # Стратегия multitask основного хода (точка расширения). Пусто — не задаём:
    # LLMAdapter/RemoteGraph плагина LiveKit сейчас не пробрасывают
    # multitask_strategy в runs.stream, поэтому отмена служебного
    # vector_checker живым ходом пока не гарантирована.
    agent_main_multitask: str = Field(default="", alias="VOICE_BOT_AGENT_MAIN_MULTITASK")

    agent_name: str = Field(default="voice-bot", alias="VOICE_BOT_AGENT_NAME")
    # true — автоподхват комнат (локальные тесты); false — только явный dispatch по имени.
    agent_auto_accept: bool = Field(default=True, alias="AGENT_AUTO_ACCEPT")
    scenario: str = Field(default="vector_ru", alias="VOICE_BOT_SCENARIO")
    log_level: str = Field(default="INFO", alias="VOICE_BOT_LOG_LEVEL")

    # --- Фоновый звук (офисный эмбиент + клавиатура в паузах thinking) ---
    bg_enabled: bool = Field(default=True, alias="BG_ENABLED")
    # Эмбиент слышен, но не перекрывает голос (при 0.15 почти не слышно).
    bg_ambient_volume: float = Field(default=0.4, alias="BG_AMBIENT_VOLUME")
    bg_thinking_volume: float = Field(default=0.6, alias="BG_THINKING_VOLUME")

    @model_validator(mode="after")
    def _validate_agent_llm_settings(self) -> Self:
        """Проверить обязательные поля при провайдере ``agent``.

        Падаем на старте воркера, а не посреди звонка: без URL графа
        собрать сессию всё равно нельзя.

        Returns:
            Те же настройки после проверки.

        Raises:
            ValueError: если ``llm_provider=agent``, а ``agent_url`` пуст;
                либо если включена предподготовка без URL/имени графа.
        """
        if self.llm_provider == "agent" and not self.agent_url.strip():
            raise ValueError(
                "VOICE_BOT_AGENT_URL обязателен при VOICE_BOT_LLM_PROVIDER=agent: "
                "без URL нельзя подключить удалённый граф"
            )
        if self.agent_partial_enabled:
            if not self.agent_partial_url.strip():
                raise ValueError(
                    "VOICE_BOT_AGENT_PARTIAL_URL обязателен при "
                    "VOICE_BOT_AGENT_PARTIAL_ENABLED=true: без URL нельзя "
                    "слать промежуточный текст на вторую точку входа"
                )
            if not self.agent_partial_graph.strip():
                raise ValueError(
                    "VOICE_BOT_AGENT_PARTIAL_GRAPH обязателен при "
                    "VOICE_BOT_AGENT_PARTIAL_ENABLED=true: без имени графа "
                    "нельзя вызвать вторую точку входа"
                )
        return self

    @property
    def proxy_url(self) -> str | None:
        """Собрать SOCKS5-URL прокси или None, если прокси не сконфигурирован.

        Формат socks5h:// — DNS резолвится на стороне прокси (нужно для обхода
        региональных блокировок).
        """
        if not (self.proxy_host and self.proxy_port):
            return None
        if self.proxy_user and self.proxy_pass:
            return (
                f"socks5h://{self.proxy_user}:{self.proxy_pass}@{self.proxy_host}:{self.proxy_port}"
            )
        return f"socks5h://{self.proxy_host}:{self.proxy_port}"

    @property
    def proxy_fields(self) -> dict[str, object] | None:
        """Поля SOCKS5-прокси для aiohttp_socks (host/port/логин/пароль).

        Возвращается отдельно от proxy_url, потому что python_socks не понимает
        схему socks5h. DNS-резолв на стороне прокси включается флагом rdns=True
        при создании коннектора, а не суффиксом h в схеме.
        """
        if not (self.proxy_host and self.proxy_port):
            return None
        fields: dict[str, object] = {
            "host": self.proxy_host,
            "port": self.proxy_port,
            "rdns": True,
        }
        if self.proxy_user and self.proxy_pass:
            fields["username"] = self.proxy_user
            fields["password"] = self.proxy_pass
        return fields


@lru_cache
def get_settings() -> Settings:
    """Вернуть единственный экземпляр настроек (кэшируется на процесс).

    Настройки читаются лениво — только при первом вызове. Это важно: пока
    функцию не вызвали, отсутствие ключей в окружении не мешает работе
    служебных команд (например, скачиванию весов моделей при сборке образа).
    """
    return Settings()  # type: ignore[call-arg]
