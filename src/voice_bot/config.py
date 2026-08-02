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
    # Обязательны только при tts_provider="elevenlabs" — проверка в валидаторе
    # ниже. При провайдере "openai" ключ и голос ElevenLabs не нужны вовсе.
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="", alias="ELEVENLABS_VOICE_ID")
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

    # --- Синтез речи: ElevenLabs или OpenAI ---
    # "elevenlabs" — как было, записанный голос компании (поведение по умолчанию).
    # "openai" — запасной провайдер на том же ключе OPENAI_API_KEY: отдельная
    # оплата и отдельный ключ не нужны, но голос другой и синтез нестриминговый.
    tts_provider: Literal["elevenlabs", "openai"] = Field(
        default="elevenlabs", alias="VOICE_BOT_TTS_PROVIDER"
    )
    # Модель синтеза OpenAI. gpt-4o-mini-tts принимает инструкции по тону;
    # tts-1 и tts-1-hd их игнорируют.
    openai_tts_model: str = Field(default="gpt-4o-mini-tts", alias="OPENAI_TTS_MODEL")
    # Голос из закрытого списка OpenAI: alloy, ash, ballad, coral, echo, fable,
    # onyx, nova, sage, shimmer. Женские по звучанию — shimmer, nova, coral, sage.
    openai_tts_voice: str = Field(default="shimmer", alias="OPENAI_TTS_VOICE")
    # Тон и манера речи словами (только gpt-4o-mini-tts). Пусто — не передаём.
    openai_tts_instructions: str = Field(default="", alias="OPENAI_TTS_INSTRUCTIONS")
    # Скорость речи, допустимый диапазон 0.25–4.0. Единица — как в модели.
    openai_tts_speed: float = Field(default=1.0, alias="OPENAI_TTS_SPEED")

    # --- Живой режим: предподготовка по промежуточному STT (vector_checker) ---
    # Выключено по умолчанию (безопасно). На стенде — окружением:
    # VOICE_BOT_AGENT_PARTIAL_ENABLED=true (чекер + контекстер, пока клиент говорит).
    # При enabled=true нужны непустые URL и GRAPH — иначе падаем на старте.
    agent_partial_enabled: bool = Field(default=False, alias="VOICE_BOT_AGENT_PARTIAL_ENABLED")
    # URL LangGraph-сервера агента (мозга). Пустой дефолт: в бою нельзя молча
    # уехать на «чужой» адрес — URL задают явно в .env.
    agent_partial_url: str = Field(default="", alias="VOICE_BOT_AGENT_PARTIAL_URL")
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

    # --- Тишина клиента и продолжение собственной речи бота ---
    #: Через сколько тишины считать, что клиент пропал. Ноль и None — выключено.
    silence_timeout: float = Field(default=6.0, alias="VOICE_BOT_SILENCE_TIMEOUT")
    #: Что сказать перед завершением звонка, когда человек так и не ответил.
    silence_goodbye: str = Field(
        default="Видимо, что-то со связью. Попробуйте, пожалуйста, перезвонить.",
        alias="VOICE_BOT_SILENCE_GOODBYE",
    )
    #: Сколько раз пробуем вернуть человека в разговор, прежде чем прощаться.
    silence_attempts: int = Field(default=2, alias="VOICE_BOT_SILENCE_ATTEMPTS")
    #: Сколько продолжений подряд разрешено, страховка от монолога.
    max_continuations: int = Field(default=3, alias="VOICE_BOT_MAX_CONTINUATIONS")

    @model_validator(mode="after")
    def _validate_required_settings(self) -> Self:
        """Проверить обязательные поля: пустые строки ловим на старте.

        Pydantic требует наличие переменной без дефолта, но пустая строка
        из окружения (``VAR=``) проходит как валидное значение — из‑за этого
        в бою бот мог молча не слать в служебный граф. Здесь явно падаем.

        Returns:
            Те же настройки после проверки.

        Raises:
            ValueError: если обязательный секрет/URL пуст; если при
                ``tts_provider=elevenlabs`` не заданы ключ и идентификатор
                голоса; если при ``llm_provider=agent`` /
                ``stt_provider=service`` не заданы адреса; если живой режим
                включён без URL/имени графа.
        """
        required_non_empty: tuple[tuple[str, str], ...] = (
            ("LIVEKIT_URL", self.livekit_url),
            ("LIVEKIT_API_KEY", self.livekit_api_key),
            ("LIVEKIT_API_SECRET", self.livekit_api_secret),
            ("OPENAI_API_KEY", self.openai_api_key),
        )
        for env_name, value in required_non_empty:
            if not value.strip():
                raise ValueError(
                    f"{env_name} обязателен: пустое значение недопустимо "
                    "(задайте в окружении или .env)"
                )

        if self.tts_provider == "elevenlabs":
            if not self.elevenlabs_api_key.strip():
                raise ValueError(
                    "ELEVENLABS_API_KEY обязателен при VOICE_BOT_TTS_PROVIDER=elevenlabs: "
                    "без ключа нельзя синтезировать голос "
                    "(или переключитесь на VOICE_BOT_TTS_PROVIDER=openai)"
                )
            if not self.elevenlabs_voice_id.strip():
                raise ValueError(
                    "ELEVENLABS_VOICE_ID обязателен при VOICE_BOT_TTS_PROVIDER=elevenlabs: "
                    "без идентификатора голоса нельзя синтезировать речь "
                    "(или переключитесь на VOICE_BOT_TTS_PROVIDER=openai)"
                )

        if self.llm_provider == "agent":
            if not self.agent_url.strip():
                raise ValueError(
                    "VOICE_BOT_AGENT_URL обязателен при VOICE_BOT_LLM_PROVIDER=agent: "
                    "без URL нельзя подключить удалённый граф"
                )
            if not self.agent_graph.strip():
                raise ValueError(
                    "VOICE_BOT_AGENT_GRAPH обязателен при VOICE_BOT_LLM_PROVIDER=agent: "
                    "без имени графа нельзя подключить удалённый мозг"
                )

        if self.stt_provider == "service" and not self.stt_service_url.strip():
            raise ValueError(
                "VOICE_BOT_STT_SERVICE_URL обязателен при VOICE_BOT_STT_PROVIDER=service: "
                "без URL нельзя вызвать сервис распознавания"
            )

        if self.agent_partial_enabled:
            if not self.agent_partial_url.strip():
                raise ValueError(
                    "Живой режим включён (VOICE_BOT_AGENT_PARTIAL_ENABLED=true), "
                    "но не задан VOICE_BOT_AGENT_PARTIAL_URL"
                )
            if not self.agent_partial_graph.strip():
                raise ValueError(
                    "Живой режим включён (VOICE_BOT_AGENT_PARTIAL_ENABLED=true), "
                    "но не задан VOICE_BOT_AGENT_PARTIAL_GRAPH"
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
