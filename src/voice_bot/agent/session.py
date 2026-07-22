"""Сборка голосовой сессии из готовых плагинов.

Здесь мы соединяем четыре кубика:

- STT (OpenAI) — распознаёт речь клиента (звук → текст);
- LLM (OpenAI) — формулирует ответ (текст → текст);
- TTS (ElevenLabs) — озвучивает ответ голосом компании (текст → голос);
- VAD (Silero) + turn detection — ловят, когда клиент начал и закончил
  говорить.

Каждый кубик — сменный: чтобы поменять провайдера, достаточно заменить одну
строку, остальной код не меняется. Позже, когда важна будет минимальная
задержка, STT и LLM можно объединить в OpenAI Realtime (режим «только текст»),
не трогая TTS и остальную логику.
"""

from __future__ import annotations

import aiohttp
import httpx
import openai as openai_sdk
from aiohttp_socks import ProxyConnector, ProxyType
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.agents import stt as stt_module
from livekit.plugins import elevenlabs, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from voice_bot.config import Settings
from voice_bot.stt import build_service_stt, build_transcription_client


def _build_openai_client(*, api_key: str, proxy_url: str) -> openai_sdk.AsyncClient:
    """Собрать OpenAI AsyncClient с httpx-транспортом через SOCKS5.

    Клиент живёт до завершения процесса воркера (один воркер = один процесс);
    явный lifecycle не требуется.
    """
    http_client = httpx.AsyncClient(
        proxy=proxy_url,
        timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=50),
    )
    return openai_sdk.AsyncClient(
        api_key=api_key,
        http_client=http_client,
        max_retries=0,
    )


def _build_elevenlabs_session(*, proxy_fields: dict[str, object]) -> aiohttp.ClientSession:
    """Собрать aiohttp-сессию с SOCKS5-коннектором для ElevenLabs.

    Использует ``ProxyConnector(proxy_type=SOCKS5, host=..., port=..., rdns=True)``,
    а не URL со схемой socks5h — python_socks её не понимает.

    Сессия живёт до завершения процесса воркера (один воркер = один процесс);
    явный lifecycle не требуется.
    """
    connector = ProxyConnector(proxy_type=ProxyType.SOCKS5, **proxy_fields)
    return aiohttp.ClientSession(connector=connector)


def _build_stt(
    *,
    settings: Settings,
    stt_kwargs: dict[str, object],
    vad: silero.VAD,
) -> stt_module.STT:
    """Выбрать провайдера распознавания речи по настройкам.

    ``openai`` — облачная модель через прокси (поведение по умолчанию, как
    было). ``service`` — локальный транскрибатор в контуре: наружу ничего не
    уходит, прокси не нужен, минуты не тарифицируются.

    Args:
        settings: настройки приложения.
        stt_kwargs: параметры облачного плагина OpenAI.
        vad: детектор речи сессии; по нему режутся реплики для сервиса.

    Returns:
        Готовый плагин STT для ``AgentSession``.
    """
    if settings.stt_provider == "service":
        client = build_transcription_client(
            base_url=settings.stt_service_url,
            timeout=settings.stt_service_timeout,
        )
        return build_service_stt(client=client, vad=vad, language=settings.language)
    return openai.STT(**stt_kwargs)


def build_session(settings: Settings) -> AgentSession:
    """Создать :class:`AgentSession` с провайдерами из настроек.

    Ключи API передаются в плагины явно из ``settings``, чтобы не зависеть
    от внутренних имён переменных окружения провайдеров (например,
    ``ELEVEN_API_KEY`` у ElevenLabs vs ``ELEVENLABS_API_KEY`` в проекте).

    Если задан прокси, внешний трафик OpenAI (``proxy_url`` / httpx) и
    ElevenLabs (``proxy_fields`` / aiohttp_socks) идёт через SOCKS5;
    иначе клиенты работают напрямую.

    Args:
        settings: настройки приложения (модели, язык, идентификатор голоса, ключи).

    Returns:
        Готовая к запуску голосовая сессия.
    """
    stt_kwargs: dict[str, object] = {
        "model": settings.stt_model,
        "language": settings.language,
        "api_key": settings.openai_api_key,
    }
    llm_kwargs: dict[str, object] = {
        "model": settings.llm_model,
        "api_key": settings.openai_api_key,
    }
    tts_kwargs: dict[str, object] = {
        "voice_id": settings.elevenlabs_voice_id,
        "model": settings.tts_model,
        "api_key": settings.elevenlabs_api_key,
        # Ровный тон между фразами: стабильность/похожесть из настроек, без style.
        "voice_settings": elevenlabs.VoiceSettings(
            stability=settings.elevenlabs_stability,
            similarity_boost=settings.elevenlabs_similarity,
            style=settings.elevenlabs_style,
            speed=1.0,
        ),
    }

    proxy_url = settings.proxy_url
    proxy_fields = settings.proxy_fields
    if proxy_url is not None and proxy_fields is not None:
        openai_client = _build_openai_client(
            api_key=settings.openai_api_key,
            proxy_url=proxy_url,
        )
        stt_kwargs["client"] = openai_client
        llm_kwargs["client"] = openai_client
        tts_kwargs["http_session"] = _build_elevenlabs_session(proxy_fields=proxy_fields)

    # VAD один на сессию: его же переиспользует StreamAdapter сервиса
    # распознавания, чтобы не держать в памяти вторую копию модели Silero.
    vad = silero.VAD.load()

    return AgentSession(
        # Речь → текст. Провайдер переключается в настройках, без правок кода.
        stt=_build_stt(settings=settings, stt_kwargs=stt_kwargs, vad=vad),
        # «Мозг»: формулирует короткие реплики по системному промпту.
        llm=openai.LLM(**llm_kwargs),
        # Текст → голос. voice_id — тот самый записанный голос компании.
        tts=elevenlabs.TTS(**tts_kwargs),
        # Слышит границы речи (начал/закончил говорить).
        vad=vad,
        # Определяет, что реплика клиента завершена (мультиязычная модель).
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
    )
