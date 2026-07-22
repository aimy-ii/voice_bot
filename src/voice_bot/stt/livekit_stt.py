"""Адаптер сервиса распознавания под интерфейс LiveKit Agents.

LiveKit ждёт от плагина STT один метод — распознать кусок звука и вернуть
событие с текстом. Сервис распознаёт реплику целиком, поэтому поверх
адаптера ставится штатный ``StreamAdapter``: он режет поток по VAD и отдаёт
нам готовые реплики.
"""

from __future__ import annotations

import logging

from livekit.agents import (
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    NotGivenOr,
    stt,
    utils,
)
from livekit.agents import (
    vad as vad_module,
)

from voice_bot.stt.audio import prepare_audio
from voice_bot.stt.client import TranscriptionClient, TranscriptionServiceError
from voice_bot.stt.constants import MIN_UTTERANCE_SECONDS, STT_LOG_PREFIX

logger = logging.getLogger("voice_bot.stt")


class ServiceSTT(stt.STT):
    """Плагин LiveKit поверх внешнего сервиса распознавания.

    Потоковый режим не поддерживается осознанно: сервис распознаёт реплику
    целиком. Границы реплик даёт VAD через ``StreamAdapter``.

    Args:
        client: клиент сервиса распознавания.
        language: язык распознавания, попадает в событие для LiveKit.
    """

    def __init__(self, *, client: TranscriptionClient, language: str = "ru") -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._client = client
        self._language = language

    @property
    def model(self) -> str:
        """Имя модели — попадает в метрики LiveKit."""
        return "gigaam-v3"

    @property
    def provider(self) -> str:
        """Идентификатор провайдера — попадает в метрики LiveKit."""
        return "stt-service"

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Отправить реплику в сервис и вернуть событие LiveKit.

        Args:
            buffer: кадры реплики, склеенные VAD.
            language: язык от вызывающей стороны; сервис русский, поэтому
                значение только пробрасывается в ответ.
            conn_options: опции повторов LiveKit.

        Returns:
            Событие с финальной расшифровкой. При пустом распознавании текст
            пустой — LiveKit такое событие просто пропустит.

        Raises:
            APIConnectionError: если сервис недоступен; LiveKit сообщит об
                ошибке штатным путём вместо падения звонка.
        """
        audio = prepare_audio(buffer)
        duration = audio.duration_seconds

        if duration < MIN_UTTERANCE_SECONDS:
            logger.debug("%s Реплика короче порога: %.3fс", STT_LOG_PREFIX, duration)
            return self._build_event("")

        try:
            text = await self._client.transcribe(audio)
        except TranscriptionServiceError as exc:
            logger.error("%s %s", STT_LOG_PREFIX, exc)
            raise APIConnectionError(str(exc)) from exc

        return self._build_event(text)

    def _build_event(self, text: str) -> stt.SpeechEvent:
        """Собрать событие финальной расшифровки для LiveKit.

        Args:
            text: распознанный текст (возможно пустой).

        Returns:
            Событие типа ``FINAL_TRANSCRIPT`` с одной альтернативой.
        """
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[stt.SpeechData(language=self._language, text=text)],
        )


def build_transcription_client(*, base_url: str, timeout: float) -> TranscriptionClient:
    """Создать клиент сервиса распознавания.

    Args:
        base_url: адрес сервиса.
        timeout: сколько ждать ответа на одну реплику, секунды.

    Returns:
        Клиент; HTTP-сессия создаётся при первом обращении.
    """
    return TranscriptionClient(base_url=base_url, timeout=timeout)


def build_service_stt(
    *,
    client: TranscriptionClient,
    vad: vad_module.VAD,
    language: str = "ru",
) -> stt.STT:
    """Собрать готовый к подключению STT для голосовой сессии.

    Args:
        client: клиент сервиса распознавания.
        vad: тот же экземпляр VAD, что и у сессии — второй раз модель
            в память не грузим.
        language: язык распознавания.

    Returns:
        Плагин STT, который принимает ``AgentSession``.
    """
    return stt.StreamAdapter(stt=ServiceSTT(client=client, language=language), vad=vad)
