"""Распознавание речи через внешний сервис: подготовка звука, клиент, плагин."""

from voice_bot.stt.audio import PreparedAudio, prepare_audio
from voice_bot.stt.client import TranscriptionClient, TranscriptionServiceError
from voice_bot.stt.livekit_stt import (
    ServiceSTT,
    build_service_stt,
    build_transcription_client,
)

__all__ = [
    "PreparedAudio",
    "ServiceSTT",
    "TranscriptionClient",
    "TranscriptionServiceError",
    "build_service_stt",
    "build_transcription_client",
    "prepare_audio",
]
