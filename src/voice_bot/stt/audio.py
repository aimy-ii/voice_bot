"""Подготовка звука для отправки в сервис распознавания.

Чистые функции: кадры LiveKit → сырой PCM int16, который принимает сервис.
Ни сети, ни моделей, ни глобального состояния — слой полностью покрывается
офлайн-тестами.
"""

from __future__ import annotations

from dataclasses import dataclass

from livekit import rtc
from livekit.agents.utils import AudioBuffer, merge_frames

from voice_bot.stt.constants import BYTES_PER_SAMPLE


@dataclass(frozen=True)
class PreparedAudio:
    """Реплика, готовая к отправке в сервис распознавания.

    Attributes:
        pcm: сырые отсчёты PCM signed 16-bit little-endian.
        sample_rate: частота дискретизации, Гц.
        num_channels: число каналов в потоке.
    """

    pcm: bytes
    sample_rate: int
    num_channels: int

    @property
    def duration_seconds(self) -> float:
        """Длительность реплики в секундах."""
        if self.sample_rate <= 0 or self.num_channels <= 0:
            return 0.0
        samples = len(self.pcm) / (BYTES_PER_SAMPLE * self.num_channels)
        return samples / self.sample_rate


def _to_frame_list(buffer: AudioBuffer) -> list[rtc.AudioFrame]:
    """Привести буфер LiveKit к списку кадров.

    Args:
        buffer: кадр или список кадров из LiveKit.

    Returns:
        Список кадров (возможно пустой).
    """
    if isinstance(buffer, list):
        return buffer
    return [buffer]


def prepare_audio(buffer: AudioBuffer) -> PreparedAudio:
    """Собрать из кадров LiveKit сырой PCM для сервиса распознавания.

    Кадры склеиваются в один и отдаются как есть: ресемплинг и сведение
    каналов делает сервис, здесь лишней работы не выполняем — это горячий
    участок цикла разговора.

    Args:
        buffer: кадр или список кадров, пришедших из LiveKit.

    Returns:
        Подготовленная реплика; для пустого буфера — нулевая длительность.
    """
    frames = _to_frame_list(buffer)
    if not frames:
        return PreparedAudio(pcm=b"", sample_rate=0, num_channels=1)

    merged = merge_frames(frames)
    return PreparedAudio(
        pcm=bytes(merged.data),
        sample_rate=merged.sample_rate,
        num_channels=merged.num_channels,
    )
