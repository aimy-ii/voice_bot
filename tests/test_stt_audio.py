"""Тесты подготовки звука для сервиса распознавания."""

from livekit import rtc

from voice_bot.stt.audio import prepare_audio
from voice_bot.stt.constants import BYTES_PER_SAMPLE


def _make_frame(
    *,
    samples_per_channel: int,
    sample_rate: int = 16_000,
    num_channels: int = 1,
    fill: bytes = b"\x01\x00",
) -> rtc.AudioFrame:
    """Собрать кадр LiveKit с заданными параметрами потока."""
    sample = fill[: BYTES_PER_SAMPLE * num_channels]
    if len(sample) < BYTES_PER_SAMPLE * num_channels:
        sample = sample.ljust(BYTES_PER_SAMPLE * num_channels, b"\x00")
    data = sample * samples_per_channel
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )


def test_prepare_audio_single_frame_keeps_pcm_and_format() -> None:
    """Один кадр отдаётся сырым PCM с той же частотой и числом каналов."""
    frame = _make_frame(samples_per_channel=160, sample_rate=16_000, num_channels=1)

    prepared = prepare_audio(frame)

    assert prepared.pcm == bytes(frame.data)
    assert prepared.sample_rate == 16_000
    assert prepared.num_channels == 1


def test_prepare_audio_merges_frame_list() -> None:
    """Список кадров склеивается в один непрерывный PCM."""
    frames = [
        _make_frame(samples_per_channel=80, fill=b"\x11\x00"),
        _make_frame(samples_per_channel=80, fill=b"\x22\x00"),
    ]

    prepared = prepare_audio(frames)

    assert prepared.pcm == bytes(frames[0].data) + bytes(frames[1].data)
    assert prepared.sample_rate == 16_000
    assert prepared.num_channels == 1


def test_prepare_audio_duration_mono() -> None:
    """Длительность моно-реплики = samples / sample_rate."""
    # 1600 отсчётов при 16 кГц → 0.1 с.
    frame = _make_frame(samples_per_channel=1_600, sample_rate=16_000, num_channels=1)

    prepared = prepare_audio(frame)

    assert prepared.duration_seconds == 0.1


def test_prepare_audio_duration_stereo() -> None:
    """Длительность стерео учитывает два канала в размере PCM."""
    # 800 отсчётов на канал при 8 кГц → 0.1 с; в PCM вдвое больше байт.
    frame = _make_frame(samples_per_channel=800, sample_rate=8_000, num_channels=2)

    prepared = prepare_audio(frame)

    assert prepared.duration_seconds == 0.1
    assert len(prepared.pcm) == 800 * 2 * BYTES_PER_SAMPLE


def test_prepare_audio_empty_list() -> None:
    """Пустой список даёт пустую реплику с нулевой длительностью."""
    prepared = prepare_audio([])

    assert prepared.pcm == b""
    assert prepared.sample_rate == 0
    assert prepared.num_channels == 1
    assert prepared.duration_seconds == 0.0
