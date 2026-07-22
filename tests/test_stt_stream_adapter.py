"""Интеграционный тест StreamAdapter + ServiceSTT с заглушкой VAD."""

from __future__ import annotations

import pytest
from livekit import rtc
from livekit.agents import stt, vad
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from voice_bot.stt.audio import PreparedAudio
from voice_bot.stt.livekit_stt import ServiceSTT, build_service_stt


class _FakeClient:
    """Заглушка клиента: запоминает вызовы и возвращает фиксированный текст."""

    def __init__(self, text: str = "проверка связи") -> None:
        self.text = text
        self.calls: list[PreparedAudio] = []

    async def transcribe(self, audio: PreparedAudio) -> str:
        self.calls.append(audio)
        return self.text


class _FakeVADStream(vad.VADStream):
    """VAD-поток: по концу входа отдаёт START/END_OF_SPEECH со всеми кадрами."""

    def __init__(self, owner: vad.VAD) -> None:
        super().__init__(owner)
        self._frames: list[rtc.AudioFrame] = []

    async def _main_task(self) -> None:
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                continue
            self._frames.append(item)

        if not self._frames:
            return

        self._event_ch.send_nowait(
            vad.VADEvent(
                type=vad.VADEventType.START_OF_SPEECH,
                samples_index=0,
                timestamp=0.0,
                speech_duration=0.0,
                silence_duration=0.0,
                frames=list(self._frames),
                speaking=True,
            )
        )
        self._event_ch.send_nowait(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=0,
                timestamp=0.2,
                speech_duration=0.2,
                silence_duration=0.0,
                frames=list(self._frames),
                speaking=False,
            )
        )


class _FakeVAD(vad.VAD):
    """Детектор речи без Silero: сразу отдаёт накопленные кадры как реплику."""

    def __init__(self) -> None:
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.1))

    def stream(self) -> vad.VADStream:
        return _FakeVADStream(self)


def _pcm_frame(
    *,
    samples_per_channel: int = 3_200,
    sample_rate: int = 16_000,
    num_channels: int = 1,
) -> rtc.AudioFrame:
    """Кадр длиннее MIN_UTTERANCE_SECONDS (0.12 с), чтобы реплика ушла в сервис."""
    data = b"\x01\x00" * (samples_per_channel * num_channels)
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )


@pytest.mark.asyncio
async def test_stream_adapter_emits_final_transcript_once() -> None:
    """Через StreamAdapter приходит FINAL_TRANSCRIPT; клиент вызван один раз."""
    client = _FakeClient(text="проверка связи")
    fake_vad = _FakeVAD()
    plugin = build_service_stt(client=client, vad=fake_vad, language="ru")  # type: ignore[arg-type]

    assert isinstance(plugin, stt.StreamAdapter)
    assert isinstance(plugin.wrapped_stt, ServiceSTT)

    frame = _pcm_frame()
    stream = plugin.stream(language="ru", conn_options=DEFAULT_API_CONNECT_OPTIONS)
    stream.push_frame(frame)
    stream.end_input()

    events: list[stt.SpeechEvent] = []
    async for event in stream:
        events.append(event)

    await stream.aclose()

    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert len(finals) == 1
    assert finals[0].alternatives[0].text == "проверка связи"

    assert len(client.calls) == 1
    prepared = client.calls[0]
    assert prepared.sample_rate == 16_000
    assert prepared.num_channels == 1
    assert len(prepared.pcm) == len(bytes(frame.data))
