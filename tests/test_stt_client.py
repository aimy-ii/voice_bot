"""Тесты HTTP-клиента сервиса распознавания (реальный локальный aiohttp.web)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import web

from voice_bot.stt.audio import PreparedAudio
from voice_bot.stt.client import TranscriptionClient, TranscriptionServiceError
from voice_bot.stt.constants import READY_PATH, TRANSCRIBE_PCM_PATH


async def _start_app(app: web.Application) -> tuple[web.AppRunner, str]:
    """Поднять приложение на свободном порту 127.0.0.1.

    Returns:
        Кортеж (runner, base_url).
    """
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    assert sockets is not None
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


@pytest.fixture
async def transcribe_server() -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Локальный сервер транскрибации; в ``seen`` складывает последний запрос."""
    seen: dict[str, Any] = {}

    async def handle_transcribe(request: web.Request) -> web.Response:
        body = await request.read()
        seen["sample_rate"] = request.query.get("sample_rate")
        seen["num_channels"] = request.query.get("num_channels")
        seen["content_type"] = request.headers.get("Content-Type")
        seen["body"] = body
        return web.json_response(
            {
                "text": "  привет мир  ",
                "language": "ru",
                "duration_seconds": 0.5,
                "elapsed_seconds": 0.012,
                "model": "gigaam-v3",
            }
        )

    app = web.Application()
    app.router.add_post(TRANSCRIBE_PCM_PATH, handle_transcribe)
    runner, base_url = await _start_app(app)
    try:
        yield base_url, seen
    finally:
        await runner.cleanup()


@pytest.fixture
async def ready_server() -> AsyncIterator[str]:
    """Локальный сервер с ``GET /ready`` и ``model_loaded: true``."""

    async def handle_ready(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "model_loaded": True})

    app = web.Application()
    app.router.add_get(READY_PATH, handle_ready)
    runner, base_url = await _start_app(app)
    try:
        yield base_url
    finally:
        await runner.cleanup()


@pytest.fixture
async def error_server() -> AsyncIterator[str]:
    """Локальный сервер, отвечающий 500 на транскрибацию."""

    async def handle_transcribe(_request: web.Request) -> web.Response:
        return web.Response(status=500, text="internal boom")

    app = web.Application()
    app.router.add_post(TRANSCRIBE_PCM_PATH, handle_transcribe)
    runner, base_url = await _start_app(app)
    try:
        yield base_url
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_transcribe_sends_pcm_params_and_reads_text(
    transcribe_server: tuple[str, dict[str, Any]],
) -> None:
    """Клиент передаёт sample_rate/num_channels, Content-Type и читает text."""
    base_url, seen = transcribe_server
    client = TranscriptionClient(base_url=base_url)
    audio = PreparedAudio(pcm=b"\x01\x02\x03\x04", sample_rate=16_000, num_channels=1)

    try:
        text = await client.transcribe(audio)
    finally:
        await client.aclose()

    assert text == "привет мир"
    assert seen["sample_rate"] == "16000"
    assert seen["num_channels"] == "1"
    assert seen["content_type"] == "application/octet-stream"
    assert seen["body"] == audio.pcm


@pytest.mark.asyncio
async def test_transcribe_http_error_raises(
    error_server: str,
) -> None:
    """Ошибочный код ответа даёт TranscriptionServiceError."""
    client = TranscriptionClient(base_url=error_server)
    audio = PreparedAudio(pcm=b"\x00\x00", sample_rate=8_000, num_channels=1)

    try:
        with pytest.raises(TranscriptionServiceError, match="500"):
            await client.transcribe(audio)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_transcribe_unreachable_raises() -> None:
    """Недоступный сервис даёт TranscriptionServiceError."""
    # Порт, на котором никто не слушает.
    client = TranscriptionClient(base_url="http://127.0.0.1:1", timeout=0.5)
    audio = PreparedAudio(pcm=b"\x00\x00", sample_rate=8_000, num_channels=1)

    try:
        with pytest.raises(TranscriptionServiceError, match="Не удалось обратиться"):
            await client.transcribe(audio)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_is_ready_true(ready_server: str) -> None:
    """is_ready возвращает True при model_loaded: true."""
    client = TranscriptionClient(base_url=ready_server)

    try:
        assert await client.is_ready() is True
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_is_ready_false_when_silent() -> None:
    """is_ready возвращает False, когда сервис молчит."""
    client = TranscriptionClient(base_url="http://127.0.0.1:1", timeout=0.5)

    try:
        assert await client.is_ready() is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_transcribe_logs_recognized_text_at_info(
    transcribe_server: tuple[str, dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """При успешном ответе сервиса в лог INFO попадает распознанный текст."""
    base_url, _seen = transcribe_server
    client = TranscriptionClient(base_url=base_url)
    audio = PreparedAudio(pcm=b"\x01\x02\x03\x04", sample_rate=16_000, num_channels=1)

    with caplog.at_level(logging.INFO, logger="voice_bot.stt"):
        try:
            await client.transcribe(audio)
        finally:
            await client.aclose()

    assert "привет мир" in caplog.text


@pytest.mark.asyncio
async def test_transcribe_logs_silence_placeholder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """При пустом тексте в логе появляется ``<тишина>``."""

    async def handle_transcribe(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "text": "",
                "language": "ru",
                "duration_seconds": 0.1,
                "elapsed_seconds": 0.001,
                "model": "gigaam-v3",
            }
        )

    app = web.Application()
    app.router.add_post(TRANSCRIBE_PCM_PATH, handle_transcribe)
    runner, base_url = await _start_app(app)
    client = TranscriptionClient(base_url=base_url)
    audio = PreparedAudio(pcm=b"\x00\x00", sample_rate=8_000, num_channels=1)

    try:
        with caplog.at_level(logging.INFO, logger="voice_bot.stt"):
            text = await client.transcribe(audio)
    finally:
        await client.aclose()
        await runner.cleanup()

    assert text == ""
    assert "<тишина>" in caplog.text
