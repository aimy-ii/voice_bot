"""HTTP-клиент сервиса распознавания речи.

Бот не держит модель у себя: он отправляет реплику в сервис и получает текст.
Соединение переиспользуется на весь звонок, поэтому накладные расходы —
единицы миллисекунд.

Прокси здесь намеренно не используется: сервис свой, внутри контура.
``aiohttp`` по умолчанию не читает переменные окружения с прокси
(``trust_env=False``), так что настройки для ElevenLabs сюда не протекают.
"""

from __future__ import annotations

import logging

import aiohttp

from voice_bot.stt.audio import PreparedAudio
from voice_bot.stt.constants import (
    READY_PATH,
    REQUEST_TIMEOUT_SECONDS,
    STT_LOG_PREFIX,
    TRANSCRIBE_PCM_PATH,
)

logger = logging.getLogger("voice_bot.stt")


class TranscriptionServiceError(RuntimeError):
    """Сервис распознавания не ответил или вернул ошибку."""


class TranscriptionClient:
    """Клиент сервиса распознавания речи.

    Держит одну HTTP-сессию на весь процесс воркера: TCP-соединение
    переиспользуется между репликами, заново не устанавливается.

    Args:
        base_url: адрес сервиса, например ``http://172.17.0.1:8137``.
        timeout: сколько ждать ответа на одну реплику, секунды.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    @property
    def base_url(self) -> str:
        """Адрес сервиса распознавания."""
        return self._base_url

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Вернуть живую HTTP-сессию, создав её при первом обращении.

        Returns:
            Открытая сессия aiohttp.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def transcribe(self, audio: PreparedAudio) -> str:
        """Отправить реплику в сервис и получить текст.

        Args:
            audio: подготовленная реплика (сырой PCM и параметры потока).

        Returns:
            Распознанный текст; пустая строка, если речи не нашлось.

        Raises:
            TranscriptionServiceError: если сервис недоступен или вернул
                ошибочный код ответа.
        """
        session = await self._ensure_session()
        url = f"{self._base_url}{TRANSCRIBE_PCM_PATH}"
        params = {
            "sample_rate": str(audio.sample_rate),
            "num_channels": str(audio.num_channels),
        }

        try:
            async with session.post(
                url,
                params=params,
                data=audio.pcm,
                headers={"Content-Type": "application/octet-stream"},
            ) as response:
                if response.status != 200:
                    detail = await response.text()
                    raise TranscriptionServiceError(
                        f"Сервис распознавания вернул {response.status}: {detail[:200]}"
                    )
                payload = await response.json()
        except aiohttp.ClientError as exc:
            raise TranscriptionServiceError(
                f"Не удалось обратиться к сервису распознавания: {exc}"
            ) from exc

        text = str(payload.get("text", "")).strip()
        logger.debug(
            "%s Реплика %.2fс → %d символов (сервис: %.3fс)",
            STT_LOG_PREFIX,
            audio.duration_seconds,
            len(text),
            float(payload.get("elapsed_seconds", 0.0)),
        )
        return text

    async def is_ready(self) -> bool:
        """Проверить, готов ли сервис принимать реплики.

        Returns:
            ``True``, если сервис ответил и модель загружена.
        """
        session = await self._ensure_session()
        try:
            async with session.get(f"{self._base_url}{READY_PATH}") as response:
                if response.status != 200:
                    return False
                payload = await response.json()
                return bool(payload.get("model_loaded"))
        except aiohttp.ClientError:
            return False

    async def aclose(self) -> None:
        """Закрыть HTTP-сессию."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
