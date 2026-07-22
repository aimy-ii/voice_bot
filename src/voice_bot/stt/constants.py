"""Константы клиента распознавания речи."""

from typing import Final

#: Префикс логов — по нему удобно грепать распознавание в звонке.
STT_LOG_PREFIX: Final[str] = "[STT]"

#: Сколько байт занимает один отсчёт PCM int16.
BYTES_PER_SAMPLE: Final[int] = 2

#: Реплики короче этого порога не отправляем в сервис — это шум.
MIN_UTTERANCE_SECONDS: Final[float] = 0.12

#: Сколько ждём ответа сервиса, прежде чем считать реплику потерянной.
REQUEST_TIMEOUT_SECONDS: Final[float] = 15.0

#: Путь горячего эндпоинта сервиса распознавания.
TRANSCRIBE_PCM_PATH: Final[str] = "/api/v1/transcribe/pcm"

#: Путь проверки готовности сервиса.
READY_PATH: Final[str] = "/api/v1/ready"
