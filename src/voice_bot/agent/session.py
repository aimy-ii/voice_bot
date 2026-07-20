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

from livekit.agents import AgentSession
from livekit.plugins import elevenlabs, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from voice_bot.config import Settings


def build_session(settings: Settings) -> AgentSession:
    """Создать :class:`AgentSession` с провайдерами из настроек.

    Args:
        settings: настройки приложения (модели, язык, идентификатор голоса).

    Returns:
        Готовая к запуску голосовая сессия.
    """
    return AgentSession(
        # Речь → текст. Язык фиксируем, чтобы модель его не угадывала.
        stt=openai.STT(model=settings.stt_model, language=settings.language),
        # «Мозг»: формулирует короткие реплики по системному промпту.
        llm=openai.LLM(model=settings.llm_model),
        # Текст → голос. voice_id — тот самый записанный голос компании.
        tts=elevenlabs.TTS(voice_id=settings.elevenlabs_voice_id, model=settings.tts_model),
        # Слышит границы речи (начал/закончил говорить).
        vad=silero.VAD.load(),
        # Определяет, что реплика клиента завершена (мультиязычная модель).
        turn_detection=MultilingualModel(),
    )
