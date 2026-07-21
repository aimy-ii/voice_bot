"""Офлайн-тесты генерации JWT-токена для LiveKit Playground."""

from voice_bot.agent.main import _rtc_session_agent_name
from voice_bot.agent.token import create_access_token


def test_create_access_token_returns_nonempty_jwt() -> None:
    """С фейковыми LIVEKIT_* ключами токен — непустая JWT-строка (без сети)."""
    token = create_access_token(
        api_key="a" * 32,
        api_secret="b" * 32,
        room="console-room",
        identity="tester",
    )

    assert isinstance(token, str)
    assert token.count(".") == 2
    assert len(token) > 20


def test_rtc_session_agent_name_auto_accept() -> None:
    """При автоподхвате rtc_session получает пустое имя (API 1.6.6)."""
    assert _rtc_session_agent_name(auto_accept=True, agent_name="voice-bot") == ""


def test_rtc_session_agent_name_explicit_dispatch() -> None:
    """Без автоподхвата передаётся явное имя агента для dispatch."""
    assert _rtc_session_agent_name(auto_accept=False, agent_name="voice-bot") == "voice-bot"
