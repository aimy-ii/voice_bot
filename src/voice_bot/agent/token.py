"""Генерация JWT-токена для входа клиента в комнату LiveKit.

Печатает токен в stdout — удобно вставить в LiveKit Agents Playground.

Запуск::

    uv run python -m voice_bot.agent.token
    uv run python -m voice_bot.agent.token --room my-room --identity alice
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

from voice_bot.config import get_settings


def create_access_token(
    *,
    api_key: str,
    api_secret: str,
    room: str = "console-room",
    identity: str = "tester",
) -> str:
    """Собрать JWT участника с правом войти в указанную комнату.

    Ключи — из настроек (``LIVEKIT_API_KEY`` / ``LIVEKIT_API_SECRET``);
    в код секреты не пишутся.
    """
    return (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(VideoGrants(room_join=True, room=room))
        .to_jwt()
    )


def main() -> None:
    """Прочитать настройки и напечатать токен в stdout (не в лог)."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="JWT-токен доступа в комнату LiveKit для Playground",
    )
    parser.add_argument("--room", default="console-room", help="Имя комнаты")
    parser.add_argument("--identity", default="tester", help="Идентичность участника")
    args = parser.parse_args()

    settings = get_settings()
    token = create_access_token(
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
        room=args.room,
        identity=args.identity,
    )
    print(token)


if __name__ == "__main__":
    main()
