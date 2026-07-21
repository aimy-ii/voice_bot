# Локальный веб-клиент: сборка официального agent-starter-react из git.
# Официального Docker-образа playground у LiveKit нет.
FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG REPO_URL=https://github.com/livekit-examples/agent-starter-react.git
ARG REPO_REF=main

RUN git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" .

RUN corepack enable \
    && corepack prepare pnpm@9.15.9 --activate \
    && pnpm install --frozen-lockfile

# /api/token в starter разрешён только при NODE_ENV=development.
ENV NODE_ENV=development
ENV HOSTNAME=0.0.0.0
ENV PORT=3000

EXPOSE 3000

CMD ["pnpm", "dev", "--hostname", "0.0.0.0", "--port", "3000"]
