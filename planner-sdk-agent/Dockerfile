# Claude Agent SDK HTTP server
FROM node:20-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally (required by Claude Agent SDK)
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY --chown=node:node . .

# Create .claude directory for tasks and accept ToS
# Create /workspace directory for agent working directory
RUN mkdir -p /home/node/.claude/tasks && \
    echo '{"hasCompletedOnboarding": true, "hasAcknowledgedCostThreshold": true}' > /home/node/.claude/settings.json && \
    chown -R node:node /home/node/.claude && \
    mkdir -p /workspace && \
    chown node:node /workspace

USER node

ENV CLAUDE_CODE_ENABLE_TASKS=true \
    CLAUDE_CODE_SKIP_ONBOARDING=true \
    HOME=/home/node \
    PORT=3000

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

CMD ["npx", "tsx", "src/server.ts"]
