# Planner Agent Dockerfile
# Multi-stage build for a lean production image with FastAPI workflow service

# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime - Lean production image
# =============================================================================
FROM python:3.12-slim AS runtime

# Install runtime dependencies including Node.js for Claude CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security (Kubernetes best practice)
RUN groupadd --gid 1000 planner && \
    useradd --uid 1000 --gid planner --shell /bin/bash --create-home planner

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Claude Code CLI globally (required by claude-agent-sdk)
RUN npm install -g @anthropic-ai/claude-code

# Set working directory
WORKDIR /app

# Copy application code - all modules for workflow service
COPY --chown=planner:planner task_manager.py .
COPY --chown=planner:planner plan_manager.py .
COPY --chown=planner:planner planner_agent.py .
COPY --chown=planner:planner workflow_service.py .
COPY --chown=planner:planner streaming.py .
COPY --chown=planner:planner main.py .
COPY --chown=planner:planner durable_agent.py .
COPY --chown=planner:planner anthropic_llm.py .

# Create directories for plans and workspace
RUN mkdir -p /app/plans /app/workspace && \
    chown -R planner:planner /app

# Switch to non-root user
USER planner

# Environment variables
# ANTHROPIC_API_KEY must be provided at runtime (via Kubernetes Secret)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/planner

# Default working directory for the agent (can be overridden)
ENV PLANNER_CWD=/app/workspace \
    PLANNER_PLANS_DIR=/app/plans

# Health check - verify FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose FastAPI port
EXPOSE 8080

# Default entrypoint - run FastAPI workflow service
ENTRYPOINT ["uvicorn", "workflow_service:app", "--host", "0.0.0.0", "--port", "8080"]

# Default command (no additional args needed)
CMD []
