FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (libsndfile for soundfile, git for autocommit, curl+node for app building)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 git curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Qwen-Agent + undeclared deps that are imported unconditionally.
# `gui` extras still pulled for the legacy Gradio path (kept for fallback during
# Chainlit cutover; remove from extras in a follow-up commit once Chainlit soaks).
# Chainlit + SQLAlchemy + aiosqlite are the new chat surface.
COPY setup.py README.md /app/Qwen-Agent/
COPY qwen_agent/ /app/Qwen-Agent/qwen_agent/
RUN pip install --no-cache-dir "/app/Qwen-Agent[gui]" croniter numpy soundfile \
    pandas requests Pillow beautifulsoup4 lxml \
    chainlit aiosqlite "sqlalchemy>=2.0"

# uv = default package manager for per-project venvs (cache-backed, near-instant
# re-installs). Its own layer so it doesn't bust the heavy install above — keeps
# rebuilds fast.
RUN pip install --no-cache-dir uv

# Copy sandbox_agent
COPY sandbox_agent/ /app/sandbox_agent/

# Create non-root user and writable data directory
RUN useradd -m agent && mkdir -p /app/data && chown agent:agent /app/data
USER agent

ENV DATA_DIR=/app/data
ENV PYTHONPATH=/app

# Chainlit creates a `.files` cache in its working directory at startup. Run
# from /app/data so it lands on the bind-mounted volume (writable by `agent`)
# instead of /app (read-only for the non-root user).
WORKDIR /app/data

# Ports: 7860 = Chainlit chat UI; 7861 = status server (set via STATUS_SERVER_PORT)
EXPOSE 7860 7861

# Chainlit replaces Gradio as the chat front door. Background lanes (cron,
# heartbeat, LLM bridge, status server) are bootstrapped at module load via
# sandbox_agent.main.bootstrap_background(), called from chat_app.py.
# `--headless` suppresses the no-op browser-open syscall in containers.
ENTRYPOINT ["chainlit", "run", "--host", "0.0.0.0", "--port", "7860", "--headless", "/app/sandbox_agent/chat_app.py"]
