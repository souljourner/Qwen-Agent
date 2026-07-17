FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (libsndfile for soundfile, git for autocommit, curl+node for app building,
# plus Chromium deps for headless Playwright browser automation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 git curl wget \
    xvfb x11-utils x11vnc \
    unzip \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# noVNC for watching the virtual display
RUN wget -qO /tmp/novnc.zip https://github.com/novnc/noVNC/archive/v1.4.0.zip \
    && unzip -q /tmp/novnc.zip -d /opt/novnc \
    && rm /tmp/novnc.zip

# WebSockify (Python VNC-to-WebSocket proxy, ships with noVNC)
RUN pip install --no-cache-dir websockify

# Copy and install Qwen-Agent + undeclared deps that are imported unconditionally.
# `gui` extras still pulled for the legacy Gradio path (kept for fallback during
# Chainlit cutover; remove from extras in a follow-up commit once Chainlit soaks).
# Chainlit + SQLAlchemy + aiosqlite are the new chat surface.
COPY setup.py README.md /app/Qwen-Agent/
COPY qwen_agent/ /app/Qwen-Agent/qwen_agent/
RUN pip install --no-cache-dir "/app/Qwen-Agent[gui,rag]" croniter numpy soundfile \
    pandas requests Pillow beautifulsoup4 lxml yfinance \
    chainlit aiosqlite "sqlalchemy>=2.0" \
    playwright "cryptography>=42.0"

# uv = default package manager for per-project venvs (cache-backed, near-instant
# re-installs). Its own layer so it doesn't bust the heavy install above — keeps
# rebuilds fast.
RUN pip install --no-cache-dir uv

# pytest + the test suite ship in the image so the agent can run
# scripts/selftest.sh (nightly self-test feeding the health-alert loop).
RUN pip install --no-cache-dir pytest
COPY tests/ /app/tests/
COPY scripts/ /app/scripts/

# Copy sandbox_agent
COPY sandbox_agent/ /app/sandbox_agent/

# Create non-root user and writable data directory
RUN useradd -m agent && mkdir -p /app/data && chown agent:agent /app/data

USER agent

# Install Playwright Chromium browser + system deps as the agent user.
# Playwright 1.60+ uses headless_shell which requires the correct user context.
# --with-deps needs root, so install system deps first as root, then browsers as agent.
USER root
RUN playwright install-deps chromium
USER agent
RUN playwright install chromium

ENV DATA_DIR=/app/data
ENV PYTHONPATH=/app

# Chainlit creates a `.files` cache in its working directory at startup. Run
# from /app/data so it lands on the bind-mounted volume (writable by `agent`)
# instead of /app (read-only for the non-root user).
WORKDIR /app/data

# Ports: 7860 = Chainlit chat UI; 7861 = status server; 6080 = noVNC (watch browser)
EXPOSE 7860 7861 6080

# Start Xvfb + noVNC if XVFB_ENABLED=true, then launch Chainlit.
USER root
COPY sandbox_agent/docker/start_xvfb.sh /usr/local/bin/start_xvfb.sh
RUN chmod +x /usr/local/bin/start_xvfb.sh

# Chainlit replaces Gradio as the chat front door. Background lanes (cron,
# heartbeat, LLM bridge, status server) are bootstrapped at module load via
# sandbox_agent.main.bootstrap_background(), called from chat_app.py.
# `--headless` suppresses the no-op browser-open syscall in containers.
ENTRYPOINT ["/bin/bash", "-c", "source /usr/local/bin/start_xvfb.sh && exec chainlit run --host 0.0.0.0 --port 7860 --headless /app/sandbox_agent/chat_app.py"]
