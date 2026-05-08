FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (libsndfile for soundfile, git for autocommit, curl+node for app building)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 git curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Qwen-Agent + undeclared deps that are imported unconditionally
COPY setup.py README.md /app/Qwen-Agent/
COPY qwen_agent/ /app/Qwen-Agent/qwen_agent/
RUN pip install --no-cache-dir "/app/Qwen-Agent[gui]" croniter numpy soundfile \
    ipykernel jupyter_client pandas requests Pillow beautifulsoup4 lxml

# Copy sandbox_agent
COPY sandbox_agent/ /app/sandbox_agent/

# Create non-root user and writable data directory
RUN useradd -m agent && mkdir -p /app/data && chown agent:agent /app/data
USER agent

ENV DATA_DIR=/app/data
ENV PYTHONPATH=/app

EXPOSE 7860

ENTRYPOINT ["python", "-m", "sandbox_agent.main"]
