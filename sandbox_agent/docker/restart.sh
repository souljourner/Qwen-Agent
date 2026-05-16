#!/usr/bin/env bash
# Restart the qwen agent in the foreground.
#
# Brings any existing stack down, then `docker compose up --build` attached to
# this terminal so logs stream live and Ctrl+C stops the container cleanly.
# Picks up code changes since the last build, .env changes, and any pulled
# updates. Data in ~/sandbox_agent_data (the bind mount) is untouched.

set -euo pipefail
cd "$(dirname "$0")"   # sit in the docker/ dir so .env is auto-loaded

echo "==> docker compose down (clearing any existing stack)..."
docker compose down --remove-orphans || true

echo "==> docker compose up --build (foreground; Ctrl+C to stop)..."
exec docker compose up --build
