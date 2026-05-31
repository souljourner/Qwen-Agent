#!/usr/bin/env bash
# Run the qwen agent in the foreground (no rebuild).
#
# Ensures the Docker daemon is running first (Docker Desktop on macOS starts
# lazily and is off after a reboot), then brings the already-built stack up
# attached to this terminal so logs stream live and Ctrl+C stops it cleanly.
# Does NOT rebuild the image — use ./restart.sh to pick up code/.env changes.
# Data in ~/sandbox_agent_data (the bind mount) is untouched either way.
#
# Chat UI:        http://localhost:7860
# Status server:  http://localhost:7861

set -euo pipefail
cd "$(dirname "$0")"   # sit in the docker/ dir so .env is auto-loaded

DOCKER_WAIT_SECS="${DOCKER_WAIT_SECS:-300}"   # how long to wait for the daemon

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker CLI not found on PATH." >&2; exit 1; }

# Make sure the Docker daemon is up before touching compose.
if ! docker info >/dev/null 2>&1; then
  echo "==> Docker daemon not responding; launching Docker Desktop..."
  open -a Docker   # macOS: start Docker Desktop

  echo -n "==> waiting for the Docker daemon (up to ${DOCKER_WAIT_SECS}s)"
  waited=0
  until docker info >/dev/null 2>&1; do
    if [ "$waited" -ge "$DOCKER_WAIT_SECS" ]; then
      echo
      echo "ERROR: Docker daemon did not come up within ${DOCKER_WAIT_SECS}s." >&2
      exit 1
    fi
    sleep 2
    waited=$((waited + 2))
    echo -n "."
  done
  echo " ready."
fi

echo "==> docker compose up (foreground; Ctrl+C to stop)..."
exec docker compose up
