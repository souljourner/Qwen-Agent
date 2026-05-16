#!/usr/bin/env bash
# Shut down the qwen agent: stop and remove the container + network.
# Data in ~/sandbox_agent_data (the bind mount) persists.

set -euo pipefail
cd "$(dirname "$0")"   # sit in the docker/ dir so the compose project is found

exec docker compose down --remove-orphans
