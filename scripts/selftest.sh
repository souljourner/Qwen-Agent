#!/usr/bin/env bash
# Run the sandbox_agent test suite and feed failures into the health-alert
# loop (weakness #6: tests only ran when someone remembered).
#
# Inside the container:  bash /app/scripts/selftest.sh
# Schedule it (timezone-aware, e.g. nightly 3am Pacific) with:
#   schedule_task(name="nightly-selftest", schedule_type="cron", cron="0 3 * * *",
#                 description="Run: exec bash /app/scripts/selftest.sh and report the outcome.")
#
# On failure it logs a `selftest_failed` activity event — health.py treats a
# single occurrence as alert-worthy and emails the user on the next heartbeat.

set -uo pipefail

TESTS_DIR="${TESTS_DIR:-/app/tests/sandbox_agent}"
OUT="$(python -m pytest "$TESTS_DIR" -q 2>&1 | tail -20)"
STATUS=$?

echo "$OUT"

if [ $STATUS -ne 0 ]; then
    python - <<PYEOF
from sandbox_agent.activity_log import log_event
tail = """$OUT"""[-500:]
log_event("selftest_failed", detail=tail)
print("selftest_failed event logged — health check will alert")
PYEOF
    exit 1
fi
echo "selftest OK"
