"""Email policy: the agent emails ONLY the owner, ONLY via the send_email tool.

Two enforcement layers (weakness #4, outbound-channel slice):
1. Recipient allowlist — send_email_message refuses any address that isn't
   the configured EMAIL_TO / ALERT_EMAIL. The send_email tool doesn't even
   expose a `to` parameter.
2. Bypass scanning — hand-rolled SMTP is rejected at every agent code entry
   point (exec commands, code_interpreter code, project file writes/patches)
   so prompt-injected exfiltration can't route around the tool.

Residual risk (documented in weaknesses_to_resolve.md #4): files that already
exist on disk (e.g. the legacy soxs signal monitors) still run; SMTP creds in
DATA_DIR/.env remain readable. Full fix = secret isolation + egress control.
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Patterns that indicate hand-rolled email sending. Deliberately narrow —
# 'mail' alone would false-positive on ordinary prose/code.
_BYPASS_PATTERNS = [
    re.compile(r"\bsmtplib\b"),
    re.compile(r"/usr/sbin/sendmail\b"),
    re.compile(r"\bsendmail\s+-"),
    re.compile(r"\bsmtps?://", re.IGNORECASE),
    re.compile(r"\bmailx\b"),
    re.compile(r"\bMIMEMultipart\b"),
]

BLOCKED_EMAIL_MSG = (
    "blocked by email policy: outbound email is only allowed through the "
    "send_email tool (which delivers to the owner). Do not hand-roll SMTP."
)


def allowed_recipients() -> List[str]:
    """The only addresses the agent may email: EMAIL_TO and ALERT_EMAIL."""
    from sandbox_agent.config import get_smtp_config
    cfg = get_smtp_config()
    out = []
    for addr in (cfg.get("to"), cfg.get("alert_to")):
        if addr and addr not in out:
            out.append(addr)
    return out


def recipient_allowed(addr: str) -> bool:
    return bool(addr) and addr.strip().lower() in {
        a.lower() for a in allowed_recipients()}


def contains_email_bypass(text: str) -> bool:
    """True when agent-authored code/commands try to send email directly."""
    if not text:
        return False
    return any(p.search(text) for p in _BYPASS_PATTERNS)


def log_blocked(channel: str, preview: str) -> None:
    """Record a blocked_email event — health treats one occurrence as
    alert-worthy (a bypass attempt is either a bug or an injection)."""
    logger.warning("email policy blocked %s: %s", channel, preview[:160])
    try:
        from sandbox_agent.activity_log import log_event
        log_event("blocked_email", channel=channel, detail=preview[:300])
    except Exception:  # noqa: BLE001
        pass
