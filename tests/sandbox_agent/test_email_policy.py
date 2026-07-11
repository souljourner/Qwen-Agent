"""Email lockdown (weakness #4, first slice): the agent may email ONLY the
owner, ONLY via the send_email tool.

- send_email loses its `to` parameter; recipient is always the configured
  owner address (EMAIL_TO/ALERT_EMAIL).
- send_email_message refuses any other recipient and logs a blocked_email
  event (health-alertable).
- Hand-rolled SMTP is blocked at every agent code entry point: exec command
  strings, code_interpreter code, and project file writes/patches.
"""

import pytest

import sandbox_agent.config as cfg
import sandbox_agent.tools.notification_tools as nt
from sandbox_agent.email_policy import contains_email_bypass
from sandbox_agent.tools.notification_tools import SendEmail, send_email_message


@pytest.fixture
def smtp_env(tmp_path, monkeypatch):
    for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "SMTP_FROM",
              "EMAIL_TO", "ALERT_EMAIL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "agent@example.com")
    monkeypatch.setenv("SMTP_PASS", "pw")
    monkeypatch.setenv("EMAIL_TO", "owner@example.com")
    monkeypatch.setenv("ALERT_EMAIL", "owner-alerts@example.com")

    sent = []

    class _FakeSMTP:
        def __init__(self, host, port, timeout=None): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def ehlo(self): pass
        def starttls(self, context=None): pass
        def login(self, u, p): pass
        def sendmail(self, f, to, payload): sent.append(to)

    monkeypatch.setattr(nt.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(nt, "log_event", lambda *a, **k: None, raising=False)
    return sent


def test_tool_has_no_to_parameter():
    assert "to" not in SendEmail().parameters["properties"]


def test_tool_always_sends_to_owner(smtp_env):
    out = SendEmail().call({"subject": "s", "body": "b"})
    assert "sent" in out.lower()
    assert smtp_env == [["owner@example.com"]]


def test_tool_ignores_injected_to_argument(smtp_env):
    # Even if the model passes a rogue `to`, it must not be honored.
    out = SendEmail().call({"subject": "s", "body": "b", "to": "evil@attacker.com"})
    assert "evil@attacker.com" not in str(smtp_env)
    assert smtp_env and smtp_env[0] == ["owner@example.com"]


def test_message_refuses_non_owner_recipient(smtp_env):
    out = send_email_message("s", "b", to="exfil@attacker.com")
    assert out.lower().startswith("error")
    assert "recipient" in out.lower() or "not allowed" in out.lower()
    assert smtp_env == []


def test_message_allows_both_configured_addresses(smtp_env):
    assert "sent" in send_email_message("s", "b", to="owner@example.com").lower()
    assert "sent" in send_email_message("s", "b", to="owner-alerts@example.com").lower()


# --- bypass-pattern scanning ---------------------------------------------------

def test_bypass_patterns():
    assert contains_email_bypass("import smtplib\ns = smtplib.SMTP('h')")
    assert contains_email_bypass("echo hi | /usr/sbin/sendmail x@y.com")
    assert contains_email_bypass("curl smtp://mail.example.com --mail-rcpt x@y.com")
    assert not contains_email_bypass("import requests\nrequests.get('https://x')")
    assert not contains_email_bypass("print('mail merge report')")


def test_exec_blocks_inline_smtp(tmp_path, monkeypatch):
    import sandbox_agent.tools.exec_tool as et
    monkeypatch.setattr(et, "DATA_DIR", str(tmp_path))
    from sandbox_agent.tools.exec_tool import ExecTool
    out = ExecTool().call({"command": "python -c 'import smtplib; ...'"})
    assert "blocked" in out.lower()


def test_code_interpreter_blocks_smtp(monkeypatch):
    from sandbox_agent.tools.code_interpreter import LocalCodeInterpreter
    out = LocalCodeInterpreter().call({"code": "import smtplib\nprint('x')"})
    assert "blocked" in out.lower() or "send_email" in out.lower()


def test_project_write_blocks_smtp(tmp_path, monkeypatch):
    import sandbox_agent.tools.project_tools as pt
    monkeypatch.setattr(pt, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pt, "PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(pt, "autocommit", lambda *a, **k: None)
    (tmp_path / "projects" / "p").mkdir(parents=True)
    from sandbox_agent.tools.project_tools import ProjectWriteFile
    out = ProjectWriteFile().call({"project": "p", "path": "mailer.py",
                                   "content": "import smtplib\n# send stuff\n" + "x" * 100})
    assert out.lower().startswith("error")
    assert "send_email" in out


def test_blocked_email_is_health_alertable():
    from sandbox_agent.health import FAILURE_EVENT_THRESHOLDS
    assert FAILURE_EVENT_THRESHOLDS.get("blocked_email") == 1
