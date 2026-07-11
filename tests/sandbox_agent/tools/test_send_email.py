"""Tests for the send_email tool + SMTP config resolution.

The tool lifts the proven soxs signal_monitor sender into the framework:
smtplib.SMTP + STARTTLS + MIME, resolved from os.environ first and
DATA_DIR/.env second, never raises, logs an email_sent event.
"""

import json

import pytest

import sandbox_agent.config as cfg
from sandbox_agent.tools import notification_tools as nt
from sandbox_agent.tools.notification_tools import SendEmail


# --- config resolution -----------------------------------------------------

SMTP_KEYS = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "SMTP_FROM",
             "EMAIL_TO", "ALERT_EMAIL")


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """No SMTP in os.environ; DATA_DIR isolated to tmp."""
    for k in SMTP_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))
    return tmp_path


def test_smtp_config_from_env(clean_env, monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("SMTP_USER", "u@example.com")
    monkeypatch.setenv("SMTP_PASS", "sekret")
    monkeypatch.setenv("EMAIL_TO", "me@example.com")
    c = cfg.get_smtp_config()
    assert c["host"] == "smtp.example.com"
    assert c["port"] == 2525
    assert c["user"] == "u@example.com"
    assert c["password"] == "sekret"
    assert c["to"] == "me@example.com"
    assert c["from"] == "u@example.com"  # SMTP_FROM unset → falls back to user


def test_smtp_config_from_data_dir_dotenv(clean_env):
    (clean_env / ".env").write_text(
        "SMTP_HOST=smtp.mail.yahoo.com\n"
        "SMTP_PORT=587\n"
        "SMTP_USER=x@yahoo.com\n"
        "SMTP_FROM=x@yahoo.com\n"
        "SMTP_PASS=abc123\n"
        "# comment line\n"
        "OTHER=ignored\n"
    )
    c = cfg.get_smtp_config()
    assert c["host"] == "smtp.mail.yahoo.com"
    assert c["port"] == 587
    assert c["password"] == "abc123"


def test_env_wins_over_dotenv(clean_env, monkeypatch):
    (clean_env / ".env").write_text("SMTP_HOST=from-file\n")
    monkeypatch.setenv("SMTP_HOST", "from-env")
    assert cfg.get_smtp_config()["host"] == "from-env"


def test_email_to_falls_back_to_alert_email(clean_env, monkeypatch):
    monkeypatch.setenv("ALERT_EMAIL", "alerts@example.com")
    assert cfg.get_smtp_config()["to"] == "alerts@example.com"


# --- the tool ---------------------------------------------------------------

class _FakeSMTP:
    """Captures the smtplib.SMTP context-manager protocol."""
    instances: list = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.logged_in = None
        self.sent = []          # (from, to, payload)
        self.starttls_called = False
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def ehlo(self):
        pass

    def starttls(self, context=None):
        self.starttls_called = True

    def login(self, user, password):
        self.logged_in = (user, password)

    def sendmail(self, from_addr, to_addrs, payload):
        self.sent.append((from_addr, to_addrs, payload))


@pytest.fixture
def smtp_ready(clean_env, monkeypatch):
    """Configured SMTP + FakeSMTP capture; returns the fake class."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "agent@example.com")
    monkeypatch.setenv("SMTP_PASS", "pw")
    monkeypatch.setenv("EMAIL_TO", "owner@example.com")
    _FakeSMTP.instances = []
    monkeypatch.setattr(nt.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(nt, "log_event", lambda *a, **k: None, raising=False)
    return _FakeSMTP


def test_sends_email_with_defaults(smtp_ready):
    out = SendEmail().call({"subject": "Daily report", "body": "# Hi\n\nAll good."})
    assert "sent" in out.lower()
    assert "owner@example.com" in out
    srv = smtp_ready.instances[-1]
    assert srv.starttls_called
    assert srv.logged_in == ("agent@example.com", "pw")
    from_addr, to_addrs, payload = srv.sent[0]
    assert from_addr == "agent@example.com"
    assert to_addrs == ["owner@example.com"]
    assert "Daily report" in payload
    assert "All good." in payload


def test_explicit_to_is_ignored_by_policy(smtp_ready):
    # Email policy (2026-07-11): recipient is fixed to the owner; a model-
    # supplied `to` must NOT be honored (see test_email_policy.py).
    SendEmail().call({"subject": "s", "body": "b", "to": "other@example.com"})
    assert smtp_ready.instances[-1].sent[0][1] == ["owner@example.com"]


def test_not_configured_error(clean_env, monkeypatch):
    monkeypatch.setattr(nt, "log_event", lambda *a, **k: None, raising=False)
    out = SendEmail().call({"subject": "s", "body": "b"})
    assert out.lower().startswith("error")
    assert "not configured" in out.lower()


def test_no_recipient_error(clean_env, monkeypatch):
    # SMTP configured, but no EMAIL_TO/ALERT_EMAIL and no `to` arg.
    monkeypatch.setenv("SMTP_HOST", "h")
    monkeypatch.setenv("SMTP_USER", "u")
    monkeypatch.setenv("SMTP_PASS", "p")
    monkeypatch.setattr(nt, "log_event", lambda *a, **k: None, raising=False)
    out = SendEmail().call({"subject": "s", "body": "b"})
    assert out.lower().startswith("error")
    assert "recipient" in out.lower() or "email_to" in out.lower()


def test_transient_failure_retries_once(smtp_ready, monkeypatch):
    calls = {"n": 0}
    orig_init = smtp_ready.__init__

    def flaky_init(self, host, port, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("connection reset")
        orig_init(self, host, port, timeout=timeout)

    monkeypatch.setattr(smtp_ready, "__init__", flaky_init)
    out = SendEmail().call({"subject": "s", "body": "b"})
    assert "sent" in out.lower()
    assert calls["n"] == 2


def test_persistent_failure_returns_error_never_raises(smtp_ready, monkeypatch):
    def always_fail(self, host, port, timeout=None):
        raise OSError("no route to host")

    monkeypatch.setattr(smtp_ready, "__init__", always_fail)
    out = SendEmail().call({"subject": "s", "body": "b"})
    assert out.lower().startswith("error")
    assert "no route" in out


def test_registered_and_in_tool_list():
    from qwen_agent.tools.base import TOOL_REGISTRY
    assert "send_email" in TOOL_REGISTRY
    assert "send_email" in cfg.TOOL_LIST
