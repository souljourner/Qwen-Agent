"""Notification tools — structured request system with pending/resolved status."""

import json
import os
import uuid
from datetime import datetime
from typing import List, Union

from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR
from sandbox_agent.tools.git_autocommit import autocommit

DIGEST_DIR = os.path.join(DATA_DIR, "digest")
REQUESTS_FILE = os.path.join(DATA_DIR, "agent_requests.json")
REQUESTS_MD = os.path.join(DATA_DIR, "agent_requests.md")


def _load_requests() -> List[dict]:
    if os.path.exists(REQUESTS_FILE):
        with open(REQUESTS_FILE) as f:
            return json.load(f)
    return []


def _save_requests(requests: List[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REQUESTS_FILE, "w") as f:
        json.dump(requests, f, indent=2, default=str)
    # Also write a human-readable markdown version
    _write_requests_md(requests)
    autocommit("agent_requests.json", "Update agent requests")


def _write_requests_md(requests: List[dict]) -> None:
    """Write a human-readable markdown view of all requests."""
    pending = [r for r in requests if r["status"] == "pending"]
    resolved = [r for r in requests if r["status"] == "resolved"]

    with open(REQUESTS_MD, "w") as f:
        f.write("# Agent Requests\n\n")
        if pending:
            f.write(f"## Pending ({len(pending)})\n\n")
            for r in pending:
                project = f" [{r['project']}]" if r.get("project") else ""
                f.write(f"### [{r['urgency'].upper()}] {r['subject']}{project}\n\n")
                f.write(f"*{r['created']}* — ID: `{r['id']}`\n\n")
                f.write(f"{r['detail']}\n\n---\n\n")
        else:
            f.write("## Pending\n\nNo pending requests.\n\n")

        if resolved:
            f.write(f"## Resolved ({len(resolved)})\n\n")
            for r in resolved[-10:]:  # Show last 10 resolved
                project = f" [{r['project']}]" if r.get("project") else ""
                f.write(f"- ~~{r['subject']}~~ {project} — resolved {r.get('resolved_at', '')}\n")
            f.write("\n")


@register_tool("read_digest")
class ReadDigest(BaseTool):
    """Read the rolling 3-day activity digest."""

    name = "read_digest"
    description = (
        "Read the agent activity digest — a rolling 3-day summary of everything you've done, "
        "organized by project and heartbeat. Use this to review recent work."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        latest = os.path.join(DIGEST_DIR, "latest.md")
        if os.path.exists(latest):
            content = open(latest).read()
            if len(content) > 16000:
                content = content[:16000] + "\n\n... (truncated)"
            return content
        return "No digest available yet."


@register_tool("request_user")
class RequestUser(BaseTool):
    """File a request to the user."""

    name = "request_user"
    description = (
        "File a request to the user when you need something you can't do yourself. "
        "Checks for duplicates automatically — won't file if same subject is already pending."
    )
    parameters = {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Short subject line.",
            },
            "detail": {
                "type": "string",
                "description": "Full explanation of what you need and why.",
            },
            "urgency": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "low (next few days), medium (today), high (blocking current work).",
            },
            "project": {
                "type": "string",
                "description": "Related project name, if any.",
            },
        },
        "required": ["subject", "detail", "urgency"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        requests = _load_requests()

        # Check for duplicate pending request
        for r in requests:
            if r["status"] == "pending" and r["subject"] == params["subject"]:
                return f"Already pending: {params['subject']} (ID: {r['id']})"

        request = {
            "id": uuid.uuid4().hex[:8],
            "subject": params["subject"],
            "detail": params["detail"],
            "urgency": params["urgency"],
            "project": params.get("project", ""),
            "status": "pending",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "resolved_at": None,
        }
        requests.append(request)
        _save_requests(requests)
        return f"Request filed ({params['urgency'].upper()}): {params['subject']} (ID: {request['id']})"


@register_tool("view_requests")
class ViewRequests(BaseTool):
    """View pending requests."""

    name = "view_requests"
    description = "View all pending requests. Check this BEFORE filing a new request to avoid duplicates."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        requests = _load_requests()
        pending = [r for r in requests if r["status"] == "pending"]
        if not pending:
            return "No pending requests."
        lines = [f"Pending requests ({len(pending)}):\n"]
        for r in pending:
            project = f" [{r['project']}]" if r.get("project") else ""
            lines.append(f"- [{r['urgency'].upper()}] {r['subject']}{project} (ID: {r['id']}, filed: {r['created']})")
        return "\n".join(lines)


@register_tool("resolve_request")
class ResolveRequest(BaseTool):
    """Mark a request as resolved."""

    name = "resolve_request"
    description = "Mark a request as resolved after the user has addressed it or it's no longer needed."
    parameters = {
        "type": "object",
        "properties": {
            "request_id": {
                "type": "string",
                "description": "The request ID to resolve.",
            },
        },
        "required": ["request_id"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        requests = _load_requests()
        for r in requests:
            if r["id"] == params["request_id"]:
                if r["status"] == "resolved":
                    return f"Already resolved: {r['subject']}"
                r["status"] = "resolved"
                r["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                _save_requests(requests)
                return f"Resolved: {r['subject']}"
        return f"Request {params['request_id']} not found."
