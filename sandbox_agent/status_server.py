"""Lightweight HTTP status server — serves /status JSON and /dashboard HTML."""

import json
import logging
import os
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

from sandbox_agent.config import DATA_DIR

logger = logging.getLogger(__name__)

_SERVER_START_TIME = None  # Set when status server process starts

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html>
<head>
    <title>Sandbox Agent — Activity Monitor</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; height: 100vh; display: flex; flex-direction: column; overflow: hidden; gap: 0; }
        h1 { color: #58a6ff; margin-bottom: 16px; font-size: 24px; flex-shrink: 0; }
        h2 { color: #8b949e; margin: 0 0 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; flex-shrink: 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 16px; flex-shrink: 0; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
        .card .label { color: #8b949e; font-size: 11px; margin-bottom: 2px; }
        .card .value { font-size: 18px; font-weight: 600; }
        .status-idle { color: #8b949e; }
        .status-chatting { color: #3fb950; }
        .status-cron_task { color: #d29922; }
        .status-heartbeat { color: #a371f7; }
        .panes { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; flex: 1; min-height: 0; }
        .panes-bottom { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; flex: 1; min-height: 0; margin-top: 16px; }
        .pane { background: #161b22; border: 1px solid #30363d; border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; }
        .pane-header { padding: 12px 16px 8px; flex-shrink: 0; }
        .pane-body { overflow-y: auto; flex: 1; padding: 0 4px 8px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #21262d; font-size: 12px; }
        th { color: #8b949e; font-weight: 500; position: sticky; top: 0; background: #161b22; }
        .type-tool_call { color: #58a6ff; }
        .type-tool_result { color: #8b949e; }
        .type-cron_start { color: #d29922; }
        .type-cron_complete { color: #3fb950; }
        .type-cron_failed { color: #f85149; }
        .type-chat_start { color: #3fb950; }
        .type-chat_complete { color: #8b949e; }
        .type-model_select { color: #a371f7; }
        .timestamp { color: #484f58; white-space: nowrap; }
        .detail { max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .auto-refresh { color: #484f58; font-size: 12px; margin-left: 12px; }
        .pane-body::-webkit-scrollbar { width: 6px; }
        .pane-body::-webkit-scrollbar-track { background: #0d1117; }
        .pane-body::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
        .pane-body::-webkit-scrollbar-thumb:hover { background: #484f58; }
        .md-content { white-space: pre-wrap; font-size: 13px; line-height: 1.6; padding: 12px 16px; }
        .md-content h1, .md-content h2, .md-content h3 { color: #58a6ff; margin: 12px 0 6px; }
        .md-content h1 { font-size: 16px; }
        .md-content h2 { font-size: 14px; }
        .md-content h3 { font-size: 13px; color: #d29922; }
        .md-content hr { border: none; border-top: 1px solid #21262d; margin: 12px 0; }
        .empty-state { color: #484f58; font-style: italic; padding: 20px 16px; }
        .request-high { border-left: 3px solid #f85149; }
        .request-medium { border-left: 3px solid #d29922; }
        .request-low { border-left: 3px solid #8b949e; }
    </style>
</head>
<body>
    <h1>Sandbox Agent <span class="auto-refresh">auto-refreshes every 5s</span></h1>

    <div class="grid" id="status-cards"></div>

    <div id="preview-row" style="margin-bottom:16px;"></div>

    <div id="models-section" style="margin-bottom:12px;">
        <h2 style="color:#8b949e;font-size:14px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">Active Models</h2>
        <div class="grid" id="model-cards"></div>
    </div>

    <div class="panes">
        <div class="pane">
            <div class="pane-header"><h2>Recent Activity</h2></div>
            <div class="pane-body">
                <table id="events-table">
                    <thead><tr><th>Time</th><th>Type</th><th>Detail</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
        <div class="pane">
            <div class="pane-header"><h2>Recent Tool Calls</h2></div>
            <div class="pane-body">
                <table id="tools-table">
                    <thead><tr><th>Time</th><th>Tool</th><th>Args</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="panes-bottom">
        <div class="pane">
            <div class="pane-header"><h2>Daily Digest (3-Day Rolling)</h2></div>
            <div class="pane-body">
                <div id="digest-content" class="md-content"><span class="empty-state">Loading...</span></div>
            </div>
        </div>
        <div class="pane">
            <div class="pane-header"><h2>Agent Requests</h2></div>
            <div class="pane-body">
                <div id="requests-content" class="md-content"><span class="empty-state">Loading...</span></div>
            </div>
        </div>
    </div>

    <script>
        function formatTime(ts) {
            if (!ts) return '';
            const d = new Date(ts);
            return d.toLocaleTimeString();
        }

        function formatUptime(secs) {
            const h = Math.floor(secs / 3600);
            const m = Math.floor((secs % 3600) / 60);
            return h > 0 ? `${h}h ${m}m` : `${m}m`;
        }

        async function refresh() {
            try {
                const resp = await fetch('/status');
                const s = await resp.json();

                document.getElementById('status-cards').innerHTML = `
                    <div class="card">
                        <div class="label">Status</div>
                        <div class="value status-${s.agent_status}">${(s.agent_status || 'idle').toUpperCase()}</div>
                    </div>
                    <div class="card">
                        <div class="label">Current Task</div>
                        <div class="value">${s.current_task || '—'}</div>
                    </div>
                    <div class="card">
                        <div class="label">Current Tool</div>
                        <div class="value">${s.current_tool || '—'}</div>
                    </div>
                    <div class="card">
                        <div class="label">Uptime</div>
                        <div class="value">${formatUptime(s.uptime_seconds || 0)}</div>
                    </div>
                `;

                // Render streaming preview if present
                {
                    const previewEl = document.getElementById('preview-row');
                    const preview = s.current_preview;
                    if (preview) {
                        const escDiv = document.createElement('div');
                        escDiv.textContent = preview;
                        previewEl.innerHTML = `
                            <div class="card" style="border-left:3px solid #3fb950">
                                <div class="label">Streaming Preview</div>
                                <div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#c9d1d9;white-space:pre-wrap;margin-top:4px;max-height:120px;overflow-y:auto">${escDiv.innerHTML}</div>
                            </div>
                        `;
                    } else {
                        previewEl.innerHTML = '';
                    }
                }

                // Render model status (included in /status response)
                {
                    const models = s.models || {};
                    const modelCards = document.getElementById('model-cards');
                    modelCards.innerHTML = Object.entries(models).map(([name, info]) => {
                        const isBusy = info.status === 'busy';
                        const color = isBusy ? '#3fb950' : '#484f58';
                        const statusText = isBusy ? 'BUSY' : 'IDLE';
                        const task = info.task || '';
                        const since = info.since ? formatTime(info.since) : '';
                        return `
                            <div class="card" style="border-left:3px solid ${color}">
                                <div class="label">${name}</div>
                                <div class="value" style="color:${color};font-size:16px">${statusText}</div>
                                ${task ? '<div style="color:#8b949e;font-size:11px;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + task + '<\/div>' : ''}
                                ${since ? '<div style="color:#484f58;font-size:10px">since ' + since + '<\/div>' : ''}
                            <\/div>
                        `;
                    }).join('');
                }

            } catch (err) {
                console.error('Status refresh failed:', err);
            }

            // Fetch activity events (historical, from activity.jsonl)
            try {
                const eventsResp = await fetch('/events');
                const eventsData = await eventsResp.json();

                const eventsBody = document.querySelector('#events-table tbody');
                eventsBody.innerHTML = (eventsData.recent_events || []).slice().reverse().map(e => `
                    <tr>
                        <td class="timestamp">${formatTime(e.ts)}</td>
                        <td class="type-${e.type}">${e.type}</td>
                        <td class="detail">${e.detail || e.tool || e.task_name || ''}</td>
                    </tr>
                `).join('');

                const toolsBody = document.querySelector('#tools-table tbody');
                toolsBody.innerHTML = (eventsData.recent_tools || []).slice().reverse().map(e => `
                    <tr>
                        <td class="timestamp">${formatTime(e.ts)}</td>
                        <td>${e.tool || ''}</td>
                        <td class="detail">${e.args || ''}</td>
                    </tr>
                `).join('');
            } catch (err) {}

            // Fetch digest and requests (less frequent — every refresh is fine)
            try {
                const digestResp = await fetch('/digest');
                const digestText = await digestResp.text();
                const digestEl = document.getElementById('digest-content');
                if (digestText && digestText !== 'No digest yet.') {
                    digestEl.innerHTML = simpleMarkdown(digestText);
                } else {
                    digestEl.innerHTML = '<span class="empty-state">No digest entries yet. Will populate after next task completes.</span>';
                }
            } catch (err) {}

            try {
                const reqResp = await fetch('/requests');
                const reqText = await reqResp.text();
                const reqEl = document.getElementById('requests-content');
                if (reqText && reqText !== 'No requests.') {
                    reqEl.innerHTML = simpleMarkdown(reqText);
                } else {
                    reqEl.innerHTML = '<span class="empty-state">No outstanding requests from the agent.</span>';
                }
            } catch (err) {}
        }

        function simpleMarkdown(text) {
            // Escape HTML first, then apply markdown formatting
            // Use a helper to create elements without literal </ in source (breaks script tag)
            var e = document.createElement('div');
            e.textContent = text;
            var escaped = e.innerHTML;
            return escaped
                .replace(/^# (.+)$/gm, function(m,p1){return '<h1>'+p1+'<'+'/h1>'})
                .replace(/^## (.+)$/gm, function(m,p1){return '<h2>'+p1+'<'+'/h2>'})
                .replace(/^### (.+)$/gm, function(m,p1){return '<h3>'+p1+'<'+'/h3>'})
                .replace(/^---$/gm, '<hr>')
                .replace(/[*][*](.+?)[*][*]/g, function(m,p1){return '<strong>'+p1+'<'+'/strong>'})
                .replace(/[*](.+?)[*]/g, function(m,p1){return '<em>'+p1+'<'+'/em>'})
                .replace(/`(.+?)`/g, function(m,p1){return '<code style="background:#21262d;padding:1px 4px;border-radius:3px">'+p1+'<'+'/code>'})
                .replace(/\\n/g, '<br>');
        }

        refresh();
        setInterval(refresh, 5000);
    </script>
</body>
</html>
"""


def _read_status_from_file() -> dict:
    """Read status from activity.jsonl file (works across processes)."""
    activity_path = os.path.join(DATA_DIR, "activity.jsonl")
    recent_events = []
    if os.path.exists(activity_path):
        try:
            # Read last 500 lines efficiently
            with open(activity_path, "rb") as f:
                # Seek to end, read backwards
                f.seek(0, 2)
                size = f.tell()
                read_size = min(size, 200_000)  # ~200KB should cover 500 events
                f.seek(max(0, size - read_size))
                lines = f.read().decode("utf-8", errors="replace").strip().split("\n")
                for line in lines[-500:]:
                    try:
                        recent_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass

    # Derive current state from the most recent events
    state = {
        "status": "idle",
        "current_task": None,
        "current_tool": None,
        "model_in_use": None,
        "started_at": None,
        "uptime_seconds": 0,
    }

    # Compute uptime from server start time
    if _SERVER_START_TIME:
        from datetime import datetime
        state["uptime_seconds"] = int((datetime.now() - _SERVER_START_TIME).total_seconds())

    # Walk backwards to find current status (the most recent state-changing event wins)
    found_status = False
    found_tool = False
    found_model = False
    for e in reversed(recent_events):
        etype = e.get("type", "")

        # Find current status (first state-changing event walking backwards)
        if not found_status:
            if etype in ("cron_complete", "cron_failed", "chat_complete"):
                state["status"] = "idle"
                found_status = True
            elif etype == "cron_start":
                state["status"] = "cron_task"
                state["current_task"] = e.get("task_name", "")
                state["started_at"] = e.get("ts", "")
                found_status = True
            elif etype == "chat_start":
                state["status"] = "chatting"
                state["started_at"] = e.get("ts", "")
                found_status = True

        # Find most recent tool (independent of status)
        if not found_tool and etype == "tool_call":
            state["current_tool"] = e.get("tool", "")
            found_tool = True
        elif not found_tool and etype == "tool_result":
            state["current_tool"] = None  # Tool finished, none active
            found_tool = True

        # Find most recent model selection
        if not found_model and etype == "model_select":
            state["model_in_use"] = e.get("model", "")
            found_model = True

        if found_status and found_tool and found_model:
            break

    # Event counts
    type_counts = {}
    for e in recent_events:
        t = e.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    recent_tools = [e for e in recent_events if e.get("type") == "tool_call"][-50:]

    return {
        "state": state,
        "event_counts": type_counts,
        "recent_events": recent_events[-100:],
        "recent_tools": recent_tools,
    }


class StatusHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/status":
            from sandbox_agent.model_tracker import read_status_from_file
            data = read_status_from_file()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode())
        elif self.path == "/events":
            data = _read_status_from_file()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode())
        elif self.path == "/models":
            from sandbox_agent.model_tracker import read_status_from_file
            data = read_status_from_file()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode())
        elif self.path == "/digest":
            digest_path = os.path.join(DATA_DIR, "digest", "latest.md")
            try:
                with open(digest_path) as f:
                    content = f.read()
            except (FileNotFoundError, OSError):
                content = "No digest yet."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content.encode())
        elif self.path == "/requests":
            # Prefer the markdown view, fall back to old format
            requests_md = os.path.join(DATA_DIR, "agent_requests.md")
            try:
                with open(requests_md) as f:
                    content = f.read()
            except (FileNotFoundError, OSError):
                content = "No requests."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content.encode())
        elif self.path == "/dashboard" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress logging


def start_status_server(port: int = 7861) -> None:
    """Start the status server as a separate process (avoids GIL blocking)."""
    import multiprocessing
    process = multiprocessing.Process(
        target=_run_server,
        args=(port,),
        daemon=True,
        name="status-server",
    )
    process.start()
    logger.info(f"Status server started on port {port} (pid={process.pid})")


def _run_server(port: int) -> None:
    """Entry point for the status server process."""
    global _SERVER_START_TIME
    from datetime import datetime
    _SERVER_START_TIME = datetime.now()
    server = HTTPServer(("0.0.0.0", port), StatusHandler)
    server.serve_forever()
