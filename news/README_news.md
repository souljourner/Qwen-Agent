# Apple News Automation — Research & Architecture

## Overview

This directory contains research, working proofs-of-concept, and architecture
for connecting the sandbox_agent to Apple News on macOS so that a vision-model
LLM can visually read articles one by one.

Date: 2026-05-31
Status: Research complete, implementation pending

---

## 1. Research Findings

### 1.1 Apple News Is a Closed Ecosystem

| Surface | Verdict |
|---------|---------|
| `news.apple.com` web | Redirect stub → `apple.com/news` (marketing page only, zero content) |
| `apple.news/XXXXX` article links | JS redirect pages that forward to canonical publisher URLs |
| RSS feeds (`/rss/`, `/feeds/`) | 404 — none exist |
| Public API | None found |
| AppleScript dictionary | **No `.sdef` file** — app is not scriptable. Only `version`, `name`, `frontmost` work. Everything else returns `-1728`. |
| On-disk cache (`~/Library/Group Containers/group.com.apple.news/`) | Sandboxed — "Operation not permitted" |
| WKWebView inside News.app | Confirmed in binary strings, but content is invisible to Accessibility API |
| URL scheme `applenews://` / `applenewss://` | Present in Info.plist, launches the app, but no documented parameters for navigation |

### 1.2 What Actually Works (Verified)

All commands below use `osascript` with **System Events** (requires Accessibility
permissions in System Settings → Privacy & Security → Accessibility).

#### Launch and Navigate

```applescript
-- Launch News
tell application "News" to activate

-- Go to Today feed (Cmd+1)
tell application "System Events" to tell process "News"
    keystroke "1" using command down
end tell

-- Navigate down one item in the feed
tell application "System Events" to tell process "News"
    key code 125  -- down arrow
end tell

-- Open the selected article
tell application "System Events" to tell process "News"
    keystroke return
end tell

-- Go back to feed (Cmd+W closes article)
tell application "System Events" to tell process "News"
    keystroke "w" using command down
end tell
```

#### Copy Article URL

```applescript
-- Cmd+Option+C copies the apple.news share URL to clipboard
tell application "System Events" to tell process "News"
    keystroke "c" using {command down, option down}
end tell
```

The URL format is `https://apple.news/AMGl1T3McSaigxSwWRV0GGA` (17-char ID).

#### Canonical URL Extraction

`apple.news/XXXXX` pages are redirect stubs containing JavaScript:

```javascript
redirectToUrl("https://www.theatlantic.com/ideas/2026/05/electric-ferrari-luce/687367/?utm_source=apple_news")
```

Extract with regex:

```python
import re, subprocess

def get_canonical_url(apple_news_url: str) -> str | None:
    r = subprocess.run(['curl', '-sL', apple_news_url], capture_output=True, text=True, timeout=15)
    m = re.search(r'redirectToUrl\("([^"]+)"\)', r.stdout)
    return m.group(1) if m else None
```

#### Window Screenshot

News.app uses SceneKit windows which are invisible to `screencapture -l` (layer
capture) and `CGWindowListCreateImage`. The only reliable approach is **region
capture** using window bounds from CoreGraphics:

```python
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGWindowListExcludeDesktopElements,
)
import subprocess

window_list = CGWindowListCopyWindowInfo(
    kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, 0
)

for w in window_list:
    if 'News' in w.get('kCGWindowOwnerName', ''):
        b = w.get('kCGWindowBounds', {})
        x, y = int(b['X']), int(b['Y'])
        w2, h = int(b['Width']), int(b['Height'])
        # Example: 501, 30, 2006, 1662
        subprocess.run([
            'screencapture', '-x', '-R', f'{x},{y},{w2},{h}', '/tmp/news.png'
        ])
        break
```

**Dependencies:** `pyobjc-core`, `pyobjc-framework-Quartz` (for `Quartz` module).

#### Image Compression

Raw PNG: ~4 MB (2006×1662), ~1.3M tokens as base64 — too large.

Compressed JPEG at 50% scale, quality 70: ~200 KB, ~50K tokens — workable.

```python
from PIL import Image
import base64, io

img = Image.open('/tmp/news.png')
w, h = int(img.width * 0.5), int(img.height * 0.5)
small = img.resize((w, h), Image.LANCZOS)
buf = io.BytesIO()
small.save(buf, format='JPEG', quality=70)
buf.seek(0)
b64 = base64.b64encode(buf.read()).decode()
data_url = f"data:image/jpeg;base64,{b64}"
```

### 1.3 Full Verified Workflow

```
1. osascript: tell application "News" to activate          → app opens
2. delay 1s
3. osascript: keystroke "1" using command down              → Today feed
4. delay 2s
5. osascript: key code 125 (×N)                             → select article
6. delay 1s
7. osascript: keystroke return                              → open article
8. delay 2s
9. osascript: keystroke "c" using {command down, option down}  → copy URL
10. pbpaste → "https://apple.news/A3d_JiFF5SfqQE94YZLVEpw"
11. curl apple.news/URL → extract canonical via regex
12. screencapture -R x,y,w,h /tmp/news.png                 → 4MB PNG
13. PIL resize + JPEG compress → ~200KB
14. base64 encode → data URL
15. Send to vision model as ContentItem(image=data_url)
16. osascript: keystroke "w" using command down             → go back
17. Repeat from step 5 for next article
```

This workflow was tested end-to-end and successfully navigated between articles,
copied URLs, extracted canonical publisher links, and captured screenshots.

---

## 2. Architecture Options

### Option A — MCP Server (Recommended)

Run a Python MCP server on the host that exposes News automation as tools.
The sandbox_agent container connects via `host.docker.internal`.

```
┌─────────────────────┐          MCP (stdio/HTTP)          ┌─────────────────────┐
│  Host Mac           │                                     │  Docker Container    │
│                     │  news_search("AI")               │                       │
│  ┌───────────────┐  │  → AppleScript + screencapture    │  sandbox_agent        │
│  │ MCP Server    │◄─┼───────────────────────────────────┤  (FnCallAgent)        │
│  │ port 8765     │  │  ← ContentItem(image=base64)      │                       │
│  └───────────────┘  │                                   │  ┌──────────────────┐  │
│                     │                                   │  │ browser_tools.py │  │
│  News.app            │  ┌─────────────────────────────┐  │  │ (for reading     │
│  (native, sandboxed) │  │  canonical URL extraction   │  │  │  canonical URLs) │  │
│                     │  └─────────────────────────────┘  │  └──────────────────┘  │
└─────────────────────┘                                   └─────────────────────┘
```

**Tools exposed by MCP server:**

| Tool | Parameters | Returns |
|------|-----------|---------|
| `news_open` | (none) | Launches News, goes to Today feed |
| `news_search` | `query: str` | Searches, returns article count |
| `news_select_next` | `steps: int` (default 1) | Moves down N items in feed |
| `news_open_article` | (none) | Opens currently selected article |
| `news_screenshot` | `scale: float`, `quality: int` | `ContentItem(image=data_url)` — compressed JPEG |
| `news_copy_url` | (none) | `str` — the `apple.news/` URL |
| `news_get_canonical` | `apple_url: str` | `str` — canonical publisher URL |
| `news_go_back` | (none) | Closes article, returns to feed |
| `news_scroll_article` | `direction: str`, `amount: int` | Scrolls within article |

**Pros:** Clean separation, uses existing MCPManager, stdio transport is simple
**Cons:** Requires MCP server setup, additional process on host

### Option B — HTTP Bridge Server

A small FastAPI/Flask server on the host (port e.g. 9876) that wraps AppleScript
and screencapture. The container calls `web_url_fetch` or a custom HTTP client.

```
Host:  python3 news_bridge.py --port 9876
       GET  /api/news/screenshot?scale=0.5&quality=70  → JSON {image: "data:image/jpeg;base64,..."}
       POST /api/news/navigate?direction=down&steps=3  → JSON {ok: true}
       GET  /api/news/url                               → JSON {apple_url: "...", canonical: "..."}
```

**Pros:** Simpler than MCP, no framework changes needed
**Cons:** Ad-hoc protocol, no tool description/schema

### Option C — Exec + SSH

The container SSHes into the host to run the automation script.

**Pros:** No server needed
**Cons:** SSH key setup, latency, fragile

### Recommendation

**Option A (MCP)** is the cleanest. The sandbox_agent already has `MCPManager` in
`qwen_agent/tools/mcp_manager.py`. We'd add a host-side MCP server config.

---

## 3. Implementation Plan

### Phase 1 — Host-side MCP Server

1. Create `news/mcp_server.py` — FastMCP or stdio-based MCP server
2. Implement all 9 tools using AppleScript + screencapture + PIL
3. Test standalone: `python3 news/mcp_server.py` → verify each tool

### Phase 2 — Container Integration

4. Create `news/mcp_config.json` — MCP server config pointing to host
5. Add to `sandbox_agent/config.py` TOOL_LIST
6. Update `sandbox_agent/main.py` to initialize MCP connection to host server
7. Test: agent calls `news_screenshot()`, receives image, vision model reads it

### Phase 3 — Reading Workflow

8. Create scheduled task `news_digest` that:
   - Opens News, searches for topics
   - Iterates articles: screenshot → vision model reads → summarizes
   - Compiles digest, posts to chat session

### Phase 4 — Dual-mode Reading

9. For paywalled/premium articles (News+ only), use screenshot mode
10. For free articles, extract canonical URL → `browser_navigate(canonical)` →
    `browser_screenshot()` in container (cheaper, no host round-trip)

---

## 4. System Requirements

### Host Mac

- macOS 14+ (Sonoma) or later
- Apple News app installed (`/System/Applications/News.app`)
- **Accessibility permissions** granted for Terminal/iTerm/VS Code:
  System Settings → Privacy & Security → Accessibility
- **Screen Recording permissions** granted:
  System Settings → Privacy & Security → Screen Recording
- Python 3.14 with:
  ```bash
  pip install pyobjc-core pyobjc-framework-Quartz pyobjc-framework-Cocoa Pillow
  ```

### Docker Container

- Existing sandbox_agent container (no changes needed for Phase 1)
- MCPManager already available in `qwen_agent/tools/mcp_manager.py`

---

## 5. Keyboard Shortcuts Reference

| Action | Shortcut | AppleScript |
|--------|----------|-------------|
| Today feed | `Cmd+1` | `keystroke "1" using command down` |
| Search | `Cmd+F` | `keystroke "f" using command down` |
| Open article | `Return` | `keystroke return` |
| Go back / close | `Cmd+W` | `keystroke "w" using command down` |
| Copy link | `Cmd+Option+C` | `keystroke "c" using {command down, option down}` |
| Down arrow | — | `key code 125` |
| Up arrow | — | `key code 126` |
| Scroll down | — | `scroll vertically by -500` (in window context) |
| Select all text | `Cmd+A` | `keystroke "a" using command down` |
| Copy | `Cmd+C` | `keystroke "c" using command down` |

---

## 6. Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| `-1728` from News | AppleScript command not supported | Use System Events instead |
| `-1719` from System Events | Accessibility not enabled | Prompt user to grant permission |
| `-25211` from System Events | Process not found or permission denied | Re-launch News, retry |
| `screencapture` fails | Screen Recording not enabled | Prompt user to grant permission |
| `could not create image from window` | SceneWindow limitation | Use `screencapture -R` with bounds instead |
| `apple.news/` → no canonical URL | Premium-only article, no redirect | Fall back to screenshot-only mode |
| Clipboard unchanged | News not focused | Re-activate, re-copy |

---

## 7. Token Budget

| Operation | Tokens |
|-----------|--------|
| Screenshot (50%, Q70 JPEG) | ~50,000 |
| Screenshot (25%, Q70 JPEG) | ~12,500 |
| Screenshot (15%, Q50 JPEG) | ~4,000 |
| Text summary of article | ~500-1,000 |
| AppleScript command | ~50 |
| Canonical URL fetch | ~200 |

**Budget strategy:** Use 25% scale for feed overview (many articles visible),
50% scale for individual article reading. Fall back to canonical URL text
extraction when possible (avoids image tokens entirely).

---

## 8. File Structure (Planned)

```
news/
├── README.md                    ← this file
├── mcp_server.py                ← MCP server exposing 9 tools
├── news_bridge.py               ← Helper: AppleScript + screencapture + PIL
├── test_workflow.py             ← Standalone test script
├── mcp_config.json              ← MCP config for container
└── screenshots/                  ← Debug screenshots
```

---

## 9. Lessons Learned

1. **AppleScript `tell application "News"` is nearly useless** — only `version`,
   `name`, `frontmost` work. All interaction must go through System Events.

2. **System Events requires two permissions:** Accessibility (for keystroke/key
   code) and Screen Recording (for screencapture). Both must be granted to the
   parent process (Terminal, iTerm, or whatever launches osascript).

3. **SceneKit windows cannot be captured** via `screencapture -l` or
   `CGWindowIDCreateImage`. Region capture with `screencapture -R` using
   bounds from `CGWindowListCopyWindowInfo` is the only reliable method.

4. **`key code 125` (down arrow) actually navigates** within News — it moves
   the selection in the feed. This was surprising given that all UI element
   counts are 0 (WKWebView content is invisible to Accessibility).

5. **`Cmd+Option+C` copies the share URL** — this is the key bridge between
   the native app and the web. The `apple.news/` URL redirects to the canonical
   publisher URL, which the container's browser can read directly.

6. **Apple News article URLs are redirect stubs** — `apple.news/XXXXX` returns
   a 200 with JavaScript that redirects to the publisher's canonical URL. This
   means the container's Playwright browser can read free articles directly
   without needing screenshots.

7. **pyobjc is essential** for proper CoreGraphics integration. Raw ctypes
   against ApplicationServices is fragile (CFNumber type mismatches, CGRect
   struct issues).

8. **Image token cost is the bottleneck** — a full-resolution PNG is ~1.3M
   tokens. Aggressive compression (25-50% scale, JPEG Q70) is mandatory.
   For free articles, extracting the canonical URL and reading via the
   container's browser is far cheaper.
