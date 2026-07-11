"""Skills system: on-demand how-to guides + auto-injection.

Skills are plain-markdown guides (line 1 `# Title`, line 2 `> description`)
extracted from SOUL.md so the system prompt stays small. Bundled defaults live
in sandbox_agent/skills/; DATA_DIR/skills/ overrides win by name (same pattern
as the DATA_DIR SOUL.md override). Loaded two ways:

- read_skill tool — the agent asks for a guide by name (SOUL's Skills section
  indexes when to use each).
- maybe_inject_skill — main.py's _logged_call_tool wrapper auto-prepends the
  matching skill to a tool result the FIRST time a trigger tool (browser_*)
  is used in a conversation. The [skill:NAME] marker in history is the dedup.
"""

import logging
import os
import re
from typing import List, Optional, Union

from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import BaseTool, register_tool

from sandbox_agent.config import DATA_DIR, SKILL_AUTOINJECT, SKILL_MAX_CHARS

logger = logging.getLogger(__name__)

_BUNDLED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _skill_path(name: str) -> Optional[str]:
    """DATA_DIR override wins; bundled fallback. None if neither exists."""
    override = os.path.join(DATA_DIR, "skills", f"{name}.md")
    if os.path.isfile(override):
        return override
    bundled = os.path.join(_BUNDLED_DIR, f"{name}.md")
    if os.path.isfile(bundled):
        return bundled
    return None


def _list_skills() -> str:
    """Merged listing (DATA_DIR overrides win per name): `name — description`."""
    names = {}
    for d, tag in ((_BUNDLED_DIR, ""), (os.path.join(DATA_DIR, "skills"), " (customized)")):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md"):
                continue
            name = fn[:-3]
            desc = ""
            try:
                with open(os.path.join(d, fn)) as f:
                    f.readline()                       # title
                    second = f.readline().strip()
                    if second.startswith(">"):
                        desc = second.lstrip("> ").strip()
            except OSError:
                continue
            names[name] = f"- {name}: {desc}{tag}"
    if not names:
        return "(no skills installed)"
    return "Available skills (read_skill(name) to load):\n" + "\n".join(names[k] for k in sorted(names))


def _load_skill_body(name: str) -> Optional[str]:
    path = _skill_path(name)
    if not path:
        return None
    try:
        with open(path) as f:
            body = f.read()
    except OSError:
        return None
    if len(body) > SKILL_MAX_CHARS:
        body = body[:SKILL_MAX_CHARS] + (
            f"\n...[truncated — skill file oversized, edit {path}]")
    return body


@register_tool("read_skill")
class ReadSkill(BaseTool):
    """Load an on-demand skill guide."""

    name = "read_skill"
    description = (
        "Load a skill guide — detailed how-to instructions for a capability. "
        "Call with no args to list available skills. The 'Skills' section of "
        "your SOUL indexes when to use each; read the relevant skill BEFORE "
        "starting that kind of work. If '[skill:NAME]' already appears in this "
        "conversation, it was auto-loaded — don't re-read it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name, e.g. 'browser-automation'. Omit to list all skills.",
            },
        },
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        name = (params.get("name") or "").strip()
        if not name:
            return _list_skills()
        if not _NAME_RE.match(name):
            return f"Invalid skill name '{name}'.\n\n{_list_skills()}"
        body = _load_skill_body(name)
        if body is None:
            return f"No skill named '{name}'.\n\n{_list_skills()}"
        return f"[skill:{name}]\n{body}\n[end skill]"


# --- auto-inject -------------------------------------------------------------

def _marker_in_messages(messages: List, marker: str) -> bool:
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, str):
            if marker in content:
                return True
        elif isinstance(content, list):
            for item in content:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if text and marker in text:
                    return True
    return False


def maybe_inject_skill(tool_name: str, result, messages: List):
    """Prepend the matching skill guide to `result` the first time a trigger
    tool is used in a conversation (dedup via the [skill:NAME] marker in
    history). Called from main.py's _logged_call_tool wrapper. Handles str
    results and multimodal ContentItem lists; anything else passes through."""
    skill = next((s for pfx, s in SKILL_AUTOINJECT.items() if tool_name.startswith(pfx)), None)
    if skill is None:
        return result
    marker = f"[skill:{skill}]"
    if _marker_in_messages(messages, marker):
        return result
    body = _load_skill_body(skill)
    if body is None:
        return result
    note = f"{marker} Auto-loaded guide for the tool you just used:\n{body}\n[end skill]\n\n"
    if isinstance(result, str):
        return note + result
    if isinstance(result, list):
        for item in result:
            if isinstance(item, ContentItem) and getattr(item, "text", None):
                item.text = note + item.text
                return result
        result.insert(0, ContentItem(text=note.rstrip()))
        return result
    return result
