"""One-time, idempotent startup migrations for the skills-system transition.

Called from main()/bootstrap_background() before load_system_message(), so the
system prompt is assembled from a migrated state.
"""

import logging
import os
from datetime import datetime

from sandbox_agent.config import DATA_DIR
from sandbox_agent.tools.git_autocommit import autocommit

logger = logging.getLogger(__name__)


def quarantine_unparseable_pipeline_state() -> None:
    """Rename pipeline state.json files that cannot parse as PipelineState
    (agent-fabricated shadow content from the phantom-promote era — missing
    project_name/description) to state.json.unparseable-<ts>. Content is
    preserved for forensics; load_state stops logging 'Corrupt pipeline
    state' on every boot. Legacy-but-real files load fine (StageState
    normalizes old status values) and are left alone. Idempotent."""
    from sandbox_agent.pipeline.models import PipelineState
    projects_dir = os.path.join(DATA_DIR, "projects")
    if not os.path.isdir(projects_dir):
        return
    import json as _json
    for proj in os.listdir(projects_dir):
        path = os.path.join(projects_dir, proj, "pipeline", "state.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                PipelineState(**_json.load(f))
            continue  # parses (possibly via legacy normalization) — keep
        except Exception:
            pass
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        quarantine = f"{path}.unparseable-{stamp}"
        try:
            os.replace(path, quarantine)
            logger.warning(
                "Quarantined unparseable pipeline state for %s -> %s "
                "(agent-fabricated or pre-schema content; preserved for forensics)",
                proj, os.path.basename(quarantine))
        except OSError:
            logger.exception("Could not quarantine state for %s", proj)


def migrate_pre_skills() -> None:
    """Archive a pre-skills DATA_DIR/SOUL.md so the slimmed bundled SOUL applies.

    Background: load_system_message historically read ONLY the bundled SOUL.md,
    so agent soul-edits accumulated in DATA_DIR/SOUL.md without ever reaching
    the system prompt. Now that the DATA_DIR override is honored, a stale
    pre-skills copy (no `## Skills` section, typically 30k+ chars) would shadow
    the slim bundled SOUL and defeat the de-clutter. Move it to soul_archive/
    and leave the agent a memory pointing at it. Idempotent: a DATA_DIR SOUL
    that already has a `## Skills` section (agent-updated post-migration) is
    left alone.
    """
    soul_path = os.path.join(DATA_DIR, "SOUL.md")
    if not os.path.exists(soul_path):
        return
    try:
        content = open(soul_path).read()
    except OSError:
        logger.exception("migrate_pre_skills: could not read DATA_DIR/SOUL.md")
        return
    if "## Skills" in content:
        return  # already post-skills — nothing to migrate

    archive_dir = os.path.join(DATA_DIR, "soul_archive")
    os.makedirs(archive_dir, exist_ok=True)
    archive_path = os.path.join(archive_dir, "SOUL-pre-skills.md")
    os.replace(soul_path, archive_path)
    logger.warning(
        "migrate_pre_skills: archived stale DATA_DIR/SOUL.md (%d chars) to %s — "
        "the slimmed bundled SOUL now applies", len(content), archive_path)

    # Leave the agent a breadcrumb so it can mine its old edits back in.
    note = (f"- [{datetime.now().strftime('%Y-%m-%d')}] SOUL.md was slimmed to a skills "
            f"index (read_skill). My pre-skills SOUL with accumulated edits is archived at "
            f"soul_archive/SOUL-pre-skills.md — mine it for anything worth re-adding via update_soul.")
    memories_path = os.path.join(DATA_DIR, "MEMORIES.md")
    try:
        existing = open(memories_path).read() if os.path.exists(memories_path) else ""
        if "SOUL-pre-skills.md" not in existing:
            with open(memories_path, "a") as f:
                if existing and "## Technical Notes" in existing:
                    pass  # append at end regardless — section placement is cosmetic
                f.write(f"\n{note}\n")
    except OSError:
        logger.exception("migrate_pre_skills: could not append migration note to MEMORIES.md")

    try:
        autocommit("soul_archive/SOUL-pre-skills.md", "Archive pre-skills SOUL.md (skills migration)")
        autocommit("MEMORIES.md", "Note pre-skills SOUL archive location")
    except Exception:  # noqa: BLE001
        pass
