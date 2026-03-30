#!/usr/bin/env python3
"""Cleanup and organize the sandbox agent's data directory.

Dry-run by default. Use --execute to apply changes.

Usage:
    python cleanup_data.py ~/sandbox_agent_data/
    python cleanup_data.py ~/sandbox_agent_data/ --execute
    python cleanup_data.py ~/sandbox_agent_data/ --execute -v
"""

import argparse
import filecmp
import fnmatch
import os
import shutil
import subprocess
import sys

# System files at root that must NEVER be moved
SYSTEM_FILES = {
    "tasks.json", "activity.jsonl", "MEMORIES.md", "HEARTBEAT.md",
    "SOUL.md", "agent_requests.json", "agent_requests.md",
}

# System directories at root that must NEVER be moved
SYSTEM_DIRS = {
    "chat_logs", "digest", "scratch", ".git", "projects", "checkpoints",
    "code_interpreter", "workspace",
}

# Junk patterns to delete
JUNK_PATTERNS = {".DS_Store", "__pycache__", ".ipynb_checkpoints"}

# Default project for all flatsixai work
PROJECT = "projects/flatsixai"


def matches_any(name, patterns):
    """Check if name matches any fnmatch pattern (case-insensitive)."""
    name_lower = name.lower()
    return any(fnmatch.fnmatch(name_lower, p) for p in patterns)


# Classification rules: (patterns, destination_subdir)
# Evaluated in order — first match wins
RULES = [
    # Trading / financial / news → trading_reports/
    (
        ["*trading*", "*news*", "*iran*", "*stock*", "*market-scan*",
         "*raw_news*", "*processed_news*", "*insights_*"],
        "trading_reports",
    ),
    # Heartbeat / session logs → project logs/
    (
        ["*heartbeat*", "*session*", "*sprint-log*", "*sprint_log*"],
        f"{PROJECT}/logs",
    ),
    # Pipeline instructions → project pipeline/
    (
        ["*pipeline-instructions*", "*pipeline-tracker*", "*pipeline-task*",
         "*pipeline-sprint*"],
        f"{PROJECT}/pipeline",
    ),
    # Sprint / ideas → project ideas/
    (
        ["*sprint*", "*new-ideas*", "*new_ideas*", "*idea*", "*brainstorm*",
         "*novel_ideas*", "*ranked-ideas*", "*ranked_ideas*"],
        f"{PROJECT}/ideas",
    ),
    # Research / analysis → project research/
    (
        ["*research*", "*analysis*", "*competitive*", "*market*",
         "*deep_dive*", "*deep-dive*", "*companies*", "*yc_w26*",
         "*yc-w26*", "*outreach*", "*outplacement*", "*interview*",
         "*customer*", "*agentguard*", "*agentops*", "*agentpayroll*",
         "*agentscale*", "*careerbridge*", "*crossstate*", "*hallucheck*",
         "*reskillbridge*", "*shadowai*", "*compliancechain*",
         "*ai-workflow*", "*cursor-legal*", "*localai*", "*privacyflow*",
         "*spoofshield*", "*humangate*", "*datasovereignty*",
         "*agentidentity*", "*agentlineage*"],
        f"{PROJECT}/research",
    ),
    # Reports / summaries → project reports/
    (
        ["*report*", "*summary*", "*findings*", "*synthesis*", "*guide*"],
        f"{PROJECT}/reports",
    ),
]

# Extension-based fallback rules (after pattern rules)
EXT_RULES = {
    ".jsonl": f"{PROJECT}/data",
    ".json": f"{PROJECT}/data",
    ".md": f"{PROJECT}/research",
    ".txt": f"{PROJECT}/research",
    ".py": f"{PROJECT}/prototypes",
    ".html": f"{PROJECT}/data",
}


def classify_file(name):
    """Classify a file by name. Returns (action, destination) or None."""
    # System files
    if name in SYSTEM_FILES:
        return ("SKIP", None)

    # Junk
    if name in JUNK_PATTERNS:
        return ("DELETE", None)

    # Pattern rules
    for patterns, dest in RULES:
        if matches_any(name, patterns):
            return ("MOVE", dest)

    # Extension fallback
    _, ext = os.path.splitext(name)
    if ext.lower() in EXT_RULES:
        return ("MOVE", EXT_RULES[ext.lower()])

    return ("UNCLASSIFIED", None)


def safe_dest(src_path, dest_path):
    """Handle conflicts. Returns (final_dest, action) where action is None or 'DELETE_DUP'."""
    if not os.path.exists(dest_path):
        return dest_path, None

    # Check if identical content
    try:
        if os.path.isfile(src_path) and os.path.isfile(dest_path):
            if filecmp.cmp(src_path, dest_path, shallow=False):
                return None, "DELETE_DUP"
    except Exception:
        pass

    # Add numeric suffix
    base, ext = os.path.splitext(dest_path)
    for i in range(1, 100):
        candidate = f"{base}-{i}{ext}"
        if not os.path.exists(candidate):
            return candidate, None

    return dest_path, "CONFLICT"


def scan_root_files(data_dir):
    """Scan root-level files and classify them."""
    ops = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            if name in SYSTEM_DIRS:
                continue
            # Handle special directories
            if name in JUNK_PATTERNS:
                ops.append(("DELETE_DIR", path, None))
            elif name == "flatsixai":
                # Duplicate dir — merge contents
                for sub in sorted(os.listdir(path)):
                    sub_path = os.path.join(path, sub)
                    if os.path.isfile(sub_path):
                        action, dest = classify_file(sub)
                        if dest:
                            dest_path = os.path.join(data_dir, dest, sub)
                            final, conflict = safe_dest(sub_path, dest_path)
                            if conflict == "DELETE_DUP":
                                ops.append(("DELETE_DUP", sub_path, dest_path))
                            elif final:
                                ops.append(("MOVE", sub_path, final))
                ops.append(("DELETE_DIR_AFTER", path, None))
            elif name == "prototypes":
                # Merge into project prototypes
                dest_dir = os.path.join(data_dir, PROJECT, "prototypes")
                ops.append(("MERGE_DIR", path, dest_dir))
            elif name == "data":
                dest_dir = os.path.join(data_dir, PROJECT, "data")
                ops.append(("MERGE_DIR", path, dest_dir))
            elif name == "tmp":
                ops.append(("DELETE_DIR", path, None))
            else:
                # Unknown dir at root — move into project
                dest_dir = os.path.join(data_dir, PROJECT, name)
                ops.append(("MERGE_DIR", path, dest_dir))
            continue

        if not os.path.isfile(path):
            continue

        action, dest = classify_file(name)
        if action == "SKIP":
            ops.append(("SKIP", path, None))
        elif action == "DELETE":
            ops.append(("DELETE", path, None))
        elif action == "MOVE" and dest:
            dest_path = os.path.join(data_dir, dest, name)
            final, conflict = safe_dest(path, dest_path)
            if conflict == "DELETE_DUP":
                ops.append(("DELETE_DUP", path, dest_path))
            elif final:
                ops.append(("MOVE", path, final))
        else:
            ops.append(("UNCLASSIFIED", path, None))

    return ops


def scan_flatsixai_root(data_dir):
    """Move pipeline-instructions and other loose files from flatsixai project root."""
    ops = []
    flatsixai_dir = os.path.join(data_dir, PROJECT)
    if not os.path.isdir(flatsixai_dir):
        return ops

    for name in sorted(os.listdir(flatsixai_dir)):
        path = os.path.join(flatsixai_dir, name)
        if not os.path.isfile(path):
            continue

        # Keep project essentials at root
        if name in ("TODO.md", "README.md", ".project.json", "pipeline-tracker.json",
                     "pipeline-task-descriptions.json"):
            ops.append(("SKIP", path, None))
            continue

        # Pipeline instructions → pipeline/
        if name.startswith("pipeline-instructions-") or name.startswith("pipeline-"):
            if name not in ("pipeline-tracker.json", "pipeline-task-descriptions.json"):
                dest = os.path.join(flatsixai_dir, "pipeline", name)
                final, conflict = safe_dest(path, dest)
                if conflict == "DELETE_DUP":
                    ops.append(("DELETE_DUP", path, dest))
                elif final:
                    ops.append(("MOVE", path, final))
                continue

        # Classify remaining files
        action, dest_rel = classify_file(name)
        if action == "MOVE" and dest_rel:
            # dest_rel is like "projects/flatsixai/research" — extract subdir
            subdir = dest_rel.replace(PROJECT + "/", "")
            dest = os.path.join(flatsixai_dir, subdir, name)
            final, conflict = safe_dest(path, dest)
            if conflict == "DELETE_DUP":
                ops.append(("DELETE_DUP", path, dest))
            elif final:
                ops.append(("MOVE", path, final))
        elif action == "DELETE":
            ops.append(("DELETE", path, None))

    return ops


def scan_junk_recursive(data_dir):
    """Find .DS_Store and __pycache__ recursively."""
    ops = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name == ".DS_Store":
                ops.append(("DELETE", os.path.join(root, name), None))
        for name in list(dirs):
            if name == "__pycache__":
                ops.append(("DELETE_DIR", os.path.join(root, name), None))
                dirs.remove(name)
    return ops


def execute_ops(ops, data_dir, dry_run=True, verbose=False):
    """Execute or report operations."""
    counts = {"SKIP": 0, "MOVE": 0, "DELETE": 0, "DELETE_DUP": 0,
              "DELETE_DIR": 0, "MERGE_DIR": 0, "UNCLASSIFIED": 0}

    rel = lambda p: os.path.relpath(p, data_dir) if p else ""

    for action, src, dst in ops:
        if action == "SKIP":
            counts["SKIP"] += 1
            if verbose:
                print(f"  SKIP: {rel(src)}")

        elif action == "MOVE":
            counts["MOVE"] += 1
            print(f"  MOVE: {rel(src)} → {rel(dst)}")
            if not dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)

        elif action == "DELETE":
            counts["DELETE"] += 1
            print(f"  DELETE: {rel(src)}")
            if not dry_run:
                os.remove(src)

        elif action == "DELETE_DUP":
            counts["DELETE_DUP"] += 1
            print(f"  DELETE (duplicate of {rel(dst)}): {rel(src)}")
            if not dry_run:
                os.remove(src)

        elif action == "DELETE_DIR":
            counts["DELETE_DIR"] += 1
            print(f"  DELETE DIR: {rel(src)}")
            if not dry_run:
                shutil.rmtree(src, ignore_errors=True)

        elif action == "DELETE_DIR_AFTER":
            # Delete empty dir after its contents were moved
            if not dry_run:
                shutil.rmtree(src, ignore_errors=True)

        elif action == "MERGE_DIR":
            counts["MERGE_DIR"] += 1
            print(f"  MERGE DIR: {rel(src)} → {rel(dst)}")
            if not dry_run:
                if os.path.exists(src):
                    os.makedirs(dst, exist_ok=True)
                    for item in os.listdir(src):
                        s = os.path.join(src, item)
                        d = os.path.join(dst, item)
                        if os.path.isdir(s):
                            if os.path.exists(d):
                                # Merge subdirectories
                                for sub_item in os.listdir(s):
                                    shutil.move(os.path.join(s, sub_item),
                                               os.path.join(d, sub_item))
                            else:
                                shutil.move(s, d)
                        else:
                            if not os.path.exists(d):
                                shutil.move(s, d)
                            elif filecmp.cmp(s, d, shallow=False):
                                os.remove(s)  # Duplicate
                            else:
                                base, ext = os.path.splitext(d)
                                shutil.move(s, f"{base}-merged{ext}")
                    shutil.rmtree(src, ignore_errors=True)

        elif action == "UNCLASSIFIED":
            counts["UNCLASSIFIED"] += 1
            print(f"  UNCLASSIFIED: {rel(src)}")

    return counts


def cleanup_empty_dirs(data_dir):
    """Remove empty directories (bottom-up)."""
    removed = 0
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for d in dirs:
            dirpath = os.path.join(root, d)
            if d.startswith("."):
                continue
            try:
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)
                    removed += 1
            except OSError:
                pass
    return removed


def git_commit(data_dir):
    """Git commit all changes."""
    git_dir = os.path.join(data_dir, ".git")
    if not os.path.isdir(git_dir):
        return
    try:
        subprocess.run(["git", "add", "-A"], cwd=data_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "cleanup: reorganize data directory per SOUL.md rules"],
            cwd=data_dir, capture_output=True,
        )
        print("\nGit: committed cleanup changes")
    except Exception as e:
        print(f"\nGit: failed to commit: {e}")


def main():
    parser = argparse.ArgumentParser(description="Cleanup sandbox agent data directory")
    parser.add_argument("data_dir", help="Path to data directory (e.g., ~/sandbox_agent_data/)")
    parser.add_argument("--execute", action="store_true", help="Actually perform changes (default: dry-run)")
    parser.add_argument("--no-git", action="store_true", help="Skip git commit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show skipped files too")
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"=== CLEANUP {mode} ===")
    print(f"Data dir: {data_dir}\n")

    # Collect all operations
    print("--- Root-level files ---")
    root_ops = scan_root_files(data_dir)

    print("\n--- Flatsixai project root ---")
    project_ops = scan_flatsixai_root(data_dir)

    print("\n--- Junk files (recursive) ---")
    junk_ops = scan_junk_recursive(data_dir)

    all_ops = root_ops + project_ops + junk_ops

    # Execute or report
    print(f"\n{'=' * 50}")
    print(f"{'EXECUTING' if args.execute else 'PLANNED'} OPERATIONS:")
    print(f"{'=' * 50}\n")

    counts = execute_ops(all_ops, data_dir, dry_run=not args.execute, verbose=args.verbose)

    # Clean up empty dirs
    if args.execute:
        removed = cleanup_empty_dirs(data_dir)
        if removed:
            print(f"\nRemoved {removed} empty directories")

    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY:")
    print(f"  Skipped:      {counts['SKIP']}")
    print(f"  Moved:        {counts['MOVE']}")
    print(f"  Deleted:      {counts['DELETE']}")
    print(f"  Dup deleted:  {counts['DELETE_DUP']}")
    print(f"  Dirs deleted: {counts['DELETE_DIR']}")
    print(f"  Dirs merged:  {counts['MERGE_DIR']}")
    print(f"  Unclassified: {counts['UNCLASSIFIED']}")
    print(f"{'=' * 50}")

    if not args.execute:
        print("\nThis was a DRY RUN. Use --execute to apply changes.")
    else:
        if not args.no_git:
            git_commit(data_dir)
        print("\nDone!")


if __name__ == "__main__":
    main()
