# Memory Maintenance — consolidating MEMORIES.md
> The compaction protocol: how to merge, prune, and archive your memories when the heartbeat asks.

Your MEMORIES.md is injected into your system prompt, but only the newest entries up to a size cap. When the file grows past the trigger threshold, a heartbeat item will ask you to consolidate. Follow this protocol:

## Protocol
1. `read_memories()` — read the FULL current file (the system-prompt copy may be capped).
2. Rewrite it, applying these rules:
   - **Merge duplicates**: multiple entries about the same topic/date → one entry carrying all the facts.
   - **Drop superseded entries**: keep only the current state (e.g. "X moved to Y" replaces "X is at Z").
   - **Drop stale task chatter**: one-off task outcomes that no longer inform future work.
   - **Always preserve**: user preferences and identity facts, hard technical rules (API quirks, footguns), anything marked important.
   - Keep the canonical `##` section structure (User Preferences / Facts & Knowledge / Technical Notes / Task Learnings) and the `- [YYYY-MM-DD] entry` format.
   - Target: comfortably under 6,000 characters.
3. Call `compact_memories(new_content=<your rewritten file>)`.
   - The old file is archived VERBATIM to `memories_archive/YYYY-MM.md` automatically — nothing is lost, so be aggressive about pruning.
   - The tool rejects content that is longer than the current file or missing the section headers.
4. Old memories stay loadable: `read_memories(archive="list")` to see archives, `read_memories(archive="YYYY-MM")` to read one.

## Judgment guide
- A memory earns its place by changing FUTURE behavior. "The user prefers kebab-case" stays forever; "researched X on May 3, saved to projects/x/" can go once the project is done.
- When unsure, compress rather than delete: fold details into a one-line summary with a pointer to the file/project that holds the full record.
