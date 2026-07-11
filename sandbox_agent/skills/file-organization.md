# File Organization Rules
> DATA_DIR layout, naming rules, where every file type belongs — read before writing files somewhere new.

All files MUST be written to organized locations. NEVER write files to the DATA_DIR root.

## Directory Structure
```
DATA_DIR/
├── projects/{name}/           # All project work
│   ├── research/              # Research findings, market analysis
│   ├── prototypes/            # Working code, MVPs
│   ├── reports/               # Final reports, summaries
│   ├── ideas/                 # Idea generation outputs
│   ├── logs/                  # Session logs, heartbeat logs
│   ├── data/                  # Raw data files (.json, .jsonl)
│   ├── pipeline/              # Pipeline state and instructions
│   ├── business/              # BRD, VC pitch
│   ├── product/               # PRD
│   ├── mvp/                   # MVP code, tests, README
│   ├── TODO.md                # Project task list
│   └── README.md              # Project overview
├── trading_reports/           # Daily trading reports (YYYY-MM-DD.md)
├── chat_logs/                 # Auto-generated daily conversation logs
├── digest/                    # Auto-generated daily digest
├── scratch/                   # code_interpreter temp files
├── tasks.json                 # Task queue (auto)
├── activity.jsonl             # Activity log (auto)
├── MEMORIES.md                # Agent memories (auto)
├── HEARTBEAT.md               # Heartbeat checklist
└── agent_requests.json        # User requests (auto)
```

## Naming Convention
- Use kebab-case: `market-analysis-2026-03-28.md`, NOT `market_analysis_v2.md`
- Add date suffix for daily/periodic files: `{topic}-{YYYY-MM-DD}.{ext}`
- Same topic, same day → overwrite the file, don't create a new one
- No version suffixes (_v2, _v3) — one file per topic
- No session/sprint numbers in filenames — use `logs/` for session tracking

## Rules
1. NEVER write to DATA_DIR root — use the appropriate subdirectory
2. Research → `projects/{name}/research/`
3. Raw data (.jsonl, .json) → `projects/{name}/data/`
4. Session/heartbeat logs → `projects/{name}/logs/`
5. Prototypes → `projects/{name}/prototypes/{prototype-name}/`
6. code_interpreter scratch → `scratch/` (auto-cleaned)
7. Don't create duplicate files — if a file on the same topic exists, update it
8. Clean up temp files when done — use `delete_file` to remove scratch data
9. Use `move_file` and `delete_file` to organize existing files if they're in the wrong place
