# Stage 5: MVP Build

## Objective
Build a working MVP in code — frontend, backend, tests, and documentation.

## Important: Token and Tool Call Management
You have a maximum of 50 tool calls and ~200k tokens of context in this session.
- **Count your tool calls.** Before each tool call, mentally increment your counter.
- **At tool call 45 or if context feels long**: wrap up the current work, save progress notes, and set status to part-completion.
- **Save notes about what you completed and what remains** — the next attempt will read these notes to continue.

## Instructions

Read the PRD thoroughly. Then build:

1. **Backend**:
   - Create the main application file (Python preferred — Flask/FastAPI)
   - Implement core API endpoints from the PRD
   - Add data models and basic database setup
   - Use project_write_file for all code files

2. **Frontend**:
   - Create a simple web interface (HTML/CSS/JS or a framework)
   - Connect to the backend APIs
   - Include all necessary copy as if ready to go live

3. **Tests**:
   - Write unit tests for core functionality
   - Use code_interpreter to run tests and verify they pass
   - Fix any failures before completing

4. **README.md** — This is critical. Include:
   - What this MVP is and what it does
   - How to install and run it
   - Database setup (which DB, creation scripts)
   - Missing tools or tech stack needed
   - Clearly defined next steps
   - Knowledge transfer: if someone else picks this up, they should be able to deploy it

5. **Database scripts** (if needed):
   - SQL or migration scripts to create the schema
   - Seed data if applicable

## If Previous Output Exists (Part-Completion)
Read the notes from previous attempts. Check what files already exist in `mvp/`. Continue from where you left off. Do NOT rewrite files that already exist and work — focus on what's missing.

## Output Format
Save all files under `mvp/` using project_write_file:
- `mvp/README.md` (required)
- `mvp/app.py` or `mvp/main.py` (backend)
- `mvp/templates/` or `mvp/frontend/` (frontend files)
- `mvp/tests/` (test files)
- `mvp/schema.sql` or `mvp/migrations/` (if DB needed)
- `mvp/requirements.txt` (Python dependencies)

## Tools to Use
- project_read_file to read the PRD
- project_write_file to save code files (use single quotes in Python strings to avoid JSON escaping)
- code_interpreter to run tests and verify code works

## Quality Bar
- The backend should have at least 2 working API endpoints
- The frontend should render and be visually presentable
- Tests should exist and at least some should pass
- README should be comprehensive enough for deployment
- Code should be clean, commented where non-obvious, and follow the tech stack from the BRD
