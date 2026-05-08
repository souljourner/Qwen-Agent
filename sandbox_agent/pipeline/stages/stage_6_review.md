# Stage 6: Review and Update Progress

## Objective
Review all artifacts from every stage. Update the project status. Capture learnings. Optionally improve the pipeline instructions.

## Instructions

1. **Read all project artifacts**:
   - research/market-research.md
   - business/brd.md
   - product/prd.md
   - business/vc-pitch.md
   - mvp/README.md
   - The `## MVP Filesystem Audit` block injected below lists every file in `mvp/` with its size. Use it — do NOT invent file counts from the README. If the audit shows 12 files totalling 60 KB, say so. If it shows only README.md, say THAT. Never hallucinate "Code Files Created: 0" when code files are clearly present in the audit, and never claim code exists when the audit shows it doesn't.

2. **Update project status**:
   - Write a comprehensive status.md with the current state of the project
   - Include what was built, what works, what needs improvement

3. **Capture learnings**:
   - What worked well in this pipeline run?
   - What were the biggest challenges?
   - What would you do differently next time?
   - Were there any surprising findings from the research?

4. **Review instruction quality** (self-improvement):
   - Read each stage instruction file (stages/stage_1_*.md through stage_5_*.md)
   - Compare the instructions with what was actually produced
   - Were the instructions clear enough? Too vague? Missing important aspects?
   - Write specific suggestions for improving each stage's instructions

5. **Do NOT update MEMORIES.md** — learnings are project-specific, not cross-project

## If Previous Output Exists
Read the existing review.md. Add new observations from this run as a new section with the date.

## Output Format
Write to these files:
- `pipeline/review.md` — Full review with learnings and analysis
- `status.md` — Updated project status (overwrite)

Sections for review.md:
- ## Summary (3-5 sentence overview of the project state)
- ## Stage Results (table: stage, status, key output, quality assessment)
- ## Learnings (what worked, what didn't, surprises)
- ## Instruction Improvements (specific suggestions per stage — this enables self-improvement)

## Writing Strategy — CRITICAL

**NEVER write the entire document in one project_write_file call.** Build section by section:
1. Write the first section: `project_write_file(mode='write', content='# Review\n\n## Section 1\n...')`
2. Append each subsequent section: `project_write_file(mode='append', content='\n\n## Next Section\n...')`
3. To fix earlier content, provide the exact text to find and its replacement:
   `project_write_file(mode='edit', old_text='exact text in file', new_text='replacement text')`

## Tools to Use
- project_read_file to read all artifacts
- project_list_files to see what exists
- project_write_file (mode='write' for first section, mode='append' for rest, mode='edit' for corrections)

## Quality Bar
- Summary should be concise and actionable
- Each stage should have a specific quality assessment (not just "good" or "bad")
- Instruction improvements should be specific and actionable (e.g., "Stage 2 should require pricing comparisons with 3 competitors")
- The review should be honest — if something is weak, say so
