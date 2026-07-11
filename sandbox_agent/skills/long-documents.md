# Writing & Editing Long Files
> Documents >2000 words: append-mode section-by-section strategy, edit mode, structured patches.

When creating documents longer than ~2000 words (market research, BRDs, PRDs, reports), NEVER try to generate the entire file in one response. Your output will be truncated. Instead, build the document incrementally:

## Strategy: Section-by-section with append mode
1. Write the title and first section: `project_write_file(mode='write', content='# Title\n\n## Section 1\n...')`
2. Research and write each subsequent section: `project_write_file(mode='append', content='\n\n## Section 2\n...')`
3. Continue until complete. Each tool call adds to the file without replacing previous content.

## Editing existing content
- **Small targeted changes**: `project_write_file(mode='edit', old_text='The TAM is $10B.', new_text='The TAM is $15B based on 2026 data.')` — old_text must exactly match existing content in the file; replaces the first occurrence only
- **Multiple edits across files**: `project_apply_patch` — apply a structured patch to one or more files in a single call:
```
*** Begin Patch
*** Update File: research/market-research.md
@@ ## Market Size
-The TAM is estimated at $10B.
+The TAM is estimated at $15B based on updated 2026 data.
+The SAM is $3.2B focusing on the AI agent segment.
*** Add File: research/appendix.md
+# Appendix
+Additional data sources...
*** End Patch
```

## Rules for pipeline stages and long reports
- **Always use append mode** to build documents section by section
- Do your research (web_search, web_url_fetch) BEFORE writing each section
- Write each section immediately after researching it — don't accumulate everything in memory
- If you need to revise an earlier section, use edit mode or apply_patch — don't rewrite the whole file
