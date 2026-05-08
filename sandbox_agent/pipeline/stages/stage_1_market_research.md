# Stage 1: Market Research

## Objective
Thoroughly research the market opportunity for this startup idea.

## Instructions

1. Use web_search to research the market size, growth trends, and TAM/SAM/SOM
2. Identify incumbents and potential competitors — include those in adjacent industries
3. Define the target customer persona in detail:
   - Demographics: age group, income level, ethnicity, occupation
   - Hobbies, lifestyle, life stage
   - Where they spend time online and offline
4. Analyze the value proposition:
   - What pain point are we solving, or what pleasure are we providing?
   - How does this differ from what exists today?
5. Evaluate timing:
   - Are we too early? Too late? What trends support this being the right time?
   - What regulatory, technological, or social changes create the opportunity?
6. Explore adjacent markets:
   - What other markets could this serve?
   - Are there markets even better suited than the primary one?
7. Note anything else interesting about the markets for this idea

## If Previous Output Exists
Read the existing market-research.md file. Identify gaps or outdated information. Search for newer data and update the report with fresh findings. Strengthen weak sections rather than starting over.

## Output Format
Write a comprehensive markdown file to `research/market-research.md` with these sections:
- ## Market Size (TAM, SAM, SOM with sources and methodology)
- ## Competitors (table: name, funding, users, strengths, weaknesses, differentiation)
- ## Target Customers (detailed persona, pain points, current alternatives, willingness to pay)
- ## Timing (why now, supporting trends, risks of being too early/late)
- ## Adjacent Markets (other markets this could serve, comparison)
- ## Key Insights (2-3 non-obvious findings from the research)

## Writing Strategy — CRITICAL

**NEVER write the entire document in one project_write_file call.** Your output WILL be truncated and the stage will fail.

Build the document section by section:
1. Research ONE section (e.g., Market Size), then write it immediately:
   `project_write_file(mode='write', content='# Market Research Report\n\n## Market Size\n...')`
2. Research the NEXT section, then append it:
   `project_write_file(mode='append', content='\n\n## Competitors\n...')`
3. Continue for each section. Each tool call adds to the file without replacing previous content.
4. To fix or update an earlier section, use edit mode with old_text (exact text to find) and new_text (replacement):
   `project_write_file(mode='edit', old_text='The TAM is $10B.', new_text='The TAM is $15B based on 2026 data.')`
   old_text must exactly match existing content in the file. Only the first occurrence is replaced.

**Do NOT accumulate all research in memory and write at the end. Write each section as you complete it.**

## Tools to Use
- web_search for market data and competitor research
- web_url_fetch for specific company pages or reports
- code_interpreter with llm_call() for processing multiple sources
- project_write_file (mode='write' for first section, mode='append' for subsequent sections, mode='edit' for corrections)

## Quality Bar
- Each section should have at least 200 words with specific data
- Competitor table should have at least 5 entries with real, verifiable data
- Market size should cite at least 3 sources
- Customer persona should be specific enough to drive product decisions
- Timing section should reference specific events or trends from the last 12 months
