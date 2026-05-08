# Stage 2: Business Requirements Document

## Objective
Define all the business components needed to build an outstanding business around this idea. Emphasis on being scrappy and getting to quickly test the idea.

## Instructions

Read the Market Research report thoroughly. Then define:

1. **Branding**: Name options, positioning statement, brand voice, visual identity direction
2. **Legal considerations**: Business structure, IP protection, regulatory requirements, potential loopholes, areas needing legal attention
3. **Scalability**: How does this grow from 1 customer to 1000 to 100,000? What breaks at each scale?
4. **Technology stack**: What to build with, build vs buy decisions, infrastructure requirements
5. **Operations support**: What operational processes are needed? Customer support model, SLAs
6. **Finance / Capital strategy**: How much money is needed to start? Revenue model, pricing strategy, break-even analysis. Bootstrap vs fundraise?
7. **Marketing**: Go-to-market strategy, customer acquisition channels, content strategy, partnerships
8. **R&D**: What additional research or development is needed beyond the MVP?
9. **Personnel**: Can one person run this? What are the first hires? (Ideally just one person can do this)

## If Previous Output Exists
Read the existing brd.md. Update sections with new market insights from the latest research. Strengthen the scrappy approach — cut anything that adds unnecessary complexity.

## Output Format
Write to `business/brd.md` with these sections:
- ## Branding
- ## Legal
- ## Scalability
- ## Technology Stack
- ## Operations
- ## Finance
- ## Marketing
- ## R&D
- ## Personnel
- ## Priority: Absolutely Necessary (what must be done to test the idea)
- ## Priority: Nice-to-Have (what can wait)

## Writing Strategy — CRITICAL

**NEVER write the entire document in one project_write_file call.** Build section by section:
1. Write the first section: `project_write_file(mode='write', content='# BRD\n\n## Section 1\n...')`
2. Append each subsequent section: `project_write_file(mode='append', content='\n\n## Next Section\n...')`
3. To fix earlier content, provide the exact text to find and its replacement:
   `project_write_file(mode='edit', old_text='exact text in file', new_text='replacement text')`

## Tools to Use
- project_read_file to read the market research
- web_search for specific business/legal questions
- project_write_file (mode='write' for first section, mode='append' for rest, mode='edit' for corrections)

## Quality Bar
- Clearly separate what is absolutely necessary from nice-to-haves
- Finance section should have specific numbers (costs, revenue projections)
- Legal section should identify at least 2 specific risks or requirements
- Technology stack should justify build vs buy decisions
- The document should read as actionable — a founder should be able to start executing from this
