# Stage 3: Product Requirements Document (PRD)

## Objective
Build the product requirements for the MVP. Define what we are building technically and how it connects to users.

## Instructions

Read the Market Research and BRD thoroughly. Then create:

1. **Problem Statement**: What specific problem are we solving? For whom?
2. **Solution Overview**: How does the product solve this problem?
3. **User Stories**: As a [persona], I want to [action], so that [outcome]
4. **MVP Scope**: What is IN the MVP and what is explicitly OUT
5. **Technical Requirements**:
   - Architecture (frontend, backend, database, APIs)
   - Key technical decisions and their rationale
   - Third-party services needed
   - Data model (key entities and relationships)
6. **Success Metrics**: How do we know the MVP works? What KPIs matter?
7. **Distribution & Deployment**:
   - How do we get this in front of users?
   - Deployment infrastructure needed
   - Launch checklist

## If Previous Output Exists
Read the existing prd.md. Update based on any changes in the BRD or market research. Tighten the MVP scope — remove anything that isn't essential for the first test.

## Output Format
Write to `product/prd.md` using standard PRD format:
- ## Problem Statement
- ## Solution Overview
- ## User Stories (at least 5 core user stories)
- ## MVP Scope (IN and OUT lists)
- ## Technical Requirements (architecture, data model, APIs)
- ## Success Metrics (specific, measurable KPIs)
- ## Distribution & Deployment (how to reach users)
- ## Timeline Estimate (rough phases)

## Writing Strategy — CRITICAL

**NEVER write the entire document in one project_write_file call.** Build section by section:
1. Write the first section: `project_write_file(mode='write', content='# PRD\n\n## Section 1\n...')`
2. Append each subsequent section: `project_write_file(mode='append', content='\n\n## Next Section\n...')`
3. To fix earlier content, provide the exact text to find and its replacement:
   `project_write_file(mode='edit', old_text='exact text in file', new_text='replacement text')`

## Tools to Use
- project_read_file to read market research and BRD
- web_search for technical feasibility questions
- project_write_file (mode='write' for first section, mode='append' for rest, mode='edit' for corrections)

## Quality Bar
- User stories should be specific and testable
- MVP scope should be ruthlessly minimal — only what's needed to test the core hypothesis
- Technical requirements should be implementable by one developer
- Success metrics should have specific numbers (not "increase engagement")
- The PRD should be detailed enough that a developer could start building from it
