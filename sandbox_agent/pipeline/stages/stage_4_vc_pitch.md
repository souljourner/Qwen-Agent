# Stage 4: VC Pitch

## Objective
Create compelling pitch materials for investors — both a short elevator pitch and a longer warm-contact pitch.

## Instructions

Read ALL documents created so far (market research, BRD, PRD). Then write:

1. **Elevator Pitch** (30 seconds, ~100 words):
   - One sentence: what the problem is
   - One sentence: what we do about it
   - One sentence: why now / why us
   - One sentence: the ask (what we need)

2. **Full Pitch** (3-5 minutes, ~800 words):
   - The problem (with data from market research)
   - The solution (with product details from PRD)
   - Market size (TAM/SAM/SOM from research)
   - Business model (from BRD)
   - Traction / validation (what we've learned so far)
   - Team (what makes us qualified)
   - The ask (funding amount, use of proceeds, milestones)
   - Why now (timing from market research)

## If Previous Output Exists
Read the existing vc-pitch.md. Sharpen the narrative with any new data from updated research/BRD/PRD. Make numbers more specific and the story more compelling.

## Output Format
Write to `business/vc-pitch.md`:
- ## Elevator Pitch (short, punchy, memorizable)
- ## Full Pitch (structured narrative with data)
- ## Key Metrics to Highlight (3-5 numbers that matter most)
- ## Anticipated Questions (top 5 questions a VC would ask, with answers)

## Writing Strategy — CRITICAL

**NEVER write the entire document in one project_write_file call.** Build section by section:
1. Write the first section: `project_write_file(mode='write', content='# VC Pitch\n\n## Elevator Pitch\n...')`
2. Append each subsequent section: `project_write_file(mode='append', content='\n\n## Next Section\n...')`
3. To fix earlier content, provide the exact text to find and its replacement:
   `project_write_file(mode='edit', old_text='exact text in file', new_text='replacement text')`

## Tools to Use
- project_read_file to read all previous artifacts
- project_write_file (mode='write' for first section, mode='append' for rest, mode='edit' for corrections)

## Quality Bar
- Elevator pitch should be under 100 words and memorable
- Full pitch should tell a compelling story, not just list facts
- Every claim should be backed by data from the research
- Anticipated questions should show awareness of weaknesses
