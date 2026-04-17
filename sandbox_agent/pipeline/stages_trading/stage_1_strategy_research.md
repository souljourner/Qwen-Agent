# Stage 1: Strategy Research

## Objective
Research the alpha hypothesis, market regime, and prior work for this trading strategy idea.

## Instructions

1. Clarify the alpha hypothesis — what inefficiency or behavioral pattern does this strategy exploit? Use web_search to find academic and industry writing on the signal.
2. Identify the market regime where the strategy is expected to work (trending vs mean-reverting, high vs low volatility, bull vs bear) and where it's expected to fail.
3. Survey prior work — similar published strategies, papers, SSRN preprints, quantitative finance blog posts, industry reports. Note which versions succeeded or failed and why.
4. Define the tradable universe in detail — specific tickers, asset classes, liquidity thresholds, market cap filters.
5. Identify known limitations or failure modes of this class of strategy (e.g., slippage, crowded trade, regime breaks, transaction cost drag).
6. Note data requirements — what historical data is needed (daily/minute/tick), what window, any alternative data.

## If Previous Output Exists
Read `research/strategy-research.md`. Identify gaps or outdated information (new academic work, recent regime changes). Search for newer data and strengthen weak sections rather than starting over.

## Output Format
Write to `research/strategy-research.md` with these sections:
- ## Alpha Hypothesis (what edge does this strategy exploit, and why should it work?)
- ## Market Regime (when does this work, when does it break?)
- ## Prior Work (papers, blogs, industry writing — with citations and brief summaries)
- ## Universe (specific tickers or selection rules, liquidity filters)
- ## Known Failure Modes (crowding, slippage, regime breaks, costs)
- ## Data Requirements (frequency, history window, sources)

## Writing Strategy — CRITICAL

**NEVER write the entire document in one project_write_file call.** Output WILL be truncated and the stage will fail.

Build the document section by section:
1. Research ONE section, then write it:
   `project_write_file(mode='write', content='# Strategy Research\n\n## Alpha Hypothesis\n...')`
2. Research the next section, then append:
   `project_write_file(mode='append', content='\n\n## Market Regime\n...')`
3. To fix an earlier section, use edit mode:
   `project_write_file(mode='edit', old_text='...', new_text='...')`

**Do NOT accumulate all research in memory and write at the end. Write each section as you complete it.**

## Tools to Use
- web_search for academic papers, strategy blogs, regime analysis
- web_url_fetch for specific papers or industry reports
- code_interpreter with llm_call() for processing multiple sources
- project_write_file (append for subsequent sections, edit for corrections)

## Quality Bar
- Alpha Hypothesis names a specific edge (not "we pick winners")
- Prior Work cites at least 3 real sources with links
- Universe is specific enough to enumerate programmatically (tickers, index name, or clear filter rules)
- Failure Modes lists at least 3 concrete ways the strategy can lose money
- Data Requirements are concrete enough for stage 3 to implement
