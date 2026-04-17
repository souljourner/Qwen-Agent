# Stage 3: Data Pipeline

## Objective
Build a reproducible data pipeline that fetches historical price data, computes features, and validates data quality. The output of this stage is code + docs, not raw data dumped into chat.

## Instructions

1. Read `strategy/spec.md` to determine exactly what data you need (tickers, frequency, history window, features).
2. Install dependencies. Use the `exec` tool, not code_interpreter:
   `exec(command="pip install yfinance pandas pandas-ta", project="{PROJECT_NAME}")`
3. **Write a pinned `data/requirements.txt`** listing exact versions. Example:
   ```
   yfinance==0.2.40
   pandas==2.2.2
   pandas-ta==0.3.14b
   ```
   This is REQUIRED. pip installs don't survive container restarts (only `/app/data` is bind-mounted), so stage 4 must reinstall from this file.
4. Write `data/pipeline.py` — a Python script that:
   - Downloads the required historical data via yfinance (or similar)
   - Computes the features defined in the strategy spec (moving averages, RSI, returns, etc.)
   - Saves the processed data to CSV or Parquet under `projects/{name}/data/` (NOT in chat)
   - Runs data quality checks: no NaN in key columns, date index is continuous within expected trading days, volume is positive
   - Logs a short summary to stdout (row count, date range, NaN counts) — nothing else
5. Run the pipeline once to make sure it works:
   `exec(command="python data/pipeline.py", project="{PROJECT_NAME}")`
6. Write `data/README.md` with these sections:
   - ## Data Sources (what we download, from where, frequency)
   - ## Features (what columns are computed and how)
   - ## Quality Checks (what's validated, what to do when a check fails)

## If Previous Output Exists
Read the existing `data/pipeline.py` and `data/README.md`. Fix bugs, tighten quality checks, update features if the spec changed. Don't rewrite from scratch.

## Output Files
- `data/pipeline.py` — the data fetcher
- `data/README.md` — sections: Data Sources, Features, Quality Checks
- `data/requirements.txt` — pinned deps for stage 4 to reinstall
- `data/processed/*.csv` or `.parquet` — processed data files (these are outputs of running the pipeline, not required artifacts)

## CRITICAL RULES
- **Never** print raw dataframes or data samples larger than 5 rows into chat — write to files and report row count + date range only.
- Use `exec` for package installs and running the pipeline; use `code_interpreter` only for quick sanity checks on small data slices.
- If yfinance rate-limits you, add `time.sleep(1)` between ticker fetches. Do not retry in a hot loop.

## Quality Bar
- pipeline.py runs end-to-end without errors when invoked via exec
- data/requirements.txt lists every non-stdlib import in pipeline.py with a pinned version
- At least one data quality check that would catch a regression (NaN rate, date gaps, or volume of zero)
- README explains enough for someone else to rerun the pipeline
