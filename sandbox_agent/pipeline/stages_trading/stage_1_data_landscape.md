# Stage 1: Data Landscape

## Objective
Catalog the **real, reachable data sources** this problem space has to offer, with working fetch code and plaintext samples. **No hypothesis yet.** Later stages choose what to test; this stage establishes what's actually available.

## Why this comes before the hypothesis
Writing a hypothesis before you know what data you have leads to hypotheses you can't test. In the prior pipeline design, this produced "the strategy" that couldn't be implemented, so the agent fabricated signals. This stage forces us to start from reality: what can we actually read, extract, and reason over?

## Instructions

0. **Setup + primary price source.** One-time: `uv pip install pandas requests numpy` via `exec(project=...)`. US equity/ETF daily prices come from the LOCAL data store (auram/QuestDB, ~3000 US tickers + leveraged ETFs, split/dividend-adjusted, refreshed daily): `from sandbox_agent.trading_data import get_daily, get_universe, health`. Catalog it FIRST in your Data Sources section — verify with `health()` and a sample `get_daily("AAPL", start="2020-01-01")`. Symbols it lacks raise `DataUnavailable` — for a US equity/ETF you can add it on demand: `request_backfill(symbol)` (same import) pulls it from EODHD into the store, blocks until the bars are queryable, and it stays refreshed; then retry `get_daily`. Coverage after backfill IS your tradable-universe constraint. yfinance is banned for US equities/ETFs; use it only for what the store lacks (futures, COT, FX, ^VIX-style indices).
1. For each candidate data source, verify it's reachable and parseable. A source you can't fetch or extract plaintext from is not a source.
2. Cover **at least 2** sources. Good starting points for a news/filings-driven strategy:
   - SEC EDGAR (10-K, 10-Q, 8-K, DEF 14A, PREM14A, 13D/G, S-1)
   - Press releases (PRNewswire, BusinessWire)
   - News aggregators with retrievable text (avoid paywalled sources you can't actually read)
   - RSS feeds with full article bodies
   - Analyst report archives, earnings call transcripts
3. For each source:
   - **Show the fetch call** as a runnable code block (how to pull one document end-to-end).
   - **Paste a 500-char plaintext sample** — AFTER running BeautifulSoup / pdfplumber / whatever converts HTML/XBRL/PDF/XML to readable text. The sample must be clean prose, not raw markup.
   - **State the volume** — approximately how many documents per ticker per year. This bounds what's feasible downstream.
   - **Name the alpha rationale** — why might an LLM find extractable signal here? Be specific (e.g., "8-K Item 1.01 tags material contracts — the LLM can classify customer-win vs. customer-loss language"). This is a **rationale**, not a hypothesis.

## Never feed HTML / XBRL / PDF / XML directly to the LLM
Raw markup wastes thousands of tokens and degrades output quality. Always extract plaintext FIRST, in the fetcher:
- **HTML** → `BeautifulSoup(html, "lxml").get_text(separator="\n")`; strip `<script>`, `<style>`, `<nav>`, `<footer>` before extraction.
- **SEC filings** → `sec-edgar-downloader` or parse EDGAR's text wrapper; strip `<XBRL>...</XBRL>` blocks before BeautifulSoup on the HTML body.
- **PDFs** → `pdfplumber` / `pypdf` for text-PDFs; `pytesseract` OCR for scanned ones.
- **XML / RSS** → `xml.etree` / `feedparser`; extract `<title>`, `<description>`, `<content>`.

## Output
Write to `research/data-landscape.md` with these sections:
- `## Data Sources` — one subsection per source: name, fetch method, volume estimate, access constraints (rate limits, auth).
- `## Sample Extractions` — one 500-char plaintext sample per source. Must be post-extraction, readable prose.
- `## Alpha Rationale` — for each source, one paragraph on what signal an LLM might extract. No backtest numbers, no hypothesis — just **why this might have alpha**.

## Writing Strategy
Write section-by-section with `project_write_file(mode='append')`. Do not buffer the whole doc in memory.

## Quality Bar
- ≥ 2 sources, each with a working fetch code block and a clean plaintext sample.
- Samples are not HTML/XML/PDF bytes — they are readable text.
- Each source has a concrete alpha rationale (no "might be useful").
- Volume / constraints are specific enough that stage 2 can pick a scope.
