# CVM Data Pipeline

`cvm_data.py` runs quarterly via GitHub Actions, fetches fundamentals for 200 curated
tickers, computes the 10-dimensional CVM scores, and commits `cvm-data.json` to the repo
root. The HTML tool fetches that JSON on page load to populate the watchlist dropdown.

## Files

| File | Purpose |
|------|---------|
| `cvm_data.py` | Main pipeline script |
| `requirements.txt` | Python dependencies (yfinance, requests, pandas) |
| `../tickers.csv` | Curated 200-ticker list (100 US + 50 EU + 50 APAC) |
| `../cvm-data.json` | Generated output (committed back by the workflow) |
| `../.github/workflows/quarterly-update.yml` | CI workflow definition |

## Running locally

To test before pushing to GitHub:

```bash
cd scripts
pip install -r requirements.txt
export FINNHUB_API_KEY=your_key_here   # optional — ESG fallback only
python cvm_data.py
```

Expect ~5–6 minutes for the full 200 tickers (yfinance rate limiting adds a 1.5s sleep
between calls). The script writes `cvm-data.json` and `cvm-data-run.log` to the repo root.

## What gets fetched per ticker

From **yfinance** (no key required):
- Market data: price, market cap, beta, 52-week range
- Valuation: P/E, P/B, PEG, EPS
- Health: current ratio, debt/equity, ROE, margins, payout ratio, dividend yield
- Growth: quarterly earnings/revenue growth
- Tech: R&D / Revenue (from income statement when reported)
- Ownership: insider % and institutional % (parsed from `major_holders` text)
- ESG: from Yahoo Sustainability when available

From **Finnhub free tier** (optional, key required):
- ESG score (fallback when Yahoo Sustainability is empty — common for non-US listings)

From **World Bank API** (no key required):
- Country GDP growth (latest available year)
- Country inflation (latest available year)

## Dimension coverage expectations

For the 200 names, realistic per-dimension live-data coverage:

| Dim | Source | Expected coverage |
|-----|--------|-------------------|
| 1. Macro Environment | World Bank | ~98% (all countries in our universe are covered) |
| 2. Financial Metrics | yfinance | ~95% |
| 3. Financial Engineering | yfinance | ~95% |
| 4. Tech Adoption | yfinance income_stmt | ~75% (non-R&D-reporting firms get semantic-zero or sector baseline) |
| 5. Strategic Transformation | yfinance | ~85% (sometimes quarterly growth fields are absent for non-US) |
| 6. Management Stake | yfinance major_holders | ~70% (US strong, non-US weaker) |
| 7. Ownership Structure | yfinance major_holders | ~70% (same constraint) |
| 8. Culture & Purpose | Sector-weighted baseline | 100% — but always baseline, never API-live |
| 9. Progressive Practices | Yahoo Sustainability + Finnhub ESG | ~65% (US-skewed) |
| 10. Market Dynamics | yfinance | ~95% |

Expected average: ~7.5 / 10 dimensions data-driven per ticker.

## Manual workflow trigger

You can run the workflow manually from the Actions tab on GitHub → "Quarterly CVM Data
Update" → "Run workflow". Useful for testing or for an on-demand refresh between quarters.

## Schedule

Cron expression: `0 6 1 1,4,7,10 *`
→ 06:00 UTC on the 1st of January, April, July, October.

These dates align with the quarterly earnings cycle (most US firms report within 4–6 weeks
of quarter-end, so by Apr 1 / Jul 1 / Oct 1 / Jan 1, most Q4/Q1/Q2/Q3 results are public).

## Limitations to know about

- **yfinance is technically scraping** Yahoo Finance's public endpoints. Yahoo can and
  occasionally does change page layouts, which can break specific fields until yfinance
  ships a patch. Workflow includes per-ticker error isolation so one broken field doesn't
  kill the whole run.
- **Free tier rate limits**: 200 tickers @ 1.5s sleep = ~5 minutes minimum runtime.
  GitHub Actions free tier (2,000 min/month for public repos) easily covers this.
- **Finnhub ESG free tier is US-focused**: international tickers will often fall through
  to sector baseline for D9. This is reflected in the data-driven flag in the JSON.
- **D8 Culture & Purpose has no live free source**. We use sector-weighted baselines
  (tech 70, energy 46, etc.) — this captures the broad signal that public Glassdoor
  distributions show, without scraping.
