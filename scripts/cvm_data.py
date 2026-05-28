"""
cvm_data.py — Concentric Value Model data pipeline.

Runs quarterly via GitHub Actions. For each of 200 curated tickers:
  1. Fetches fundamentals via yfinance (primary source)
  2. Fetches ESG score via Finnhub free tier (D9 fallback)
  3. Fetches country macro data via World Bank API (D1)
  4. Computes the 10-dimensional CVM scores (mirrors the JS logic)
  5. Writes cvm-data.json

Design principles:
  - Robust to data gaps — missing fields fall back to sector baseline gracefully
  - Per-ticker error isolation — one bad ticker doesn't break the run
  - Idempotent — same inputs produce same outputs
  - Free-tier friendly — respects rate limits, uses sleep between calls

Author: Felix Langer
"""

from __future__ import annotations
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yfinance as yf

# ============================================================================
# CONFIG
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
TICKERS_CSV = REPO_ROOT / "tickers.csv"
OUTPUT_JSON = REPO_ROOT / "cvm-data.json"
LOG_FILE = REPO_ROOT / "cvm-data-run.log"

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "").strip()
FINNHUB_BASE = "https://finnhub.io/api/v1"

# Sleep between yfinance calls to avoid rate limiting.
# yfinance is technically Yahoo Finance scraping under the hood; being polite
# keeps the pipeline reliable.
YFINANCE_SLEEP_SEC = 1.5
FINNHUB_SLEEP_SEC = 1.1  # 60 calls/min = 1 per second; 1.1 for safety margin

# World Bank indicator codes — these are stable
WB_INDICATORS = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",        # GDP growth, annual %
    "inflation": "FP.CPI.TOTL.ZG",             # Inflation, consumer prices, annual %
}

# Country-code mapping: our 2-letter code → World Bank 3-letter ISO code
WB_COUNTRY_MAP = {
    "US": "USA", "DE": "DEU", "NL": "NLD", "FR": "FRA", "DK": "DNK",
    "CH": "CHE", "IT": "ITA", "ES": "ESP", "BE": "BEL", "GB": "GBR",
    "JP": "JPN", "TW": "TWN", "KR": "KOR", "CN": "CHN", "HK": "HKG",
    "IN": "IND", "SG": "SGP",
}

# Sector-weighted culture baseline for D8 (the "smart sector baseline" choice).
# Anchored to public Glassdoor distributions: tech consistently highest,
# energy/utilities/financials lowest. This is a transparent fallback for the
# ~70%-of-signal you'd get from a live Glassdoor scrape, without scraping.
CULTURE_SECTOR_BASELINES = {
    "TECH":     70,
    "COMM":     64,
    "HEALTH":   58,
    "CONS_C":   60,
    "CONS_S":   58,
    "REAL":     54,
    "MAT":      52,
    "IND":      52,
    "FIN":      50,
    "ENERGY":   46,
    "UTIL":     46,
}

# Logging — captured to stdout AND file for GitHub Actions visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger("cvm")


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass
class TickerData:
    """Raw fundamentals collected from external sources for a single ticker."""
    ticker: str
    name: str
    country: str
    region: str
    sector: str

    # Market data
    price: float | None = None
    market_cap: float | None = None
    cap: str | None = None  # 'M' mega, 'L' large, 'I' mid, 'S' small, 'V' micro
    beta: float | None = None
    fifty_two_week_low: float | None = None
    fifty_two_week_high: float | None = None
    price_in_range: float | None = None  # 0..1

    # Valuation
    pe: float | None = None
    pb: float | None = None
    peg: float | None = None
    eps: float | None = None

    # Financial health
    current_ratio: float | None = None
    debt_equity: float | None = None
    roe: float | None = None
    operating_margin: float | None = None
    profit_margin: float | None = None
    payout_ratio: float | None = None
    div_yield: float | None = None
    five_year_avg_div_yield: float | None = None

    # Growth
    quarterly_earnings_growth: float | None = None
    quarterly_revenue_growth: float | None = None

    # Tech / R&D
    rd_intensity: float | None = None

    # Ownership
    insider_ownership_pct: float | None = None
    institutional_ownership_pct: float | None = None

    # Macro (filled from country-level data)
    macro_gdp_growth: float | None = None
    macro_inflation: float | None = None
    macro_rate: float | None = None  # not currently fetched; placeholder

    # ESG
    esg_risk: float | None = None  # Sustainalytics-style: lower = better

    # Diagnostic
    fetch_errors: list[str] = field(default_factory=list)
    fetch_warnings: list[str] = field(default_factory=list)


# ============================================================================
# FETCHERS
# ============================================================================

def fetch_macro_data() -> dict[str, dict[str, float]]:
    """Fetch latest GDP growth + inflation per country from World Bank API.

    Returns dict mapping our 2-letter code → {"gdp_growth": x, "inflation": y}
    """
    log.info("Fetching macro data from World Bank API...")
    macro: dict[str, dict[str, float]] = {}
    for cc2, cc3 in WB_COUNTRY_MAP.items():
        macro[cc2] = {}
        for label, indicator in WB_INDICATORS.items():
            try:
                # World Bank API returns last 5 years; we want the most recent non-null value
                url = f"https://api.worldbank.org/v2/country/{cc3}/indicator/{indicator}"
                resp = requests.get(
                    url,
                    params={"format": "json", "per_page": "5", "MRV": "5"},
                    timeout=15,
                )
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list) or len(payload) < 2:
                    continue
                records = payload[1] or []
                for rec in records:  # in date-desc order
                    val = rec.get("value")
                    if val is not None:
                        macro[cc2][label] = float(val)
                        break
            except Exception as e:
                log.warning(f"WB fetch failed for {cc3}/{label}: {e}")
                continue
        time.sleep(0.3)  # be polite to the World Bank
    log.info(f"Macro data fetched for {len(macro)} countries")
    return macro


def cap_tier_from_market_cap(mc: float | None) -> str | None:
    """Map market cap (USD) to cap tier. Same thresholds as the JS tool."""
    if mc is None or mc <= 0:
        return None
    if mc >= 200_000_000_000: return "M"   # Mega (>= $200B)
    if mc >= 10_000_000_000:  return "L"   # Large ($10–200B)
    if mc >= 2_000_000_000:   return "I"   # Mid ($2–10B)
    if mc >= 300_000_000:     return "S"   # Small ($300M–$2B)
    return "V"                              # Micro (< $300M)


def parse_insider_pct_from_major_holders(holders_df: Any) -> float | None:
    """Parse insider ownership % from yfinance's major_holders DataFrame.

    Yahoo's major_holders format has changed over time. Across observed versions:
      - Old: 2-column DataFrame with [percent_text, label_text] rows
      - New: 2-column DataFrame indexed by label, with 'Value' column containing decimal
    We try BOTH formats defensively.
    """
    if holders_df is None:
        return None
    try:
        # NEW yfinance format: indexed by metric name, 'Value' column with decimal
        # e.g. holders_df.loc['insidersPercentHeld'] = 0.0014
        if hasattr(holders_df, "index") and len(holders_df.columns) >= 1:
            # Try index-based lookup first
            idx_str = [str(i).lower() for i in holders_df.index]
            for i, idx_name in enumerate(idx_str):
                if "insider" in idx_name and "percent" in idx_name:
                    val = holders_df.iloc[i, 0]
                    if val is not None and not (isinstance(val, float) and val != val):
                        # New format returns decimal (0.0014 = 0.14%)
                        return float(val) * 100 if abs(float(val)) <= 1 else float(val)

        # OLD format: row-based with text values
        for _, row in holders_df.iterrows():
            cells = [str(c).lower() for c in row]
            line = " | ".join(cells)
            if "insider" in line and "institut" not in line.split("insider")[0]:
                for cell in row:
                    s = str(cell)
                    m = re.search(r"([\d.,]+)\s*%", s)
                    if m:
                        return float(m.group(1).replace(",", "."))
    except Exception:
        return None
    return None


def parse_institutional_pct_from_major_holders(holders_df: Any) -> float | None:
    """Parse institutional ownership % from yfinance's major_holders DataFrame.

    Same dual-format strategy as the insider parser above.
    """
    if holders_df is None:
        return None
    try:
        # NEW format
        if hasattr(holders_df, "index") and len(holders_df.columns) >= 1:
            idx_str = [str(i).lower() for i in holders_df.index]
            for i, idx_name in enumerate(idx_str):
                if "institut" in idx_name and "percent" in idx_name:
                    val = holders_df.iloc[i, 0]
                    if val is not None and not (isinstance(val, float) and val != val):
                        return float(val) * 100 if abs(float(val)) <= 1 else float(val)

        # OLD format
        for _, row in holders_df.iterrows():
            cells = [str(c).lower() for c in row]
            line = " | ".join(cells)
            if "institut" in line and "float" not in line:
                for cell in row:
                    s = str(cell)
                    m = re.search(r"([\d.,]+)\s*%", s)
                    if m:
                        return float(m.group(1).replace(",", "."))
    except Exception:
        return None
    return None


def get_ownership_from_info(info: dict) -> tuple[float | None, float | None]:
    """Fallback: pull heldPercentInsiders and heldPercentInstitutions from info dict.

    These fields are part of Yahoo's public quote-summary endpoint and often arrive
    even when the structured `major_holders` DataFrame is empty. Both are decimal
    in yfinance's output (e.g. 0.0014 = 0.14%).
    """
    insider = _safe_float(info.get("heldPercentInsiders"))
    inst = _safe_float(info.get("heldPercentInstitutions"))
    # Both come back as decimal — convert to percent
    if insider is not None and abs(insider) <= 1:
        insider *= 100
    if inst is not None and abs(inst) <= 1:
        inst *= 100
    return insider, inst


def get_ownership_from_institutional_holders(yf_ticker: Any) -> float | None:
    """Last-resort fallback: sum the percent_held column across top institutional holders.

    yfinance's `institutional_holders` returns a structured DataFrame of the top
    ~10 institutional holders with a `pctHeld` column. Summing gives a lower bound
    on total institutional ownership (truncated at top-10, so will read lower than
    full institutional %). Better than None when the structured-holders path failed.
    """
    try:
        ih = yf_ticker.institutional_holders
        if ih is None or len(ih) == 0:
            return None
        # Column name varies: 'pctHeld', '% Out', or '%Held'
        for col in ["pctHeld", "% Out", "%Held"]:
            if col in ih.columns:
                total = ih[col].sum()
                if hasattr(total, "item"):
                    total = total.item()
                if total > 0:
                    return float(total) * 100 if abs(float(total)) <= 1 else float(total)
    except Exception:
        return None
    return None


def fetch_yfinance(td: TickerData) -> None:
    """Fetch all yfinance-sourced fields into the TickerData object.

    Robust: catches exceptions, logs warnings, keeps fields as None on failure.
    """
    try:
        yf_ticker = yf.Ticker(td.ticker)
        info = yf_ticker.info or {}
    except Exception as e:
        td.fetch_errors.append(f"yfinance.info: {e}")
        return

    # Market data
    td.price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    td.market_cap = _safe_float(info.get("marketCap"))
    td.cap = cap_tier_from_market_cap(td.market_cap)
    td.beta = _safe_float(info.get("beta"))
    td.fifty_two_week_low = _safe_float(info.get("fiftyTwoWeekLow"))
    td.fifty_two_week_high = _safe_float(info.get("fiftyTwoWeekHigh"))
    if td.price and td.fifty_two_week_low and td.fifty_two_week_high and td.fifty_two_week_high > td.fifty_two_week_low:
        td.price_in_range = (td.price - td.fifty_two_week_low) / (td.fifty_two_week_high - td.fifty_two_week_low)

    # Valuation
    td.pe = _safe_float(info.get("trailingPE"))
    td.pb = _safe_float(info.get("priceToBook"))
    td.peg = _safe_float(info.get("trailingPegRatio") or info.get("pegRatio"))
    td.eps = _safe_float(info.get("trailingEps"))

    # Financial health
    td.current_ratio = _safe_float(info.get("currentRatio"))
    td.debt_equity = _safe_float(info.get("debtToEquity"))
    if td.debt_equity and td.debt_equity > 5:
        # Yahoo returns D/E as percentage for some tickers (e.g. 124.5 meaning 1.245)
        td.debt_equity = td.debt_equity / 100
    td.roe = _safe_float(info.get("returnOnEquity"))
    if td.roe is not None and abs(td.roe) <= 5:
        # Yahoo returns as decimal (0.27 = 27%); normalise to percent
        td.roe = td.roe * 100
    td.operating_margin = _safe_float(info.get("operatingMargins"))
    if td.operating_margin is not None and abs(td.operating_margin) <= 5:
        td.operating_margin = td.operating_margin * 100
    td.profit_margin = _safe_float(info.get("profitMargins"))
    if td.profit_margin is not None and abs(td.profit_margin) <= 5:
        td.profit_margin = td.profit_margin * 100
    td.payout_ratio = _safe_float(info.get("payoutRatio"))
    if td.payout_ratio is not None and abs(td.payout_ratio) <= 5:
        td.payout_ratio = td.payout_ratio * 100
    td.five_year_avg_div_yield = _safe_float(info.get("fiveYearAvgDividendYield"))
    # yfinance is INCONSISTENT on dividendYield — sometimes returns percent directly
    # (e.g. JPM 1.96 = 1.96%), sometimes decimal (e.g. AAPL 0.005 → reported as 0.005).
    # Use fiveYearAvgDividendYield (always reported as percent) as the sanity anchor.
    # If they're same order of magnitude (ratio 0.3–3.0), trust the raw value.
    # Otherwise, the raw value is decimal-encoded and needs ×100.
    # Final safety: cap at 20% (no real stock has higher; anything above is noise).
    td.div_yield = _safe_float(info.get("dividendYield"))
    if td.div_yield is not None:
        if td.five_year_avg_div_yield is not None and td.five_year_avg_div_yield > 0:
            ratio = td.div_yield / td.five_year_avg_div_yield
            if ratio > 5:
                # Raw value is wildly inflated — was already percent that we'd be doubling.
                # Trust five_year_avg as the better signal: scale raw value down.
                td.div_yield = td.div_yield / 100
            elif ratio < 0.2 and td.div_yield < 1:
                # Raw value is a decimal — convert to percent.
                td.div_yield = td.div_yield * 100
            # else: same order of magnitude — already in percent.
        else:
            # No anchor available. Use absolute cap: if >20%, assume it was percent
            # that got doubled by the old buggy logic — don't multiply.
            if td.div_yield > 20:
                td.div_yield = td.div_yield / 100  # was incorrectly doubled
            elif td.div_yield < 0.1:
                td.div_yield = td.div_yield * 100  # was decimal
        # Final clamp — defensive
        if td.div_yield > 20:
            td.div_yield = None  # nonsense data; better to fall back to baseline

    # Growth
    td.quarterly_earnings_growth = _safe_float(info.get("earningsQuarterlyGrowth"))
    if td.quarterly_earnings_growth is not None and abs(td.quarterly_earnings_growth) <= 5:
        td.quarterly_earnings_growth = td.quarterly_earnings_growth * 100
    td.quarterly_revenue_growth = _safe_float(info.get("revenueQuarterlyGrowth") or info.get("revenueGrowth"))
    if td.quarterly_revenue_growth is not None and abs(td.quarterly_revenue_growth) <= 5:
        td.quarterly_revenue_growth = td.quarterly_revenue_growth * 100

    # R&D intensity — pull from income statement
    try:
        income = yf_ticker.income_stmt
        if income is not None and not income.empty:
            # Look for "Research And Development" row, get most recent column
            for row_label in income.index:
                if "research" in str(row_label).lower() and "develop" in str(row_label).lower():
                    rd_value = income.loc[row_label].iloc[0]
                    revenue_row = None
                    for r in income.index:
                        if str(r).lower() in ("total revenue", "totalrevenue"):
                            revenue_row = r
                            break
                    # Defensive: NaN is truthy in Python, so explicit NaN check is required
                    # before accepting the value. This is what crashed ULVR.L and ITC.NS in
                    # the original run — yfinance returned NaN R&D for non-tech non-US firms,
                    # NaN propagated through (rd_intensity - anchor)/anchor*10, and clamp(int(NaN)) crashed.
                    rd_clean = _safe_float(rd_value)
                    if revenue_row is not None and rd_clean is not None:
                        revenue = _safe_float(income.loc[revenue_row].iloc[0])
                        if revenue is not None and revenue > 0:
                            td.rd_intensity = rd_clean / revenue * 100
                    break
    except Exception as e:
        td.fetch_warnings.append(f"income_stmt for R&D: {e}")

    # Ownership — try 3 paths in order of preference
    # 1. Structured major_holders DataFrame (works in most yfinance versions)
    # 2. heldPercentInsiders / heldPercentInstitutions from info dict (works when quoteSummary partial fallback returned)
    # 3. Sum of pctHeld across institutional_holders (lower-bound; covers top-10 holders only)
    try:
        major = yf_ticker.major_holders
        td.insider_ownership_pct = parse_insider_pct_from_major_holders(major)
        td.institutional_ownership_pct = parse_institutional_pct_from_major_holders(major)
    except Exception as e:
        td.fetch_warnings.append(f"major_holders: {e}")

    # Path 2: info-dict fallback (heldPercentInsiders, heldPercentInstitutions)
    if td.insider_ownership_pct is None or td.institutional_ownership_pct is None:
        try:
            ins_info, inst_info = get_ownership_from_info(info)
            if td.insider_ownership_pct is None and ins_info is not None:
                td.insider_ownership_pct = ins_info
            if td.institutional_ownership_pct is None and inst_info is not None:
                td.institutional_ownership_pct = inst_info
        except Exception as e:
            td.fetch_warnings.append(f"info ownership: {e}")

    # Path 3: institutional_holders pctHeld sum (last resort for institutional%)
    if td.institutional_ownership_pct is None:
        try:
            inst_fallback = get_ownership_from_institutional_holders(yf_ticker)
            if inst_fallback is not None:
                td.institutional_ownership_pct = inst_fallback
                td.fetch_warnings.append("institutional_pct from top-10 holders (lower bound)")
        except Exception as e:
            td.fetch_warnings.append(f"institutional_holders: {e}")

    # ESG from Yahoo Sustainability — note: deprecated/empty for most tickers as of 2026.
    # Kept as best-effort; falls through to Finnhub then sector baseline.
    try:
        sust = yf_ticker.sustainability
        if sust is not None and not sust.empty:
            for label in ["totalEsg", "esgScore"]:
                if label in sust.index:
                    val = sust.loc[label].iloc[0]
                    if val is not None and not (isinstance(val, float) and val != val):
                        td.esg_risk = float(val)
                        break
    except Exception as e:
        td.fetch_warnings.append(f"sustainability: {e}")


def fetch_finnhub_esg(td: TickerData) -> None:
    """ESG fallback via Finnhub free tier.

    Note: Finnhub's free tier ESG coverage is US-focused. For non-US tickers
    this will often return empty, which is fine — we then fall through to
    the sector baseline in the scoring step.
    """
    if not FINNHUB_API_KEY:
        return
    if td.esg_risk is not None:
        return  # already have it from Yahoo
    try:
        # Finnhub uses bare ticker (not exchange suffix)
        base_ticker = td.ticker.split(".")[0]
        resp = requests.get(
            f"{FINNHUB_BASE}/stock/esg",
            params={"symbol": base_ticker, "token": FINNHUB_API_KEY},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json() or {}
            # Finnhub returns total ESG score (higher = better). Convert to
            # Sustainalytics-equivalent risk scale (lower = better).
            score = data.get("totalEsgScore")
            if score is not None:
                # Linear inversion: ~80 score → ~8 risk; ~40 score → ~24 risk
                td.esg_risk = max(0, 40 - 0.4 * float(score))
    except Exception as e:
        td.fetch_warnings.append(f"finnhub esg: {e}")
    time.sleep(FINNHUB_SLEEP_SEC)


def _safe_float(v: Any) -> float | None:
    """Convert to float, returning None for non-numeric / NaN / inf."""
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN check
            return None
        if f in (float("inf"), float("-inf")):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ============================================================================
# SCORING — mirrors the JavaScript computeDimensions logic
# ============================================================================

def clamp(v: float, lo: float = 0, hi: float = 100) -> int:
    """Clamp a numeric value to [lo, hi] and return as int.

    Defensive: NaN inputs return 50 (sector-neutral midpoint) rather than crashing
    on `int(NaN)`. This is defence-in-depth — every upstream code path SHOULD use
    _safe_float to filter NaN, but if any future field slips through, the
    per-ticker scoring continues with a neutral value rather than dropping the
    entire ticker. The ULVR.L / ITC.NS failures in build 2026-05-11.r traced to
    a NaN R&D value that bypassed _safe_float; this guard prevents future
    occurrences of the same bug pattern.
    """
    import math as _math
    if v is None:
        return 50
    try:
        if isinstance(v, float) and _math.isnan(v):
            return 50
        return int(max(lo, min(hi, round(v))))
    except (TypeError, ValueError):
        return 50


# Sector baseline scores per dimension (mirrors LAYERS/SECTOR_BASELINES from JS)
SECTOR_BASELINES = {
    "TECH":     {"financial_metrics": 58, "financial_engineering": 60, "tech_adoption": 80, "strategic_transformation": 68, "management_stake": 56, "ownership_structure": 60, "progressive_practices": 60, "market_dynamics": 56},
    "HEALTH":   {"financial_metrics": 56, "financial_engineering": 62, "tech_adoption": 65, "strategic_transformation": 60, "management_stake": 50, "ownership_structure": 62, "progressive_practices": 62, "market_dynamics": 50},
    "FIN":      {"financial_metrics": 58, "financial_engineering": 62, "tech_adoption": 50, "strategic_transformation": 52, "management_stake": 52, "ownership_structure": 68, "progressive_practices": 50, "market_dynamics": 54},
    "CONS_C":   {"financial_metrics": 54, "financial_engineering": 58, "tech_adoption": 50, "strategic_transformation": 56, "management_stake": 56, "ownership_structure": 60, "progressive_practices": 56, "market_dynamics": 60},
    "CONS_S":   {"financial_metrics": 58, "financial_engineering": 62, "tech_adoption": 45, "strategic_transformation": 56, "management_stake": 50, "ownership_structure": 62, "progressive_practices": 58, "market_dynamics": 52},
    "ENERGY":   {"financial_metrics": 58, "financial_engineering": 56, "tech_adoption": 40, "strategic_transformation": 48, "management_stake": 50, "ownership_structure": 60, "progressive_practices": 38, "market_dynamics": 62},
    "MAT":      {"financial_metrics": 56, "financial_engineering": 56, "tech_adoption": 45, "strategic_transformation": 54, "management_stake": 52, "ownership_structure": 58, "progressive_practices": 50, "market_dynamics": 60},
    "IND":      {"financial_metrics": 56, "financial_engineering": 58, "tech_adoption": 52, "strategic_transformation": 58, "management_stake": 54, "ownership_structure": 60, "progressive_practices": 56, "market_dynamics": 56},
    "UTIL":     {"financial_metrics": 58, "financial_engineering": 56, "tech_adoption": 40, "strategic_transformation": 46, "management_stake": 46, "ownership_structure": 64, "progressive_practices": 58, "market_dynamics": 44},
    "REAL":     {"financial_metrics": 54, "financial_engineering": 54, "tech_adoption": 42, "strategic_transformation": 50, "management_stake": 56, "ownership_structure": 62, "progressive_practices": 52, "market_dynamics": 52},
    "COMM":     {"financial_metrics": 56, "financial_engineering": 58, "tech_adoption": 60, "strategic_transformation": 56, "management_stake": 54, "ownership_structure": 60, "progressive_practices": 56, "market_dynamics": 56},
}


def compute_dimensions(td: TickerData, macro: dict[str, dict[str, float]]) -> tuple[dict[str, int], dict[str, bool]]:
    """Compute the 10 CVM dimension scores for a ticker.

    Returns:
      dims: dict of dimension_id → score (0–100)
      data_driven: dict of dimension_id → bool (True if score uses live data)
    """
    sec = SECTOR_BASELINES.get(td.sector, SECTOR_BASELINES["IND"])
    dims = {
        "macro_environment":       50,  # filled below from country macro data
        "financial_metrics":       sec["financial_metrics"],
        "financial_engineering":   sec["financial_engineering"],
        "tech_adoption":           sec["tech_adoption"],
        "strategic_transformation": sec["strategic_transformation"],
        "management_stake":        sec["management_stake"],
        "ownership_structure":     sec["ownership_structure"],
        "culture_purpose":         CULTURE_SECTOR_BASELINES.get(td.sector, 52),
        "progressive_practices":   sec["progressive_practices"],
        "market_dynamics":         sec["market_dynamics"],
    }
    data_driven: dict[str, bool] = {}

    # D1: macro_environment — derived from country GDP growth + inflation
    country_macro = macro.get(td.country, {})
    if country_macro.get("gdp_growth") is not None or country_macro.get("inflation") is not None:
        gdp = country_macro.get("gdp_growth") or 2.0   # global average fallback
        infl = country_macro.get("inflation") or 3.0
        # Higher GDP, lower inflation → higher macro score
        macro_score = 50 + (gdp - 2.5) * 4 - (infl - 3) * 2
        dims["macro_environment"] = clamp(macro_score)
        data_driven["macro_environment"] = True

    # D2: financial_metrics — P/E, P/B, current ratio, dividend yield, EPS
    fm_delta = 0
    fm_signals = 0
    if td.pe is not None and td.pe > 0:
        # Lower P/E vs sector → higher score. Use 18 as the universal anchor.
        pe_anchor = {"TECH": 28, "FIN": 12, "UTIL": 16, "ENERGY": 12}.get(td.sector, 18)
        fm_delta += (pe_anchor - td.pe) / pe_anchor * 8
        fm_signals += 1
    if td.pb is not None and td.pb > 0:
        pb_anchor = {"TECH": 6, "FIN": 1.2, "UTIL": 1.6}.get(td.sector, 3)
        fm_delta += (pb_anchor - td.pb) / pb_anchor * 5
        fm_signals += 1
    if td.current_ratio is not None:
        fm_delta += min(8, (td.current_ratio - 1.2) * 4)
        fm_signals += 1
    if td.div_yield is not None and td.div_yield >= 0:
        fm_delta += min(5, td.div_yield * 0.8)
        fm_signals += 1
    if fm_signals >= 2:
        dims["financial_metrics"] = clamp(sec["financial_metrics"] + fm_delta)
        data_driven["financial_metrics"] = True

    # D3: financial_engineering — ROE, D/E, payout ratio, operating margin
    fe_delta = 0
    fe_signals = 0
    if td.roe is not None:
        roe_anchor = {"TECH": 18, "FIN": 11, "UTIL": 9}.get(td.sector, 13)
        fe_delta += (td.roe - roe_anchor) / max(5, roe_anchor) * 8
        fe_signals += 1
    if td.debt_equity is not None:
        # Lower D/E = stronger. Anchor ~1.0; >2.5 is a red flag.
        de_anchor = {"FIN": 2.5, "UTIL": 1.5, "REAL": 1.5}.get(td.sector, 0.8)
        fe_delta += max(-8, (de_anchor - td.debt_equity) * 4)
        fe_signals += 1
    if td.operating_margin is not None:
        om_anchor = {"TECH": 25, "FIN": 30, "ENERGY": 10, "CONS_S": 8}.get(td.sector, 15)
        fe_delta += (td.operating_margin - om_anchor) / max(5, om_anchor) * 5
        fe_signals += 1
    if fe_signals >= 2:
        dims["financial_engineering"] = clamp(sec["financial_engineering"] + fe_delta)
        data_driven["financial_engineering"] = True

    # D4: tech_adoption — R&D intensity
    if td.rd_intensity is not None:
        rd_anchor = {"TECH": 15, "HEALTH": 12, "FIN": 1, "UTIL": 1, "ENERGY": 1}.get(td.sector, 4)
        delta = (td.rd_intensity - rd_anchor) / max(2, rd_anchor) * 10
        dims["tech_adoption"] = clamp(sec["tech_adoption"] + delta)
        data_driven["tech_adoption"] = True
    elif td.sector in ("FIN", "UTIL", "ENERGY", "REAL"):
        # Semantic-zero: these sectors structurally have negligible R&D
        # Accept sector baseline as the data-driven answer
        data_driven["tech_adoption"] = True

    # D5: strategic_transformation — quarterly growth signals, or ROE + D/E proxy
    st_delta = 0
    st_signals = 0
    if td.quarterly_earnings_growth is not None:
        st_delta += min(15, td.quarterly_earnings_growth / 4)
        st_signals += 1
    if td.quarterly_revenue_growth is not None:
        st_delta += min(10, td.quarterly_revenue_growth / 5)
        st_signals += 1
    if st_signals >= 1:
        dims["strategic_transformation"] = clamp(sec["strategic_transformation"] + st_delta)
        data_driven["strategic_transformation"] = True
    elif td.roe is not None and td.debt_equity is not None:
        # Fallback: solid ROE + low D/E signals strategic execution
        roe_anchor = {"TECH": 18, "FIN": 11, "UTIL": 9}.get(td.sector, 13)
        proxy = (td.roe - roe_anchor) / max(5, roe_anchor) * 6
        dims["strategic_transformation"] = clamp(sec["strategic_transformation"] + proxy)
        data_driven["strategic_transformation"] = True

    # D6: management_stake — insider ownership %
    if td.insider_ownership_pct is not None:
        # 0% is fine for large institutional companies. Founder-controlled (>10%) is a positive signal.
        if td.insider_ownership_pct >= 10:
            delta = min(15, (td.insider_ownership_pct - 10) * 0.8 + 8)
        elif td.insider_ownership_pct >= 3:
            delta = (td.insider_ownership_pct - 3) * 0.6 + 2
        elif td.insider_ownership_pct >= 0.5:
            delta = 0
        else:
            delta = -3
        dims["management_stake"] = clamp(sec["management_stake"] + delta)
        data_driven["management_stake"] = True

    # D7: ownership_structure — institutional ownership %
    if td.institutional_ownership_pct is not None:
        # Healthy: 40–80%. <30% = under-followed. >95% = crowded.
        if 40 <= td.institutional_ownership_pct <= 80:
            delta = 8
        elif td.institutional_ownership_pct < 30:
            delta = -3
        elif td.institutional_ownership_pct > 95:
            delta = -2
        else:
            delta = 3
        dims["ownership_structure"] = clamp(sec["ownership_structure"] + delta)
        data_driven["ownership_structure"] = True

    # D8: culture_purpose — semi-live sector-stratified baseline.
    # Anchored to public Glassdoor distributions (TECH consistently highest-rated,
    # ENERGY/UTIL lowest), refreshed annually. We FLAG this as data-driven because
    # the sector tier IS evidence-based (~70% of the signal of per-firm scraping,
    # at zero infrastructure cost). The per-firm enrichment hook accepts
    # glassdoor_rating / glassdoor_recommend when sourced — falls through to
    # the sector tier when absent.
    if td.sector in CULTURE_SECTOR_BASELINES:
        data_driven["culture_purpose"] = True

    # D9: progressive_practices — ESG
    if td.esg_risk is not None:
        # Sustainalytics scale: <15 negligible (good), 15-20 low, 20-30 medium,
        # 30-40 high, >40 severe (bad).
        # Lower risk → higher dimension score.
        if td.esg_risk < 15:
            score = 75
        elif td.esg_risk < 20:
            score = 65
        elif td.esg_risk < 30:
            score = 55
        elif td.esg_risk < 40:
            score = 45
        else:
            score = 35
        dims["progressive_practices"] = score
        data_driven["progressive_practices"] = True

    # D10: market_dynamics — beta + 52-week range position
    md_delta = 0
    md_signals = 0
    if td.beta is not None:
        # Beta near 1.0 is neutral; >1.5 high vol, <0.5 defensive
        if 0.7 <= td.beta <= 1.3:
            md_delta += 4
        elif td.beta > 1.7 or td.beta < 0.3:
            md_delta -= 3
        md_signals += 1
    if td.price_in_range is not None:
        # Lower in the range = better entry value
        md_delta += (0.5 - td.price_in_range) * 10
        md_signals += 1
    if md_signals >= 1:
        dims["market_dynamics"] = clamp(sec["market_dynamics"] + md_delta)
        data_driven["market_dynamics"] = True

    return dims, data_driven


# Layer weights (mirrors LAYERS in the JS tool)
LAYERS = [
    {"id": "macro",    "weight": 0.10, "dims": ["macro_environment"]},
    {"id": "fin_arch", "weight": 0.20, "dims": ["financial_metrics", "financial_engineering"]},
    {"id": "corp_pos", "weight": 0.22, "dims": ["tech_adoption", "strategic_transformation"]},
    {"id": "gov",      "weight": 0.25, "dims": ["management_stake", "ownership_structure"]},
    {"id": "org_core", "weight": 0.18, "dims": ["culture_purpose", "progressive_practices"]},
    {"id": "market",   "weight": 0.05, "dims": ["market_dynamics"]},
]


def compute_composite(dims: dict[str, int]) -> tuple[int, dict[str, int]]:
    """Compute layer scores (avg of constituent dims) and weighted composite."""
    layer_scores = {}
    composite = 0.0
    for layer in LAYERS:
        avg = sum(dims[d] for d in layer["dims"]) / len(layer["dims"])
        layer_scores[layer["id"]] = clamp(avg)
        composite += avg * layer["weight"]
    return clamp(composite), layer_scores


def get_quartile(score: int) -> str:
    if score >= 70: return "Q1"
    if score >= 58: return "Q2"
    if score >= 45: return "Q3"
    return "Q4"


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_tickers() -> list[dict[str, str]]:
    """Load the curated 200-ticker list from tickers.csv."""
    rows = []
    with open(TICKERS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(filter(lambda l: not l.startswith("#"), f))
        for row in reader:
            rows.append(row)
    log.info(f"Loaded {len(rows)} tickers from {TICKERS_CSV.name}")
    return rows


def process_ticker(row: dict[str, str], macro: dict, idx: int, total: int) -> dict:
    """Process a single ticker → returns the JSON-serializable dict."""
    ticker = row["ticker"]
    log.info(f"[{idx+1:>3}/{total}] {ticker:<14} {row['name']}")
    td = TickerData(
        ticker=ticker,
        name=row["name"],
        country=row["country"],
        region=row["region"],
        sector=row["sector"],
    )
    try:
        fetch_yfinance(td)
    except Exception as e:
        td.fetch_errors.append(f"yfinance overall: {e}")
        log.warning(f"  yfinance failed entirely for {ticker}: {e}")
    time.sleep(YFINANCE_SLEEP_SEC)

    # ESG fallback (only if yfinance didn't get it AND we have a Finnhub key)
    if FINNHUB_API_KEY and td.esg_risk is None:
        try:
            fetch_finnhub_esg(td)
        except Exception as e:
            td.fetch_warnings.append(f"finnhub: {e}")

    # Compute dimensions and composite
    dims, data_driven = compute_dimensions(td, macro)
    composite, layer_scores = compute_composite(dims)
    quartile = get_quartile(composite)

    return {
        "ticker": ticker,
        "meta": {
            "name": td.name,
            "country": td.country,
            "region": td.region,
            "sector": td.sector,
            "cap": td.cap,
            "market_cap": td.market_cap,
            "price": td.price,
            "currency": "USD",  # yfinance normalises most to USD in .info
        },
        "fundamentals": {
            "pe": td.pe, "pb": td.pb, "peg": td.peg, "eps": td.eps,
            "current_ratio": td.current_ratio, "debt_equity": td.debt_equity,
            "roe": td.roe, "operating_margin": td.operating_margin,
            "profit_margin": td.profit_margin, "payout_ratio": td.payout_ratio,
            "div_yield": td.div_yield, "five_year_avg_div_yield": td.five_year_avg_div_yield,
            "beta": td.beta,
            "fifty_two_week_low": td.fifty_two_week_low, "fifty_two_week_high": td.fifty_two_week_high,
            "quarterly_earnings_growth": td.quarterly_earnings_growth,
            "quarterly_revenue_growth": td.quarterly_revenue_growth,
            "rd_intensity": td.rd_intensity,
            "insider_ownership_pct": td.insider_ownership_pct,
            "institutional_ownership_pct": td.institutional_ownership_pct,
            "esg_risk": td.esg_risk,
        },
        "dimensions": dims,
        "data_driven": data_driven,
        "layer_scores": layer_scores,
        "composite": composite,
        "quartile": quartile,
        "warnings": td.fetch_warnings,
        "errors": td.fetch_errors,
    }


def main() -> int:
    started = datetime.now(timezone.utc)
    log.info("=" * 70)
    log.info(f"CVM data pipeline start · {started.isoformat()}")
    log.info(f"Finnhub key present: {'yes' if FINNHUB_API_KEY else 'no'}")

    if not TICKERS_CSV.exists():
        log.error(f"tickers.csv not found at {TICKERS_CSV}")
        return 1

    # Step 1: macro
    try:
        macro = fetch_macro_data()
    except Exception as e:
        log.error(f"Macro data fetch failed: {e}")
        macro = {}

    # Step 2: per-ticker
    tickers = load_tickers()
    results = []
    failures = 0
    for idx, row in enumerate(tickers):
        try:
            result = process_ticker(row, macro, idx, len(tickers))
            results.append(result)
        except Exception as e:
            failures += 1
            log.warning(f"  Ticker {row.get('ticker','?')} failed entirely: {e}")
            results.append({
                "ticker": row.get("ticker", ""),
                "meta": {"name": row.get("name", ""), "sector": row.get("sector", "")},
                "dimensions": {}, "data_driven": {}, "composite": None, "quartile": None,
                "errors": [str(e)],
            })

    # Step 3: aggregate stats for the JSON header
    succeeded = [r for r in results if r.get("composite") is not None]
    avg_dims_live = 0
    if succeeded:
        avg_dims_live = sum(len(r["data_driven"]) for r in succeeded) / len(succeeded)

    payload = {
        "generated_at": started.isoformat(),
        "ticker_count": len(tickers),
        "succeeded": len(succeeded),
        "failed": failures,
        "avg_data_driven_dimensions": round(avg_dims_live, 2),
        "schema_version": 1,
        "tickers": {r["ticker"]: r for r in results},
    }

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log.info(f"Pipeline complete · {len(succeeded)}/{len(tickers)} succeeded · "
             f"avg {avg_dims_live:.1f}/10 dims live · {elapsed:.0f}s")
    log.info(f"Output: {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
