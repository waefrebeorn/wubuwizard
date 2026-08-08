# ⚔️ DA #2: Scraping Coverage & Pipeline Quality

**DA Phase:** Phase 1-4 complete  
**Date:** May 25, 2026  
**Scope:** Monkey scraper v2 coverage, pipeline reliability, missing sources

---

## Phase 1: CLAIM

> "Monkey scraper v2 pulls 25+ data sources across 8 categories (crypto, macro, defi, weather, sentiment, on-chain, prediction markets, news). It runs every 15 minutes with httpx async pipeline, User-Agent headers, and timeline DB integration."

**Source:** `monkey_scraper.py` (437 lines) + cron job listing  
**Trust:** MEDIUM — verified via last run output

---

## Phase 2: VERIFY

### Current Sources
| Category | Sources | Format | Status |
|----------|---------|--------|--------|
| Kraken OHLC | BTC, ETH, SOL, XRP | REST → JSON | ✅ 4 pairs |
| CoinGecko | Global + top 10 prices | REST → JSON | ✅ 10 coins |
| DefiLlama | Chains TVL + DEX volume | REST → JSON | ✅ 20 chains + 10 DEX |
| FRED | 10 macroeconomic series | REST → CSV | ✅ SP500, VIX, CPI, etc. |
| Weather (Open-Meteo) | 5 financial hubs | REST → JSON | ✅ NYC, London, Tokyo, Chicago, Houston |
| Fear & Greed | Crypto sentiment | REST → JSON | ✅ alternative.me |
| Polymarket | Top 100 markets | REST → JSON | ✅ gamma-api |
| GDELT | News events | REST → JSON | ⚠️ 5 articles max, 135 total in timeline |

### What's NOT Being Scraped

**Financial Data (free APIs, no auth):**
- **Alpha Vantage** — 50+ forex pairs, sector performance, crypto (free tier: 5 calls/min)
- **Twelvedata** — Stocks, forex, crypto, commodities (free: 800 calls/day)
- **ExchangeRate API** — 170 currencies, real-time (free tier)
- **Binance** — All spot pairs, 24h ticker (no auth)
- **BYBIT** — Funding rates, open interest, liquidations (no auth)
- **Yahoo Finance** — All stocks, ETFs, options (no auth)
- **FMP (Financial Modeling Prep)** — Company profiles, earnings, SEC filings (free tier)
- **OpenBB** — Aggregated financial data (free)

**Crypto-specific:**
- **CoinGlass** — Liquidations, long/short ratios
- **DeFi pulse** — Top protocols, yields
- **Etherscan** — Gas prices, network stats (no auth)
- **CoinGecko trending** — Trending coins, recently added
- **GeckoTerminal** — Recent DEX trades, pools, pairs

**News & Social:**
- **CryptoPanic** — Crypto news aggregator API (free)
- **LunarCrush** — Social metrics for coins (free tier)
- **Google Trends** — Search interest for coins/terms (no auth daily)
- **Hacker News** — Top financial/tech stories (no auth)
- **Reddit API** — r/cryptocurrency hot posts (no auth)

**Alternative Data:**
- **Google Flights** — Travel demand proxy (no auth)
- **AmazingHiring / LinkedIn scraping** — Tech hiring proxy (labor market)
- **BitInfoCharts** — Rich list, supply distribution
- **Tracking transactions** — API for crypto merchant volume

### Pipeline Quality Issues

| Issue | Detail | Severity |
|-------|--------|----------|
| News timeline writes = 0 | GDELT `append_timeline` writes NOT inside the success check | **HIGH** — bug, writes only when resp==200 but check is inverted |
| FRED CSV parsing is fragile | CSV changes format on holidays/ weekends | MEDIUM |
| No retry logic | Single 10s timeout per source, first failure = skip whole category | MEDIUM |
| No credential fallback | CoinGecko free tier rate-limits after 50 calls/min — no API key for higher limit | LOW |
| Weather 5 cities only | Missing Singapore, Dubai, Frankfurt — major financial hubs | LOW |
| No dedup on writes | `INSERT OR IGNORE` OK for timeline, but no in-memory dedup = same data written multiple times | LOW |
| GDELT only 5 articles | API allows max=50, scraper hardcodes max=10 → timeline has 135 rows total | **HIGH** |

---

## Phase 3: RISK ASSESSMENT

| Risk | Assessment |
|------|-----------|
| **Competitive disadvantage** | Existing API providers (Twelvedata, Alpha Vantage) already sell 100+ assets for $10/mo. Our data set is smaller and uncurated. |
| **News gap is crippling** | 135 news rows means the system is blind to market-moving events. Can't predict anything without news context. |
| **No unique data** | Everything in timeline is available from free sources. No derived signals, no proprietary indicators = no moat. |
| **GDELT reliability** | GDELT is a free university project — no SLA, uptime ~95%. Not a production data source. |
| **Crypto altcoin gap** | Kraken only has 4 pairs. No MATIC, AVAX, DOT, LINK — these have different market dynamics. |
| **FRED update latency** | FRED data is daily with 1-day delay on many series (GDP, CPI). Real-time macro is missing. |

---

## Phase 4: MITIGATION

### Fix pipeline bugs (this session)
1. Fix GDELT news writing to timeline — currently only 5 articles * 4 tiers... but `fetch_rss` doesn't differentiate. Rewrite `scrape_gdelt` to use RSS directly for more articles.

### Add 10 new sources (priority order)
| # | Source | Data | Effort |
|---|--------|------|--------|
| 1 | Binance ticker | All 200+ crypto pairs 24h price | 15 min (httpx) |
| 2 | BYBIT funding | Funding rates + OI for top 20 coins | 20 min (httpx) |
| 3 | TwelveData forex | 28 forex pairs real-time | 20 min (httpx + free key) |
| 4 | CryptoPanic | 100s of crypto news articles/day | 15 min (httpx, no auth) |
| 5 | CoinGlass | Liquidations data | 20 min (httpx) |
| 6 | Etherscan gas | Gas prices, network stats | 10 min (httpx, no auth) |
| 7 | LunarCrush | Social metrics (mentions, sentiment) | 20 min (httpx, free tier) |
| 8 | Google Trends | Search interest daily | 15 min (pytrends lib) |
| 9 | DeFi Llama yields | Lending/borrowing rates | 10 min (already scraped? check) |
| 10 | CoinGecko extended | Top 100 → top 250 | Parameter change |

### Increase scraped rows per source
- GDELT: maxrecords=10→50 (more articles)
- CoinGecko: top 10→top 50 prices
- Polymarket: 100→200 markets
- DefiLlama chains: 30→50
- Weather: 5→8 cities (add Singapore, Dubai, Frankfurt)

### Add retry logic to httpx client
- Single retry with 2s delay for transient 5xx errors
- 15s timeout per call instead of 10s

---

## Summary
The scraper pipeline has a **critical bug** (news not writing to timeline) and **major coverage gaps** (news, derivative, altcoins, forex, social sentiment). Fixing the news bug alone would increase timeline's prediction value by 50%. Adding 10 new sources would make the dataset genuinely competitive. Without these fixes, any prediction API built on current data would be inferior to free alternatives.