# ⚔️ DA #1: Trading Data Gaps & Currency Expansion

**DA Phase:** Phase 1-4 complete  
**Date:** May 25, 2026  
**Scope:** Current trading coverage vs what's needed for real prediction API

---

## Phase 1: CLAIM

> "The trading ecosystem tracks crypto, macro, forex, prediction markets, and sentiment across 9.3M timeline rows. It's comprehensive enough to start selling predictions."

**Source:** Walkway state survey + cron/script inventory  
**Trust:** LOW — not verified against real market needs

---

## Phase 2: VERIFY

### Crypto — 7.6M rows
| Pair | Rows | Source | Live? |
|------|------|--------|-------|
| BTC/USD | 7,569,588 | Bitstamp 1-min historical | ❌ 2012-2021 historical only |
| BTC/USD | 6,116 | Kraken OHLC | ✅ 1-min live |
| ETH/USD | 6,116 | Kraken OHLC | ✅ 1-min live |
| SOL/USD | 6,116 | Kraken OHLC | ✅ 1-min live |
| XRP/USD | 1,790 | Kraken OHLC | ✅ 1-min live |

**VERIFIED GAP:** 7.5M BTC historical rows are from Bitstamp (2012-2021, no live feed). Live Kraken only started recently. No altcoins beyond BTC/ETH/SOL/XRP. No derivatives (funding rates, OI, liquidations). No stablecoin pegs tracked.

### Forex — 3.9K rows
| Pair | Rows |
|------|------|
| EUR/USD | 1,300 |
| GBP/USD | 1,301 |
| USD/JPY | 1,300 |

**VERIFIED GAP:** 3 pairs, all daily. No 1-min data. Missing: AUD/USD, USD/CAD, NZD/USD, USD/CHF, EUR/GBP, EUR/JPY, GBP/JPY, CHF/JPY, and 20+ cross pairs.

### Equities — 34K rows
All indices via FRED (SP500, VIX, Dow, Nasdaq, FTSE, Nikkei). **No individual stocks.** No earnings data. No sector ETFs.

### News — 135 rows total
GDELT API writes only 5 articles per scrape. No Twitter/X feed. No Reddit. No Telegram channels. No crypto-specific news sources (The Block, CoinDesk, CoinTelegraph).

### On-chain — ZERO
No exchange inflow/outflow. No whale wallets. No stablecoin mint/burn. No active addresses. No miner flows. No DEX volume by token. No staking yields.

### Order Book — ZERO
No bid/ask spread tracking. No liquidity depth. No CLOB snapshot data.

### Derivatives — ZERO
No futures funding rates. No open interest. No liquidations. No basis (perp vs spot). No options implied volatility.

---

## Phase 3: RISK ASSESSMENT

| Risk | Assessment |
|------|-----------|
| **Data too thin** | BTC-only prediction API is worthless. Need 20+ assets minimum for credibility. |
| **Backtest limited** | Can't backtest multi-asset strategies with only BTC/3 forex. No cross-asset correlations. |
| **News blind** | 135 news rows = can't do event-driven trading. No sentiment edge. |
| **Derivatives blind** | Can't detect squeezes, carry trades, or volatility events without funding/OI. |
| **On-chain blind** | Can't track whale accumulation, exchange inflows (sell pressure), or stablecoin liquidity. |
| **Competition risk** | Existing APIs (TradingView, CoinGecko, FRED) already cover what we have. No unique data = no sale. |
| **API product weak** | No one pays for "Kraken BTC price" — already free. Need unique derived signals. |
| **Stale data volume** | 7.5M historical rows are impressive-looking but add zero live prediction value. |

### Key Insight
The timeline data is **volume-heavy, value-light**. 9.3M rows but 81% is a single historical BTC dataset (Bitstamp 2012-2021). Live usable data is ~1.8M rows spread thin across 30 sources. The ecosystem looks bigger than it is.

---

## Phase 4: MITIGATION

### Expansion Targets (Priority Order)

**P0 — Fix news pipeline (highest ROI, lowest effort)**
- Mitigation: Rewrite live_news.py to write to timeline DB
- Monitoring: News rows in timeline should go from 135 → 5,000+/day
- Revert threshold: If GDELT rate-limits, switch to RSS-only

**P1 — Add 8 more forex pairs (free APIs exist)**
- Mitigation: Add EUR/JPY, GBP/JPY, AUD/USD, USD/CAD, NZD/USD, USD/CHF, EUR/GBP from ExchangeRate API / Frankfurter
- Monitoring: Forex rows/tick should double weekly

**P2 — Expand crypto to top 20 coins**
- Mitigation: Add Kraken SOL/XRP/ADA/DOT/LINK/AVAX/MATIC/UNI/ATOM/ALGO
- Add CoinGecko 24h price for all top 100 (already in scraper, just pipe to timeline)
- Monitoring: All top-20 coins present in timeline

**P3 — Order book snapshots**
- Mitigation: Kraken depth API every 5m for BTC/ETH
- Monitoring: bid/ask spread in timeline

**P4 — Derivatives data**
- Mitigation: Binance/BYBIT funding rate API (free, no auth)
- Monitoring: Funding rates appear in timeline

**P5 — On-chain basics**
- Mitigation: Glassnode free tier / CoinMetrics / Blockchain.com
- Monitoring: Exchange inflow/outflow columns fill

---

## Summary
Current 9.3M rows = illusion of completeness. Only 1.8M rows are live, and 80%+ of that is crypto. The prediction product can't launch until we have news, forex depth, derivatives, and order book data. Minimum viable prediction API needs 6 asset classes with 50+ signals each.