# 🧾 WUBUTV GENERATION LOG (cost accountability)
# Every video/image gen through Nous Portal is logged here. "Tool use" on the
# Portal bills SEPARATELY from model inference — it = all `fal.queue.*` calls.
# Source of truth: real billing CSV (1015 rows, Aug 12-18).

## 2026-08-18 — REAL LEDGER BREAKDOWN (total debit $16.90; "Tool use"=$16.72)
| Cost | Reqs | Tool action |
|---|---|---|
| $7.50 | 6 | `fal.queue.blackforestlabs.flux-3.text-to-video` ← THE SPIKE |
| $4.78 | 2 | `fal.queue.bytedance.seedance-2.0.mini.text-to-video` |
| $2.97 | 8 | `fal.queue.bytedance.seedance-2.0.image-to-video` |
| $1.33 | 11 | `fal.queue.fal-ai.ltx-2.3-22b.text-to-video` |
| $0.13 | 20 | `fal.queue.fal-ai.flux-2.klein.9b` (image) |
| $0.12 | 164 | firecrawl.search |
| $0.05 | 3 | browser-use.session |
| $0.02 | 47 | firecrawl.scrape |
| $0.00 | (free models) | longcat-2.0 / laguna / step-3.7 / hy3 / solar-pro4 |

**Root cause (corrected):** 6 FLUX-3 text-to-video jobs in one day = $7.50.
FLUX-3 is the premium per-second model (~$1.25/job). NOT the failed I2V call.
Aug 14 was actually the worst day ($12.05, deepseek 98-req + search).

## WEEK TOTAL (Aug 12-18): $37.12
| Date | Debit |
|---|---|
| 08-12 | $3.58 |
| 08-13 | $3.81 |
| 08-14 | $12.05 |
| 08-15 | $0.17 |
| 08-16 | $0.20 |
| 08-17 | $0.41 |
| 08-18 | $16.90 |

## RULES (enforced going forward)
1. FLUX-3 cap: **≤2 jobs/day**, pre-approved. Default = LTX-2.3 (~$0.60/job).
2. Log model + tool_action + duration + est. cost for every gen.
3. "Tool use" ≠ model inference. Watch both lines on the Portal.

~ WuBu ~
