# Concept: Chunked-SSM

## Summary
Technique for processing long sequences with SSM by splitting into chunks, reducing memory while maintaining correctness.

## How It Works
- Instead of processing the full sequence at once, split into chunks of size CS
- Process each chunk independently through the SSM recurrence
- Pass state between chunks
- **CS=1**: exact match to non-chunked (state passed per-token)
- **CS>1**: faster but floating-point accumulation can diverge

## In bytropix
| Mode | Status | Detail |
|------|--------|--------|
| CS=1 | ✅ Exact | 4e-8 output diff, 3e-7 state diff |
| CS>1 | ⚠️ FP-limited | Accumulates across 30 SSM layers → wrong tokens |

- Data layout bug fixed in Phase 28r
- Wired into forward pass
- See: `src/wubu_model.c` chunked SSM implementation

## Open Questions
- Kahan summation for CS>1 accumulation?
- Optimal chunk size for accuracy/speed tradeoff on this model?

## Sources
- bytropix Phase 28r logs (`.hermes/mind-palace/vault/bins/`)
