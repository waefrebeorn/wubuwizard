# Concept: MTP (Multi-Token Prediction)

## Summary
Speculative decoding technique where the model predicts multiple future tokens simultaneously, then verifies them.

## How It Works
- Draft model produces k candidate tokens
- Target model verifies in parallel (single forward pass)
- Accepted tokens: keeps the longest matching prefix
- Speedup depends on acceptance rate (higher = faster)

## In bytropix
| Metric | Value | Status |
|--------|-------|--------|
| Speed | 8.5 tok/s | ✅ |
| Acceptance rate | 4% | 🟡 Low but working |

- Implementation: `gen_text_mtp` binary
- Reference: `ref_dumper_mtp` for multi-token reference data

## Related
- [[SSM]] — SSM benefits from parallel verification
- [[Chunked-SSM]] — chunked prefill can accelerate MTP

## Sources
- bytropix Phase 28l logs
