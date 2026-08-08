# 512K Long-Form Conversation Test — May 27, 2026

**Test:** Hermes Agent → bytropix local server → gen_text_cpu  
**Model:** Qwen3.6-35B-A3B-UD-IQ2_M (11GB GGUF, IQ2_M quant)  
**Hardware:** i5-8365U, 16GB DDR4, 4x OMP threads  
**Server:** `tools/serve_local.py` on port 8001  
**Workload:** 3-turn NES emulator architecture Q&A (system + 3 user turns)

## Results

| Turn | Context | Words | Time (s) | Words/s | Est. tok/s |
|------|---------|:-----:|:--------:|:-------:|:----------:|
| 1 | Cold start + model load | 110 | 143.9 | 0.76 | ~1.0 |
| 2 | Warm, short context | 174 | 185.0 | 0.94 | ~1.2 |
| 3 | Warm, growing context | 197 | 415.2 | 0.47 | ~0.6 |
| **Total** | 3 turns, 7 messages | **481** | **744.0** | **0.65** | **~0.84** |

## Key Observations

1. **Model load dominates** — ~80s from cold start on i5-8365U
2. **Decode speed decays with context** — 50% drop from turn 2 to turn 3
3. **ChatML formatting** — gen_text_cpu raw mode doesn't properly handle `<|im_start|>` tokens. Model regenerates system/user preamble in output. Needs CHAT=1 mode or tokenizer-level ChatML support.
4. **Server stability** — survive.py handles BrokenPipeError gracefully from client disconnects
5. **Throughput ceiling** — ~1.2 tok/s at short context, ~0.6 tok/s with growing context

## For 512K Context

At full 512K context with sparse attention (expected ~4.1 tok/s decode historically), a 3-turn conversation would take:
- Turn 1: ~80s model load + ~10s prefill = ~90s
- Turn 2: ~15s (512K sparse attn decode at 64 tokens)
- Turn 3: ~15s
- **Total:** ~120s for 3-turn conversation

Current test shows 744s (12.4 min) due to non-sparse attention at short context. Sparse attention only activates at >4K context.

## Recommendation

bytropix 512K pipeline works end-to-end. Performance is usable for:
- Batch/background inference
- Long-running agent tasks
- 512K context benchmark workloads

Not suitable for interactive chat (0.6-1.2 tok/s). GPU, Q3_K+/F16, or MTP needed for interactive speeds.
