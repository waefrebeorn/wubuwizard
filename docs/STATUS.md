# STATUS — Inkling-2-Mini engine (wubuwizard)

One page. Everything here is either a measurement or explicitly marked as retracted. Discipline: `docs/LONG-CONTEXT-ROADMAP.md`
holds the history; this file holds only what is true now.

## How to gate any change

    bash tools/oracle_gate.sh             # must print GATE PASS (15/15 tokens)
    bash tools/oracle_gate.sh --rebuild   # after editing

Golden line is 3 prompt + 12 generated = **15** tokens (not 12 — the command's `12` is the generate count). The gate refuses
to pass on an empty or short reference, which is how four false passes were avoided this session.

## Architecture fact that governs all arithmetic

**Looped transformer.** GGUF metadata: `inkling2.num_loops = 3`, `loop.core_start = 6`, `loop.core_end = 18`,
`loop.gate_rank = 64`. Layers 6..17 run **three times per token** (engine: `load.c:78 is_loop_layer`). So the stack costs
**72 layer-passes**, and MoE (layers 0..23) costs **48 passes**: 6 + 12x3 + 6 = 48 - exactly the 48 MoE calls `IN2_MOE_NTOK`
prints. Any estimate assuming 48 layer-passes is 1.5x low.

## Measured state (60-token prefill, 8 threads, unless noted)

    prefill            5.3 s total        88 ms/token
    decode            172 ms/token         5.8 tok/s   (entirely in in2_forward_at; "other" is 0.000 s)
    oracle            15/15 identical in every configuration tested, throughout

Phase split (`IN2_TIMETY`):

    moe        41.9 %   expert-major, omp over 128 experts, per-token FLAT (38.8 -> 38.2 ms/token)  -> healthy
    q          16.2 %   the wq matvec at 33.7 GMAC/s — the best CPU rate measured in this engine   -> nothing to win
    attn        7.5 %   partly parallelised this session (that change gave 3.8 % overall and was kept)
    gate+wo     6.8 %
    kv+conv     3.9 %
    rest        <2 %

Thread scaling: **2.29x at 8 threads** on 6 physical cores (Ryzen 5 8645HS, 2x8 GB DDR5-5600 dual channel). Not bandwidth
(sliding traffic is 443 MB/s against ~90 GB/s), not allocation (fixed, flat), not granularity (scaling got *worse* with bigger
buckets).

## What the GPU work concluded

The resident tier hits 97.7 %; the 8-column IQ2_XXS tile is bit-identical to the shipping kernel (max|diff| = 0 against
max|ref| = 1.266e+04) and 1.73–1.76x faster at M>=8, and it is wired behind `n_tok >= 8`. **It moved prefill by 2 %**, and the
batched MoE/projection hooks moved it by nothing or less. The measurements say why: the engine is not kernel-bound in any
place the GPU was offered. The GPU line is closed on evidence, not on opinion.

One real win came out of it: the per-visit KV mirror (`in2_kv_k`/`in2_kv_v`), which is flat in the default f32 config but
**4.8x** in TQ-6 compressed-KV mode (28.7 s -> 5.93 s at 60 tokens, within 4 % of f32 while holding 12.06 KB/token). That is
what makes the 512k-context configuration practical.

## Next actions, in order

1. **Per-position logits in `in2_forward_at`** — the prerequisite for speculative decoding. `model.c:295` computes the head for
   the last position only. Extract `:295-389` into a helper taking `pos` and `out`, call it with `pos = n_tok-1` so the
   existing path runs identical instructions (the gate validates the refactor alone), THEN add optional `logits_all` and loop.
   Note the entanglement: the block uses locals `d`, `eps`, `head_done` and the `_fw1`/`_h0` timers, so it needs judgement per
   line, not a script.
2. **Draft/verify loop in `tools/inkling_gen.c`** — suffix-match over `toks[]`, one `in2_forward_at` for K+1 tokens, accept
   while argmax matches. Parity is preserved by construction (the drafter only proposes).
   Economics: head is 13.6 ms/position against 172 ms/saved forward — 8 % — so K=4 nets +634 ms per block even before counting
   acceptance; at 50 % acceptance still +290 ms.
3. Only then re-examine MoE (41.9 %), which is already the healthiest path in the engine.

## Standing corrections (each cost a session turn)

- `sliding` is a **layer class**, not the attention op; attention proper is 7.5 %, not 27 % or 80 %.
- `q` is **not** single-threaded; my operation count was 3x low (24 vs ~68 calls), which made a correct timing look like one
  thread. Four attempts died downstream of that arithmetic.
- The model has 48 blocks with experts on **layers 0-23 only** (a hybrid stack); the engine was right and my note was wrong.
- OpenMP does not nest by default — an outer `parallel for` around an inner parallel helper buys nothing.
- A phase label is a bookmark, not a definition: read the two marker lines before attributing time.
- Pattern-searching a file (grep, brace scan, Nth match, call graph) has been the failing component six times. Use printed line
  numbers as the edit anchor, and `git add` explicit paths when the tree may hold in-flight work.
