# INKLING2 KV CACHE — design

**Problem, measured:** 14 tokens took **1,079 s** (~77 s/token) because `in2_generation` recomputes the whole
prefix on every step: step *k* runs 24 layers × 3 passes over *k* positions, so generation is O(n²) in work for
n tokens. At a 19-token chat prompt the first step already costs ~60 s and every later step costs more.

This is not a kernel-tuning problem. It is an algorithmic one, and the cache is the whole fix.

## What must survive between steps

| state | per layer | size (this model) | why |
|---|---|---|---|
| sliding K/V | 20 layers × 1,024 pos × 1,024 (8 kv heads × 128) | 20 × 1024 × 1024 × 4 B × 2 = **168 MB** | window never exceeds 1,024 keys, so a ring buffer |
| full K/V | 4 layers × n_pos × 1,024 | 4 × n × 1024 × 4 B × 2 = 33 KB/token | grows with context |
| conv history | 4 convs × 3 rows × (2,304 or 1,024) | ~50 KB/layer | 4-tap causal convs need the last 3 inputs |
| block states | current position only | 8 × 2,304 | the depth mix is **within a position** (see below) |

**Block states do not need history.** `in2_mix` indexes blocks with `block_stride = n_tok*d` and
`pos_off = t*d`, so a position's mix attends to that position's states across blocks — verified in
`transformers`' `Inkling2BlockAttentionResidual`, which stacks `[block_states, partial]` per position and
normalises each state separately. A decode step of one token therefore needs only its own block states.

## Two traps this model sets

1. **Conv histories are per *visit*, not per layer.** `docs/INKLING2.md:233` — "the KV cache and the conv
   histories are a visit's". Layer 6 runs three times per token; each visit must see its own 3 rows of history,
   not the previous visit's. A cache keyed only by layer index would silently carry pass 2's history into
   pass 3 and corrupt every looped layer.

2. **The loop restarts from the prelude.** Each pass begins at the state and block bank the core was first
   given, so a cached state must be snapshotted after the pre-loop layers and after pass 1 separately; the
   resume point differs per pass.

## Interface

```
in2_gen_state_t s;                     /* opaque: kv rings, conv histories, prelude snapshot */
in2_gen_begin(ctx, m, cfg, &s, prompt_tokens, n_prompt, h, logits);   /* prefill: fills the cache */
in2_gen_step(ctx, m, cfg, &s, token, pos, h, logits);                  /* decode: 1 position, O(1) work */
in2_gen_end(&s);
```

`in2_forward_seq` stays as it is — it is the correctness reference and the prefill path. The rule for the cache
is the one this project already uses for the message path: **a cached step and a full recompute must produce the
same token**, and that is a gate, not a hope.

## Gate

For a 19-token templated prompt, generate 8 tokens two ways:

- `in2_forward_seq` full recompute (current behaviour), and
- `in2_gen_begin` + 8 × `in2_gen_step`.

Requirements: identical id sequence, and `max|logit difference| < 1e-4` at every step. Anything else means the
cache is wrong, not that the model is subtly different — the arithmetic is the same arithmetic.

Then the speed claim is measured, not asserted: `n_gen / elapsed` for both, on a quiet box, recorded in
`ENGINE-THROUGHPUT-BASELINE.md` with the prompt, the file and the flags.

## Order of work

1. `in2_gen_state_t` + allocation, keyed `[visit][conv_idx]` for histories and `[layer]` for the sliding rings.
2. Sliding layers first (20 of 24, and the ring is where off-by-one bugs live — the window is `dist < 1024`).
3. Full layers (grow-on-append, no ring).
4. The loop's per-pass snapshot.
5. The gate above, then timing.

Non-goal for this pass: batching, sampling beyond greedy, and any GPU work. Those come after decode is linear.

## Status (2026-11-25)

Done and verified:
- `src/wubu_inkling2_kv.c` — per-visit storage, ring slots for the 20 sliding visits and linear slots for the 4
  full ones, raw pre-conv rows, residency predicate. Self-test covers slot reuse, visit aliasing, expiry, rotation.
- `in2_kv_conv_row` — the 4-tap static conv one row at a time from cached history. **Bit-identical to
  `in2_shortconv`** over 9 rows (worst |diff| 0.000e+00).
- `in2_kv_conv_row_dyn` — the same for the full layers' dynamic conv (tap = stat + per-position projection).
  **Bit-identical to `in2_shortconv_dynamic`** over 7 rows.
- Store path wired and gated: `IN2_KV=1 IN2_KV_VERIFY=1` writes post-clip K / post-norm V plus raw rows every
  sliding visit and reads them back — `worst=0.000e+00`, logits 27.1610 with and without the cache.

Remaining, and it is one coherent change, not a set of independent ones:
1. `in2_forward_step(ctx, m, cfg, tokens, n_tok, pos0, h, logits, kv)` — the current entry point always starts at
   position 0 and the layers always receive `pos0 = 0` from the macro in `wubu_inkling2_model.c`.
2. Both layer bodies must compute K/V only for `[pos0, pos0+n_tok)` and attend over `[max(0,p-EXT+1), p]` from the
   cache. Sliding and full must land together: caching one and not the other leaves the other layer type
   attending to a partial prefix, which is silently wrong rather than obviously broken.
3. The gate: incremental step logits vs a full recompute of the same final token, `max|dlogit| < 1e-4`.
   Note the activations already diverge at ~1e-2 from int8 quantisation, so this gate must be run **within one
   activation mode** (both int8 or both fp32), not across them.
4. Size the real cache: 60 sliding visits x 1024 pos x 1024 dim x 4 B x 2 = ~504 MB, plus 12 full visits growing
   with context. The store-path test allocation is deliberately tiny (72 x 64) and must not be mistaken for it.

## Numerics that are not free (measured 2026-11-25)

The attention QK dot must accumulate in double. Products reach ~1e5 before the 1/sqrt(HD) scale, and the error
is carried through all 48 layer visits: the same edit in the full-attention layer moved the final logit from
**27.1610 to 25.0769** - a 2-logit shift from a change that looks purely mechanical.

The AVX2 float version was measured in the sliding layer and **bought nothing**: 8,021 vs 8,041 ms at a 5-token
prefix. It was reverted. Rule: the attention core stays double until attention is measured to be the bottleneck at
a realistic prefix length; the projections are where the time goes, and those are already integer kernels.

## Full-attention conversion: open bug (2026-11-25)

Sliding layer is cache-capable and verified. The 4 full-attention layers are NOT; the attempt was reverted twice.

Found and fixed in the attempt:
- `logit`/`alpha` in `in2_layer_full_seq` were sized `n_tok`. Once a cache is attached `n_key = pos0 + t + 1`, so
  the first incremental step writes past a 3-element buffer. That is what "no output at all" looks like. Now sized
  `n_tok + pos0 + 1`.

Still crashing with the cache attached, so at least one more cause exists. Leads, in the order worth checking:
- `in2_layer_full_seq` stores V into the cache **after** attention, but attention reads V from the cache. For the
  new positions that reads calloc'd zeros on the first pass - wrong numbers, and if the read happens for a slot the
  visit never wrote, `in2_kv_v` can hand back a row that is not the one intended.
- visit numbering: sliding and full both call `in2_kv_advance` once per visit for 72 total against a 72-slot test
  allocation. Any path that calls a layer twice, or skips the advance on an early `rc` return, puts later rows in
  the wrong visit - and an off-by-one in the last layer walks off the end of the array.
- the `continue` on a NULL `in2_kv_k` result shrinks `n_key` without telling the sink softmax why.

Current verified baseline (double QK accumulation, no cache): 3-token prompt 26.9567, top1 149876; 5-token 29.2490,
top1 151451, 8,043/8,045 ms. IN2_KV=1 is safe today only because the sliding store path is a no-op on the numbers.

## Conv replacement diverges (2026-11-25)

With the batch conv left in place, the cache path is exact: 26.9567 / top1 149876 with and without IN2_KV.
Both attempts to make the conv position-local broke it:

| attempt | result |
|---|---|
| `in2_kv_conv_row_dyn` per row | **process died**, no output |
| prepend 3 cached raw rows to the batch, keep only new outputs | ran, but **22.0042 / top1 220** vs 26.9567 |

The prepend case is the informative one: with a fresh cache the history rows are zeros, which is exactly the
padding the batch conv applies, so the two must agree for the first forward - and they do not. Either the
prepended buffer is not what the batch path would have seen (dyn input `hx` rows, tap order, or the fact that
`in2_shortconv_dynamic` derives `dw` per row from `dyn_input`), or the cache path has a second divergence that
only surfaces once the conv reads anything at all.

Next step is therefore an isolated test, not an integration attempt: the same comparison the static-conv selftest
does (whole-sequence vs prepended-history, both against `in2_shortconv_dynamic`), with a non-zero history so a
tap-order bug cannot hide behind zeros.

The batch conv stays. It is correct for any full recompute, and full recompute is what runs today.

## Incremental step diverges - isolated (2026-11-25)

Smallest possible case, 1-token prefill then 1 step:

| | gen[0] (prefill) | gen[1] (one step) |
|---|---|---|
| full recompute | 220 / 27.9802 | 61840 / 19.2437 |
| incremental | 220 / 27.9802 | **43663 / 20.6381** |

`gen[0]` matching exactly is the useful part: **the prefill path is exact with the cache attached**, including the
conv-history push. The divergence appears on the step. The 3-token run matching earlier (149876/149917/151451) was
coincidence, not correctness - it agreed on tokens while carrying a 0.5-logit difference.

So the bug is in the read path, not the store path, and it is not the four convs (those are now cached and the
prefill proves the push order is right).

Next test, in order:
1. Dump `h` at the layer boundary (`IN2_DUMPX_LAYER`) for the step and for a 2-token recompute at the same position,
   compare layer by layer, and find the FIRST layer that differs. That splits "everything upstream is wrong" from
   "this layer's read is wrong" in one run, instead of guessing.
2. With that layer known, check in order: does `in2_kv_k/v(.., j)` for j < pos0 return the row the recompute used
   (compare against a dumped prefill row), then the bias `dist = gp - j`, then the tau/log-scale factor.
3. Only after a layer is proven equal, remove the conv history from suspicion - it is currently proven only for the
   prefill, where history is zeros.

Do not treat the cache as working until the 1+1 case matches to 1e-4. The tokens agreeing at 3+1 is not evidence.

## Divergence localized to the pass boundary (2026-11-25)

`IN2_LTRACE=1` prints the L2 of the last position's hidden state after every layer. Comparing a 2-token recompute
against a 1+1 incremental step, aligned on the step's last position:

```
LT il=06 ntok=2 l2_last=3.025383022053e+02   (recompute)
LT il=06 ntok=1 l2_last=8.184486260684e+02   (incremental)
```

Layers 0-5 are identical to twelve digits. Layer 6 is the **first layer of the second loop pass**, and that is the
only place it diverges. So the read path, the four convs, K/V, the relative bias and the sink are all correct for
pass 1 - what is wrong is what crosses between passes.

`in2_messages(prev_blocks, n_prev, prev_h, ...)` builds the message for a position from the PREVIOUS pass's hidden
states and block states. In a full recompute those buffers hold every position in the prefix; on a 1-token step they
hold one. Nothing caches them.

That is the same class of bug as the four convs, one level up: per-visit state that depends on the whole prefix.
The fix is to cache the pass-boundary h and block states per position (d_model and n_blocks x d_model per pass),
keyed the same way, and have the message construction read prefix rows from that cache.

Acceptance stays the same: 1+1 must match to 1e-4, then 3+1, then a 19-token prompt.

## Layer 6: hypotheses cleared (2026-11-25)

`IN2_VTRACE=1` prints the visit index for `il=6`. Prefill (pos0=0, ntok=1) and the step (pos0=1, ntok=1) both
report `vvis=6, 18, 30` - so the three passes of layer 6 address the same cache slots in both modes. Visit
alignment is correct and is not the bug.

Also cleared: the read path (positions are addressed globally and agree to 12 digits through layers 0-5), K/V
storage for pass 1, the four convs' history push (the prefill is exact), the relative bias and the sink.

Still open at layer 6. Remaining candidates, in the order that costs least to test:
1. **The K/V rows themselves.** Dump the row `in2_kv_k(kv, 6, 0)` in the step and the row the 2-token recompute
   used for position 0 at the same visit, and diff. If they differ, the prefill stored something different from
   what the recompute computes for position 0, which would mean the *prefill* path and the recompute path disagree
   at that layer even though the final h agreed - possible, because layers 0-5 comparing equal does not prove
   their stored K/V are equal.
2. **The MoE router.** Layer 6 is a fresh visit; if a row differs by 1e-7 the top-8 selection can flip and the
   output moves by far more than the input error. That would also explain a divergence that starts at exactly one
   layer and then persists.
3. Only after 1 and 2: messages/block states, which the trace has not yet implicated (a message bug would have to
   act at layer 6 specifically).

Do not spend more attempts on integration guesses. The next action is the row diff in (1).
