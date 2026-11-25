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
