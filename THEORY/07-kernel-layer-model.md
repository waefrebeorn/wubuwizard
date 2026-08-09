# THEORY/07 — The Kernel-Layer Model: Byte-Budget, Arena, Quant Cascade

> The Brain runs as a kernel-layer C11 engine. This doctrine redesigns the
> 35M model from “research allocator” to “kernel component”: every byte is
> accounted for, one allocator of record owns all memory, and precision is a
> negotiated cascade (not a fixed typedef). Derived from own-c-kernel-eng,
> wubu-kvfs-namespace, and the mGBA timing lessons in research/062.

## The kernel contract (non-negotiables, from own-c-kernel-engineering)

1. **One allocator of record.** The kernel exposes `mem_alloc(size, align)`
   (a bump over the boot heap, backed by the physical page pool). Every
   submodule — weights, KV, activations, intermediates — routes through it.
   A second libc `calloc` that is never initialized returns NULL and kills
   the boot silently. → The model uses a **single arena** (`wubu_arena_t`)
   initialized from `mem_alloc`; no bare `malloc` in `wubu_load`.
2. **Alignment as correctness, not speed.** AVX-512 needs 64-byte alignment;
   GQA/KV striding needs block-granularity. Misalign one slab and the
   `lea (%rip)` load becomes a `mov (%rip)` garbage read → intermittent
   hal. → All arena slabs carry an explicit alignment tag (f32=4, f16=2,
   int8=1, AVX=64).
3. **Opaque seams at every boundary.** The public header (`wubu.h`) exposes
   `typedef struct wubu_block wubu_block_t;` + accessors. The engine never
   sees raw struct layouts of its peers. → The weight tensor type is opaque;
   the consumer calls `wubu_tensor_bytes(t)` / `wubu_tensor_f32(t, out)`.
4. **No libc in the hot path.** `<math.h>` is a *boot-time* luxury; the
   speed kernel runs `expf_ps`, `rsqrtf_nr` (Newton-Raphson), `hmul`
   (hex-float mul) inline. → The forward body uses the own-C math shims in
   `include/wubu_math.h`; `libm` is linked only for tests.

## Byte budget — the 35M as a kernel slab map

The released checkpoint is F32 (`seed-sft2.safetensors`). The kernel does
NOT keep it F32. The arena is carved once at init, sized in bytes, and each
slab is tagged with its precision + alignment:

```
┌────────────────────────────────────────────────────────┐
│  wubu_arena_t  (one bump from mem_alloc, 64-byte aligned)   │
├────────────────────────────────────────────────────────┤
│ [0]  embedding/lm_head  F16  [16384 × 448]  = 14.73 MB  │ ←  tied, so ONE slab
│ [1]  final_norm  F32    [448]         = 1.8 KB           │
│ [2]  blocks[12]                                           │ ← array of opaque handles
│   [2.0]  q_proj  F16 [448×448]  = 798 KB  │ int4 449 KB │ ← quant cascade
│   [2.0]  k_proj  F16 [448×64]   = 112 KB  │ int4  56 KB │ (50% BW win @ decode B=1)
│   [2.0]  v_proj  F16 [448×64]   = 112 KB  │ int4  56 KB │
│   ...per-block (12 blocks)                               │
│ [3]  selectors[12] F16 [12×448]  = 21 KB                  │
│ [4]  KV cache  Q8_0  [max_ctx×kv_heads×head_dim×2×layers]  │ ← tiered at init
│ ...intermediates (activations, attn scores, swiglu) are TRANSIENT — │
│     borrowed from a scratch slab, freed/reset per step, never owned    │
└────────────────────────────────────────────────────────┘
```

Concrete byte budget for the 35M (F32→F16 weights, Q8_0 KV):

| Component | F32 | F16 (kernel default) | Savings |
|---|---|---|---|
| embedding (tied) | 29.5 MB | 14.7 MB | **2×** |
| q/k/v/o/gate_up/down ×12 | 31.1 MB | 15.5 MB | 2× |
| all weights | 60.6 MB | 30.3 MB | **~2× = 30.3 MB** |
| KV @ ctx=2048, F32 | 136 MB | — | — |
| KV @ ctx=2048, Q8_0 | 136 MB | 48 MB (Q4)→77 MB (Q8) | **~1.8×** |
| KV @ ctx=2048, F16 | — | 68 MB | |
| **Total resident (F16+Q8KV)** | 202 MB | **~100 MB** | **~2×** |
| **Total resident (F16+Q4KV)** | 202 MB | **~80 MB** | **~2.5×** |

The Roofline crossover (doc 013, WUBU_ROOFLINE=2607.02558) says: at decode
**B=1, s≤2048**, weights are the dominant footprint → compress to **int4**
if the kernel is compute-bound, **F16** if BW-bound. The selector is
`wubu_kv_select(cfg, P_params, B, s)` → returns `wubu_kv_choice_t` with
both KV scheme AND weight bits (4/16). The arena is carved *after* the
choice: the weight slab is `P_params × weight_bytes_per_elem`.

## The quant cascade (the kernel's negotiation)

```
wubu_model_t (opaque)
  └─ wubu_arena_t *arena        ← single allocator of record
  └─ wubu_kv_choice_t  choice   ← negotiated at init (P, B, s, BW)
  └─ int weight_bits            ← 4 (int4) or 16 (f16), from choice
  └─ int n_gqa_layers           ← S6: runtime KV cap (gqa_max_ctx)
  └─ wubu_kvfs_t *kvfs          ← KV-as-FS (ADR-003)

wubu_load(path):
  1. probe_dims(path)              → WUBU35_DIMS (S1)
  2. kv = wubu_kv_select(roofline, P=35e6, B=1, s=gqa_max_ctx)
  3. arena = wubu_arena_create(total_bytes_from(variant, kv))
  4. for each tensor: arena_push(name, data, dt)
        → dt = F16 if weight_bits==16, INT4_PACKED if weight_bits==4
        → slab aligned to .align
  5. wire kvfs layer handles (resolve-once)
```

The forward then dispatches on `weight_bits`:
- `weight_bits == 16` → `gemv_f16` (AVX F16C)
- `weight_bits == 4`  → `gemv_int4` (nybble unpack, per own-c-kernel pitfalls:
  pack unsigned 0..15 = `round(w/scale)+8`, unpack `(b&0xF)-8`)

## KV tiering as a memory-space contract (P9)

```
g_kv_scheme (global, set at init):
  WUBU_KV_F32 → 4 bytes/elem  (fallback; full precision)
  WUBU_KV_F16 → 2 bytes/elem  (baseline; bandwidth-adequate)
  WUBU_KV_Q8  → 1.125 bytes/elem (near-lossless block-32, the kernel default)
  WUBU_KV_Q4_0 → 0.56 bytes/elem (long-context; Q4_0 block-32)
  WUBU_KV_KIVI → ~1.03 bytes/elem (KIVI per-token V: K!=V)
```

The scheme is a **memory-space decision**, not just a speed knob: at
`ctx=32768` the F32 KV cache is **1.75 GB** — larger than the entire model.
The kernel tiers: KV slab is allocated at `kv_bytes_per_elem` and the
`kv_cache_*` macros in `wubu_model.h` already dispatch on `g_kv_scheme`.
The redesign makes the **choice** kernel-resident (one allocator, one
decision, logged).

## Event-driven timing (the mGBA P2/P6 lesson)

> “No emulator has ever come close to emulating prefetch accurately.” —
> research/062, hop 6 (mGBA)

A fixed `for (t=0; t<seq; t++)` decode loop is a static cycle table. The
kernel redesign makes timing event-driven: the decode head **yields control**
after each produced token, returning the KV slab offset consumed. A
scheduler (the amoeba loop, or the Styx namespace) decides when to re-enter:

```
wubu_kvslab_t {               // per-layer KV slab (resolve-once handle)
  void *base; size_t off;     // off = tokens written (the event cursor)
  int bits;                   // WUBU_KV_* precision tag
};
wubu_decode_step(m, slab, &n_produced):  // produces 1-token, advances off
  → returns the slab offset; caller re-enters when ready
```

This is the *same pattern* as KVFS handles (`wubu_kvfs_open` →
`wubu_kvfs_handle_read`): resolve once, advance a cursor, hand back
control. The difference: now the cursor is a **timing event**, not a
position. Idle cycles (waiting on I/O, waiting on the user, waiting on a
parallel decode) are filled by the prefetcher (P5: idle-opportunistic),
not wasted on a fixed tick.

## The slab allocator for the kernel

```c
// include/wubu_arena.h
typedef struct { uintptr_t start, end; uintptr_t pos; } wubu_arena_t;
wubu_arena_t *wubu_arena_create(size_t bytes);          // one mem_alloc
void  *wubu_arena_push(size_t sz, size_t align);        // bump, zero-retained
void   wubu_arena_reset(wubu_arena_t *a);               // rewind cursor (transient)
void   wubu_arena_free(wubu_arena_t *a);                // single free
```

- Weights slab: allocated ONCE, never reset (the frozen cartridge).
- KV slab: reset only on `n_layers` change (model swap = cylinder reload).
- Intermediates slab: reset per step (the scratch workspace).

The 14 separate `calloc`s in `wubu_load` (b->x, b->q, b->k, …) collapse to
ONE 4 KB scratch slab per block — re-borrowed, never individually owned.
14 × `calloc` → 1 × `wubu_arena_push` per block.

## ADR-004: the model struct goes fully opaque

The current `wubu.h` exposes `wubu_block_t` layout (q_proj, k_proj, …).
That violates axiom 3 (opaque seams). ADR-004:

```
// wubu.h (public) — the opaque contract
typedef struct wubu_model wubu_model_t;
typedef struct wubu_block wubu_block_t;

wubu_model_t *wubu_model_create(const char *path);  // probes + arena + KV
void         wubu_model_destroy(wubu_model_t *m);
float        wubu_model_forward(wubu_model_t *m, int token, int pos);
const wubu_kv_choice_t *wubu_model_choice(const wubu_model_t *m);  // introspection
wubu_kvfs_t  *wubu_model_kvfs(wubu_model_t *m);                    // KV-FS handle
```

The `.c` struct holds the arena, the byte-budget slab table (name→base,
size, precision, align), and the resolve-once KVFS handles. A Styx export
(/kv/layer_02) reads the slab table — zero forward coupling.

## Pitfall: the slab-alignment trap (kernel-level)

AVX-512 GEMM reads 64 bytes at a time. A 448×448 F16 slab is 399 KB — a
multiple of 64, fine. BUT the int4-pack slab is `ceil(d*d / 2)` = 99824
bytes → 99824 % 64 = 48. Unaligned tail → the unrolled `vmovdqa` on the
last block reads past the slab into the next slab → silent corruption.
Fix: **pad each slab to alignment** (arena_push rounds up). The byte-budget
table must include the padding (30.3 MB → 30.5 MB with 12 weight slabs
padded). Budget for 2% padding; verify with `assert((uintptr_t)base % 64 == 0)`.

## Verification (the kernel must be byte-exact)

- `tools/test_wubu_kernel.c`: loads `seed-sft2.safetensors`, runs the
  quant cascade, and asserts:
  - `arena.bytes_used == expected_budget` (within padding tolerance)
  - every slab base is aligned to its `.align`
  - a forward pass with F16 == a forward pass with F32 within maxdiff
    (weights dequantize identically) — the invariance test from
    own-c-kernel-engineering
  - `wubu_kv_select` returns the documented scheme for each (B, s) pair
  - the arena has exactly 3 reset-points (init, model-swap, step), not 14
- Byte budget printed: `Arena: 30.3 MB weights (F16) + 68 MB KV (F16) = 98.3 MB`
  vs the naïve 202 MB.

## Cross-model bridge (P14 — the merge path)

The slab table is a mount table (KVFS pattern). A cross-model KV transfer
(AN25 arXiv:2608.03893) becomes a `wubu_arena_transfer(src, dst, map)` —
the slabs are addressed by name, so a linear-mapped layer copies its slab
bytes directly. No re-prefill needed when the slabs match.
