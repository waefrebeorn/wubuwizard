# Slerm notes — upstream code pulled 2026-11-20

Clones live OUTSIDE the engine tree at `C:/Users/eman5/WuBu/repos/_sota_research/`.
Read them, reimplement the mechanisms in C11. Never link or vendor them.

## DeepInfer — DSpark confidence-budgeted speculation

Source: `_sota_research/DeepInfer/deepinfer/model_executor/layers/tilelang_ops/{asc,cuda}/schedule_mtp_varlen.py`.
`compute_seqlens_q_by_confidence_rates_tl` computes a **cumulative product of per-draft
confidence rates** along each sequence (`acc *= prev_confidence_rates[prev, j]`), then
uses that cumulative rate plus `total_q - num_seqs` extra budget to pick each
sequence's verification length `seqlens_q`. Invalid/remapped sequences get `acc = 0`.

C11 mapping for LP-07: store draft confidences in a flat `[n_seq][nextn]` array,
prefix-multiply in F32, rank sequences by cumulative confidence, allocate the
fixed query budget longest-first, clamp to `nextn`. No GPU kernel needed for the
small batch sizes wubu runs; the verifier stays unchanged.
Also read `csrc/engine/zos.cpp` for the CPU/device overlap idea: distinguish fresh
rows from rows remapped onto previous outputs, validate metadata, keep staging
tensors pinned so scheduling hides under compute.

## kvcached — all-or-nothing exact block claims

Source: `_sota_research/kvcached/csrc/inc/exact_blocks.hpp` and
`_sota_research/kvcached/csrc/page_allocator.cpp`.

`claim_exact_blocks(free_blocks, requested)`:
1. reject negative or duplicate requested IDs;
2. reject duplicate IDs already in the free list;
3. check **every** requested ID is available before touching the list;
4. build a replacement `remaining` vector first, then commit — so a failure or
   throw leaves the allocator unchanged.
`alloc_exact()` is a thin wrapper; the Python manager adds preflight against
physical page capacity before calling it, and rolls back created pages on failure.

C11 mapping for LP-05: `can_alloc_exact()` = validate + membership check only;
`alloc_exact()` = preflight, build a new free-ID array, commit with a single
swap, return distinct status codes for bad IDs / capacity / quota. Gate must
assert no partial mutation on each failure path.

## What this changes in the roadmap

- LP-07 gets an executable specification, not just a paper: prefix-confidence
  budget allocation with a fixed query budget.
- LP-05 gets a concrete invariant and failure taxonomy from `exact_blocks.hpp`.
- Both are host-side algorithms that fit the memory-bandwidth-bound C11 spine
  without adding a dependency.
