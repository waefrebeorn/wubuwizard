# MTP Acceptance Campaign — Gameplan (May 27, 2026)

**Objective**: MTP acceptance >50% within 11GB WSL RAM

## Implemented: IQ Raw-Quant Cache (Phase C ✅) — v2

**Status**: 16% acceptance (measured May 27) — matches IQ2_XXS baseline within noise
- 16-slot LRU, ~24MB heap — stores native IQ2_XXS/IQ3_XXS bytes (no dequant/requant)
- Fill = memcpy from blob. Use = original `quantized_matmul_from_q8` vec_dot path
- v1 (Q8_K cache) had 3% acceptance loss from requant + Q8_K×Q8_K dot. Fixed by storing raw IQ bytes.
- 24MB vs 41MB: 57% smaller cache
- No buffer overflow, no crash on exit

**Key finding**: IQ raw-quant cache preserves native accuracy (16% vs 17% baseline).
Reached 15% threshold → next step is P3 (expert prefetch matrix).

## Core Insight: Q8_K Draft Head (Achieved, Below Target)

blk.40's IQ2_XXS weights are ALREADY in the mmap'd GGUF blob (part of 10.7GB). No separate alloc needed.

**DA found**: F32 lazy dequant is 10ms/expert × 8 = 80ms first draft, plus 96MB LRU overflows 11GB WSL. 
**CORRECTION**: Use **Q8_0** instead of F32.
- Q8_0 cache: 3.4MB/expert (vs 12MB F32). 8-slot LRU = 27MB heap. Fits.
- Q8_0 vec_dot: 4× faster than F32 SGEMM. ~1ms/expert matmul.
- Q8_0 acceptance estimate: 25-35% (from vault paper).
- Combined: 25ms first draft, 8ms subsequent. At K=4 speculation: 25+8+8+8=49ms draft, 380+25×3=455ms verify = 507ms for 4 tokens = 7.9 tok/s at 100% acceptance. At 50% acceptance (2 of 4): 3.9 tok/s > 2.8 baseline. **VIABLE**.

**Lazy Q8_0 cache**: Dequant IQ2_XXS → Q8_0 on demand for 8 selected experts. LRU cache (8 slots) for reuse.

## Game Plan (recursive DA loop: implement → review → palace → loop)

### Phase 1: Q8_0 Lazy Dequant Cache (1 session)
```
State: blk40.moe.ffn_gate_exps_q = IQ2_XXS blob ptr (current)
Goal: dequantize 8 router-selected experts to Q8_0 on demand

Implementation:
1. Allocate 8-slot LRU cache: q8_gate[8][BLOCK_COUNT][sizeof(block_q8_0)], same for up/down
   - Each slot: 3 × 1.13MB = 3.4MB. Total 27MB heap. Fits 11GB WSL.
2. In wubu_moe_forward, AFTER router selects 8 experts on draft path (blk.40 only):
   - Check LRU cache for each expert (uint64 hash of expert index)
   - Cache miss → dequant IQ2→temp F32 → requant temp F32→Q8_0, store in LRU
   - Cache hit → use stored Q8_0 pointer
   - Use Q8_0 vec_dot for gate/up matmuls (exists in quantized_dot_generic.c)
3. Only active for MTP draft path (blk.40). Main model layers 0-39 stay IQ2_XXS.

Memory: LRU cache = 8 × 3.4MB = 27MB heap. Acceptable.
Speed: ~2.5ms per cache miss (dequant + requant). ~1ms/expert matmul with Q8_0 vec_dot.
Acceptance: ~25-35% (Q8_0 has 0.5% cos-sim error per matmul vs IQ2_XXS)
```

### Phase 2: DA Review → Update Mind Palace
- Measure acceptance rate with F32 blk.40 vs IQ2_XXS blk.40
- If <30%: need deeper changes (prefetch matrix next)
- If >50%: MTP viable → measure throughput

### Phase 3: Expert Prefetch Matrix (1 session)
```
Matrix: prompt_hash → expert_indices[layer][0..7] for layers 0-39
Build: each forward pass records expert selections per layer
Use: next forward with same prompt prefix → prefetch those experts

Implementation:
1. Hash prompt prefix (first 16 tokens) with xxhash
2. On each layer forward, save selected expert indices
3. On subsequent runs, prefetch from LRU cache during SSM forward
```

### Phase 4: DA Review → Update Mind Palace
### Phase 5: C Voodoo Pass (1-2 sessions)
```
Targets:
1. Top-8 selection: replace bubble-sort with SIMD comparison tree
2. Softmax: fast_expf intrinsic (cut expf latency from 20→5 cycles)
3. LRU cache: uint64 hash lookup, branchless eviction
4. Dequant: Duff's device on IQ2_XXS block dequant
5. Routed expert dispatch: computed goto instead of function pointer
6. _mm_prefetch: call exactly 4KB before use (tune to DDR4 timing)
7. Aligned loads: guarantee 32-byte alignment for IQ2 blocks
```

### Phase 6: DA Review → Update Mind Palace
### Phase 7: Math Vault Sweep (2-3 sessions)
```
Sweep ALL 2334 lines of unused math:
- wubu_poincare_gqa.c (257 lines) — wired into GQA forward
- wubu_mobius_linear.c (200 lines) — Möbius transformation for MoE expert projection
- wubu_hyperbolic_output_proj.c (243 lines) — hyperbolic output projection
- wubu_mobius_gyrate.c (75 lines) — gyration for rotation in hyperbolic space

Each: find if applicable to MTP draft head, wire in if cos-sim improves
```

### Phase 8: Final DA Review → Battleship Finalized

## Recursive DA Loop (applied after EVERY phase)

```
1. IMPLEMENT → change code
2. DA REVIEW → 4-phase check (CLAIM→VERIFY→RISK→MITIGATE)
3. MIND PALACE → update state.md, ARCHITECTURE.md, battleship cells
4. RECURSE → if acceptance <50% or tok/s < expected, return to Phase N+1
```

## Acceptance Thresholds

| Phase | Target | Result | Next |
|-------|--------|--------|------|
| P1 (IQ raw-quant cache) | >15% acceptance | **16% measured** ✅ | → P3 prefetch matrix |
| P3 (prefetch matrix) | >30% acceptance | ⬜ | → P4 DA review |
| P5 (C voodoo) | >40% acceptance | ⬜ | → P6 DA review |
| P7 (math vault sweep) | >50% acceptance | ⬜ | → New hardware required |

## C Voodoo Targets (specific)

| Trick | Where | Before | After | Expected Gain |
|-------|-------|--------|-------|:------------:|
| Duff's dequant | iq2_xxs_dot_block | for-loop per block | do-while unroll | 1.5× |
| SIMD top-8 | wubu_moe_router_only | bubble-sort | _mm256_cmp_ps | 4× |
| fast_expf | softmax | expf() | expf intrinsic (bit trick) | 4× |
| Computed goto | quant matmul dispatch | function pointer | static goto table | 1.2× |
| LRU hash | F32 cache | linear scan | uint64 open addressing | 2× |
| Aligned prefetch | _mm_prefetch | any offset | cache-line-aligned stride | 1.3× |
| Branchless eviction | LRU | if-else | arithmetic min | 1.5× |