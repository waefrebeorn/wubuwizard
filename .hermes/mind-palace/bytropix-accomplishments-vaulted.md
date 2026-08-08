# bytropix — Accomplishments Vaulted (May 26, 2026)

> All completed work vaulted to clear the board for a fresh gap analysis.
> Triple Devil's Advocate review: claims verified against actual code state.

## 🏆 Completed Campaigns

### Campaign 1: Accuracy Recovery (May 25-26)
**Status:** ✅ All cells verified
- [x] **tgt_wrap removal** — Removed from wubu_ssm.c:1476,1664 (attention scores). Root cause: fmod(x+π,2π)-π wrapped large positive dot products (best matches) to negative, inverting softmax probabilities.
- [x] **Chunked SSM FP accumulation fix** — SSM_CHUNK_MIN changed from 2→4096. Root cause: CS>1 amplifies FP errors across 30 SSM layers, corrupting token selection.
- [x] **Verification** — RAW "capital of France?"→"Paris". CHAT produces coherent thinking output. Cos-sim vs llama.cpp ~0.994.
- [x] **Commit:** `066ff74 fix(cpu): two accuracy bugs`

### Campaign 2: Prefill Speed (May 26)
**Status:** ✅ 3.9x improvement (1.1→4.3 tok/s)
- [x] **Batched SSM projections** — quantized_matmul_batched replaces per-token quantize+matmul loop. Weight read ONCE from RAM per layer.
- [x] **Batched SSM output projection** — Single Q6_K weight pass for all N tokens.
- [x] **Batched GQA projections** — Q+K+V in single weight pass.
- [x] **Output projection nested OMP fix** — omp_set_num_threads(1) inside outer parallel region. Eliminated 16-thread oversubscription on 4-core i5. 1609ms→31ms (52x).
- [x] **AVX2 L2 Norm + RMSNorm** — Vectorized sum-of-squares with _mm256_fmadd_ps reduction.
- [x] **AVX2 Conv1d depthwise** — 8-channel SIMD, broadcast kernel, FMA.
- [x] **Commit:** `(pending: batched projections, AVX2 norms/conv1d, OMP fix)`

### Campaign 3: DRAM Characterization (May 26)
**Status:** ✅ Measured
- [x] **tREFI probe** — 5M probes on i5-8365U. TSC: 1.896 GHz. DRAM refresh period: 7.62µs (2.3% from expected 7.8µs). Spike rate: 2.4%. Verdict: PERIODIC.
- [x] **Impact assessment** — ~10ms/token wasted on refresh stalls (negligible vs 400ms decode). Tailslayer hedging would save <1%.
- [x] **Vault doc:** `vault/tailslayer-dram-hedged-reads.md`

### Campaign 4: Baseline & Tooling (May 25-26)
**Status:** ✅ Complete
- [x] **CPU battleship doc** — `.hermes/battleship-cpu-inference.md` with Row A baselines
- [x] **Mind palace adoption** — state.md, index.md, plan.md in `.hermes/mind-palace/`
- [x] **CPU infer skill** — `mlops/cpu-inference-optimization/SKILL.md` updated
- [x] **Accuracy verified:** Both RAW and CHAT modes produce correct output
- [x] **Benchmarked:** llama.cpp comparison on same hardware

---

## DA Review Sign-off

All claims above cross-referenced against:
- `git log --oneline cpu-optimize-may26` (3 commits)
- `git diff src/wubu_ssm.c` (tgt_wrap removal, SSM_CHUNK_MIN, projections)
- `git diff src/wubu_model.c` (OMP fix, profile extension)
- Benchmark output (PROFILE=1 runs)
- trefi_probe binary output

Signed: Hermes Agent, May 26, 2026
