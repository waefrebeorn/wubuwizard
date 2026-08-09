# SD CPU Inference — 50 Speedup Gaps (researched 2026-08-09)

Research: 15 web searches (7-step cascade: conv micro-kernels → GEMM → memory
layout → threading → precision → math → compiler → pipeline) + the sibling
agent's wuburvc `knowledge/50_CPU_SPEED_FIXES.md` + live profiling on this
box (Ryzen 5 8645HS, 2 cores/4 threads, WSL2, anything-v5 Q4_0).

Measured baseline this session: UNet step 74s → **5.0s** (14.8×), VAE 7min →
**25s** (17×), blob load 60s → **1s**, full 12-step txt2img 152s → (in flight).

Status legend: **DONE** (measured, in the tree) / **TODO** (queued) /
**REJECTED** (measured or analyzed against OUR kernel — with reason).

## A. GEMM / conv micro-kernels (12)
1. **Register-blocked GEMM 2×8** (16 ymm accs, no spills). DONE — the 74s→5s
   step. [deep-kondah.com high-performance GEMM]
2. **Raw-blob quantized GEMM** — F16/Q4_0 weights read straight from the mmap'd
   GGUF, dequant per-block IN REGISTERS; zero dequant scratch, zero F32 weight
   copies. DONE — this is the user law AND the memory win (Q4_0 = 0.56 B/elem
   vs 4 B F32). [llama.cpp vec_dot; user law 2026-08-08]
3. **F16 xcol (activations)** — halves the im2col memory traffic; F16 act is
   what GPUs run anyway; parity vs sd.cpp unchanged (corr 0.990341 both ways).
   DONE. [tensorflow half precision]
4. **Wider j-block 16 with 2 rows** — 32 accs spills (32 ymm = all regs);
   measured slower. REJECTED. [own profile]
5. **4×8 / 6×16 register blocks** — same spill ceiling on AVX2; needs packing
   to pay off. TODO (see 12). [salykova.github.io gemm-cpu 16×6 kernel]
6. **Software prefetch of weight blocks** — measured 2× SLOWER (18B strided
   blocks pollute L1 with 64B lines). REJECTED for Q4_0; TODO for packed F16.
   [oneDNN/CUTLASS mainloop]
7. **Winograd F(2,3) 2D** — 2.25× fewer mults on 3×3 convs; sd.cpp paper claims
   2.76×/conv, 4.79× total. REJECTED for OUR big convs: the weight transform
   U=GgGᵀ costs ~30 adds × C_out×C_in per call (0.15s for the 1280-conv) ≈ the
   GEMM saving, and caching U (16× weight size) violates the no-weight-cache
   law. Applies to sd.cpp because their GEMM runs at 12% of OpenBLAS. The 1D
   F(2,3) (wuburvc t_winograd_avx.c) stays in RVC where K is tiny.
   [arxiv 2412.05781; dl.acm.org 10.1145/3472456.3472464]
8. **Direct conv (no im2col) with OC-blocking** — input tile stays L1-hot
   across 4+ output channels (input traffic ÷4); no xcol at all. The VAE's
   N=128 convs at 512×512 build 600MB xcol each — direct conv is the fix.
   TODO — top queue item. [wuburvc conv_avx OC_BLK=4; sciencedirect
   S1383762122002910: direct beats im2col for small channels]
9. **im2win** — im2col's cousin, −41.6% memory. TODO if 8 doesn't close it.
   [arxiv 2306.14320]
10. **2+ independent FMA chains** (ILP) — the 2×8 already gives 16 independent
    accs; compiler interleaves. DONE by construction.
11. **Kernel specialization per (k, stride)** — 3×3 s1 vs 1×1 vs s2 paths.
    PARTIAL: 1×1 goes through the same im2col; TODO to shortcut 1×1 (proj_in/
    out, skip convs) as a plain GEMM without xcol.
12. **GEMM packing (L2-friendly B)** — pack the raw W bytes per (k-chunk,
    j-block) into contiguous scratch. Raw-byte copy, not dequant; only helps
    after 5. TODO. [salykova gemm-cpu; XNNPACK packing]

## B. Memory / layout (8)
13. **mmap the GGUF blob** — DONE: was malloc+fread (1.5GB copy at load).
14. **madvise(WILLNEED|HUGEPAGE)** — DONE: the 60s was 375K × 4KB page faults
    (disk is 4GB/s); 2MB huge pages + readahead fix it.
15. **64B alignment for AVX buffers** — DONE (posix_memalign xcol/yt).
16. **Buffer pooling** (reuse xcol/yt across calls) — ~80 mallocs/step of big
    buffers; allocator is fast, but TLB/VA churn is real. TODO.
    [MemoMalloc]
17. **NHWC activations** — channels-last for direct convs; the xcol already
    sidesteps this for the GEMM path. TODO alongside 8. [IPEX/NNCF blog]
18. **Weight repack once at load** (XNNPACK weights cache) — raw-byte transpose
    of the blob; 1.5GB copy once. Conflicts with the no-weight-copy law —
    REJECTED unless the law is amended. [XNNPACK]
19. **Kill redundant memcpy** — the resblock copies x→xn for gn (in-place
    safety) and the transformer copies x→xorig; 2 full-tensor copies per
    block. TODO: fuse gn+silu+conv1 into one pass. [wuburvc 50-fixes #18]
20. **Huge pages for activation scratch** (xcol/yt >2MB) — madvise per-buffer.
    TODO micro-win.

## C. Threading (6)
21. **Thread-count sweep** — 4 threads on 2 physical cores (SMT×2); tested 4 =
    best for the GEMM. DONE (measurement). [llama.cpp thread tuning]
22. **OMP_MAX_ACTIVE_LEVELS=1** — stop nested oversubscription (matmul inside
    conv inside step). DONE by construction (single level everywhere).
    [Oracle nested parallelism docs]
23. **Static vs dynamic schedule** — static for uniform tiles. DONE
    (schedule(static) on the i-loop). [OpenMP schedule semantics]
24. **Core pinning / affinity** — WSL2 vCPU pinning is unreliable; NUMA is
    single-node. TODO if moved to bare metal. [oxmaint numa]
25. **One worker per physical core** (2, not 4) for pure-memory stages —
    measured: GEMM wins at 4 (FMA-bound), no stage is pure-memory here. DONE.
26. **Spin-wait barriers** — OpenMP barrier cost per call (~80 GEMM calls ×
    4 barriers); omp_set_nested false + OMP_WAIT_POLICY=active. DONE env.
    [llama.cpp spin locks]

## D. Precision / quantization (7)
27. **No weight dequantization at all** — DONE (the law; the biggest single
    philosophy win). [user law; llama.cpp]
28. **F16 weights with F32 accumulate** — DONE (raw F16 read, cvtph on load).
    [TF half precision]
29. **W8A8 INT8 GEMM** — 1.4–2.4× on AVX2, but needs per-channel scales +
    calibration; breaks the raw-blob Q4_0 model. TODO (new model format).
    [openreview SklzIjActX]
30. **W4A8 mixed** — up to 3×; same caveat. TODO. [CVPRW 2025 dual precision]
31. **Groupwise Q4_K / Q6_K** — the model is Q4_0; re-quantizing is a model
    change. TODO. [emergentmind llama.cpp]
32. **Quantized intermediates** (q8 residual chains) — activations already F16
    via xcol; going q8 would break the 0.99 corr parity. REJECTED.
33. **BF16** — no 2× on Zen4 (no bf16 FMA advantage over fp32 AVX2). REJECTED.

## E. Math / activations (6)
34. **fast_expf (IEEE-754 bit trick)** for softmax — DONE (attn 2.3s→0.9s).
    [QuAKE dotLLM#55]
35. **FTZ+DAZ flush denormals** (MXCSR 0x8040) — DONE. [Intel oneMKL
    denormals; wuburvc 50-fixes #36]
36. **FMA contraction** (-ffp-contract=fast, default) — DONE.
37. **-ffast-math** — risks the F16 path's tiny diffs + silu/gelu edges;
    test per-layer. TODO/risky. [gcc docs]
38. **Vectorized softmax (max-subtract in ymm)** — the attn core already
    max-subtracts in AVX2. DONE.
39. **Approximate GELU/silu poly** — silu = x·σ(x): a degree-3 poly on the
    F16 path could shave the ffn. Measured earlier: the gelu loop is NOT the
    ffn bottleneck (linears are) — REJECTED (no win measured).
    [wuburvc 50-fixes #39]

## F. Compiler / runtime (6)
40. **-march=native** — DONE.
41. **-mno-avx512f on Zen4** — AVX-512 is double-pumped 256-bit here;
    measured 5.9→6.3s with zmm. DONE (keep AVX2). [own measurement]
42. **-flto -funroll-loops** — DONE (in build).
43. **PGO** — typical 5–30%; wuburvc has build_pgo.sh. TODO (needs a training
    run per build). [learn.microsoft PGO]
44. **mimalloc/jemalloc** — the per-call xcol/yt allocs are few (80/step);
    measured allocator cost negligible. REJECTED for now. [mimalloc]
45. **-mno-xsave etc. CPU flags hygiene** — micro; TODO with a flags sweep.
46. **Static linking / -Wl,--gc-sections** — start time only. TODO.

## G. Pipeline / algorithm (6)
47. **ToMe token merging** (r=0.3–0.5) — up to 2× on the transformer, minimal
    visual impact per CVPRW; but the boss says NON-SLOP — the merge smears
    fine anime detail. OPTIONAL flag, default OFF. [tomesd; CVPRW 2023]
48. **Sampler/steps** — DDIM holds quality at 12 steps; the tool's discrete
    schedule under-denoises at 12. TODO: expose DDIM + CFG rescale.
    [stable-diffusion-art samplers]
49. **Skipped-step CFG** — run CFG only on every other step; ~1.5× on the
    UNet budget. TODO (config flag). [baseten SDXL guide]
50. **Online-softmax attention** — the 64×64 attn is small; flash-style
    tiling adds overhead on CPU. REJECTED (measured attn already 0.9s).
    [flash-attention; hermes mlops-flash-attention: CPU prefers
    memory-efficient, not flash]
51. **fused gn+silu+conv1 epilogue** — one pass instead of three buffers.
    TODO (with 19). [oneDNN fused primitives]
52. **Async I/O / double-buffer the noise+prompt** — load already 1s. N/A.

## Verdict
The box is 2 cores/4 threads — a 12s/step-class machine at 40% GEMM peak.
The step floor is GEMM-bound (FMA-limited); the remaining big levers are
direct-conv for the VAE (#8, ~10s off), the 1×1 shortcut (#11), buffer
pooling (#16), and PGO (#43). Everything above is measured or sourced; no
claims without a timestamp.
