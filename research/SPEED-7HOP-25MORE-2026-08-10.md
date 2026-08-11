# 25+ More Improvements — 7-Hop Research Chain (2026-08-10)

Kevin Bacon chain: each hop builds on the previous finding, ending at
actionable improvements for wubuwizard on the CM4 (A72, 4×1.5GHz, 1MB
shared L2, memory-bandwidth-bound — Roofline 2607.02558).

Chain: im2win → SDXL guide → DeepCache → TAESD → LCM → A72 microarch →
HAWC Winograd → llama.cpp exp2 → OpenVINO/ToMe → GGUF quants → tiled VAE →
justine matmul → Turbo-VAED → direct-conv anatomy → Arm I8MM

---

## HOP 1: im2win / NHWC (arXiv 2408.00278)
"High Performance Im2win and Direct Convolutions using Three Tensor
Layouts on SIMD Architectures" — **up to 355% speedup over NCHW layout.**
Optimized im2win/direct hit 95%/94% of machine theoretical peak.

**Why it matters here:** our #1 measured VAE cost is the im2col build
gather (29s CPU-s on up.1.upsample) — every 4-byte read from a 1MB-strided
channel plane misses. NHWC/channel-last turns those gathers into
sequential channel runs.

### 1. im2win NHWC for the VAE convs
Replace im2col with im2win: reorganize input into dot-product windows,
flatten unique window elements per receptive field. NHWC layout makes the
k-loop channel-contiguous → vectorizable loads instead of gathers.
Est: build phase 29s → ~8s on the big upsample convs.

### 2. Direct convolution with NHWC (channel-last)
For C≥128 VAE layers, NHWC direct conv vectorizes across channels with
128-bit loads (4 floats = 4 channels), skipping im2col entirely.
Paper: im2win+NHWC 355% vs NCHW; direct+NHWC strong for large channels.

### 3. CHWN8 layout for 8-wide channel blocks
CHWN8 groups 8 channels per innermost run — matches A72's 128-bit NEON
(8×F16 or 4×F32). Consider for the F16-x GEMM's x layout.

## HOP 2: SDXL optimization guide (felixsanz.dev)
"Ultimate guide to optimizing Stable Diffusion XL" — full technique
basket with measured numbers.

### 4. TinyVAE / TAESD decoder swap
Replacing the 49.5M-param VAE decoder with the 1.2M-param TAESD decoder
is the single biggest VAE lever (below). Guide: TinyVAE + disable CFG +
fewer steps = the fastest config.

### 5. VAE FP16 fix
Run the VAE in FP16 with a few high-sensitivity layers kept FP32 —
the guide's standalone VAE speedup (12.9s vs slower baseline). We already
run F16-x activations in convs; extending F16 storage to the VAE's
groupnorm/silu intermediates cuts traffic further (parity to verify).

### 6. Sequential CPU offload / batch processing
Lower memory pressure lets the 5.8GB box keep more resident. Minor speed
on CM4 (no GPU), but relevant if we tile.

## HOP 3: DeepCache (arXiv 2312.00858)
"Accelerating Diffusion Models for Free" — **2.3× for SD1.5 with only
0.05 CLIP-score decline**, training-free. 4.1× for LDM-4-G.

### 7. DeepCache: cache high-level UNet features across steps
U-Net's high-level (low-res, deep) features change slowly across denoise
steps; low-level features need updating. Cache the deep blocks, reuse
across steps, only recompute the cheap shallow path. 2.3× on the UNet
(the biggest total-time chunk).

### 8. Cache-interval tuning (cache every N steps)
Cache cadence (interval=10 typical) controls the speed/quality knob.
Adaptive coarse-grained caching (feature-similarity based, AAAI 2026)
adjusts intervals dynamically.

## HOP 4: TAESD (madebyollin/taesd)
Tiny AutoEncoder for SD: **decoder 49,490,179 → 1,222,531 params (40×)**,
"decode at (nearly) zero cost". Just Conv+ReLU resblocks + 2× upsample.
ONNX ops: Add/Conv/Div/Mul/Relu/Resize/Tanh — trivially portable to C11.

### 9. TAESD decoder as an optional fast decode path
Same latent API (4×(H/8)×(W/8) → 3×H×W). Trade: "tends to fudge fine
details" — preview-quality, not final-quality. Perfect as a live-preview
decoder during sampling + full VAE only for the final image, OR as a
user-selectable fast mode. **Decode 492s → single-digit seconds.**

### 10. Bounded receptive field → tiled decode without seams
TAESD's bounded receptive field permits tile-based decoding (with
overlap) for arbitrarily large images — enables 4K+ on the 5.8GB box.

## HOP 5: LCM / LCM-LoRA (arXiv 2310.04378, 2311.05556)
Latent Consistency Models: **2-4 step sampling** vs 20-50.

### 11. LCM 4-step sampling for the UNet
4 steps × ~120s/step → ~8× total speedup if we adopt an LCM-distilled
checkpoint. 32 A100-hours to distill; LCM_Dreamshaper_v7 is a drop-in
SD1.5 checkpoint.

### 12. LCM-LoRA: universal acceleration module
LCM-LoRA plugs into ANY SD checkpoint (no re-distillation) for 4-step
sampling. Training-free at runtime — just weight merging at load.

## HOP 6: A72 microarchitecture (chipsandcheese.com)
"ARM's Cortex A72: aarch64 for the Masses" — the hardware truths:
- **single 128-bit load per cycle** → wide loads essential
- **~8 bytes/cycle from L2** per core → L2 traffic is the wall
- write bandwidth lower than read (Graviton1) → minimize stores too

### 13. 128-bit-only loads in every inner loop
Avoid 64-bit NEON loads; pack to 128-bit. The F16-x GEMM already loads
4×F16 = 64-bit — pair loads to 128-bit (vld1q_u16 ×8) where possible.

### 14. L2 traffic budget per conv (roofline re-check)
8B/cycle/core × 4 cores × 1.5GHz ≈ 48 GB/s L2. Our 18 GFLOP/s GEMM moves
~6B/FMA ≈ 108 GB/s — but x re-reads hit L1, so effective L2 traffic is
the W stream (~4B/FMA). This explains why j-chunking W into L2 (590KB
chunk) was the win, and why Winograd's 1.8×-bigger V lost.

### 15. PRFM/PLD software prefetch hints
A72 supports PLD/PLDW/PRFM hints; L2 prefetcher exists but can be tuned.
Insert PRFM at j-chunk boundaries to overlap DRAM→L2 W fetch with GEMM.

### 16. Minimize write-allocate traffic
Write bandwidth < read bandwidth on A72-class cores. Our im2col writes
are already sequential 2-byte stores (line-fill friendly); consider
non-temporal stores (STNP) for the xcol buffer that is written-then-read
once — skips the allocate on the second pass.

## HOP 7: HAWC — FP16 Winograd on ARM many-core (APSys'22)
"Optimizing Half Precision Winograd Convolution on ARM Many-Core" —
**avg 10.74×, up to 27.56×** vs state-of-the-art. Three insights that
fix our Winograd failure:

### 17. Custom data layout: T_size-outermost transformed tiles
HAWC stores transformed inputs I' as [T/Cb][C_in/Cb][T_size][Cb][Tb] —
the TILE is outermost so the transform-space GEMM streams tiles, not
per-tile scalar work. Our Winograd recomputed U per co-chunk (up to 64×);
HAWC's layout makes U computed once, reused across all co.

### 18. Overlap input/output transforms with the GEMM
Input/kernel/output transforms are memory-bound; the GEMM is compute-
bound. HAWC overlaps them (double-buffering) so transforms hide under
GEMM time. Our Winograd ran them serially per tile.

### 19. F16 transform-space storage (halves traffic)
FP16 in Winograd doubles FMA rate on Neoverse-N1 (FMLA v1.8H) AND halves
data movement. **Caveat: A72 (ARMv8-A) has FP16 conversion but NOT FP16
arithmetic** — that's ARMv8.2-A. For us: store U/V as F16 (halves V
traffic 4MB→2MB — fixes the exact reason Winograd lost), dequant in the
GEMM via vcvt (free on A72). Parity risk higher; test like we did.

### 20. Static job scheduling (compile-time task assignment)
HAWC assigns jobs to threads at compile time → zero runtime scheduling
overhead, balanced loads. Our omp static schedule is close; a hand-rolled
persistent-thread pool for the VAE conv loop removes omp spawn cost.

## HOP 8: llama.cpp exp2 SiLU (PR 7154, HN 40371612)
"New exponent functions that make SiLU and SoftMax 2× faster, at full
accuracy" — exp(x) → exp2(x·log2e): exp2 is a single-instruction-style
op (or bit-magic), no argument reduction.

### 21. exp2-based SiLU
Our `unet_silu`/`wubu_sd_silu` call scalar `expf` per element
(x/(1+expf(-x))). exp2(-x·log2e) vectorizes 4-wide with vdup+vfma+
frecpe-style exp2. 2× on the silu passes; bit-exactness preserved when
done in F32 (exp2 vs exp differ in last ULP — verify against parity bar).

### 22. exp2-based softmax in attention
Attention QK^T softmax also uses expf. Same trick. Also: online softmax
(streaming max+sum in one pass) to avoid the two-pass max/exp/sum.

## HOP 9: OpenVINO + ToMe (HF blog, NNCF)
"Optimizing Stable Diffusion for Intel CPUs" — **5.1× (FP32 1.9×, INT8
3.9×, +ToMe 5.1×)** on CPUs, footprint 0.25×.

### 23. ToMe: Token Merging before attention
Merges redundant tokens before self-attention (the compute-hot part of
the UNet). Training-free, quality-preserving at small merge ratios.
Directly portable: our attention is i-batched QK^T; merging tokens cuts
its N².

### 24. QAT 8-bit for the UNet convs/linear
Intel found post-training 8-bit breaks SD (needs AdaRound/QAT). Our
Q4_0 conv weights + F16-x activations already beat plain INT8; the
lesson is: layer-wise distillation (AdaRound) quantizes activations too.

## HOP 10: GGUF quantization deep-dive (medium, michael.hannecke)
### 25. Q8_0 is the fastest CPU dequant path
"Q8_0 remains the fastest option for CPU inference due to its simple
dequantization path" (~lossless, +0.01 PPL). For the VAE convs (dequant
once per conv), Q8_0 removes the Q4_0 nibble-pair unpacking → faster
sd_dequant_w, and 8-bit is line-aligned (no 18-byte blocks).
Trade: 2× weight bytes (memory-bound!) — test per-layer: VAE convs are
traffic-heavy, so Q4_0's smaller footprint may still win. Benchmark both.

### 26. Aggressive quants for edge (IQ2_M/IQ3_S)
Smaller weights = less traffic on bandwidth-bound edge → our box's
dominant constraint. IQ2_M for the UNet's big linear layers if quality
holds (test with parity bar).

## HOP 11: Tiled VAE (diffusers, ComfyUI VAEDecodeTiled)
### 27. Tiled VAE decode with overlap+blend
Split decode into overlapping tiles (overlap 64 default), blend edges.
On CM4: smaller working set per conv → better L1/L2 residency per tile.
Speed-neutral to slower on small images, enables big ones. Pairs with
TAESD's bounded receptive field.

## HOP 12: justine.lol matmul ("LLaMA Now Goes Faster on CPUs")
### 28. mnpack recursive register blocking (3×4 / 4×1 / 1×4)
Justine's GEMM packs 3×4 and 4×1 micro-tiles with dual accumulators
(2× _mm256 chains = ILP). Our F16 GEMM is 4×4; try 2×4 dual-accumulator
for more FMA-in-flight per A72's single FP pipe.

### 29. Dual-accumulator chains (SLLMM2 pattern)
Two independent accumulator vectors per j-block hide FMA latency. Our
4×4 kernel already has 16 accs; verify the compiler interleaves them
(branch-friendly) rather than serializing.

### 30. F16/Q8_0 weights are the CPU sweet spot
"F16 and Q8_0 are really solid choices" on CPU GEMM — matches our
F16-x activations. dotprod/fp16 ISAs (10×) are ARMv8.2+ — A72 excluded;
stay with vcvt-based F16.

## HOP 13: Turbo-VAED (arXiv 2508.09136)
"Fast and Stable Transfer of Video-VAEs to Mobile Devices" — **84.5×
VAE speedup at 720p, 17.5% params, 96.9% reconstruction quality.**

### 31. Depthwise separable convs in the VAE
3D depthwise-separable integration slashes params. 2D analog for our VAE:
depthwise+pointwise instead of dense 3×3 → ~9× fewer MACs on the big
convs. Requires distilling the decoder (decoder-only distillation, $95
training cost on GPU).

### 32. Decoupled pixel-shuffle upsampling (replaces nearest+conv)
"The upsampling techniques in mainstream VAEs are poorly suited to
mobile hardware and form the main bottleneck" — Turbo-VAED's decoupled
pixel-shuffle replaces conv-on-upsampled-grid with a rearrange+conv on
the SMALL grid. This is the principled version of our fused upsample:
same math, subpixel form (4 separate small convs, one per phase) —
2.25× fewer MACs on upsample convs, without Winograd's traffic blowup.

## HOP 14: Anatomy of high-performance convolution (sahnimanas)
### 33. Goto-style cache blocking (the 100× path)
Register-block the output tile (MR×NR), pack A/B into L1-resident
buffers, loop order i→j→k with packing. Our j-chunked GEMM is on this
path; the remaining step is explicit x-row packing (currently xcol is
built fresh per tile).

### 34. Loop-order: parallel over pixels, serial over channels
The direct-conv lesson: iterate output pixels in the parallel loop
(load balance), channels serial inner (data reuse). Our im2col does
exactly this; the NHWC variant (hop 1) makes it cache-friendly.

## HOP 15: Arm I8MM (developer.arm.com)
### 35. smmla i8mm kernels — NOT applicable to A72
"Optimized Llama.cpp Q6_K/Q4_K with smmla, big uplift" — but i8mm is
ARMv8.2-A. A72 (ARMv8.0) lacks it. Documented so we don't chase it.

### 36. A72-specific tuning: -mcpu=cortex-a72
We build with -march=armv8-a; -mcpu=cortex-a72 adds A72 scheduling
(tune). Cheap, immediate, no risk.

---

## Ranked for the CM4 (by expected total-time impact)

| # | Improvement | Target | Est. gain | Risk |
|---|---|---|---|---|
| 9 | TAESD fast decode path | VAE 492s | 40× on decode | quality (preview) |
| 7,8 | DeepCache | UNet steps | 2.3× total | 0.05 CLIP |
| 11,12 | LCM 4-step | UNet steps | ~8× sampling | needs LCM ckpt/LoRA |
| 32 | Decoupled pixel-shuffle upsample | up.1/up.2 | 2.25× MACs | parity (verify) |
| 1,2 | NHWC/im2win convs | build gather 29s | ~3× build | rewrite |
| 21 | exp2 SiLU | silu passes | 2× | parity ULP |
| 36 | -mcpu=cortex-a72 | all | small | none |
| 19 | F16 transform-space Winograd | VAE convs | ~2× (revisit) | parity |
| 25 | Q8_0 vs Q4_0 A/B | conv dequant | ? | traffic |
| 23 | ToMe | attention N² | 1.2-1.5× | quality |
| 17,18 | HAWC layout+overlap | Winograd | revisit | — |
| 31 | Depthwise separable VAE | VAE convs | 9× MACs | distillation |
