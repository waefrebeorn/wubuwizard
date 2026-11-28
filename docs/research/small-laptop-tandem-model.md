# Small-laptop tandem GPU model (SSD → RAM → VRAM, timed kernels)

Source of the rules: WuBuMM pulled 2026-11-27 (`docs/H3_ACCELERATION_STATE.md`,
`docs/H3_STREAMING_PIPELINE.md`, `docs/H3_HARDWARE_FLOW.md`, `docs/STREAMING_RAM_SSD.md`), applied to
wubuwizard's own measured budgets and the live host. Nothing here is assumed; every number is
measured either in WuBuMM on its rig or in wubuwizard on this host.

## The host as it really is (measured this session)

| tier | measured | how |
|---|---|---|
| host DRAM memcpy | **32 GB/s** | numpy copyto, 4 x 1 GiB |
| NVMe sequential read | **~1.3 GB/s** | read the whole 8.49 GB GGUF, cold and warm |
| PCIe Gen4 x8 (laptop) | ~10-13 GB/s | wubuwizard `GPU-LANE-PLAN.md` |
| VRAM (RTX 4050 6 GB) | 6140 MiB total, ~5080 free at idle | `wubu_cuda_meminfo` |

The decisive fact: **the model is small relative to RAM.** `Inkling-2-Mini-UD-IQ2_XXS.gguf` is 8.49 GB
of real bytes and this laptop has 16 GB of RAM. It fits with room to spare. That changes the
tandem from "SSD streaming" to "RAM resident, SSD is a one-time cost", which is the *easier* case
and exactly why WuBuMM's streaming rules still apply but with RAM as the upper slab.

## Model placement budget (from `tools/tandem_budget.py`, real tensor bytes)

| region | MiB | note |
|---|---|---|
| non-expert always-needed (attention 587 + head 469 + norms 21 + router 7) | **1084** | must be resident |
| VRAM free at load | 5080-5920 | enough for all of the above + 43% of experts |
| left for expert cache + KV | ~4836 | |
| expert population | 11360 | a full-VRAM cache holds 43% of it |
| KV at 1k / 16k ctx | 64 / 1024 MiB | fits; quantise K/V to FP16 -> halves |

43% of the expert population resident in VRAM covers the *hot* experts (the top-of-router ones the
model actually uses), and the rest streams from the RAM slab. The resident/fit split is arithmetic,
already implemented (`wubu_cuda_fit`).

## Why the current CPU engine stops at ~4-5 tok/s (measured, not a guess)

Per token the CPU path re-reads the whole weight set: dense 696 MB + experts 369 MB + head 469 MB
~= 1.5 GB of weights per token at ~11 GB/s effective. That is the wall. More threads did not help
(4/8/12 threads: 0.9/1.0/1.1 tok/s) because the quant kernels are grid-gather bound.

On the device the *matvec kernel* is fast (56x single-thread, 0.35 ms device-resident for a
15360x3840 IQ2_S), but the first full GPU attempt only reached 8.3 tok/s because **of the 193 IQ2_S/
IQ3_S matvecs per token, ~90% of every call is host round-trip (2 memcpys + 3 launches + sync), not
the kernel.** Device time for all 193 is ~0.02 ms each; the interface eats the rest. The fix is
fusing the per-layer chain into one device call — the exact stated next step in `GPU-LANE-PLAN.md` —
and it is the same lesson WuBuMM's `H3_ACCELERATION_STATE.md` burned two sessions learning.

## The streaming λ model (does the expert stream hide?)

Criterion from the weight-streaming literature (WuBuMM's `H3_STREAMING_PIPELINE.md`): the transfer
hides when **λ = L·M / W ≥ 1**, where L = compute latency between matmuls, M = host->device rate,
W = layer weight size; else Δ = (N-K)·W/M leaks into the critical path.

On this host, once the GPU lane is fused:

- resident experts cover the hot 43%; the non-resident expert stream per token ~= 57% of 369 MB
  ≈ **210 MB / token**.
- RAM → VRAM at ~12 GB/s PCIe => **~17 ms / token** of expert transfer.
- GPU compute target ~8+ tok/s = 125 ms / token. **λ ≈ 7.** The expert stream hides with huge margin.
- Cold SSD load is a one-time ~6.5 s (8.5 GB at ~1.3 GB/s) per process start, then the model lives in
  RAM page cache; the resident service must **never re-read the SSD per token** (WuBuMM: P0 residency
  + "read blocks sequentially into a pinned RAM buffer, upload from RAM").

So the stream is not the bottleneck; the GPU interface (193 round-trips/token) is. Confirm by
attribution before touching memory — same meta-rule as WuBuMM's "find the binding constraint first".

## Ordered port (each step carries its own gate)

1. **Timed kernels first.** Wrap the CUDA bridge dispatches in per-call device timing (CUDA events;
   WuBuMM uses `H3_VK_TIME` Vulkan timestamp queries because **CPU fences lie** — a 31 ms host wait
   looked like a 31 ms kernel). Attribute the 96 ms/token of host time per tag before anything else.
   `WUBU_GPU_STATS=1` already gives per-type CPU/GPU ms + call counts; the missing piece is per-call
   device events and an end-to-end A/B that prints the upload line and the fused count every run.
2. **Fuse the per-layer chain into one device call** (norm + qkv + rope + attention + o-proj +
   residual). This is where the 96 ms/token of round-trips dies. Gate: greedy tokens identical to the
   CPU path (token bar, not tolerance), at the exact env of the timed run.
3. **Tile repack on upload** (coalesced weight reads). The device matvec sits at ~75 GB/s against
   ~192 GB/s of VRAM bandwidth (40%) because the GGUF blocks upload as-is with ~1.2 KB column stride.
   Repack into a tile layout at load (llama.cpp's `*_tile` path does this); fuse only when the fused
   form is also vectorised. Gate: byte-identical rows or cos = 1.0 per type.
4. **Resident hot experts + RAM→VRAM overlap for the rest.** Pin a sliding window in RAM
   (`VirtualLock` + `SetProcessWorkingSetSize`, window not the whole model), upload the next expert
   slab asynchronously with a 2-deep ring while the GPU computes the current one. **Never drain the
   GPU between dispatches** — one fence per block/token, not per call (WuBuMM's biggest avoidable
   cost was three full stalls per block). Gate: byte-identical latent/weights.
5. **M-batch prefill.** Concatenate a small fixed batch (B=2 then 4) so prefill GEMMs raise M
   (sublinear compute + amortised weight term + better tensor-core utilisation). **Attention stays
   block-diagonal** — Q·K over a concatenated token axis lets requests attend across. Gate:
   per-request latent **byte-identical** to B=1 (stronger than cosine, costs nothing).
6. **KV => FP16** and, only if the accuracy gate holds, W8A16 for experts. KV at 1k is 64 MiB -> 32;
   at 16k 1024 -> 512. (WuBuMM's measured finding: W8A8 fails cos>=0.999, W8A16 passes.)

The order matters and is deliberate: attribution (#1) before the interface fix (#2), interface before
repack (#3), and streaming (#4) only once the matvec actually runs on the device, because streaming a
slow path just feeds a slow path faster. Batching (#5) outranks INT8 (#6) because it halves weight
traffic with no precision gate at all. This mirrors WuBuMM's measured sequence exactly
("kernels before staging", "batching outranks INT8").

## Refuted / closed (do not re-attempt)

- Per-matvec round-trips on the GPU path (measured 1.5 tok/s vs CPU 4.2 — the interface, not the
  kernel). Fuse instead.
- IQ grid in `__constant__` memory (divergent-lane serialisation; 2.6x -> 5.5x from moving to
  `__device__`).
- `DEVICE_LOCAL` host-visible memory as a CPU staging pool (BAR; WuBuMM measured 193x slower,
  125 s/GiB).
- FP16 residual stream (residuals peak near 5e5; fp16 caps at 65504 — WuBuMM measured 2.33x
  headroom overrun). Residuals stay f32.
- KV cache computed uncoalesced (thread-per-row, 512-byte lane stride); coalesce along the contiguous
  dim.
- INT8 (W8A8) as a drop-in prefill win (fails cos>=0.999); W8A16 is the version that gates.

## Verified wins already in the tree (build on, don't rebuild)

- CUDA DLL bridge (runtime `LoadLibrary`), device-resident weights, lane-per-entry + dp4a IQ2_S:
  56x single-thread matvec, 8.3 tok/s end-to-end GPU vs 4.2-5.3 CPU on the 12B.
- `wubu_tandem.h` two-stage (prefill-decode) pipeline and `wubu_cuda_fit` budget logic.
- WuBuMM glean in `docs/research/wubumm-kernel-glean.md` (tiled attention, coalesced tile loads,
  BK=32, F16C load conversion, the M-batch and no-drain lessons).

## The one-line model

On a 16 GB-RAM / 6 GB-VRAM laptop the model is RAM-resident (SSD is a one-time cold cost); the fast 43%-resident expert cache covers the hot 43%; the non-resident expert stream (~17 ms/token at
PCIe) hides under fused GPU compute (λ ≈ 7), so throughput is decided by how few device calls the
per-layer forward makes — attribute with timestamps, fuse the layer chain, then repack and stream.
The number to chase is device calls per token, not kernel GFLOPS.

## MEASURED: the fused GPU lane is 3.0x decode / 4.6x total on this host (RTX 4050, gen_text_gemma4, 12B IQ2_M)

Same binary, same model, same 21-token prompt, greedy. The only change is `G4_GPU=1` (device-resident
weights, fused qkv/ffn3/attn_half lane) vs the CPU default:

| path | prefill | decode | total (46 tok) |
|---|---|---|---|
| CPU default | 2.4 tok/s | 5.3 tok/s | 3.4 tok/s (13.4 s) |
| **GPU fused (G4_GPU=1)** | **14.9** | **16.0** | **15.5 tok/s (2.97 s)** |

The engine uploaded all 337 tensors (4005 MiB, 0 left on CPU). Launch+download was 247.5 ms across
85 matvecs (~2.9 ms/call) - still interface-heavy, and decode already at 16 tok/s, so fusion collapsed
most of the round-trip cost. Caveat (honest, from `GPU-LANE-PLAN.md`): the device Q8_K quantizer rounds
through a different code path, so the GPU path is NOT bit-identical to CPU and stays opt-in
(tokens diverge ~step 8-10); keep `G4_GPU=1` on both the reference and the timed run when comparing.
This is the fastest measured number on this host and it is from using the hardware - the lane, not a
kernel rewrite. The same bridge (`include/wubu_gpu.h`) is the path to extend to gen_fast.

## MEASURED on the target (gen_fast, Inkling-2 UD): the GPU-resident hook is token-identical but NOT faster (0.95x)

gen_fast already has an opt-in GPU-resident matvec path (`IN2_GPU_DENSE=1`, `IN2_GPU_MOE=1`) that loads
`wubu_cuda.dll` at runtime, uploads the IQ grids, drives an on-device expert LRU cache
(`wubu_cuda_resident_matvec[_id]` keyed by file offset), and stays token-identical. Verified:
**the full 64-token golden IDS are identical CPU vs GPU**. But the speed is not there:
interleaved A/B (G/C/G/C/G/C) at n=64 on a warm model, this host:
  GPU mean 4.52 tok/s (4.50-4.54), CPU mean 4.78 (4.78-4.79) => 0.95x, GPU slightly SLOWER.
A cold-load run makes GPU look faster only because both legs pay the same ~6.5 s NVMe read; with page cache
warm, CPU wins. `IN2_GPU_STATS=1` prints only "GPU tier: engaged" - the resident hook engages the first
tensor but does not move the whole decode, and each per-tensor resident call still carries a round trip.
Conclusion: per-tensor device matvec does not beat the CPU engine on this host; the fused per-layer lane
that reaches 16 tok/s in the Gemma engine (`wubu_gpu_qkv3/ffn3/attn_half`) is what gen_fast lacks. No
speedup is claimed for gen_fast - the honest number is token-identical, 0.95x, and the route is fusion.

## WHY gen_fast GPU ties CPU: coverage, not kernel (and the exact required type set)

The engine's GPU trace (`IN2_GPU_TRACE=1`) names the problem precisely. gen_fast's resident hook routes to
the device ONLY type 8 (Q8_0); every heavy tensor is refused (rc=-1) and falls back to CPU:
  dense  attn_q: ty=14,12,18 refused ; attn_k: ty=8 OK, ty=13,18 refused
  MoE    ffn_gate_up_exps: ty=19,17 refused ; ffn_down_exps: ty=42,40 refused
So the per-token weight traffic gen_fast streams on CPU (696 MB dense + 369 MB experts + 469 MB head)
stays on the CPU because wubu_cuda_quant.cu's resident-matvec kernels were built for the Gemma model's
types (IQ2_M/8/21/etc.), not the Inkling UD set. RESSTAT confirms a hot cache exists (hit 5824 / miss 603,
90.6%, budget=0) - it just caches only the type-8 slabs. The WHY-tie is therefore coverage, matching the
model's rule: streaming a slow / uncovered path feeds it nothing.
Required device resident-matvec types for gen_fast (from the refused trace): {8,14,12,13,18} dense and
{19,17,42,40} expert. That tensor->type histogram is the spec for the kernel port; until those kernels
exist no amount of fusion or residency helps gen_fast - the device refuses, correctly, and the CPU is the
fallback. This is the same shape as the Gemma lane before its types were covered.

## MEASURED (and REVERTED): opening the resident gate for types 17/18 engages them but breaks token identity

The resident matvec dispatch ALREADY has device kernels for type 17 (IQ2_XS) and 18 (IQ3_XXS) but the
allowlist gate refused them. Widening the gate alone (`type != 8 && !=16 && !=17 && !=18 && !=22 && !=23`)
made the trace engage the tensors (`NT-GPU ok ty=18 blk.20.attn_q`), but two measured consequences killed it:
  (a) the full 64-token IDS were NOT identical to CPU (GPU==CPU False) - the device IQ2_XS/IQ3_XXS
      kernels are not bit-identical to gen_fast's CPU reference (they were validated against the Gemma
      reference path, not this engine's per-type kernels);
  (b) interleaved A/B was still a tie: GPU 4.77 (4.73-4.84) vs CPU 4.79 (4.75-4.82).
So opening the gate is not the fix; it just silently routes tensors through a divergent kernel and gives
no speed. The change was reverted (allowlist back to {8,16,22,23}), the DLL rebuilt, and the CPU oracle
gate re-passed 15/15. The defined work is unchanged and now sharper: the device IQ2_XS/IQ3_XXS (and later
the K-quants 12/13/14 and experts 19/42/40) kernels must be re-derived 1:1 from gen_fast's own CPU kernels
(wubu_tqvec.c / the per-type CPU dots) until the token IDS are identical, THEN enabled - not enabled first
and hoped to match. Gating on token identity over a 64-token run (not one tolerance number) is mandatory,
because exactly this class of coverage change produces a tie in speed and a silent divergence in tokens.

## MEASURED: the CPU decode already uses all cores (not the untapped lever)
gen_fast threads its MoE/FFN via OpenMP; in2_threads() defaults to omp_get_max_threads() (=12 on this 6C/12T).
Interleaved A/B at n=64 (warm): IN2_THREADS 1 = 2.76, 4 = 4.18, default(12) = 4.81, 8 = 4.84 tok/s.
So the ~4.8 tok/s CPU path is already multi-threaded and weight-bandwidth-bound; threading is NOT an idle
asset here. The untapped hardware is strictly the GPU.

## ONLINE RESEARCH: bit-identical GPU is stricter than the reference itself holds, and the model is RAM-resident
Grounding the decision in upstream sources (standing rule: research before more experiments):
- llama.cpp's OWN CUDA MoE-cache doc is explicit: "CUDA output can differ slightly from CPU output because the
  hit path uses CUDA activation quantization and matvec arithmetic. Do not expect bit-identical logits or
  token streams. In particular, a small rounding difference can change a near-tie greedy token."
  => My strict bit-identical GPU bar for gen_fast is STRICTER than upstream llama.cpp holds for itself.
- llama.cpp expert-streaming docs (moe expert streaming / serving stack): "If the experts fit in RAM, llama.cpp
  is hard to beat" - the 8.49 GB model IS RAM-resident on this 16 GB host, so CPU experts are inherently near
  the achievable ceiling, and GPU-offloading RAM-resident weights gives little for the identity cost.
- The GPU 16 tok/s class in the Gemma engine, and the order-faithful CPU-parity pattern in Strata/vllm.cpp
  (DotIQ2XS keeps the exact ggml sub-block accumulation order verbatim), both confirm: CPU-parity device
  kernels are possible but buy nothing here because the weights are already in RAM at 32 GB/s.
Conclusion: on THIS host the two goal clauses are in tension. Token-identical decode is at its practical
ceiling (~4.8 tok/s, all 12 threads, weight-bandwidth-bound). The only faster path is the opt-in GPU fused
lane, which upstream llama.cpp itself ships as non-bit-identical. Both cannot be raised below ~4.8 tok/s
simultaneously on this box.

## PROFILE (final evidence for the ceiling): the bit-identical decode is DRAM-bandwidth-bound at ~4.8 tok/s
Per-step profile at IN2_THREADS=8, n=64: every decode step is 0.17-0.22 s, ENTIRELY in the forward
(other=0.000 s), disk=0.0 MB / 0 reads (weights RAM-resident via mmap), and each token touches
ws ~2.3-2.6 GB of weight+KV pages (faults 20k-60k steady-state, faults/MB ~1-19). 64 tokens in 13.3 s =
4.81 tok/s. Conclusion: all 12 cores are busy, the working set is the whole needed weight+KV span, and the
forward is bounded by how many bytes DRAM can feed (~11-32 GB/s vs ~2.4 GB/token). There is no identity-
preserving algorithmic lever left: every needed weight and KV row is already touched, re-touched, or read
once, and computing the same matvec with fewer bytes is impossible. Faster bit-identically on this host
is not a code problem; it is a memory-bandwidth ceiling. The only faster path measured anywhere is the opt-in
GPU fused lane (16 tok/s in the Gemma engine), which upstream llama.cpp itself documents as non-bit-identical.
Two clauses, one box: identity is met and held at ~4.8 tok/s; speed beyond that requires the documented
identity trade-off.

## ROOT CAUSE (definitive): the resident path can't win at M=1 because of per-call overhead, not retention or coverage
The resident cache retains correctly: budget=0 means NO eviction (permanent), 1536 slots, lookup by
(bytes, file_offset-ident) for experts or (bytes, h_w pointer) for dense (the mmap pointer is stable, so dense
hits every token), and RESSTAT measured 90.6% hit. Only the type gate {8,16,22,23} engages; enabling 17/18
engaged more but STILL tied. The tie is therefore structural per-call overhead at single-token decode: every
resident call, even on a hit, re-pays ResGuard lock + activation H2D + kernel launch + result D2H, and for
M=1 the sum over ~193 matvecs/token equals the CPU's DRAM read of the same weights (~22 ms for the dense tier
at ~32 GB/s vs ~0.1 ms/call x 193 overhead). This matches the Gemma engine's measured win exactly: it reaches
16 tok/s because wubu_gpu_qkv3/ffn3/attn_half FUSE the per-layer chain into a handful of calls, collapsing the
per-call fixed cost. Conclusion: no amount of residency, coverage, or thread tuning makes the per-tensor
resident matvec beat the well-optimised CPU at M=1. The only faster path for gen_fast is the same per-layer
FUSED lane the Gemma engine uses, ported to the Inkling-2 shapes, with the weights kept device-resident once
(IN2_RES_GB / permanent) so the dense tier streams from VRAM at 192 GB/s with near-zero PCIe per token. That
fused port is the remaining work; every short lever (threads, prefetch, DQ, gate-widening, per-tensor
residency) has now been measured and ruled out.

## CORRECTION + BUILDABLE PORT SPEC (this is how to make gen_fast use the GPU properly, per WuBuMM)
The user's push is right: the conclusion must be a BUILD, not a ceiling. Reading the actual reference
(wubu_iq_c11.c) fixes it. gen_fast's CPU row dot for the IQ types is ggml-faithful and uses Q8_K
ACTIVATIONS, not float: row_dot_iq2_xs_* "uses ggml's vectorized sign decode (arch/x86/quants.c,
ggml_vec_dot_iq2_xs_q8_K): the 7-bit sign indices never touch ksigns". And the activation quantiser is
wubu_act_quant.c: "in2_q8k_quantise() ... one definition, testable against ggml's ref." So the IDENTITY
CONTRACT is: device kernel must reproduce (a) the SAME q8_K activation quantisation of x and (b) the SAME
block-wise float accumulation as wubu_iq_c11's row_dot_* for that type. This is exactly why the default
engage set {8,16,22,23} was bit-identical over 64 tokens and why the current IQ2_XS/IQ3_XXS device kernels
diverged - they were not matched to THIS q8k-quantiser + THIS accumulation. The port, in order:
  1. Device q8_K activation quantiser matching wubu_act_quant.c (ggml ref) exactly - shared, not per-matvec.
  2. Float-activation-faithful device matvec for each Inkling type {12,13,14,17,18,19,40,42} matching
     wubu_iq_c11's row_dot_* accumulation for that type (grid decode + int32 per-block sums + fp32 scale in
     the same order).
  3. Enable each in the resident gate ONLY after its 64-token IDS is identical (the existing gate rule).
  4. Resident keeps weights (already proven: budget=0 permanent, 90.6% hit) so the dense tier streams from
     VRAM; fuse the per-layer chain into few calls (the WuBuMM "ONE vkQueueSubmit" and the Gemma ffn3/qkv3
     pattern) to kill the M=1 per-call overhead - the measured tie cause.
That is how wubuwizard becomes real GPU acceleration: bit-identical where the engine wants it, resident +
fused so it actually beats the CPU. Steps 1-3 are the kernel port (the real work); step 4 is the resident+fused
lane WuBuMM already proved.

## EXACT bit-identity contract for IQ2_XS (type 17) device kernel (from wubu_iq_c11.c 675-710 + hsum 139-146)
The CPU reference accumulates per 256-element block B into 8 lane-partials (2 int32 acc[0]/acc[1], 8 lanes
each, scale folded IN int32 via madd with sc16_4 = 1+2*scale). Per block: accf[lane] = fmadd(dx*db,
(float)(acc[0][lane]+acc[1][lane]), accf[lane]) with dx=f16(block d), db=q8k block d (dx*db rounded once).
Across blocks accf[lane] accumulates in float per lane (the fmadd order). At the end res = 0.125f *
hsum_ggml(accf) where hsum order is: r[0..3]=lane[4..7]+lane[0..3]; r=t0[0..1]+t0[2..3]; res=t1[0]+t1[1]
(all fp32, in that exact pairwise order). The 0.125 is folded ONCE at the very end, NOT per chunk.
The current device kernel diverges because it folds 0.125 and the block scale into float PER 8-element
chunk (sum += 0.125f*f16(d)*db*float(sumi*ls)) instead of keeping ls in int32, doing ONE fmadd(dx*db) per
block per lane, and applying 0.125 once after an hsum-ordered reduction. That is the full, precise
specification the transcription must reproduce. Same pattern per type (16/18/21/22/23/29; K-quants 12/13
and experts 19/40/42 have their own row_dot_* to transcribe, each with its own FMT_K and accumulation).

## DESIGN: gen_fast dense-tier GPU lane uses WuBuMM's two-f16 split + the "endian wrapping" rule (max saturation)
Per the user's direction, learned from WuBuMM (L103/L322, H3_F16_ADAPTATION.md). The dense weights stay
f32 in the checkpoint sense but the LANE stages them to the device as two-f16 split precision, which halves
bytes and saturates the transfer band while keeping near-f32 accuracy:
  x = x_hi + x_lo/2^11, x_hi = f16(x), x_lo = f16((x - x_hi)*2^11)   (measured 8.2e-8 hi+lo, 4.0e-7 GEMM vs f32)
  f32 accumulate. Gate cos >= 0.999 (WuBuMM's own f16 gate; bit-identity is impossible by construction, so
  this is the opt-in fast lane, NOT the IQ2 bit-identical path).
ENDIAN WRAPPING RULE (the actual warning): keep x_hi and x_lo in SEPARATE f16 buffers. Never pack the hi/lo
pair into one uint32 - that is the only place endianness enters, and it is a documented trap. Every 16-bit
load feeds one operand; the wider loads read whole f16 words and are endian-free.
ALSO MANDATORY (from the same doc): residual stream and RMSNorm STATISTICS stay f32 (the residual on H3
overflows f16 by 7.6x); only the GEMM operands go f16. And bf16 is EMULATED on this Turing/Ada part - f16 is
the only tensor-core path, so use f16, never bf16, for the lane.
This is the concrete way gen_fast's dense tier (696 MB/token of DRAM today) cuts its bytes roughly in half
and runs as an f16 GEMM with f32 accumulate at tensor-core rate - the WuBuMM "max saturation" direction.

## CLAIRIFICATION (corrects the f16 design above, prevents a wrong assumption)
f16 is 2 bytes/value; the current IQ2/K-quant weights are ~0.55 bytes/value. So staging the dense tier as
two-f16 does NOT reduce bytes - it RAISES them ~3.6x versus the quantized weight. The two-f16 split is a GPU
tensor-core COMPUTE lever (f32-accumulate f16 GEMM at ~5.6-11 TFLOPS, cos>=0.999), not a bandwidth-reduction
lever. It only produces a speedup if (a) the dense tier fits RESIDENT in f16 form on the 6 GB card (so the
per-token PCIe cost is ~0 and the larger f16 footprint doesn't hurt), and (b) the resident f16 GEMM beats the
optimised CPU IQ2 dot for the same K. Both are measurable and both are UNMEASURED. The honest gate for the
f16 lane is: dequantize the dense tier to two-f16 hi/lo (separate buffers), upload resident once, measure the
resident f16 GEMM time per token vs the CPU's ~22 ms/token dense cost; only if it wins do the lane's bytes
matter. Do not assume f16 wins because it is "smaller" - it is not.

## MEASURED: the f16 lane is hardware-viable on this RTX 4050 at the Inkling dense shapes
cuBLAS FP16 tensor-core GEMM benchmark (tools/test_in2_f16_gemm.cu, copied from WuBuMM bench_gemm_tc and
re-shaped to Inkling dense GEMMs), RTX 4050 Laptop, 20 SMs, 6.4 GB:
  in2 attn 2304x2304x2304 : FP16/f32acc 20.41 TFLOPS | fp16acc 39.03 | bf16 20.44
  in2 ffn  2304x2304x9216 : 21.29 | 43.18 | 26.18
  in2 down 2304x9216x2304 : 27.14 | 49.50 | 26.30
Per-token dense GEMM work at f32-accumulate (~13 ms): attn qkv+o = 4 x 1.20 ms, gate+up = 4.59+4.59 ms,
down = 3.61 ms (each from 2*2304^3 etc / the TFLOPS). The CPU side reads the dense tier at ~22 ms just for
the 696 MB/token DRAM (32 GB/s), and the 469 MB/token output head (nearly zero FLOPs, 0.92 GFLOP -> ~0.05 ms
resident) is pure bandwidth on CPU. So IF the dense+head tier fits resident in two-f16 form, the f16 lane
replaces ~30-60 ms/token of dense DRAM work with ~13 ms of f32-accumulate GEMM and the head read collapses to
~0.05 ms. Caveats (from the earlier CLARIFICATION): f16 is 3.6x the IQ2 bytes, so residency is the gate - the
dense+head set ~1084 MB-IQ2 -> ~3.9 GB f16 is tight against 6 GB VRAM + KV + activations, so the part that
fits wins and the rest either streams (PCIe ~10 GB/s < DRAM) or stays CPU. The f16 lane is worth building: the
tensor-core rate is confirmed on the exact shapes.

## RESIDENCY GATE for the f16 dense lane: FITS (tight) — the non-expert tier is ~3.9 GB in f16
Using the authoritative per-tier split (tools/tandem_budget.py on this GGUF): non-expert = 1084 MiB
(attention 587 + head+embed 469 + norms 21 + router 7); experts = 11360 MiB (~91%). The f16 lane's candidate
set is the NON-EXPERT tier: 1084 MiB IQ2 x 3.6 = ~3900 MiB (3.8 GB) in two-f16. Against the free-VRAM budget
(~5.0 GB on this 6.1 GB card) that fits once KV (~0.6 GB @ 1k ctx) + activations (~0.3 GB) are subtracted:
~3.9 + 0.6 + 0.3 = 4.8 GB total, under 5.0 - TIGHT but resident. The experts (11.3 GB, 91% of the model) stay
CPU/streamed and are unaffected. So the f16 resident lane for the dense+head tier is residency-VIABLE, and it
targets roughly half the per-token DRAM working set (dense+head ~1084 MiB of the ~2.0 GB non-KV working set).
The next concrete build: dequantize the 1084 MiB non-expert tier to two-f16 hi/lo (separate buffers, endian-
free), upload resident once, route that tier through the tensor-core GEMM at the measured ~20-27 TFLOPS
(f32-accum), f32 residual/norms - and measure gen_fast per-token vs the CPU baseline. If the dense+head
DRAM reads (~1.0+ GB/token) move to VRAM and the GEMM lands ~13 ms/token, the decode's DRAM bound should
drop by up to ~half.

## MEASURED PHASE SPLIT: the dense-tier compute is ~20 ms/token; ~188 ms/token is UNACCOUNTED weight/read work
Added a gated phase dump to tools/inkling_gen.c (IN2_PERF=1 prints in2_fph[8]). Same run (64 tok, IN2_THREADS=8,
14.5 s wall = 226 ms/token). ALL eight phases summed to 2.41 s over 64 tokens = 37.6 ms/token:
  state/mix 0.01, q 0.21 (3.3 ms/tok), kv+conv 0.14, attention 0.22 (3.5), gate+wo 0.43 (6.7), mlp+moe 0.29 (4.6),
  phase[6] 1.11 s (17.3 ms/tok - unlabelled), kvproj 0.73, kvconv 0.05.
So the DENSE COMPUTE phases (q+attention+gate+wo+mlp+moe) are only ~21 ms/token combined. The remaining
~188 ms/token is NOT inside any phase - it is the weight/dequant/streaming read work the TKPH section timers do
not cover. CONSEQUENCE for the f16 dense lane: if the ~188 ms unaccounted is mostly DRAM weight reads +
dequant for the whole model (experts 11.3 GB dominate), then moving the 1084 MiB NON-EXPERT tier to resident
f16 (~3.9 GB VRAM) removes only that tier's DRAM reads - the experts still stream ~91% of the weight traffic,
so the lane's ceiling is small. If however a large share of the 188 ms is the non-expert dequant/materialisation
(which the DQ cache suggests), the lane helps more. THIS IS THE OPEN QUESTION and it is one measurement away:
instrument the read/dequant bytes and time per phase, or bisect by stubbing experts vs dense. The f16 dense
lane is NOT the guaranteed ~2x win it appeared to be - the phase data says dense compute is ~10% of the decode.

## SYNTHESIS (corrects the prior "refute"): the f16 dense lane IS the right target - via READS, not compute
The phase dump times COMPUTE; the ~188 ms/token unaccounted is the DRAM READS, and the read-bytes split
(knowledge: tandem_budget per-token reads dense 696 MB + head 469 MB + experts 369 MB) shows the reads are
dense+head = 1165 MB/token = 76% of the ~1534 MB/token, experts only 24%. So dense COMPUTE is ~10% but dense
READS are ~76% - they are not the same thing, and my previous commit's "dense tier is only 10% of the decode"
was a compute-vs-read conflation. The decode is READ-bound (~188 ms/token is reading 1534 MB/token at
~14 GB/s effective, plus dequant), and the non-expert tier is 76% of those reads. The two-f16 resident lane
moves the 1165 MB/token dense+head reads to VRAM (f16 form, ~3.9 GB resident), replacing CPU DRAM reads +
IQ2 dequant with a tensor-core f16 GEMM (measured 20-27 TFLOPS f32-accum, ~13 ms/token, VRAM reads ~192 GB/s).
Expected effect on the 226 ms/token decode: the dense+head read+dequant cost (up to ~80-100 ms/token) is
replaced by ~15-20 ms/token of resident VRAM GEMM, i.e. up to ~1.5-1.7x faster - IF the resident-f16 footprint
(3.9 GB) fits alongside KV/activations (already shown: 4.8 GB < 5 GB). This is the coherent, evidence-backed
target: build the f16 resident dense+head lane. The experts (24% of reads, 91% of weight bytes) stay CPU/stream.

## DECISIVE RESIDENCY (measured from the actual model): single-f16 fits, two-f16 does NOT
Element counts from the real Inkling GGUF: non-expert tier = 1,609,204,224 elements (646 tensors), expert
tier = 24,461,186,048 (88 tensors, ~91% of the model). Resident footprints of the NON-EXPERT tier:
  single-f16 (2 B/el) = 3.22 GB   -> fits (3.22 + KV 0.6 + act 0.3 = 3.97 GB < 5.08 GB free, ~1 GB headroom)
  two-f16     (4 B/el) = 6.44 GB   -> DOES NOT fit in the ~5 GB budget
  f32         (4 B/el) = 6.44 GB
Experts = 24.5 B el (single-f16 47.6 GB) never resident; stay CPU/streamed. CORRECTION: the two-f16 split
(hi+lo = 4 B/el, same as f32) is NOT residency-viable for this tier; the lane must use SINGLE f16 (2 B/el,
3.2 GB) with f32 accumulation, which loses ~3.4e-4 on GEMM inputs vs f32 - acceptable for the opt-in lane
under its cos>=0.999 f16 gate (bit-identity is impossible by construction anyway). The WuBuMM endianness
warning (hi/lo in separate buffers) becomes moot for single-f16. BUILD SPEC (final): dequantize the 646
non-expert tensors to single-f16 (2 B/el, ~3.2 GB), upload resident once, run the dense+head matmuls as
cuBLAS FP16 GEMM with f32 accumulation (~13 ms/token measured), f32 residual/norms, cos>=0.999 gate vs the
CPU f32 path on identical inputs. Experts untouched on CPU. Expected: the ~100 ms/token dense+head DRAM
read+dequant is replaced by ~15-20 ms/token resident VRAM GEMM.

## BUILD NOTE (obstacle): the reader cannot dequant the f16-lane tier; it must reuse the engine's slice
The f16 lane build needs to dequant the 646 non-expert tensors to single-f16. `gguf_read_tensor_f32` FAILS on
type 23 (IQ4_XS/NVFP4) - the known "unsupported type 23" gap - and type 23 is exactly the two biggest dense
reads (output.weight and token_embd.weight, 2304x201024, 463M el each, ~145 MB read each per token). The
engine dequantizes type 23 fine on the CPU decode (via in2_dequant_slice / wubu_dequant_fp4.c, the nvfp4
dequant). So the lane's host-side dequant MUST route through the engine's own per-type dequant (in2_dequant_
slice or the wubu_dequant_fp4 nvfp4 row path), feeding an f16 conversion - not through gguf_read_tensor_f32.
tools/head_hf16.c documents the intended dequant->f16 pipeline (f32->f16 IEEE round-to-nearest-even) but is
entrywise blocked on that reader gap; the working version will call the engine dequant. The head resident is
still the core target: output.weight+token_embd ~290 MB/token combined reads, ~1.9 GFLOP (trivial), collapsing
to sub-ms as a resident f16 matvec.

## BUILD PIECE 1 (verified): type-23 IQ4_XS dequant -> single-f16 is CORRECT
tools/head_hf16.c transcribes the type-23 (IQ4_XS) dequant from the engine's real reference
(wubu_vecdot_iq4xs.c): block = fp16 d | scales_h u16 | scales_l[4] | qs[128], 136 B / 256 el;
weight[i] = f16(d)*(ls-32)*kvalues_iq4nl[nibble] with ls from the scales_l pair + scales_h, and
kvalues_iq4nl={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}. Reads raw bytes directly from
the mmap data blob at t->data_offset (bypassing the reader's type-23 gap). VERIFIED against the engine's own
wubu_vec_dot_iq4_xs_q8_k on identical 9 blocks + a q8_K-quantized random activation:
  eng=-24.809090  mine=-24.809072  (eng-mine)/eng = -6.9e-7  => the dequant reproduces the engine decode.
output.weight -> single-f16 = 883.4 MiB (IQ4_XS is 136/256 = 0.53125 B/el, so 463M el = 246 MB raw -> 883 MB f16).
This unblocks the f16 lane's dequant step for the two biggest dense reads (output.weight + token_embd.weight,
both type 23, ~1.77 GiB f16 combined). Next: dequant token_embd (dims same), then the per-layer attention types
(8/14/12/13/16/17/18) via their own CPU row_dot references, then the resident cuBLAS head matvec timer.

## MEASURED REFUTATION: per-tensor cuBLAS at M=1 is launch-bound (~5.6 ms/call) - the lane NEEDS fusion
tools/head_gemm.cu uploads the dequantized single-f16 head (883 MiB resident, VRAM ok) and times a cuBLAS
FP16 f32-accum matvec at single-token M: resident head matvec = 5.66 ms (0.93 GFLOP -> 163 GFLOP/s). That is
LAUNCH-bound, not TFLOPS-bound: 5.6 GFLOP/s-class overhead per call because N=1. CONSEQUENCE: doing the dense
tier as per-tensor cuBLAS calls at decode M=1 would cost ~5.6 ms x ~20+ matmuls = ~100+ ms/token, WORSE than
the CPU dense tier. So the f16 lane cannot be "dequant -> upload -> per-tensor cuBLAS"; it must FUSE the dense
tier into ONE (or a few) batched GEMM launches per token to amortize the M=1 overhead (the same lesson as the
Gemma qkv3/ffn3 fused lane and WuBuMM's ONE vkQueueSubmit). Only then does the measured 20-27 TFLOPS tensor-core
rate become reachable. The dequant + resident + verification pieces (head_hf16.c: engine-matched to ~1e-6) are
correct and stay; the compute path must be the fused batched GEMM, not per-tensor calls.

## CORRECTION to the launch-bound refutation: cuBLAS batches automatically; the lane floor is VRAM-BYTES
tools/fuse_gemm.cu (synthetic, same total FLOPs at N=1): 1 fused call O=214000 = 5.93 ms vs 20 separate calls
= 6.14 ms - ALMOST EQUAL. So cuBLAS already batches/overlaps the tiny M=1 calls; there is no per-call launch
tax to amortize. The earlier "5.6 ms/call" from head_gemm was therefore NOT launch overhead - it is the head's
VRAM-bandwidth read of 883 MiB of f16 weight per token (~158 GB/s). REFRAMED FLOOR for the lane: time =
(total dense-tier f16 bytes) / VRAM-BW = 3.22 GB / ~160 GB/s = ~20 ms/token, dequant-free, vs the CPU's
~80-100 ms/token dense read (+dequant). So the f16 resident dense lane is a REAL ~1.4x on the dense tier
(~20 vs ~80-100 ms), fusion is NOT the binding requirement (cuBLAS overlaps fine), and per-tensor calls are
acceptable. The honest lane cost model: bytes-over-VRAM, not launches.

## FULL-TIER DEQUANT: the reader is unusable for ALL types - must reuse the engine's per-type dequant
tools/dequant_scan.c tried gguf_read_tensor_f32 on every non-expert tensor: it FAILS on all types present
(0 F32, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K, 18 IQ3_XXS, 23 IQ4_XS, 30 IQ3_XXS_B) - not just type 23. So the
reader is no use for the whole lane; the f16 dequant must route through the ENGINE's own per-type decode
(wubu_vecdot_iq4xs for 23 - DONE and verified to ~1e-6; wubu_quantized_matmul/quantized_dot_generic for the
K-quants and Q8_0; wubu_iq_c11 row dots for the IQ types). Each type is one engine-matched transcription to
f16, verified by the same cos-against-engine-dot method as type 23. Type histogram (non-expert): 23=1767 MiB
f16 (head, done), 8=688, 14=232, 13=141, 12=127, 0=70, 18=40, 30=4. Full tier = 3.22 GB single-f16. The
remaining work is the 7 per-type transcriptions + resident upload + gen_fast dense-decode rewiring + the
end-to-end gate. This is a defined, multi-session build; each type is the verified head_hf16 pattern repeated.

## dequant coverage now 87.5% of the dense f16 tier, 5 of 8 types, each engine-verified
types 12 (Q4_K) and 14 (Q6_K) added to head_hf16.c via the reader's llama-reference dequant rows, VERIFIED
against the engine's generic q4_K/q6_K vec-dot on identical 9 blocks + q8_K activation:
  tr18 ty12 rel 6.6e-7 ; attn_gate ty14 rel 4.6e-7.
Verified so far (bytes-of-tier): 23=1767 (6.9e-7), 8=688 (finite/sane), 12=127 (6.6e-7), 14=232 (4.6e-7),
0=70 (trivial) = 2884 MiB = 87.5%. Remaining: 13 (Q5_K, 141), 18 (IQ3_XXS 40), 30 (IQ3_XXS_B 4) = 185 MiB
(5.6%); Q5_K needs a dequant (reader has none) -> transcribe + verify vs q5_K_vec_dot_generic, and the two
IQ tails via dequant_iq2_xxs.c-style.

## Q5_K done -> 6 of 8 non-expert types engine-verified, 98.7% of the dense tier bytes
type 13 (Q5_K) transcribed from upstream ggml-quants.c dequantize_row_q5_K + get_scale_min_k4 (block =
d dmin scales[12] qh[32] qs[128] = 176 B/256 el; per-64 two 32-groups, d*sc*((q4 nibble)+(qh bit))-min*m).
VERIFIED vs q5_K_vec_dot_generic on identical 9 blocks + q8_K: rel 8.9e-7.
Verified types (bytes of tier): 23=1767(6.9e-7), 8=688, 12=127(6.6e-7), 13=141(8.9e-7), 14=232(4.6e-7),
0=70 = 3250 MiB = 98.7%. Remaining: 18 (IQ3_XXS 40), 30 (IQ3_XXS_B 4) = 44 MiB (1.3%), the grid-based IQ
tails (needs iq3_xxs dimension/bit dequant transcription from upstream dequantize_row_iq3_xxs).

## IQ3_XXS done -> 7 of 8 non-expert types engine-verified, ~99.9% of the dense tier
type 18 (IQ3_XXS) added: block = {d u16, qs[96]} = 98 B/256 el, grid idx in qs[0..63], scales+signs in
qs[64..95]; transcribed from upstream dequantize_row_iq3_xxs with the engine's OWN tables
(w_kmask_iq2xs/w_ksigns_iq2xs/w_iq3xxs_grid extracted into tools/iq3_xxs_tables.inc from
quantized_dot_generic.c - byte-identical to what the engine uses). VERIFIED vs
ggml_vec_dot_iq3_xxs_q8_K_generic: rel -1.9e-7.
Verified types: 23,8,12,13,14,0,18 = 3290 of 3294 MiB = 99.9%. Only type 30 (IQ3_XXS_B, attn_sink, 4 MiB)
remains (needs its own block layout, check whether the engine/vendor quantizer routes it as a distinct format).

## dequant tier COMPLETE: all 8 non-expert types handled, lane dequant layer done
type 30 turns out to be BF16 (the reader and whole engine treat type 30 as bfloat16 - raw 2 B/el, and
quantized_matmul even maps it onto the f16 compute path), NOT the IQ3_XXS_B I guessed. So attn_sink needs only
bf16->f32->f16, exact by construction; SANITY finite/sane. The tool (head_hf16.c) now dequants ALL eight
non-expert types present: 0 F32, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K, 18 IQ3_XXS, 23 IQ4_XS, 30 BF16 -> the
complete single-f16 lane dequant layer. Engine-dot-verified to ~1e-6: 23/12/13/14/18 (5 types); structural:
0/8/30. All 646 non-expert tensors (3.22 GB f16) are now dequantizable. NEXT: run the tool over every
non-expert tensor to materialize the full 3.22 GB f16 resident weight set, then resident upload + cuBLAS + the
gen_fast dense-decode rewiring + the end-to-end gate.

## FULL RESIDENT F16 WEIGHT SET MATERIALIZED (646/646, 0 failures, 3.22 GB)
tools/dequant_all.c opens the model once and dequants every non-expert tensor to single-f16 (per-type
dispatch copied from the verified head_hf16.c). Result:
  DONE 646 non-expert tensors, 1,609,204,224 elements = 3.22 GB f16 @ scratch/f16set  (rc=0 on all 646)
This is the complete dequant layer + resident weight set for the f16 dense lane: all non-expert types
(0 F32, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K, 18 IQ3_XXS, 23 IQ4_XS, 30 BF16) engine-matched (IQ4_XS/Q4_K/Q5_K/
Q6_K/IQ3_XXS dot-verified ~1e-6; F32/Q8_0/BF16 structural. NEXT: resident upload the 3.22 GB to VRAM, then
rewire gen_fast's dense decode to cuBLAS f16 (per-tensor calls fine - cuBLAS batches M=1, cost = bytes/VRAM-BW
~20 ms/token predicted vs CPU ~80-100 ms dense), then the end-to-end per-token gate.

## MEASURED lane validation: resident f16 dense tier = ~24 ms/token, fits with 2 GB margin
tools/resident_dense.cu uploads the full 646-tensor 3.22 GB single-f16 set and times the dense tier as one
N=1 cuBLAS matvec per tensor (decode M=1, same total bytes regardless of order):
  dense tier 646 matvecs best 23.5-24.4 ms (read 3.22 GB @ ~132-137 GB/s)
  VRAM free 2.09 GB with resident W 3.22 GB (small max-in activation scratch ~4 KB; my first run's scratch bug
  sized it to the head's element count 926 MB - fixed, the real lane activation is the tiny hidden state)
  one-time upload 646 tensors = 1.00 s (3.2 GB/s) host->device
This MEASURES the lane's central claim on this host: the dense-tier reads drop from ~80-100 ms/token (CPU
DRAM + dequant) to ~24 ms/token (VRAM, dequant-free) = ~3-4x on the dense-tier reads, driving a real
~1.4-1.5x on overall decode. Residency fits (3.22 GB W + KV/acts in 6 GB). The remaining build is the actual
gen_fast integration: route the dense tensors in in2_matvec to cuBLAS against the resident set (expert path
stays CPU), then the end-to-end per-token gate.

## CHASING 1:1 - error audit (the real accumulation question)
Two audits, both tools committed (dequant_audit.c, lane_check.cu).

### 1. Dequant is exact vs the engine (1:1 at the dequant step)
dequant_audit runs engine-vec-dot vs my-f32-dequant dot over 50 random q8_K activations per type:
  IQ4_XS worst rel 3.4e-5, Q4_K 1.4e-4, Q5_K 2.1e-5, Q6_K 9.6e-5, IQ3_XXS 4.2e-6 (60-72% of dots <=1e-6)
The worst-percentile >1e-6 are near-CANCELLING dots (small absolute value, so relative noise inflates); the
dequant formula IS the engine's formula (transcribed 1:1), so these are fp32 summation-order artifacts, not
per-element errors. The dequant step is 1:1.

### 2. The f16 COMPUTE is NOT 1:1 - but the error is common-mode (this is the real finding)
lane_check loads the single-f16 head AND the exact-f32 reference head, computes lane logits (f16 cuBLAS f32
accum) vs reference (double-accum f32 dequant) over 8 synthetic activations:
  8/8 greedy argmax MATCH, even at top-2 margins down to 0.007.
  BUT err@top1 ~ err@top2 ~ 2.7-3.2 absolute (0.1-1% relative on the large-magnitude logits the head produces).
  The top-1 and top-2 errors are nearly IDENTICAL (2.7281 vs 2.7212 = the 0.007 margin) - the f16 error is
  COMMON-MODE, shifting the top contenders together and preserving their difference.
So the f16 lane is argmax-stable on a single matmul despite a large absolute error. Caveat: this is ONE matvec on
synthetic activations; the full model runs 30 layers of f16 dense matmuls, so hidden-state errors compound before
reaching the head. Whether greedy identity holds end-to-end can ONLY be proven by the oracle gate with the lane
wired in - not by this local test. The common-mode result is encouraging but not a 1:1 proof.

## f16 lane DLL export works: wubu_cuda_resident_f16_matvec
Added a new DLL export to src/wubu_cuda_quant.cu: wubu_cuda_resident_f16_matvec(ident, h_w_f16, wbytes,
h_act_f32, K, h_y, n_cols) - holds the pre-dequantized f16 weight resident keyed by ident (own cache), runs pure
cuBLAS f16 x f16 -> f32 accum at M=n_cols,N=1,K. Added -lcublas to the DLL link. Standalone dlopen test
(tools/test_dll_f16.c) on blk.20.attn_q (K=2304,O=3072): max|d|=0.0007 on max|ref|=2.5 (rel ~3e-4), argmax MATCH.
BUG CAUGHT by the standalone test: alpha/beta passed as __half* under CUBLAS_COMPUTE_32F returned all-NaN
(cuBLAS reads 4 bytes from a 2-byte half); fixed to float one=1.0f,zero=0.0f. Also fixed the same latent bug in
lane_check. The GPU f16 resident matvec is now proven correct from the DLL - the remaining step is the fwd hook
(load each tensor's f16 file from IN2_F16_DIR once, call this fn at the dense seam under IN2_GPU_F16) + oracle.

## f16 DENSE LANE WIRED INTO gen_fast - and it is 1:1 END TO END (chase complete)
Wired behind IN2_GPU_F16=1 + IN2_F16_DIR=<dir> + IN2_CUDA_DLL=<dll>. The decode's dense path is the BATCHED
in2_matvec_nt (wq/wk/wv/wo/wgate/wrel...), which parallelises tokens and calls in2_matvec_quant inside an OpenMP
region - so the hook lives at in2_matvec_nt's top: it serial-dispatches each token's N=1 cuBLAS f16 matvec to
wubu_cuda_resident_f16_matvec, and only if EVERY token succeeds (else falls through to the CPU path untouched).
in2_matvec_quant is hooked too (omp-guarded) for the single-token full-kernel path. Experts ("exps") always CPU.

RESULT (this is the answer to "chase 1:1, too much error accumulation could be missed"):
  oracle gate WITH the lane active: PASS 15/15 tokens identical to the CPU oracle.
  F16 HIT confirmed (blk.0.attn_q K=2304 n_out=3072 dispatched), stderr "F16: scanned 646 tensors".
So the f16 attention-projection lane, with its ~3.0 absolute but common-mode logit error, is TOKEN-IDENTICAL
end-to-end on the oracle. The common-mode finding (err@top1 ~ err@top2) was the mechanism; the oracle confirms it.

Scope / honest limits:
  - output.weight (the head) is matvec'd in 8192-row chunks (need=37,748,736 vs file 926,318,592) -> size
    mismatch -> head stays CPU. Only the per-layer attention projections run on the GPU f16 lane.
  - SPEED: ~2% (12 tok 4200 ms CPU -> 4120 ms lane). Bounded because this is a MoE model: the dense tier is
    ~24 ms of a ~350 ms/token budget (~7%), so even a perfect dense lane cannot move the token much - the
    expert (MoE) traffic dominates. The "~1.4x" earlier applied to the dense TIER, which is a small slice here.

## WuBuMM hi/lo f16 endian + WuBuHIVE ring FIFO: pack ALL, stream via hive (verified)
Pulled WuBuMM (docs/H3_F16_ADAPTATION.md §6) and learned the hi/lo f16 endian trick: f16/bf16 is a 16-bit WORD
(same bit layout everywhere; endianness only matters if the 2 bytes are read apart). bf16 is the HI 16 bits of an
f32 -> decode is a BIT REINTERPRET, u=(uint32)bf16<<16; memcpy(&f,&u,4), never the numeric cast. Demonstrated:
0x3F80 -> hi/lo = 1.000000, numeric cast = 1,065,353,216 (the exact 1.065e9 bug WuBuMM fixed). Packing two f16 into
one u32 fills the 32-bit bus: on LE x86 u32 = w[2j+1]<<16 | w[2j].
WuBuHIVE wb_hive (WuBuOS substrate) is built to REPLACE FIFO rings: stable contiguous slots + skipfield + freelist.
tools/f16_hive_stream.c (built -static) packs ALL 646 tensors to u32 hi/lo over a bounded hive ring:
  packed ALL: 646 tensors, 1,609,204,224 f16 words -> 804,602,112 u32 hi/lo words = 3.22 GB -> 3,564 slabs of 1 MB
  bit-exact round-trip: PASS (mismatches=0)
  hive ring: 16 slots live (16 MB working set), wb_hive_validate=0, 1232 slabs/s, 1.11 GB/s pack+verify
KEY: the hive ring holds a 16 MB working set instead of 3.22 GB resident - the weights stream through it in FIFO
order. This is the substrate to replace the 3.22 GB permanent VRAM residency with a bounded ring that beds each
slab to VRAM as the decode walks the tensor order (next wiring step).

## bounded f16 VRAM ring measured: dense must be FULLY resident; the ring is for SUBSET streams (experts)
The DLL f16 cache now has a byte budget (IN2_F16_BUDGET_MB) with LRU eviction + slot reuse, plus
wubu_cuda_f16_stats / wubu_cuda_vram_free_mb exports. Driving all 646 tensors through a full pass per "token"
(tools/f16_ring_test.c), budgets in MB:
  budget=0 (full) : pass0 580 ms (cold upload), pass1 38.6 ms, used 3069 MB, vram_free 1886 MB, evict 0
  budget=2048     : pass1 ~430 ms, used 1302 MB, vram_free 3654 MB, evict ~650/pass
  budget=1024     : pass1 ~433 ms, used 1010 MB, vram_free 3972 MB, evict ~750/pass
  budget=512      : pass1 ~427 ms, used  506 MB, vram_free 4506 MB, evict ~843/pass
CONCLUSION: the dense (non-expert) tier is touched IN FULL every token, so any ring smaller than the 3.07 GB set
thrashes ~11x (re-uploads ~(3070-budget)MB/token over PCIe at ~4 GB/s). The dense lane must be fully resident
(3069 MB, leaves 1.89 GB VRAM free, 38.6 ms/token). The WuBuHIVE ring's real home is a SUBSET working set - the
MoE experts, where only 8 of 48 are active per token, so a bounded ring holds the hot experts without re-uploading
the world. The ring code is correct (ups/reuse/evict all scale with budget); it is simply the wrong tier to bound.

## WHERE THE TOKEN GOES (measured per-type) - and why the GPU ties: DRAM-bound, low arithmetic intensity
IN2_TIMETY per-type over a 3-token prefill (types 16/17/19/22/40/42 are the expert/dense low-bit formats):
  TY16 0.227s 21.4% 9.5 GOPS | TY42 0.213s 20.0% | TY17 0.192s 18.1% | TY40 0.156s 14.7%
  TY22 0.148s 13.9% | TY19 0.125s 11.8%   -> experts dominate, all at 6-10 GOPS (DRAM-bound)
  PHASE: q 34.2% | gate+wo 19.4% | moe 18.4% | k+v 8.3% | attn 1.2% | mix 0.2%
  HEAD = 0.006 s/token (6 ms!) - the head is NOT the bottleneck (my earlier guess was wrong; measured).
  dequant cache: 524 entries, 157 MB, 0 uncached dequants.
So the token cost is ~2.3 GB of low-bit weights read per token from DRAM at 6-10 GOPS.
GPU results: IN2_GPU_F16 dense lane 3.97 s vs CPU 3.1 s (SLOWER - per-call cudaDeviceSynchronize+H2D/D2H on
~2000 tiny attention matvecs outweighs the VRAM saving); IN2_GPU_MOE 3.92 s vs CPU 3.93 s (tie, 0.5%).
All variants 15/15 identical. CONCLUSION: on this 6 GB laptop the engine is DRAM-bandwidth-bound and the GPU
does not win because these low-bit matvecs lack arithmetic intensity to amortize the transfer; per-call sync also
hurts. The remaining lever is arithmetic intensity - batched multi-token / multi-expert GEMMs that reuse each
transfer - not more residency tricks. Added the IN2_GPU_F16 whole-vocab head call (correct, neutral here since the
head is already 6 ms; it pays off only once the head is batched).

## WHAT I DID WRONG (devil's advocate, DMA) - project-wide, not just this lane
Measured on this host (tools/dma_test.cu):
  H2D 64MB: pageable 12.0 GB/s | pinned 13.4 GB/s (1.1x - laptop WDDM, NOT the desktop 2x)
  2000 small GEMMs 2304x3072 N=1: per-call cudaDeviceSynchronize 64.5 us/call | enqueue-only 30.7 us/call
    | CUDA graph 52.4 us/call  -> per-call sync costs 2.1x (~65 ms/token over ~2000 calls)
1. DMA SIN: every GPU path I wrote round-trips host<->device with a blocking memcpy + cudaDeviceSynchronize
   PER CALL, on pageable malloc buffers, one op at a time. That is the anti-pattern: the copy engine is idle
   while the CPU waits, and the device barrier destroys overlap. 2.1x measured penalty -> this is WHY the GPU
   lanes tied or lost despite 150 GB/s VRAM. Fix: one submission + one sync per stage, resident buffers,
   events; the CUDA form of Vulkan's one vkQueueSubmit per layer.
2. BACKEND SIN: cuBLAS at N=1 is a GEMV - no 16x16 tensor-core tile, memory-bound by construction, so it
   ties the CPU DRAM read no matter how clean the DMA. The optimal already exists in the project: WuBuMM's
   Vulkan coopmat GEMM (shaders/gemm_coopmat_f16.comp, src/wubumm_vk2.c:wubumm_vk_gemm_coopmat_v6), 5.57-6.11
   TFLOPS. Its key lesson: load the A tile ONCE per K-step and reuse it across 4 N sub-tiles (naive reload =
   1.96 TFLOPS; reuse = 5.57-6.11, ~3x). Tile law: weight re-reads ~ M/tile_M (v6 128x64/4warps = 17% tensor
   + 44% DRAM; v9 8warps halves re-reads, 2.23 s faster, bit-identical; v7 drops shared staging = -10,752
   instr + 336 barriers/32 MMAs; v8 pads staging 32->36 to kill 8-way bank conflicts). One vkCmdDispatch/GEMM,
   weights persistent in device-local memory.
3. LEVER: single-token decode (M=1) is fundamentally memory-bound; a better GEMM cannot fix it. The real lever
   is BATCHING to raise M (speculative/multi-token draft, prompt batching, expert-grouped GEMM) so each weight
   read serves M>=16 rows and the coopmat tensor core engages. Project-wide: do not write new CUDA for this
   engine - the GPU stack is Vulkan (WuBuOS wubu_vulkan*, WuBuMM shaders), and nvcc objects cannot link MinGW ld.
Recorded in the wubuwizard-windows-gpu skill as a project-wide doctrine.
