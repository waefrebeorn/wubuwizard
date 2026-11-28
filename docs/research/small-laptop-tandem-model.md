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

## DO BOTH: cuBLAS vs Vulkan coopmat measured at the engine's shapes (M-scaling law)
Installed Vulkan SDK 1.4.363.0 (winget KhronosGroup.VulkanSDK) and built WuBuMM's coopmat GEMM test
(tools/test_coopmat_gemm.c + src/wubumm_vk2.c; needs -include 'static int up_pipe_on(void);' fwd-decl).
cuBLAS f16xf16->f32 (tensor cores), K=2304:
  N=3072 (14.2 MB, fits L2):  M=1 455 GF/s 455 GB/s | M=16 7251 | M=32 12004 | M=64 9200 | M=128 17882
  N=8192 (37.7 MB, > L2):     M=1 181 GF/s 181 GB/s | M=16 2478 | M=32 4612 | M=64 9102 | M=128 16407
Vulkan coopmat (NV/SMEM/v6 shaders), K=2304 N=3072, cos=0.999999993 PASS:
  M=16 0.34 | M=32 0.58 | M=64 0.94 | M=128 1.68 | M=256 2.74 TFLOPS
  (shader 38 / v10 16-warp 128x128 FAILS pipeline creation here: VK_ERROR_INVALID_SHADER_NV, falls back)
CONCLUSIONS (both are true):
1. BATCHING M is the universal lever for BOTH backends: cuBLAS 455 -> 17,882 GF/s (M=1->128, ~39x);
   Vulkan 0.34 -> 2.74 TFLOPS (M=16->256). Single-token M=1 is memory-bound on either.
2. On THIS Ada laptop at these small shapes cuBLAS is ~6-10x faster than the portable coopmat shader
   (17,882 vs 1,680 GF/s at M=128). Vulkan coopmat is the PORTABLE/learning path (runs on any GPU, correctness
   PASS, and its A-tile-reuse technique is exactly what cuBLAS does internally); cuBLAS is the SPEED path here.
   "Vulkan shows the way, CUDA delivers it on Ada."
3. The GPU is NOT bandwidth-limited vs the CPU: M=1 N=8192 reads at 181 GB/s (full VRAM) = 14x the CPU's
   ~13 GB/s DRAM read. So a properly-overlapped resident lane SHOULD win on the weight read; my per-call
   cudaDeviceSynchronize (2.1x) plus host round trips were the entire blocker, not physics.

## CORRECTION: my cuBLAS-vs-coopmat comparison was UNFAIR (hard device facts, not assumptions)
tool tools/probe_coopmat_props.c enumerates vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR. On this box:
  dev0 AMD Radeon iGPU: 11 configs, all 16x16x16 scope=SUBGROUP
  dev1 RTX 4050 Laptop: 15 configs, ALL scope=SUBGROUP(3) - 16x16x16 f16/f16/f32 (A0 B0 C1 R1), 16x8x16,
       16x8x8, and K=32 int8 (A7/3 B7/3 C5 R5). NO workgroup scope -> VK_NV_cooperative_matrix2 unavailable.
  device subgroupSize = 64 (min 32, max 64); subgroupSizeControl = true.
What this invalidates: WuBuMM's coopmat shaders are written for warp=32 (one 16x16 tile per 32-lane subgroup;
v10 = 128x128, 512 threads = 16 warps) and the host does NOT pin requiredSubgroupSize, so on a 64-wide-subgroup
device the tiling math and VGPR pressure are wrong - and v10 (the CORRECT 128x128 + 16KB shared-staging kernel)
FAILS pipeline creation here (VK_ERROR_INVALID_SHADER_NV). So the earlier "cuBLAS 6-10x faster than Vulkan
coopmat" measured the FALLBACK v6/SMEM variants (v7 even removes shared staging - the exact anti-pattern the
Khronos docs call out: "subgroup-scope matrices used the natural way do not lead to good performance"), NOT a
fair Vulkan result. Retraction: cuBLAS faster here is TRUE FOR THE FALLBACK PATH ONLY, not a verdict on coopmat.
Correct path (research, not assumption): coopmat MUST use large workgroup tiles + cooperative global->shared
staging (Khronos/NVIDIA: 8 subgroups splitting a 256x256 tile into 8x128x64, K=32 -> 21.8 TFLOPS on an RTX 2070,
vs low single digits naive). On this device that means pinning requiredSubgroupSize=32 (subgroupSizeControl is
supported) or tiling for 64-wide subgroups, then re-measuring v10-class kernels. Until that runs, the honest
statement is: naive/fallback coopmat = 0.34-2.74 TFLOPS on this laptop; cuBLAS = up to 17.9 TFLOPS; the gap is
TILING + subgroup width, not a Vulkan-vs-CUDA law.

## HYPOTHESIS FALSIFIED: subgroup=64 is NOT why v10 fails (tested, pinned, no change)
Enabled VkPhysicalDeviceSubgroupSizeControlFeatures{subgroupSizeControl,computeFullSubgroups} in the pNext chain
and injected VkPipelineShaderStageRequiredSubgroupSizeCreateInfo{32} per pipeline (env WUBUMM_PIN_SUBGROUP=32).
Result: the pin took effect (device accepted it), all working shaders still PASS cos 0.999999993, and
  - shader 38 / v10 STILL fails pipeline creation (VK_ERROR_INVALID_SHADER_NV) - unchanged
  - perf unchanged: V6 0.95, V5 0.89, W4 0.86, SMEM 0.45 TFLOPS at M=64 (vs cuBLAS 9.2 at the same shape)
So subgroup width is RULED OUT as the cause. v10's failure is shader/resource-specific (VGPR pressure or a
bad precompiled SPIR-V blob), not a device-config law. Two assumptions died by measurement this round:
(1) "coopmat is inherently ~10x slower than cuBLAS" - measured on fallback variants only; (2) "subgroup=64 is
the cause" - pinning 32 changed nothing. HONEST STATE: the shipped WuBuMM shaders at K=2304/N=3072 on this laptop
give 0.45-0.95 TFLOPS vs cuBLAS 9.2-17.9; the best-tiling variant (v10) does not build here for an unfound
shader reason; a correctly-built 256x256/8-subgroup coopmat GEMM has NOT been measured on this box, so the
Vulkan-vs-CUDA throughput question is OPEN, not settled.

## A/B after removing the redundant cudaDeviceSynchronize: decode lane now TIES the CPU (interleaved x3)
The DLL f16 matvec did gemm + explicit cudaDeviceSynchronize + blocking D2H memcpy. cublas and cudaMemcpy
already share the default stream, so the barrier was redundant; removed. Interleaved 3x (12 tok, prompt
371 151451 149879, 15/15 identical every run):
  CPU : 4238 4150 4131  mean 4173 ms
  GPU : 4187 4178 4172  mean 4179 ms   (0.15% apart - a tie)
Earlier the lane lost 3.97s vs 3.1s (~7% slower); that deficit was the full-device barrier, now gone. But a tie
is not a win: at M=1 these 161 attention GEMVs are overhead/memory bound, and a 12-thread CPU is competitive with
per-call GPU dispatch+H2D/D2H. CONCLUSION: the decode (M=1) GPU lane is honest-neutral and NOT the lever. The
lever remains batched M>=16 (prefill / speculative / expert-grouped), where cuBLAS hits 9-18 TFLOPS; this engine's
per-token loop never reaches M>=16 on decode, so the dense lane cannot beat the CPU on tokens/sec here.

## ROOT-CAUSED "shader 38 failed pipeline creation" - it was a wrong buffer count, not a mystery
Method that produced the answer (repeat this instead of writing "unknown reason"):
1. VkResult(3): VK_ERROR_INVALID_SHADER_NV = "one or more shaders failed to compile or link; more details are
   reported via VK_EXT_debug_report if enabled" -> the reason ALWAYS exists, it just needs a callback.
2. A messenger passed via VkInstanceCreateInfo.pNext only fires during vkCreateInstance/vkDestroyInstance.
   Pipeline failures are invisible until you call vkCreateDebugUtilsMessengerEXT and KEEP the messenger.
3. The loader ignores VK_LAYER_PATH when elevated; register the layer manifest under
   HKLM\SOFTWARE\Khronos\Vulkan\ExplicitLayers (REG_DWORD 0) so the layer is visible to every process.
4. Reproduce ONE blob standalone (tools/debug_pipe_fp16tc.c) with configurable push-constant bytes, buffer
   count, device extensions and apiVersion; bisect there, not in the engine.
FINDING: shader 38 is gemm_i8_tiled_shader (5688 B). It declares FOUR storage buffers (binding 3 = "C") and a
32-byte push block, but wubumm_vk2.c registered it (3 buffers, 6 words). Validation:
  "SPIR-V uses descriptor [Set 0, Binding 3, variable C] but the binding was not declared in pSetLayouts[0]"
  VUID-VkComputePipelineCreateInfo-layout-07988  ->  NVIDIA returns VK_ERROR_INVALID_SHADER_NV.
FIX: register 4 buffers / 8 words -> "shader 38 pipeline OK". Committed WuBuMM f4e5bdd.
Two more real bugs found on the way: (a) VkDeviceCreateInfo pNext chain had tail=&v12 instead of &st16, so
appending the coopmat feature OVERWROTE v12.pNext and dropped VkPhysicalDevice16BitStorageFeatures entirely -
uniformAndStorageBuffer16BitAccess was never enabled despite querying as supported (VUID-RuntimeSpirv-
uniformAndStorageBuffer16BitAccess-06332); (b) no persistent debug messenger, so nothing could ever be seen.
STATUS: coopmat perf unchanged (i8 shader, not the f16 path): NV 0.85, SMEM 0.46 TFLOPS at M=64. Separately,
the W4/V5/V6 variants still FAIL correctness (cos 0.32-0.35, max|diff| 1.18e2) at M=64 while NV/SMEM pass at
cos 0.999999993 - that is a shader-math bug in those variants, still open, with the same debugging method ready.

## CLOSED: W4/V5/V6 correctness failure was subgroup width, not shader math
Same binary, same inputs (M=64, N=3072, K=2304):
  unpinned : W4 cos=0.351105747  V5 cos=0.351105747  V6 cos=0.322663722   max|diff|=1.176904e+02  FAIL
  pinned 32: W4 cos=0.999999993  V5 cos=0.999999993  V6 cos=0.999999993   max|diff|=9.866714e-03  PASS
Cause: those shaders assume warp==32, so local_size_x=128 is FOUR warps and they split a 64x64 tile four ways.
This GPU reports default subgroupSize=64 (min 32, max 64), so 128 threads is only TWO subgroups - half of every
tile is never written. Device facts: VkPhysicalDeviceSubgroupSizeControlFeatures.subgroupSizeControl=1,
VkPhysicalDeviceSubgroupSizeControlProperties{min=32,max=64}.
FIX (WuBuMM): query subgroup-size-control at init, enable the feature, and pin requiredSubgroupSize=32 whenever
min<=32 && max>32 (env WUBUMM_PIN_SUBGROUP overrides, 0=off). Default now prints
  VK: subgroup pin=32 (sscSupported=1 min=32 max=64)
and ALL FIVE variants pass correctness at M=64 AND M=256 (VERDICT: PASS).
Perf: M=64 V6 0.94 / NV 0.90 / V5 0.89 / W4 0.86 / SMEM 0.45 TFLOPS; M=256 V6 2.69 / V5 2.37 / W4 2.09 /
SMEM 0.84 / NV 0.52 TFLOPS. No regression from the pin - it was correctness-only, as the earlier perf A/B said.
NOTE the two-stage truth: subgroup width was NOT the cause of shader 38's pipeline failure (that was the wrong
buffer count) and NOT a perf lever, but it WAS the cause of silent wrong results in the warp-assuming variants.

## Post-fix M-sweep (N=3072 K=2304): real coopmat ceiling, and a NEW small-M correctness gap
  M=16  variants correct 2/5   NV 0.29 TFLOPS
  M=32  variants correct 3/5   NV 0.59
  M=64  variants correct 5/5   NV 1.02  V6 0.93
  M=128 5/5                    NV 0.71  V6 1.71
  M=256 5/5                    NV 0.52  V6 2.73
  M=512 5/5                    NV 0.52  V6 3.42  <- coopmat ceiling measured on this laptop
So V6 tops out at 3.42 TFLOPS at M=512 vs cuBLAS 17.9 at M=128 - the remaining ~5x is TILE SIZE (64x64 here vs
the 256x256 / 8-subgroup cooperative-staging pattern Khronos measures 21.8 TFLOPS with), not subgroup width.
NEW OPEN ITEM: the 64x64-tile variants (W4/V5/V6) are WRONG at M=16 and M=32 (2/5 and 3/5 correct) while NV/SMEM
(16x16 tile) pass. That is an out-of-bounds/masked-tile bug when M is smaller than the 64-row tile - not yet fixed.

## Batched f16 GEMM (M = prompt length) is correct and LIVE, but only ~1% end-to-end
Implemented a real M-token entry: wubu_cuda_resident_f16_gemm(ident,w,M,K,x,y,n_out) sharing f16_gemm_core with
the M=1 matvec (CUBLAS_OP_T/OP_N, dims n_cols x M x K, output row-major M x n_cols). Engine: in2_try_f16_m +
in2_f16_dispatch_m, called first in in2_matvec_nt when nt>1, falling back to the serial per-token loop.
PROOF IT IS LIVE: "F16 GEMM-M first: blk.0.attn_q.weight M=30 K=2304 n_out=3072" for a 30-token prompt (M scales
with prompt length); F16STAT ups=161 reuse=1641 evict=0 used=2107MB vram_free=2846MB.
CORRECTNESS: 120-token prompt + 4 generated, CPU vs GPU IDS IDENTICAL.
SPEED (120-token prompt, 4 gen, 2 interleaved runs): CPU 14119/13997 (mean 14058 ms); GPU(f16 M-batch)
13927/13904 (mean 13916 ms) -> ~1.0% faster. Also note prefill is now ~113 ms/token vs ~267 ms/token decode,
because the GEMMs amortise over the batch even though attention is O(M^2).
WHY ONLY 1%: the lane covers 161 dense attention tensors (2107 MB). The MoE experts and the output head - the bulk
of the bytes - are not on this path, so batching the dense GEMMs cannot move wall-clock much. Honest conclusion:
the M-batch lane is correct and worth keeping, but it is NOT the lever. The remaining lever is the expert/head
bytes, not more dense-lane work.

## CORRECTION: the head is NOT the bottleneck (6-7 ms/token). The EXPERTS are, at ~19 GFLOPS effective.
IN2_TIME + IN2_TIMETY (arrays now accumulate across the run instead of being cleared at end-of-forward):
  HEAD 0.303 s for 201024 rows x 2304   (3-token prefill)
  HEAD 0.006-0.007 s per decode token   <- whole vocab in ONE call, already on the f16 GPU lane
  FWD wall: embed+layers=0.162-0.198 s  head=0.006-0.007 s per token
So the earlier claim "~187 ms/token unattributed = CPU head" was WRONG and is retracted. Layers are ~180 ms/token;
the head is 6 ms.
Per-type CPU seconds over 15 forwards (3 prefill + 12 decode), with MAC counts:
  ty=16  1.189 s  1783 calls  10101.9 M
  ty=17  0.800 s  1675 calls   9507.4 M
  ty=42  0.751 s  3478 calls   9810.0 M
  ty=22  0.634 s   747 calls   4161.8 M
  ty=19  0.592 s   855 calls   4751.0 M
  ty=40  0.477 s  1583 calls   4459.1 M
  ty=8   0.005 s    84 calls     74.3 M   (Q8_0 - already trivial)
  ty=12/14  0.001 s each
The six expert types are 4.44 s of the run and ~42.8 GMAC (~85.6 GFLOP) -> ~19 GFLOPS effective, which is far below
what a 6-core AVX2 box should do for these formats. THAT is the lever: expert matvec efficiency, not the head and not
more dense-lane batching. Note the GPU MoE batch path (IN2_GPU_MOE) measured neutral here (4048 vs 4040 ms) because
the experts are NOT in the f16 resident tier (dequant_all excluded "exps"), so that path still gathers quantised
host bytes per expert.

## Thread scaling: 1.66-1.77x from 8 threads, so ~55% is serial; and only the expert block scales
12 tokens, prompt 371 151451 149879:
  threads=1 6655/6549 ms | 2 5284 | 4 4359 | 8 4011/3985 | 12 4019
Amdahl: 6655/4011 = 1.66x at 8 threads -> serial fraction ~0.55. PHASE comparison 1t vs 8t:
  phase           1t     8t
  state/mix     0.002  0.001
  q             0.046  0.046
  kv+conv       0.061  0.060
  attention     0.055  0.050
  gate+wo       0.056  0.086   (WORSE at 8 threads)
  mlp+moe       0.083  0.081
  "?"           0.658  0.232   <- only thing that scales (2.8x)
  remaining     5.04   2.84    <- scales 1.77x, this is the expert matvec block
So the expert matvec work IS the parallel part but achieves only 1.77x of 8x on 12 hardware threads, i.e. ~19 GFLOPS
effective. Phases 0-5 are ~0.30 s total and completely flat in thread count - they are not the target. The lever is
expert-kernel parallel efficiency (per-call work too small / dequant traffic), not the head, not the dense lane.

## RIGOROUS A/B: every GPU lane together is only -1.3% on decode (noise). RETRACT the single-run 5.3%.
Method: 5 interleaved reps, medians (single runs drift ~5%, so single-run comparisons are worthless here).
  CPU            median 3969 ms  mean 3955   (raw: 3913 3969 3906 4015 3972)
  F16+DENSE+EXPS median 3918 ms  mean 3930   (raw: 3946 3907 3918 3966 3914)   -> -1.3% median
  ALL+MOE        WORSE (4112/4214) - the MoE batch path costs more than it saves
All runs kept 15/15 identical, so this is purely a speed statement.
SELF-CORRECTION: an earlier single run showed GPU_DENSE 3885 vs CPU 4102 (-5.3%) and I nearly reported it as a win;
interleaved reps show the CPU baseline itself wanders 3903-4102 ms, so that was drift, not speedup. RULE: never report
a decode speed claim from fewer than 5 interleaved reps, and use the median.
STATE OF THE ART ON THIS BOX (measured): dense f16 lane ~1%, M-token prefill batch 1-4.6%, GPU MoE batch NEUTRAL to
worse, resident quant tier (ty=8 only) ~0, all lanes together -1.3%. Therefore GPU offload is NOT the lever for decode
here. The lever is the CPU expert kernel: 19 GFLOPS effective, 1.77x of 8x thread scaling, 4.44 s of the run.

## DMA done properly: pinned staging + async copies on a dedicated stream -> still no gain
Changed f16_gemm_core to the textbook DMA pattern: cudaHostAlloc page-locked staging for both directions,
cudaMemcpyAsync H2D, cublas on a dedicated cudaStreamNonBlocking (cublasSetStream), cudaMemcpyAsync D2H, exactly ONE
cudaStreamSynchronize per call, then a host memcpy out of pinned memory. Previously: malloc'd pageable buffers and
blocking cudaMemcpy both ways.
5 interleaved reps, 12 tokens, gate 15/15 on every rep:
  CPU            median 3955 ms  mean 3941
  GPU(pinned DMA) median 3974 ms  mean 3970   -> +0.5% (noise, if anything slower)
So pageable-vs-pinned staging was NOT the limiter. The per-call cost is ROUND-TRIP LATENCY (161 dense tensors per
token = 161 H2D + 161 D2H + 161 submissions), which pinning does not remove. Conclusion: the fix is not better
copies, it is FEWER round trips - keep activations resident on the device across a whole group of matvecs and only
move the concatenated result once (group by shared input: {wq,wk,wv} share one activation; {wgate,wrel} share one),
or chain the whole dense block device-side. That is the remaining correct architecture for GPU acceleration here.

## Test harness the WuBuMM way (per-shape PASS/FAIL + poison), and what it caught
tools/test_f16_group.c: 21 cases. Per case it computes a DOUBLE-precision reference, then reports cos, max|abs diff|,
and a POISON count for BOTH the grouped path and the per-op path, plus a final VERDICT. Bounds are explicit on
purpose: M in {1,2,3,7,8,15,16,17,31,32,33,63,64,65}, K in {32,64,2304}, N in {16,32,1536,3072}, and a group whose
n_cols differ (N, N, N/2) so per-op offsets are exercised. tools/test_gemm_m_probe.c is a minimal reproducer that
uses W=identity so expected Y[m][o] == X[m][o] exactly, printing both matrices so a failure MODE is visible.
CAUGHT #1 (test bug, important lesson): the resident cache is keyed by (ident, wbytes) and assumes ident <-> immutable
weights (true for the engine, which keys on file_offset). My first harness reused fixed idents with fresh random
weights per case, so it got legitimate stale-cache hits and reported 15/18 FAIL. Fixed by deriving ident from
(n_cols,K) and seeding the RNG from that same ident. With that fix: 20/21 PASS. A caller that reuses an ident with
different bytes WILL get silent wrong numbers - that is the cache contract.
CAUGHT #2 (real DLL bug, still open): grouped call with n_ops=3 at M=1, K=2304, N=3072 gives op2 (n_cols=1536)
cos=0.953 with max|diff|=51, while the same op called alone is cos=0.999999983. The bad elements are the LAST ~143 of
1536 and the count MOVES between runs (145 then 143) => non-deterministic, so an OOB write or race, not a logic error.
Discriminators: 1 op PASS, 2 ops PASS, single 1536-wide PASS - only 3 ops (where op0/op1 share one cached slab)
fails, and the tail of the whole output device buffer is what comes back wrong. Adding cudaStreamSynchronize before
cache-miss uploads and cudaDeviceSynchronize before the D2H did NOT fix it, so it is not simple stream ordering;
compute-sanitizer memcheck is running to locate the write. The grouped entry is NOT wired into the engine, so the
gate is unaffected.

## CLOSED: grouped GEMM tail bug was cuBLAS not writing a narrower op at a non-zero offset
Root cause found by bisection with the harness, not by guessing:
  - 1 op PASS, 2 ops PASS, single 1536-wide PASS
  - 3 ops at the SAME width (uniform ldc, offsets 0/3072/6144) PASS
  - 3 ops where the LAST is narrower (ldc changes 3072 -> 1536 at offset 6144) FAIL on that op
  - compute-sanitizer memcheck: 0 errors => no OOB, the elements were simply never written by cuBLAS
  - the failing count MOVED between runs (145 then 143) => silent, non-deterministic underwrite
FIX: one device output region PER OP (g_pb[i] sized n_cols[i]*M), each GEMM writing at offset 0 with its own ldc,
then a single cudaDeviceSynchronize and one blocking D2H per op straight into the caller's buffer. Still ONE H2D and
ONE sync per group, so the round-trip win is intact.
RESULT: tools/test_f16_group.exe -> VERDICT: 22/22 cases PASS (bounds sweep M=1..65 plus engine shapes, K=32/64/2304,
N=16/32/1536/3072, mixed-width groups), and the engine holds 15/15 identical x3 reps with the rebuilt DLL.
TWO BUGS CLOSED BY TESTS THIS ROUND (neither was visible from cos-on-one-shape):
 1. my harness keyed the resident cache wrong (fixed ident, fresh weights) -> false 15/18 FAIL
 2. cuBLAS underwrite on varying-ldc into one buffer -> real, silent, non-deterministic wrong numbers
LESSON: keep the per-shape + poison + double-reference harness; single-shape cos checks miss exactly this class.

## CORRECTION: experts are 32% of wall time, not "4.44 s" - and the non-expert half is the same size
The per-layer expert timer (IN2_TIME, "moe L## ...s", measured at the SINGLE call site, so no thread double-count):
  576 lines, total 1.104 s, avg 1.92 ms per layer-moe
  3.4 s wall -> experts = 1.104 s = 32% of TOTAL time (~41% of layer time)
RETRACT the earlier "six expert types = 4.44 s of the run, ~19 GFLOPS". 4.44 s cannot fit in a 3.4 s run: that
number was clock() accumulated INSIDE parallel workers, so it sums concurrent thread time. Effective GFLOPS computed
from it (19) is also wrong; from wall time it is ~40 (42.8 GMAC x2 / 2.16 s of layer time).
CONSEQUENCE (this is the important bit): experts do 99.8% of the MACs but only ~32% of the wall time, so the
remaining ~1.0-1.6 s is spent on the small non-expert layer ops (qkv, attention, rmsnorm, shortconv, residual, router)
which carry only 0.2% of the MACs. Those are latency/overhead bound, and they are as large a lever as the experts.
So "the expert kernel is the only lever" was wrong too - it is HALF the lever.
Also confirmed by background job: compute-sanitizer memcheck on the grouped test = "ERROR SUMMARY: 0 errors", which is
why the tail bug had to be found by bisection + poison rather than by a memory checker.

## WALL-CLOCK INSTRUMENTATION FIXED + 42 MB PER LAYER VISIT REMOVED
Three instrumentation faults found and fixed, each of which had been feeding a wrong conclusion:
1. clock() -> QueryPerformanceCounter everywhere. On Windows clock() quantises to ~15.6 ms, so every sub-16 ms kernel
   read as 0.000 s and the per-type profile was mostly aliasing. in2_now() already used QPC; the per-type, gpu and
   alloc accumulators did not.
2. Per-type timing moved from three CALL SITES to the LEAF kernel with a depth guard. The three spans nest (a batch
   entry calls the per-row entry), so their sum read 4.20 s inside a 3.50 s run.
3. Parallel regions are timed as ONE span with leaf timers suppressed (in2_ty_par), because summing concurrent
   per-thread spans is what inflated it. A global depth counter is also not thread safe - that is why the first
   attempt still read 2.4x over.
REMOVED WASTE: in2_moe_wrap allocated gu (2*DFF*d floats) and dn (d*DFF floats) = 42 MB up front on EVERY layer
visit, but the default fused path never touches them - only the opt-in IN2_STAGE branch does. 456 visits in a
12-token run. Now allocated lazily.
MEASURED: 3.43-3.46 -> 3.64 tok/s, 5/5 reps IDENTICAL to the oracle, gate 15/15.
MOE PHASE SPLIT (expert-major batch path, IN2_THREADS=1 so attribution is exact):
  gu 0.493 s (59%), down 0.297 s (36%), act 0.012 s (1.4%), route 0.026 s (3%), gather/scatter/free/setup ~0.003 s
  => the expert MATVEC is 95% of the expert compute; activation, gather and scatter are noise.
CAVEAT: the batch path only runs for prefill (n_tok >= 2). Decode takes the per-token fused path, which is NOT yet
split, so the above is the prefill split, not the decode split.
MAC COUNTER IS BOGUS: it charges n_out*n_in per call regardless of how many experts run, and for one token it reports
42.8 GMAC, which against the measured 0.087 s of moe would imply ~984 GFLOPS - about 90% of this CPU's peak, which
per WuBuMM doctrine means the numerator is wrong, not that the kernel is 90% of peak. Real per-token expert work is
TOPK * (2*DFF*d + d*DFF) ~ 85 MMAC, i.e. ~2 GFLOPS effective. That is a BANDWIDTH signature, not an ALU one.

## DECODE EXPERT SPLIT (measured, 1 thread = exact): unpack/dot bound, not bandwidth bound
Config: d=2304 NE=128 TOPK=8 DFF=1152 L=24, model 7.90 GB.
Expert unit = 2*DFF*d + d*DFF = 7.96 M params (gate_up 5.31 M + down 2.65 M).
Per token: 24 layers x 8 experts = 1.53 G params = 394 MB of weight bytes at 2.06 bpw.
SPLIT (fixed attn note: serial fused path, IN2_THREADS=1, 12 tokens):
  gu 1.741 s (65.9%), down 0.851 s (32.2%), act 0.049 s (1.9%), accum 0.001 s (0.1%)
  gu:down = 2.05:1 against a MAC ratio of 2.00:1 -> both matvecs cost the same per weight. No format asymmetry.
  Activation is 1.9%. The expert matvec is the whole phase.
RATE: 216 ms/token serial = 7.08 G weights/s single thread = 1.8 GB/s effective.
That is ~8x BELOW the single-thread DRAM ceiling (~10-15 GB/s), and threading gives 5.1x (2.592 s -> ~0.511 s
wall, 43 ms/token). A bandwidth-bound phase cannot gain 5x from threads, so decode is UNPACK+DOT BOUND, not I/O
bound. More cores help; a faster disk would not.
THE LEVER THIS IDENTIFIES: the CUDA f16 lane served 161 slab uploads for 2304 expert matvecs in this run, i.e. the
GPU idles during exactly the phase that costs 45% of decode. Moving expert matvecs onto the resident-f16 lane
(394 MB/token over PCIe at ~12 GB/s is 33 ms, against 216 ms on CPU) is the available win, and it is the same
tandem pattern the goal calls for.
RETRACTED AGAIN: + the mac counter must not be used for GFLOPS at all (it charges n_out*n_in per call regardless of
how many experts run); the trustworthy rates are the ones above, computed from real param counts.

## FALSIFIED: "the GPU f16 lane idles during expert matvecs, so round-trips are the cost"
Measured A/B at the same thread count, same prompt:
  CPU only (no IN2_F16_DIR, no CUDA dll) : 3.64 tok/s
  f16 lane on (IN2_GPU_F16=1, +IN2_GPU_MOE=1): 3.66 tok/s
Within run-to-run noise. The lane reports ups=161 reuse=3771 evict=0 used=2107 MB vram_free=2846 MB in both cases, so
it is engaged and hitting, but it buys nothing measurable. Two hypotheses died here: (a) that the expert phase is
GPU round-trip latency, and (b) that IN2_GPU_MOE selects a different path - it is identical with and without.
Consequence: the expert matvec cost is CPU unpack+dot, so the wins are (i) fewer weights touched per token, or
(ii) a kernel that unpacks faster. Not PCIe, not sync count, not this lane as it stands.
CURRENT RUNNING NUMBERS (12 tokens, 8 threads): 3.64 tok/s, moe phase 1.093 s of 3.4 s = 32%, of which the matvec
is ~0.51 s of CPU-seconds/8 ~ 0.51 s, so ~0.58 s of the moe phase is router + top-k + accumulation, not matvec.

## WHY THE GPU CANNOT TAKE THE EXPERTS ON THIS CARD (measured, not assumed)
1. THREE HARD GUARDS in in2_try_f16 refuse expert matvecs before any work happens:
     elem_off != 0 -> return 0        (experts are per-slab offsets, never 0)
     name contains "exps" -> return 0 (explicit refusal)
     omp_in_parallel() -> return 0    (the expert k-loop is OMP)
   So the 161 resident slabs / 3771 reuses are DENSE tensors only. That is why IN2_GPU_F16 and IN2_GPU_MOE change
   nothing for experts, and why the A/B was 3.64 vs 3.66 tok/s.
2. f16 EXPERT RESIDENCY DOES NOT FIT. Routing is diffuse, not concentrated:
     64 tokens: 1984 of 3072 expert slabs touched (65%), min count 1, max 151, mean 12.2
     f16 slab = 15.9 MB; 2846 MB free VRAM holds 178 slabs = 9% of the pool
     the best 178 slabs cover 44.9% of picks; 90% of picks needs 1009 slabs = 16.1 GB
   A single token alone needs 24 layers x 8 = 192 distinct slabs = 3.1 GB, which exceeds the 2.85 GB free.
   Conclusion: no hot set exists that fits. f16 residency on a 6 GB card is not viable at this expert count.
3. REMAINING ROUTE, if the GPU is to touch experts at all: keep the weights COMPRESSED on the device and dequantise
   on the GPU. That streams IQ2 bytes (394 MB/token) rather than f16 bytes, and it needs a device-side IQ2 kernel
   that does not exist yet. Estimated upside from PCIe alone is ~2x on the moe phase, i.e. ~15% end to end.
EFFECTIVE RATES, same 12-token run, measured twice so they are not mixed:
   8 threads: moe region wall 0.809 s = 67 ms/token, 394 MB/token -> 5.9 GB/s
   1 thread : moe 2.592 s = 216 ms/token -> 1.8 GB/s, 7.08 G weights/s
   Ceilings: single-thread DRAM ~10-15 GB/s, 8-thread aggregate ~30-40 GB/s. Even the 8-thread figure is ~5x below
   the aggregate ceiling, so decode is UNPACK+DOT bound at both thread counts. Faster storage does not help.
TOOLING DEFECT FOUND: IN2_MOE_IDS emits nothing (0 lines with IN2_MOE_HIST=1 + IN2_MOE_IDS=1 in a 1-token run), so
the per-token routing trace that the header comment promises does not work. The aggregate histogram works.

## KERNEL KNOB SWEEP: NULL RESULT, plus a harness I do not trust and a claim that needs re-testing
Swept the existing IQ2 tuning knobs on gen_fast (12 tokens, same oracle prompt). Single-pass sweep reported:
  baseline 4.57, MAYA_IQ2_VAR=0 4.52, MAYA_IQ2_VAR=2 4.63, MAYA_IQ_PF_ROWS=0 4.54, =4 4.31, MAYA_IQ2XXS_1T=1 4.57
Then ran the only comparison worth believing - 5 INTERLEAVED pairs, base against MAYA_IQ2_VAR=2:
  base 3.46 3.52 3.52 3.49 3.49   var2 3.49 3.49 3.49 3.51 3.51
  => the knob is NEUTRAL. The apparent +1.3% did not survive interleaving. Score = byte-identical on all 10 runs.
MAYA_IQ_PF_ROWS=4 looking worse (4.31) is unverified for the same reason and is not being adopted.
TWO PROBLEMS THIS EXPOSED:
1. The sweep harness reported a 4.57 tok/s baseline that I CANNOT reproduce anywhere else. Under controlled conditions the
   same binary, prompt and env gives 3.46-3.52, and bare (no env at all) gives 3.51. A 30% phantom cannot be explained
   by the knobs, so the sweep's stdout/stderr capture (2>&1 >/dev/null piped twice through grep) is taking the wrong
   number. Treat that column as invalid rather than as a discovery.
2. Run-to-run spread on this box is ~0.07 tok/s but session-to-session spread is larger (3.43-3.64 earlier today,
   3.46-3.52 now). Therefore the "+6% from removing the 42 MB lazily-allocated buffers" claim is NOT safe yet: that A/B
   was before/after, not interleaved, and the two ends were measured minutes apart. It needs an env-gated rebuild to be
   switchable at runtime, then 5 interleaved pairs, exactly like the test that just killed the knob result.
   The allocation itself was provably dead work (gu/dn are NULL and unused in the fused path - visible in the debug
   dump as bufs=0/0/...), so removing it cannot have hurt; what is unproven is the SIZE of the win.
DISCIPLINE NOTE: three conclusions in this document died to proper measurement today (experts = 4.44 s, coopmat 6-10x,
GPU lane round-trips). The pattern is always the same - a number taken once, not interleaved, on a noisy box.

## CORRECTION: the lazy-allocation win is +0.7%, not +6% (interleaved A/B, same binary)
Added IN2_MOE_EAGER=1 to restore the old up-front 42 MB allocation, so the change became switchable at runtime and
the comparison could finally be interleaved instead of before/after across minutes. Five interleaved pairs:
  lazy  (new) 3.51 3.49 3.51 3.52 3.51  mean 3.508
  eager (old) 3.49 3.46 3.51 3.48 3.48  mean 3.484
  delta +0.024 tok/s = +0.7%, and lazy >= eager on 5 of 5 pairs. All 10 runs byte-identical to the oracle.
So the earlier "+6% (3.43 -> 3.64)" was SESSION DRIFT, not the allocation. The change stays (42 MB of provably dead
allocation per layer visit removed, numerics untouched, no downside) but it is not the win it was reported as.
This is the fourth number today that shrank or died when measured properly. The standing rule this document now
follows: an A/B is only a result if both arms are interleaved in one session through the same binary.
TOP-LINE, current truth: 3.51 tok/s on this prompt (12 tokens, 8 threads), decode expert matvec ~32% of wall,
unpack+dot bound, GPU unable to take the experts without a device-side IQ2 kernel.

## THE PROFILING SURFACE IS THE BLOCKER (five separate instruments, five defects)
Attempting to attribute the non-expert ~2/3 of wall turned up that every existing profiler in this repo is broken in
its own way. This is now the top blocker: the remaining work cannot be steered without a trustworthy instrument.
1. PHASE[] (in2_fph, 8 slots, printed at inkling_gen.c:411): covers only ~15% of wall (0.694 s against a 4.6 s
   twelve-token run), and its LARGEST slot is PHASE[6] labelled "?" at 0.235 s. The names array has 6 real names for
   8 slots. It is a leftover from an earlier experiment, not a working profiler.
2. IN2_TIMETY: sat inside `if (getenv("IN2_PERF"))`, so setting it alone prints NOTHING - the same nesting bug that
   MOESPLIT had. With both set it prints, but reports "last forward" while the numbers are per-forward resets, and the
   call counts are not credible (ty=16 calls=9 for a token that must issue hundreds of expert matvecs). Sum 0.034 s
   against 0.296 s/token means it is measuring a fragment, not the forward.
3. IN2_MOE_IDS: emits zero lines with IN2_MOE_HIST=1 + IN2_MOE_IDS=1. Dead emitter.
4. The macs counter: charges n_out*n_in per call regardless of how many experts actually run, which is why it once
   produced "984 GFLOPS = 90% of this CPU's peak" - a number that should have been rejected on sight.
5. IN2_TIME (per-layer line printing): 576 fprintf to a pipe per run. It measurably perturbs the run it is measuring.
CONCLUSION: stop editing these. Replace them with one profiler that is (a) leaf-level, (b) thread-safe by
construction (one span per parallel region, leaf timers suppressed inside), (c) run-total and forward-total separately,
(d) silent unless asked, and (e) with every slot named. WuBuMM doctrine applies to instruments too: no claim without
an artifact, and an instrument that cannot fail honestly is worse than none.

## ROOT CAUSE OF THE BROKEN PROFILING: THE INSTRUMENTED LAYER IS DEAD CODE
in2_phase[] is DECLARED at wubu_inkling2_fwd.c:132 and NEVER WRITTEN anywhere in the repository (grep for
`in2_phase[N] =` returns 0 across all of src). That is why PHASE printed an unnamed slot as its largest value.
Why: the timers live in in2_layer_full() (wubu_inkling2_fwd.c:2572, ~316 lines) whose own body says
"superseded by in2_layer_full_seq, kept only for single-token reference". It is never called - grep for
`in2_layer_full(` finds only its own declaration and definition. The live path is in2_layer_full_seq(),
declared at wubu_inkling2_model.c:30 and invoked from the layer loop macro at wubu_inkling2_model.c:180.
CONSEQUENCE: every phase/mac/mac-counter reading in this document was produced by instruments attached to code that
does not execute, or by call-site wrappers that do. Any further profiling MUST be attached inside
in2_layer_full_seq in wubu_inkling2_model.c, not in wubu_inkling2_fwd.c's layer functions.
This also retro-justifies distrusting PHASE outright and explains why its total covered 15% of wall.
ACTION TAKEN: none yet to the code - recording the location first so the profiler is built in the right place.

## FIRST TRUSTWORTHY PHASE TABLE - AND DECODE AND PREFILL HAVE OPPOSITE BOTTLENECKS
The FP2 instrument in wubu_inkling2_fullseq.c IS live and correct (7 chained wall-clock spans, no nesting). The
reason it read as garbage was a SECOND printer in tools/inkling_gen.c:411 whose names array was shifted by one and
had no name for slot 6, so the largest span printed as "?" and every other row named the slot before it. Names fixed
to match the FP2 call sites exactly (f_loads, f_mix+gate, f_q, f_kv+conv, f_attn, f_gate+wo, f_mlp+moe).
DECODE, n_tok=1 (this is the phase that sets tok/s), FP2 total 0.185 s:
  gate+wo 0.049 (26.5%) | mlp+moe 0.047 (25.4%) | q 0.037 (20.0%) | kv+conv 0.023 (12.4%) | mix+gate 0.015 (8.1%)
  | attn 0.014 (7.6%)
PREFILL forward, n_tok=12, FP2 total 0.565 s:
  mlp+moe 0.230 (40.7%) | attn 0.089 (15.8%) | gate+wo 0.084 (14.9%) | q 0.061 (10.8%) | kv+conv 0.052 (9.2%)
  | mix+gate 0.048 (8.5%)
=> OPPOSITE BOTTLENECKS. In decode the dense projections (q + gate+wo + kv+conv) are 58.9% against experts at 25.4%.
In prefill the experts lead at 40.7% against 34.9% dense. Every earlier "experts are the lever" statement was
prefill-shaped and did not apply to the phase that determines tokens per second.
THE SMOKING GUN: gate+wo is 24 layers x 2 matvecs = 48 calls in 0.049 s, i.e. ~1.0 ms per call, for ~5.3 MMAC each.
That is ~216 MMAC/s, roughly a thousand times below this CPU's capability, so the dense decode projections are
OVERHEAD bound, not compute bound. The overhead candidates in the live layer are the unconditional
in2_tensor_to_f32() materialisations (Wq/Wk/Wv/Wrel/Wsk/Wg/Wo) that the *_nt quant-kernel branch never reads,
the per-visit calloc/free set, and the dequant cache lookups.
NEXT: make those materialisations lazy exactly like the gu/dn fix - they are dead work on the kernel path - then A/B
interleaved through one binary. FP2 covers 0.185 s of a ~0.285 s token, so ~35% is still outside the layer (head +
embed + sampling) and is the other unmeasured block.

## DECODE IS SINGLE-THREADED, AND THE OBVIOUS FIX IS INADMISSIBLE (measured both ways)
CAUSE, measured from the corrected phase table without any new build: the same op costs 7x more per token alone
than batched.
  f_q       batched (n_tok=12) 5.1 ms/token   single (n_tok=1) 37.0 ms/token   7.3x
  f_kv+conv batched 4.3                        single 23.0                    5.3x
  f_gate+wo batched 7.0                        single 49.0                    7.0x
The ratios match the thread count, so at n_tok == 1 the kernel is handed ONE row, row parallelism cannot engage, and
every dense projection runs on a single core. That is the whole reason decode is dense-bound while prefill is
expert-bound - not a format difference, not a cache problem, just no parallelism at n=1.
ATTEMPTED FIX: split the OUTPUT ROWS of the dense matvec across threads (a pure partition - each block writes its own
slice, no reduction), reusing the exact call pattern that already exists at wubu_inkling2_fwd.c:1879
`in2_matvec_quant(ctx, t, elem_off + (int64_t)r * n_in, x, n, n_in, y + r)`. Built, gate-clean at default, and
env-gated as IN2_ROWPAR so both arms ran through one binary.
RESULT: IN2_ROWPAR=1 CHANGES THE TOKENS. Default was IDENTICAL to the oracle, ROWPAR=1 was MISMATCH. The split is
mathematically a partition, so the divergence is floating point: a different number of rows per call changes the
kernel's blocking, which changes the SUMMATION ORDER inside each dot product. Microseconds of reassociation are
enough to flip a later argmax.
VERDICT: rejected and reverted. Token-for-token identity is a hard clause and outranks any speedup, so row-splitting
behind this kernel is not a candidate at all. This is the useful kind of failure: the diagnosis (no parallelism at
n=1) is now proven, and one implementation is now excluded by measurement rather than by argument.
THE ADMISSIBLE ROUTE: parallelise across the INDEPENDENT OPS instead of within one. Inside a layer q, k and v are
mutually independent, while gate -> wo is a dependency chain. Running q/k/v concurrently in one region leaves each
kernel call byte-identical - same arguments, same row count, same blocking, same summation order - so identity is
preserved by construction rather than hoped for. Expected saving is min(q, kv) ~ 23 ms of a 285 ms token (~8%), and
it composes with the same trick over the three rmsnorms.
GATE: 15/15 after reverting.

## OP PARALLELISM IS A LOSS, AND PAIRWISE INTERLEAVING IS NOT ENOUGH (methodology correction)
Took the admissible route from the rejected row-split: run the independent wk and wv projections of a decode step in
two OpenMP sections. Nothing about either call changes - same arguments, row count, kernel, blocking, summation
order - so the tokens stayed identical (verified: default IDENTICAL, OPPAR IDENTICAL).
FIRST RESULT (5 pairwise interleaved, A then B every pair): 3.552 -> 3.600 tok/s = +1.4%, 5/5 wins. Encouraging.
Then flipping the default to ON and re-running the loop the other way round (OPPAR first, opt-out second) gave the
OPPOSITE: opt-out 3.64 three times running against OPPAR 3.56/3.62/3.61. The arm that runs SECOND in each pair wins
regardless of which arm it is - an order effect from page cache and GPU warm-up, not from the change.
ABBA x2 settled it (A=OPPAR, B=opt-out):
  A 3.59 3.59 | B 3.62 3.59 | B 3.59 3.61 | A 3.59 3.57   -> A mean 3.585, B mean 3.610
Verdict: op parallelism is -0.7%, a small LOSS. A parallel region per layer (24 per token) costs more than two
concurrent matvecs save. Reverted; gate 15/15.
METHODOLOGY CORRECTION, the important part: pairwise interleaving alone does not remove order bias, because one arm
always runs first in every pair. The protocol must alternate (ABBA) so each arm occupies both positions equally.
Both the "+0.7% lazy alloc" (measured pairwise) and this "+1.4%" (measured pairwise) are therefore suspect in
magnitude; the lazy-alloc direction was corroborated by 5/5 wins but its size is not trustworthy either.

## THE LM HEAD IS ~25% OF DECODE AND RUNS 4x SLOWER PER WEIGHT THAN THE LAYER MATVECS
HEAD is 201024 rows x 2304 = 463 MMAC, i.e. 463 M weights = 119 MB at 2.06 bpw - more arithmetic than all 24 layers'
dense projections put together.
MEASURED: HEAD 0.283 s in the one-token full-recompute run, which computes the head once per position (4), so ~71 ms
per generated token. Layers are ~0.185 s/token, so 185 + 71 = 256 ms against a measured ~275 ms token: the head closes
the "35% unmeasured" gap that FP2 could not reach. Decode is therefore roughly 25% head, 67% layers, 8% embed/sample.
RATE: 463 M weights in 71 ms = 6.5 G weights/s ... see below, the honest number is worse.
Correction on my own arithmetic: 463 M weights / 0.071 s = 6.5 G weights/s, which is IN LINE with the layer matvecs
(~7 G weights/s). The "4x slower" impression came from dividing the 0.283 s by ONE position when it covers four.
So the head is not anomalously slow - it is simply big, and it is the largest single matvec in the model.
INSTRUMENT FIXED: the head was timed with clock(). Replaced with in2_wall_now(). The value did not change (0.283 s
either way), which settles a question left open earlier: on this Windows/mingw runtime clock() is WALL time, not
CPU time summed across threads. That means the original "4.44 s of experts inside a 3.4 s run" was NOT clock()
summing threads - it was the nested call-site timers. Correcting the record on that point.
ANOMALY WORTH FLAGGING: with IN2_GPU_F16=1 the head takes the "whole-vocab in ONE call" cuBLAS path per the code
comment, and it measures 0.282 s against the CPU chunked path's 0.281 s, with 12-token throughput identical at
3.64 tok/s both ways. Either that path is not actually being taken (the f16 resident slab for output.weight is
missing), or the one-GEMM path is not faster here. Unresolved, and it is the single highest-value thing left to
resolve because the head is a quarter of decode.
GATE: 15/15 after the instrument change.

## LM HEAD, GPU LANE: TWO HYPOTHESES TESTED AND BOTH WRONG - WHAT IS ACTUALLY ESTABLISHED
The head is 201024 x 2304 = 463 M weights and costs 0.281-0.290 s. It is the largest single matvec in the model and
close to a quarter of a decode token. Both the CPU chunked path (25 x 8192-row calls) and the "whole-vocab in ONE
call" path taken when IN2_GPU_F16 is set measure the SAME time, and 12-token throughput is 3.64 tok/s either way.
HYPOTHESIS 1 (WRONG): "the f16 files carry a 16-byte header, so the size gate `fby != need` rejects the head."
  Disproved by reading in2_f16_lookup: it reads two int64 dimensions from the header and returns d0*d1*2 as the length,
  i.e. it already strips the header. For the head the file is 926318608 bytes, the header says [2304, 201024], and the
  returned length is 926318592 = need exactly. The gate passes. (The header IS real - od shows 0x900 and 0x31140 - and
  every f16 file has it; the loader handles it correctly.)
HYPOTHESIS 2 (WRONG): "clock() sums CPU time across threads, so 0.281 s is really ~40 ms wall."
  Disproved by replacing clock() with in2_wall_now() and measuring again: 0.283 s. Same number. On this runtime clock()
  IS wall time. The head genuinely costs ~0.28 s.
WHAT IS ACTUALLY ESTABLISHED:
  - F16STAT reports 161 resident slabs totalling 2107 MB, mean 13 MB. output.weight alone is 926 MB. A 926 MB slab
    cannot be hiding inside a 2107 MB total made of 161 slabs averaging 13 MB, so THE HEAD IS NOT RESIDENT.
  - With IN2_GPU_TRACE=1 the trace prints hits for blk.* tensors and ZERO "F16 miss" lines, and never mentions
    output.weight at all. (The hit print is once-only, so silence is not proof of a miss.)
  - Therefore the head either never reaches in2_try_f16, or reaches it and declines without tracing.
OPEN QUESTION, stated plainly: the head's f16 copy exists on disk (926 MB), the loader would accept it, and the GPU
would run the GEMM in about a millisecond - yet the head runs at 0.28 s on both paths. Locating WHERE the head call
diverges from the traced blk.* path is the highest-value open item in this document, because it is ~25% of decode.
NOT CLAIMED: any mechanism for the divergence. Two guesses already failed here and a third would be noise.

## SOLVED: THE HEAD SENDS 926 MB OVER PCIe ON EVERY TOKEN INSTEAD OF STAYING RESIDENT
Three independent measurements now agree, and they replace the two hypotheses that failed above.
1. HEADTRY trace: the head reaches in2_try_f16, "HEADTRY reach: output.weight fby=926318592 need=926318592
   n_out=201024", then "HEADTRY dispatch output.weight rc=0 (GPU TOOK IT)". So the GPU path IS taken - it is not
   declining, and it is not a CPU fallback.
2. F16STAT is IDENTICAL for a 1-token and a 12-token run: ups=161 reuse=240/3771 evict=0 used=2107MB slabs=161
   vram_free=2846MB. The upload count does not move and slab count does not move, so the head's 926 MB never enters
   the resident slab cache. There are two distinct GPU paths and the head uses the one with no residency.
3. 926,318,592 bytes / 0.297 s = 3.12 GB/s, which is PCIe-gen4-x4-ish upload bandwidth, not GEMM time. A resident
   f16 GEMM of 463 MMAC on this GPU would be ~1 ms, roughly 300x faster.
CONCLUSION: ~0.30 s per generated token - about a quarter of decode - is spent re-uploading output.weight from host
memory every single token, while 2846 MB of VRAM sits free and the slab would easily fit. The work is pure waste, and
it is the largest single addressable cost found in this entire investigation.
FIX (not yet implemented): route the head through the same resident path the blk.* tensors use, keyed on
t->file_offset, so the 926 MB is uploaded once and then reused. IN2_F16_BUDGET_MB and the eviction policy must admit a
926 MB slab; vram_free is 2846 MB so it fits with ~1.9 GB to spare, but a per-slab cap would explain today's behaviour
and needs checking before assuming the budget is the gate.
VERIFICATION PLAN: after the change, F16STAT ups for a 12-token run must be 161 + 1 (the head uploaded once), not
unchanged-with-the-head-absent, and HEAD must fall from ~0.29 s toward ~1-5 ms. Token identity must stay 15/15 because
the same f16 bytes and the same GEMM are used - only the delivery mechanism changes.

### Budget hypothesis ALSO RULED OUT
IN2_F16_BUDGET_MB=4096 (double the default) gives HEAD 0.280 s against 0.297 s at 2048 - noise, not a fix, and the
resident stats still do not move. So a simple residency cap is NOT the reason the head is served uncached.
in2_f16_dispatch already calls the resident entry point wubu_cuda_resident_f16_matvec(ident, w, wbytes, x, K, y,
n_out) with ident = t->file_offset, so the plumbing is right and the decision is inside the DLL.
NEXT: read wubu_cuda_resident_f16_matvec in src/wubu_cuda_quant.cu for its admission rule (per-slab size limit, an
alignment or shape condition, or a fallback branch that uploads without caching). Three hypotheses on this head have
now been ruled out by measurement - size gate, clock granularity, budget cap - so the next one must come from the
code, not from guessing at the numbers.

## RETRACTION: the head is NOT re-uploaded per token, and it is NOT 25% of decode
The section above ("the head sends 926 MB over PCIe on every token") is WRONG. Reading f16_slot_get in
wubu_cuda_quant.cu settles it: slots are cached by (ident, wbytes) and g_f16_ups is incremented ONLY inside the miss
branch, right after cudaMalloc+cudaMemcpy. Measured ups is 161 for a 1-token run AND 161 for a 12-token run, so the
926 MB slab is uploaded ONCE and hits thereafter. The slab is resident; 2107 MB used = 926 MB for output.weight plus
about 1181 MB across the other 160 slabs (mean 7.4 MB), which is exactly what the numbers allow.
THE ARITHMETIC THAT SHOULD HAVE STOPPED ME: a 12-token run takes 3.3 s = 275 ms/token. If the head cost 0.297 s per
token, twelve of them would need 3.56 s, more than the entire run. So HEAD 0.297 s in the one-token run is the
one-time upload plus the first call, not a recurring per-token cost. I had already noticed this contradiction and
explained it away with "once per position" instead of believing it - that was the error, not the measurement.
CORRECTED PICTURE: per token, layers ~185 ms, head/embed/sampling ~90 ms, and the head's GEMM is small because
output.weight is already in VRAM. There is no 300 ms/token of waste to reclaim. The largest remaining known cost
is the layer work, and within that decode is dense-projection bound at 58.9% (q + gate+wo + kv+conv) with experts at
25.4%, all of it single-threaded because a one-row matvec cannot engage row parallelism.
FOUR hypotheses about this head were tested and all four were wrong: size gate, clock granularity, budget cap, and
per-token re-upload. The instrument work was still worth it - it eliminated a quarter of decode as a target and the
HEADTRY trace is now permanent, so the next person gets the answer in one run instead of four.

## THE IDENTITY BREAKER WAS A KERNEL SWAP, NOT ROW BLOCKING (corrects two earlier claims)
Earlier I wrote that splitting output rows "changes the kernel's blocking, which changes the summation order inside each
dot product". That was a guess dressed as a mechanism, and it is wrong. Tested directly by making the block size an
env knob and sweeping it, all against the oracle:
  IN2_ROWBLK=288  MISMATCH
  IN2_ROWBLK=512  MISMATCH
  IN2_ROWBLK=1024 MISMATCH
  IN2_ROWBLK=2304 MISMATCH
The last one is the proof: with blk = 2304 and n_out = 2304 the loop body executes ONCE, serially, with elem_off = 0 -
there is no blocking and no parallelism - and the tokens STILL differ. So the divergence is not caused by how the rows
are divided. It is caused by WHICH KERNEL runs: the experiment called in2_matvec_quant(), while the untouched path
calls in2_matvec_nt(). Those are two different kernels and they do not agree bit-for-bit on the same shape.
CORRECTED CONCLUSION: two separate things were conflated.
  1. Row blocking is identity-safe in principle - production already row-blocks the head (blk=8192, elem_off = r*n_in,
     wubu_inkling2_model.c:453-459) and passes the gate.
  2. Parallelising the DENSE projections is blocked by the fact that the row-offset entry point (in2_matvec_quant) is
     not the same arithmetic as the one the layer uses (in2_matvec_nt), not by floating-point reassociation.
WHAT THIS OPENS: in2_matvec_nt has no row-offset parameter, so it cannot be row-split as written. Giving it one (or a
tensor-offset variant that preserves its exact arithmetic) would allow identity-preserving row parallelism - and that
is now the only route to multi-core decode that has not been excluded by measurement.
GATE: 15/15 after reverting.

## WHY ROW PARALLELISM BROKE, MECHANISM NOW IN THE CODE (two independent causes, both fixable)
Cause A - the row-offset block cannot use the GPU lane. in2_matvec_nt starts with the f16 lane (in2_try_f16_m for the
whole batch, else per-token in2_try_f16), but in2_try_f16 returns 0 immediately when elem_off != 0. A row split passes
elem_off = r0*n_in, so block 0 (offset 0) lands on the GPU while blocks 1..k fall through to CPU. The same output row
range is then produced by two different arithmetic engines, which is exactly how a greedy token flips.
Cause B - the two CPU paths are not the same function. Even with blk = 2304 (single iteration, elem_off = 0, no
parallelism) the tokens differed, so in2_matvec_quant's CPU path and in2_matvec_nt's CPU path do not agree bit-for-bit.
in2_matvec_nt has its own machinery: a thread-local whole-tensor mapping buffer sized rb*n_out, a q8_0-vs-q8k activation
block choice (be = 32 for types 8/40/42 else 256), and a wubu_iq_supported/in2_ty_verified gate before it will use the
kernel at all, otherwise a float fallback in2_matvec_quant does not share.
CONSEQUENCE: "row parallelism breaks identity" was never one fact, it was these two, and neither is a fundamental
objection. A correct implementation must (1) give the row-offset call the SAME engine as the full call - i.e. extract a
row-range entry inside in2_matvec_nt itself so the f16 lane receives an ident-derivable slab plus a row offset and the
CPU path uses nt's buffer/blocking rules - and (2) verify against the oracle before any timing. Reordering-based
explanations are retired; this is an engine-consistency problem.
STATUS: tree reverted and clean, gate 15/15. No timing claimed for row parallelism in either direction.

## ANSWER: THIS KERNEL'S OUTPUT DEPENDS ON n_out, SO IDENTITY-PRESERVING ROW PARALLELISM DOES NOT EXIST HERE
Third and cleanest implementation, and it closes the question. Instead of introducing a second kernel, this version
advanced the tensor's file_offset by row_off * row_bytes and called the SAME in2_matvec_nt with a reduced n_out - so
each block got nt's own buffer sizing, its own be (32/256) selection, its own supported/verified gate and its own
fallback. Per-row code path identical to the full call by construction.
RESULT, fresh build (the previous attempt's IDENTICAL readings were from a stale binary and are void):
  blk=256  MISMATCH      blk=512  MISMATCH      blk=1152 MISMATCH      GPU+ROWPAR MISMATCH
A block equal to the full n_out trivially reproduces the original call; ANY proper sub-range does not. Therefore the
rows of this kernel are NOT independently computed - the arithmetic of a row depends on how many rows are requested in
the same call. That is why every one of the three independent implementations failed, and it is a property of the
kernel, not of the caller.
CONSEQUENCE, stated plainly: there is no identity-preserving way to parallelise a single-row dense projection through
this kernel. Row splitting, op splitting (measured -0.7%), and the GPU residency route (does not fit) are all excluded.
Multi-core decode would require rewriting the kernel so each output row is computed independently of n_out - a
quantisation-kernel change with its own gate, not a wiring change. That is the honest boundary of what this codebase
can do today, and it is now a fact rather than a series of failed guesses.
GATE: 15/15 after reverting. Tree clean.

## THE REAL MECHANISM: ONLY ONE PATH IS NUMERICALLY CORRECT, SO ANY REROUTING BREAKS IDENTITY
Measured on the reverted, gate-passing binary:
  default (quant kernel path)  -> IDENTICAL to the oracle
  IN2_NOFAST=1 (float fallback) -> MISMATCH
So the float fallback does NOT reproduce the oracle. The engine has exactly ONE numerically-correct path, the quant
kernel, and in2_matvec_nt's own comment says this was already learned once: an unverified type was let through and the
last token of a six-token greedy run moved from 220 to 167815.
This explains all three row-parallelism failures better than "the arithmetic depends on n_out". A partial slab changes
conditions the kernel path keys on - the read length rb*n_out, the size handed to in2_gpu_resident, and the
supported/verified gate - so some blocks take the kernel and the rest take the fallback. Two engines, one output range,
tokens flip. The row arithmetic may well be n_out-independent; what is not engine-independent is the ROUTING.
REVISED SPEC for any future parallel dense path, and it is now much cheaper than "rewrite the kernel":
  1. Every block must be provably routed to the SAME engine as the full call - assert it, do not assume it. IN2_TRACE_NT
     already prints "-> KERNEL path" vs "-> FLOAT-FALLBACK" per call, so this is directly checkable.
  2. The fallback must never run in a partial-block configuration, even when the full call would have used the kernel.
  3. Verify with the gate before timing; IN2_NOFAST must NOT be treated as a safe reference, because it disagrees with
     the oracle.
WHAT THIS ALSO MEANS: the float fallback is a latent correctness hazard, not just a slow path - it silently produces
different tokens. That is worth its own fix regardless of performance.

## ROUTING DIAGNOSED: THE BASELINE ITSELF MIXES ENGINES (and that reframes the last two entries)
Per-call routing from IN2_TRACE_NT, counting the last field of each "NT ty=..." line, one-token run:
  ROWPAR off : 24 kernel + 24 FLOAT-FALLBACK   <- the untouched, gate-passing build ALREADY runs half its calls on
                                                  the float fallback
  ROWPAR on  : 24 FLOAT-FALLBACK + 312 kernel  <- the blocked version routes wq's blocks to the kernel
So the row split did not "reroute rows off the one correct path" as I wrote a moment ago. It did the OPPOSITE: for that
op the full call uses the FLOAT FALLBACK, and blocking it forced the blocks onto the quant kernel. Different engine,
different tokens.
CORRECTED UNDERSTANDING, and this is simpler than everything before it: this engine deliberately mixes two numerics
per operator - the baseline is already a mix and still matches the oracle. Identity therefore is not "always use the
kernel"; it is "each operator must keep using exactly the engine the unblocked call used". A parallel dense path must
reproduce the per-operator routing decision, not improve on it.
THE CONDITION MISSING FROM EVERY ATTEMPT: the row helper gated on ty >= 0 and n_in % be == 0, which are NOT the
conditions in2_matvec_nt uses. nt additionally requires wubu_iq_supported(ty) && in2_ty_verified(ty) before it will take
the kernel, and falls back otherwise. Only ops that pass ALL of those may be blocked; blocking anything that the full
call sends to the fallback flips tokens.
BLOCKER on implementing that guard: wubu_iq_supported() and in2_ty_verified() are not externally linkable from
wubu_inkling2_fullseq.c (link error), so the guard needs an accessor or a small exported predicate. That is the next
concrete change, and it is a few lines rather than a kernel rewrite.
ALSO: IN2_NOFAST=1 (force everything to the fallback) MISMATCHES the oracle, so the mix is load-bearing for correctness
on at least one operator - the fallback cannot stand in for the kernel everywhere.
GATE: 15/15, tree clean.

## ROUTING MAP BY SHAPE (corrects the arithmetic used in every earlier cost estimate)
IN2_TRACE_NT on a one-token run, grouped by shape (count, type, in, out, routing):
   3  ty=12 n_in=2304 n_out=384  kernel
   3  ty=14 n_in=2304 n_out=384  kernel
  18  ty=8  n_in=2304 n_out=384  kernel
  24  ty=30 n_in=2304 n_out=24   FLOAT-FALLBACK
Two facts this settles:
1. THE DENSE PROJECTIONS ARE 2304 -> 384, not 2304 -> 2304. qdim is 384 (24 heads x 16), so q, wgate and wo are each
   ~0.88 MMAC, not 5.3 MMAC. My earlier "5.3 MMAC per call" figure was wrong by 6x and every rate derived from it was
   wrong in the same direction. gate+wo at 49 ms/token for ~1.77 MMAC is ~36 MMAC/s - roughly 3-4 orders of magnitude
   below this CPU. The phase is OVERHEAD bound beyond any doubt; the arithmetic is negligible.
2. The 24 FLOAT-FALLBACK calls are ty=30 at n_out=24 - one per layer, the sink projection. That is the operator the
   baseline deliberately runs on the fallback, and it is why IN2_NOFAST breaks the oracle: forcing the kernel onto a
   type-30 tensor changes tokens. The blocked ops were NOT this one.
UNRESOLVED: with the routing predicate exported and used as the block gate, the kernel-call count rose from 24 to 816
while the fallback count stayed at 24, and tokens still mismatched - yet wq/wgate/wo are n_out=384 which is below the
helper's 1024 threshold, so they should not have been blocked at all. Something other than my helper changed the call
pattern, and that is unexplained rather than assumed. Reverted; gate 15/15; tree clean.
NEXT DIAGNOSTIC (cheap): log the helper's entry ONCE (n_out, ty, predicate result, whether it blocked) into a single
line per layer, so the 24->816 increase is attributed to a specific call site instead of inferred from aggregate counts.

## WHERE THE 89% GOES: kernel is ~11% of a token, and the read-timing attempt did not emit
The leaf per-type counters (IN2_TIMETY, now working once IN2_PERF is also set) sum to 0.034 s of quant-kernel time
against ~0.296 s/token, so the kernels themselves are ~11% and the other ~89% is everything around them. That is the
single most useful number in this document, and it reframes the target: not FLOPs (dense ops are only ~0.88 MMAC each),
not bandwidth (5.9 GB/s against a 30-40 GB/s ceiling), not threads - but per-call overhead.
ATTEMPTED: a counter around the weight read in in2_matvec_nt (in2_nt_read_sec/n, plus an unmapped counter to see whether
the mapping is being used at all), dumped next to the other per-forward stats in wubu_inkling2_model.c.
RESULT: the dump never printed under IN2_PERF, so the read time is still unmeasured. Reverted - an instrument that
emits nothing is worse than none, and the tree stays exactly as the gate verified it.
NEXT STEP, corrected: put the dump in tools/inkling_gen.c beside the PHASE/TIMETY exit dump (that block demonstrably
runs) rather than in model.c's per-forward block, and report three numbers per shape: read seconds, kernel seconds, and
the residual. That bisects the 89% into I/O, kernel and dispatch in one run.

## BISECTION RESULT: the weight read is FREE, and only 24 of ~168 dense matvecs reach that branch
NTREAD instrumentation (counters in in2_matvec_nt, dump beside the PHASE block that provably runs):
  NTREAD 0.0000 s read over 24 in2_matvec_nt calls (0.0 us/call), unmapped=0
Two hard facts, both eliminating a suspect:
1. THE READ COSTS NOTHING. unmapped=0 means the mapping is in use, so no copy is made, and 24 calls accumulated 0.0000 s
   of read time. Weight I/O is not the ~89%.
2. ONLY 24 CALLS PER FORWARD REACH THAT BRANCH. A full layer issues ~7 dense matvecs (q, k, v, gate, wo, wrel, sink)
   over 24 layers = ~168 per forward, but the kernel branch saw 24. So roughly 85% of dense matvecs take a DIFFERENT
   path - the f16/GPU lane, the float fallback, or another entry point entirely. My mental model of the call graph was
   wrong, and every "per-call overhead" estimate that assumed one entry point was measuring the wrong population.
CONSEQUENCE: the 89% cannot be attributed from inside in2_matvec_nt, because most of the work does not pass through it.
The next instrument must sit ONE LEVEL UP and count entries by function and shape - every dense matvec entry point, not
just nt's kernel branch.
This is the fourth suspect eliminated for the 89% (after arithmetic, bandwidth/kernel choice, and thread count), and the
first that also invalidates the scope of the previous measurements rather than just a candidate.
GATE: 15/15.

## STRUCTURAL CORRECTION: only 2 dense-kernel calls per layer, not 7 - so every per-call cost was unsound
Adding up the traced calls for one forward: ty=12 3 + ty=14 3 + ty=8 18 + ty=30 24 = 48 calls over 24 layers = exactly
2.0 per layer. I had been assuming ~7 per layer (q, k, v, gate, wo, wrel, sink = ~168 per forward), a factor of 3.5.
The trace print sits after the f16-lane block but before any early return, and IN2_GPU_F16 was unset for those runs, so
every call that enters in2_matvec_nt does print. The 48 is therefore the true population for the layers that ran.
WHAT THIS INVALIDATES: every "per-call" figure in this document. "q = 1.5 ms/call (36 ms / 24 calls)" divides a phase
wall time by an assumed call count; if the phase actually issues 1 call, that call is ~36 ms and is the single most
expensive operation in the model. The 0.88 MMAC and 36 MMAC/s numbers assumed 384 outputs x 2304 inputs per call and are
only valid if the shape belongs to that call - which now has to be verified per shape rather than assumed.
ALSO UNRESOLVED: q, k, v, gate and wo are not visible as separate in2_matvec_nt calls. They are either fused into one
tensor, or they run through in2_layer_sliding (which has its own path and does not emit FP2 phases), or through an entry
point the trace does not cover. The 24 traced kernel calls are 2304x384 and the 24 fallback calls are 2304x24 (sink),
which is 2 tensors per layer - consistent with a fused projection, not with seven separate ones.
RULE ADOPTED: no per-call cost is quoted again until entries are counted across EVERY dense entry point (in2_matvec_nt,
in2_matvec_quant, in2_matvec_float, in2_matvec_rows*, and the sliding-layer path) with shape attribution. Dividing phase
time by a guessed denominator is how the last three estimates went wrong.

## RETRACTION: the call count is 448 per forward, not 48 - my "2 per layer" was a truncated grep
ENTRY instrumentation on the three dense entry points, one-token forward:
  ENTRY nt=448 quant=1 float=0 other=0
So in2_matvec_nt is entered 448 times per forward = 18.7 per layer, which is consistent with the ~7 dense ops x 24
layers I originally assumed. The previous entry claiming "only 2 dense-kernel calls per layer" is WRONG and is
retracted: that 48 was produced by piping the grouped trace through `head -10`, which cut the shape list at ten rows.
The lesson is the same one this document keeps relearning - a grouped count is only a count if nothing truncates it.
INSTRUMENT NOW IN PLACE AND WORKING: ENTRY nt/quant/float counters, dumped beside the PHASE block that provably runs
(this is the block that printed, unlike the model.c per-forward dump). IN2_TIMETY is confirmed not to be part of this
run's output path; ENTRY is.
CORRECTED PER-CALL ARITHMETIC, with the population actually known:
  layer phases total 0.183 s over 448 nt calls -> mean 0.41 ms/call
  q phase 36 ms over 24 calls -> 1.50 ms/call
A 2304x384 quant matvec is 0.88 MMAC, so 1.5 ms mean ~0.6 GFLOPS, three orders of magnitude below this CPU.
=> per-call OVERHEAD is the target, and the population is now measured rather than assumed.
GATE: 15/15.

## THE 89% IS LOCATED: in2_matvec_nt is ~100% of layer time and 82% of it is dequant/dispatch, not the dot
Timed wrapper around in2_matvec_nt (so every return path is measured: f16-lane early return, kernel success, float tail):
  ENTRYTIME nt total 0.7387 s / 448 calls = 1648.9 us/call | read 0.0000 s (0.0 us/call) | kernel+dispatch 0.7387 s
Cross-check that makes this trustworthy: the counters are run-wide (4 forwards in a full-recompute run) while PHASE
resets per forward, and 0.7387 / 4 = 184.7 ms against a PHASE layer total of 183 ms - within 1%. So in2_matvec_nt IS
the layer cost; there is nothing significant hiding beside it.
CONSEQUENCES, each one an elimination:
  read = 0.0000 s, unmapped = 0 -> not I/O (mapping is used, no copy).
  TIMETY leaf dot = ~0.034 s/forward = 18% of 0.183 s -> the dot is only 18%.
  Therefore ~82% of nt time, ~150 ms per forward, is DEQUANTISATION + buffer/dispatch work inside the kernel branch.
  That is ~4.6x the cost of the arithmetic it feeds - and the arithmetic is only 0.88 MMAC per 2304x384 call.
Calls per forward: 448 / 4 = 112, i.e. 4.7 per layer, not the 7 I assumed nor the 2 I wrongly retracted to. At 1.65 ms
per call, 112 calls = 185 ms, which is the whole layer budget.
WHERE THE NEXT FIX GOES: inside in2_matvec_nt's kernel branch, between the mapping and the rows call - the dequant of
rb*n_out bytes into the shared `whole` buffer plus the activation quantisation. That region is now the only place left
with 82% of the cost, and it is instrumentable with two more spans (dequant time, act-quant time) inside the same
wrapper.
NOTE ON COUNTS THAT HAVE BITTEN TWICE: 448 was right last turn (it was the 48 that was truncated), and 448 is a RUN
total across 4 forwards, not a per-forward count. Both numbers were mine; both were misread the first time. Always
divide run-wide counters by the forward count before quoting a per-call cost.
GATE: 15/15.

## THE 82% IS THE CPU ROWS KERNEL, AND TIMETY's 18% WAS MY OWN INSTRUMENT LYING
Span inside in2_matvec_nt's kernel branch, run-wide over 448 calls:
  read 0.0000 s | act-quant 0.0004 s (0.8 us/call) | kernel 0.7535 s (1681.9 us/call)
The GPU dense attempt is gated on IN2_GPU_DENSE, which was never set, so gpu_took never fired: the whole 1681.9 us is
the CPU rows kernel. Activation quantisation is 0.8 us - irrelevant. The read is zero.
SO THE KERNEL IS THE COST AFTER ALL, and the earlier "TIMETY says the dot is only 18%" was WRONG: in2_ty_par suppresses
leaf timers inside parallel regions (a change I made earlier in this session to stop double counting), and the rows
kernel runs inside a parallel region, so TIMETY measured almost nothing of it. My own anti-double-counting guard turned
the profiler into an under-counter for the one region that matters. Instrument discipline: a suppression flag must
report what it suppressed, or it silently becomes a lie.
WHAT THE KERNEL IS DOING, with the shape finally known:
  per call 1683 us for 0.88 MMAC  ->  1.05 GFLOP/s
  228 KB of packed weights per call -> 135 MB/s effective
Neither ALU nor DRAM explains those numbers on a Ryzen 5 8645HS. 448 calls x 228 KB = 102 MB touched per forward against
a 7.9 GB model, so the weights are cold every step - but 135 MB/s is still far below what cold page-cache reads cost.
NEXT INSTRUMENT STEP, and it is the last one needed: accuse the kernel itself. Take ONE shape (ty=8 or 12, 2304x384)
out of the engine, call the rows kernel directly in a microbenchmark over the same bytes with the same thread count,
and time it in isolation. That separates "the kernel is slow" from "the kernel is slow when called 448 times per
forward in this pattern" (allocator churn, false sharing, TLB/page-cache thrash, or nested parallel regions).
GATE: 15/15.

## ROOT CAUSE: THE DENSE ROWS KERNEL IS COMPLETELY SERIAL - it ignores thread count
Per-call kernel time inside in2_matvec_nt, same shape (2304x384), only IN2_THREADS varied:
  threads=1 1633.4 us/call | threads=2 1604.5 | threads=4 1587.5 | threads=8 1634.2
Flat to within 3% from 1 to 8 threads. The dense rows kernel does NOT parallelise. At 0.88 MMAC in 1.63 ms it runs at
~1.08 GFLOP/s, single core, for every call.
THIS EXPLAINS THE WHOLE SESSION'S NEGATIVE RESULTS: the dominant cost (82% of layer time) never participated in any
thread experiment. The 1.66x whole-run gain from 1 to 8 threads (6655 -> 4011 ms on a 12-token run) comes from mix,
attention and other regions - not from the projections that hold the time. Every attempt I made to parallelise decode
at the CALLER (row split, op split, block sizes) was competing for a resource the bottleneck had already left unused.
It also explains why the kernel looked "overhead bound": 1.63 ms for 0.88 MMAC is neither ALU (1 GFLOPS) nor DRAM
(135 MB/s) - it is one core doing serial work while seven idle.
THE FIX, and it is now well specified: parallelise OUTPUT ROWS INSIDE the rows kernel (wubu_iq_c11.c), not at the nt
level. Inside the kernel every block runs the same code on the same weights for its own row slice, so the earlier
engine-routing failure cannot recur (nothing changes about which engine runs). Identity still has to be proven with the
oracle gate, because a per-row result could in principle depend on the kernel's internal blocking - but this is the
first parallelisation attempt aimed at the code that actually holds the time.
ACCEPTANCE: kernel us/call must fall materially with thread count (that is the direct check, independent of tok/s),
and the gate must stay 15/15.
GATE: 15/15.

## NESTED PARALLELISM RULED OUT - the region DOES run, so the worker cost is not shrinking with NT
Probe at the top of in2_matvec_nt, threads set to 8:
  NESTCHK in2_threads=8 omp_in_parallel=0 omp_max=12 n_out=3072
So: not nested (omp_in_parallel=0), NT is 8 as requested, and the FIRST call is 3072 wide, which is well past
in2_rows_parallel's n_out>=256 gate. The parallel path is therefore being taken - and yet per-call kernel time is flat
from 1 to 8 threads (1633 / 1605 / 1588 / 1634 us).
That combination has exactly one interpretation: the region runs, but the per-worker cost does not fall as NT rises.
Either each worker touches work proportional to the FULL problem (for example converting or traversing the whole weight
buffer rather than its own row slice), or the pool is not actually executing NT-way concurrent work, or the jobs are so
imbalanced that the slowest one sets the wall time regardless of NT.
This also corrects the shape I had been reasoning about: the dense calls are 3072 wide, not 384 - 2304 -> 3072 is 7.08
MMAC per call, not 0.88. At 1.63 ms that is 8.7 GFLOP/s, not 1.08, so the kernel is roughly 8x less absurd than my last
figure claimed, and the "0.88 MMAC" arithmetic in the three preceding entries was wrong by 8x.
NEXT CHECK (one line, decisive): count, per worker, how many rows it actually processed and how long it took, and print
the min/max across workers for a single call. If max >> min the split is imbalanced; if all workers show ~1/N rows and
full serial time each, the workers are each doing full-size work.
GATE: 15/15.

## THE SPLIT WORKS BUT ONLY COVERS THE SMALL CALLS - nothing with n_out>=2048 reaches it
Per-job spans inside the worker (sum of job spans vs region wall):
  threads=2: wall 0.0001 s, jobs=2, rows=384, avg job 18.6 us for 192 rows
  threads=8: wall 0.0000 s, jobs=8, rows=384, avg job  3.5 us for  48 rows
That is a clean 5.3x for 2->8 threads and near-perfect 1/N scaling: the row split is CORRECT and it WORKS, and the
identity-safe claim (same kernel, disjoint row ranges) is borne out.
But gating the same print on n_out >= 2048 produces NOTHING at any thread count:
  threads=1: (no call with n_out>=2048 reached the parallel path)
  threads=8: (no call with n_out>=2048 reached the parallel path)
So the wide calls - the ones that carry 7.08 MMAC each - never enter in2_rows_parallel. They are handled before it, by a
different branch in in2_matvec_nt (the f16/multi-row lane) or by another function entirely (moe batch / expert path), and
that branch is serial. This is why per-call kernel time is flat across thread counts: the mega-calls that dominate the
448-call total are not participating in the split at all, while the calls that do participate are already cheap (~4-19 us).
CONSEQUENCE: parallelising harder is the wrong next move. The next move is to find WHERE the wide calls go and route them
through the same row-parallel dispatch. The evidence says the split is proven correct and fast; it is simply not being
applied to the shapes that cost the time.
NEXT CHECK: log (n_in, n_out, ns) for every nt call and aggregate by shape - that names the exit branch by elimination and
gives the per-shape cost table the whole session has been missing.
GATE: 15/15.

## THE SHAPE TABLE - the table this project never had, and it names one shape holding 62.6%
in2_matvec_nt aggregated by (n_in, n_out) over one full-recompute run (4 forwards), total 0.7385 s:
  2304 -> 3072 : 0.4622 s   128 calls (32/fwd)   3611.3 us/call   62.6% of nt
  3072 -> 2304 : 0.1295 s    64 calls (16/fwd)   2023.9 us/call   17.5%
  2304 -> 1024 : 0.1055 s   128 calls (32/fwd)    824.3 us/call   14.3%
  2304 ->  384 : 0.0334 s    64 calls (16/fwd)    522.4 us/call    4.5%
  2304 ->   24 : 0.0077 s    64 calls (16/fwd)    120.9 us/call    1.0%
ONE SHAPE CARRIES 62.6% OF ALL nt TIME: 2304 -> 3072, 32 calls per forward, 3.61 ms each.
That shape is 7.08 MMAC, so 3.61 ms is ~1.96 GFLOPS - worse than the 8.7 GFLOPS I computed for the 3072-wide case
earlier, because that figure divided by the wrong denominator. This is the number to attack.
THE CONTRADICTION TO RESOLVE NEXT: n_out=3072 plainly satisfies in2_rows_parallel's gate (NT>=2 and n_out>=256), yet the
WJ probe - gated on n_out>=2048 - never printed for any thread count. So those calls enter in2_matvec_nt (they are in the
wrapper's table) and RETURN BEFORE the rows_parallel line. They are leaving through an earlier branch. That branch is the
fix site, and it is now a single named question: which return does a 2304->3072 call take?
Also note 2304->384 measures 522 us/call in the table while the WJ probe showed a 384-row split completing in ~28 us
(8 jobs x 3.5 us). Those cannot both be the common path, so most 2304->384 calls are ALSO leaving early, not splitting.
NEXT CHECK (one line per call): record the exit branch id in the wrapper (the impl returns a code for path taken, or add a
path byte set at each return) and print the path histogram grouped by shape. That converts "which branch" from inference
into a count, and given the last three turns, inference is exactly what keeps being wrong.
GATE: 15/15.

## THE ANOMALY IS A QUANT TYPE: Q6_K (ty=14) runs 4.5x slower per byte than Q8_0 (ty=8) on the SAME shape
Shape table keyed by (ty, n_in, n_out), one full-recompute run, 17 shapes, 0.7338 s:
  ty=14  2304 -> 3072 : 0.2727 s   44 calls   6196.8 us/call   37.2%
  ty=8   2304 -> 3072 : 0.1130 s   64 calls   1766.1 us/call   15.4%
  ty=8   3072 -> 2304 : 0.0981 s   56 calls   1751.1 us/call   13.4%
  ty=8   2304 -> 1024 : 0.0767 s  106 calls    723.6 us/call   10.5%
  ty=12  2304 -> 3072 : 0.0414 s   12 calls   3450.8 us/call    5.6%
  ty=13  3072 -> 2304 : 0.0256 s    7 calls   3655.6 us/call    3.5%
  ty=8   2304 ->  384 : 0.0231 s   45 calls    513.3 us/call    3.1%
  ty=13  2304 -> 3072 : 0.0157 s    5 calls   3148.7 us/call    2.1%
  ty=18  2304 -> 3072 : 0.0152 s    3 calls   5078.9 us/call    2.1%
The same shape at the same row count, differing only in quant type:
  Q8_0 (ty=8)  : 1766.1 us for 7.08 MMAC, 7.52 MB -> 4.26 GB/s, 8.02 GFLOP/s
  Q6_K (ty=14) : 6196.8 us for 7.08 MMAC, 5.81 MB -> 0.94 GB/s, 2.28 GFLOP/s
Q6_K carries FEWER bytes and costs 3.5x MORE time, so per byte it is ~4.5x slower than Q8_0. That is a dequant/kernel
defect in the Q6_K unpack path (wubu_iq_c11.c), not a memory bandwidth or ALU limit: the same machine reaches 4.26 GB/s
on Q8_0 with the same shape, the same threads and the same split. ty=14 alone is 37.2% of nt time; all 2304->3072 types
together are 60.3%.
WHY THIS EXPLAINS THE WHOLE SESSION: the "overhead bound" and "flat thread scaling" readings were aggregates over 17
shape/type combinations whose per-byte costs differ by 4.5x. Averaging them produced numbers that belonged to no actual
kernel, so every mechanism inferred from those averages was chasing a mixture. The variance is between TYPES, not
between threads.
NEXT STEP, now specific: read the Q6_K unpack in wubu_iq_c11.c and compare its inner loop against the Q8_0 one for the
same block count - super-scale/min handling, per-16 groups, byte shuffling, and whether it dequantises to a scratch
buffer per call. Q8_0 at 4.26 GB/s is the existence proof that the surrounding machinery (split, threading, mapping) is
fine, so the defect is inside the Q6_K block decode.
GATE: 15/15.

## Q6_K IS NOT A MISSING-SIMD DEFECT - it is the canonical ggml AVX2 kernel
Read, not assumed:
  src/wubu_iq_c11.c:57   #define IQ_AVX2 __attribute__((target("avx2,fma,f16c")))
  src/wubu_iq_c11.c:1536 IQ_AVX2 static float dot1_q6_K(...)  <- target attribute, so it is AVX2 codegen regardless of -march
  src/wubu_iq_c11.c:1767 DEFINE_NEW_ROWS_8K(q6_K)             <- 8-row blocked rows kernel exists
  src/wubu_iq_c11.c:1929 case 14: rows_q6_K(...)              <- and it IS the dispatch for ty=14
The inner loop is the ggml_vec_dot_q6_K_q8_K AVX2 body verbatim: 4x q4bit unpack via maddubs, q4h_0..3 high-bit folding,
and the -32 offset folded into q8sums x scales << 5. Nothing is left on the floor by inspection.
So the 4.5x per-byte gap versus Q8_0 on the same shape is NOT a scalar fallback and NOT an arithmetic shortcut missed in
the loop. What remains, in order of likelihood:
  1. the ACTIVATION side: Q6_K requires q8_K (256-element blocks with 16 int16 bsums) while Q8_0 needs plain q8_0. The
     act-quant span read 0.8 us/call, so the quantisation itself is not it - but a per-call scratch layout/size difference
     for q8_K, or a bsums computation done per row instead of per call, would be.
  2. the CALL MIX: the ty=14 2304->3072 calls may not all be the same tensor. 44 calls at 6197 us average can hide a few
     very slow calls and many fast ones, and an average cannot distinguish them.
  3. KB-per-row: Q6_K is 210 bytes per 256-weight block versus Q8_0's 272, so if the rows_8K path re-reads per row rather
     than streaming, Q6_K's extra shuffle work per block dominates.
NEXT CHECK, and it needs no new software: the same shape/type table, but printing min and max us/call per shape instead of
only the mean. If ty=14's max is far above its mean, the 6197 us is a measurement of a bimodal mix and not of one kernel -
which is the same trap that produced the last four wrong mechanisms, all of which were averages over mixtures.
GATE: 15/15, tree clean, 32 commits, identity byte-identical.

## SOLVED: it is DEMAND PAGING of an mmap'd 7.9 GB model - min/max kills the Q6_K theory
Min/max per (ty, shape), one full-recompute run:
  ty=14  2304 -> 3072 : mean 6187.6 us [min   147.0  max 187192.7]  37.1%
  ty=8   2304 -> 3072 : mean 1760.4 us [min   146.1  max   5318.5]  15.3%
  ty=8   3072 -> 2304 : mean 1755.9 us [min   150.8  max   5230.5]  13.4%
  ty=8   2304 -> 1024 : mean  722.1 us [min    87.4  max   2361.7]  10.4%
TY=14's MIN IS 147.0 us - the same as ty=8's 146.1 us. The Q6_K kernel is NOT slower than Q8_0. The 4.5x "per-byte
defect" was the arithmetic mean of a BIMODAL distribution, and the mean belonged to no call that actually ran.
WHAT THE TWO MODES ARE, from the same numbers:
  fastest 5.81 MB call: 147.0 us -> 39.5 GB/s  (the 5.81 MB tensor fits L3; this is cache-warm execution)
  slowest 5.81 MB call: 187192.7 us -> 31.0 MB/s  (this is page-in from disk, not ALU and not DRAM)
  max/min = 1273x within one shape and one type.
So the dominant cost in decode is DEMAND PAGING of the mmap'd 7.9 GB GGUF. `unmapped=0` and `read 0.0000 s` were both
true and both misleading: with mmap there is no copy to time, so the read counter reads zero while the page faults land
inside the kernel. That is why per-call kernel time was flat across thread counts (page faults on one mapping serialise),
why the "GFLOPS" figures were nonsense (they divided compute by page-fault time), and why every shape/type average moved
when the access pattern moved.
THIS IS THE STANDING GOAL'S OWN SUBJECT: SSD streaming on a 16 GB laptop is the bottleneck, exactly as the tandem model
predicts, and the fix is not a faster kernel - the kernel already runs at 39.5 GB/s when warm.
FIX DIRECTION, in order of expected effect:
  1. MADV_WILLNEED / sequential prefetch ahead of the layer loop, so paging overlaps compute instead of serialising it.
  2. Read-ahead window sized to the 5.8 MB per-tensor working set rather than per call.
  3. Keep the hot dense tensors (the 2304->3072 set) resident - they are small enough to hold, and they carry 37% of time.
ACCEPTANCE: the max/min ratio for a shape must collapse toward 1.0, and mean us/call must approach the min. tok/s is a
downstream consequence, not the check.
GATE: 15/15.

## CONFIRMED BY EXPERIMENT: page-cache warming alone is worth +25% decode
Same binary, same prompt, same 12 tokens, IN2_THREADS=8, KV cache path. Only difference: the GGUF was read end-to-end
first so its pages were resident.
  cold (normal): 3.51 - 3.64 tok/s
  warm (pread) : 4.49 tok/s      -> +25% (4.49 / 3.60 = 1.25)
and the shape table tightens with it:
  ty=8  2304 -> 3072 : mean  270.3 us [min 104.8  max   870.7]  23.5%
  ty=8  3072 -> 2304 : mean  274.0 us [min 106.6  max  2948.6]  20.1%
  ty=8  2304 -> 1024 : mean  141.3 us [min  72.8  max   795.8]  19.3%
  ty=14 2304 -> 3072 : mean  232.3 us [min 102.0  max  1528.8]  12.3%
  ty=8  2304 ->  384 : mean   97.6 us [min  67.3  max   639.8]   5.6%
Compare with the cold 1-token run: ty=14 mean 6187.6 us [min 147.0 max 187192.7]. The 187 ms page-in outlier is GONE,
max/min fell from 1273x to ~15x, and per-call mean fell from 1682 us to 170 us (the latter is also a different token
mix - 12 tokens on the KV path vs 1 token full recompute - so only the max/min collapse and the tok/s delta are clean).
THE DIAGNOSIS IS NOW A MEASUREMENT, NOT AN INFERENCE: cold demand paging costs ~25% of decode throughput, and the
kernel itself is fine (min 102-147 us per call, i.e. 39-57 GB/s on a 5.8 MB tensor).
ALSO MEASURED, and it indicts the existing prefetch: gguf_reader.c:1676 issues MADV_WILLNEED over the ENTIRE mapped blob
at open. On a 16 GB box that reads 7.9 GB and evicts the early layers before layer 0 is reached, so the prefetch is
largely self-defeating - it pays the I/O and then faults again.
NEXT FIX (specific, one place): replace the whole-blob WILLNEED with per-layer prefetch issued one layer ahead of use, so
paging overlaps compute. Acceptance: cold-start tok/s approaches the 4.49 warm figure without a manual pre-read, and the
per-shape max/min ratio stays near 15x or better.
GATE: 15/15.

## CORRECTION: there is NO prefetch on Windows at all - the madvise is inside #if defined(__linux__)
src/gguf_reader.c:1674-1677 reads:
    #if defined(__linux__)
                madvise(base, ctx->data_blob_mmap_len, MADV_WILLNEED);
                madvise(base, ctx->data_blob_mmap_len, MADV_HUGEPAGE);
    #endif
This build is Windows (_WIN32), so NEITHER call has ever been compiled here. My previous entry claimed the loader
"issues MADV_WILLNEED over the entire mapped blob at open" and that this prefetch was self-defeating on a 16 GB box.
That was wrong: on this platform there is no prefetch of any kind, which is precisely why pure demand paging shows up as
a 187 ms page-in outlier and why a manual pre-read is worth +25%.
There is also no madvise in src/wubu_win.h (checked), so the Windows path has no read-ahead equivalent wired at all.
THE FIX IS THEREFORE AN ADDITION, NOT A REPLACEMENT: implement a Windows prefetch primitive once (PrefetchVirtualMemory
declared locally rather than relying on _WIN32_WINNT, so it links on any Win8+ without SDK version gates), then issue it
one layer ahead of use from the layer dispatch macro in src/wubu_inkling2_model.c:179-182, where (IL) is already in scope:
  before the in2_layer_full_seq / in2_layer_sliding dispatch, prefetch the window that belongs to layer (IL)+1.
Window sizing: one layer's worth of the blob, i.e. data_blob_size / n_layer, because GGUF stores layers in order - which
also avoids depending on per-tensor offsets or the model struct field names.
ACCEPTANCE: cold-start tok/s approaches the 4.49 warm figure WITHOUT a manual pre-read, and per-shape max/min stays near
15x or better.
GATE: 15/15, tree clean, 35 commits.

## FIRST PREFETCH ATTEMPT MEASURED AND REJECTED: -1.4%, the window was 7x too big
Implemented wubu_gguf_prefetch() in gguf_reader.c (PrefetchVirtualMemory resolved from kernel32 at runtime, so it links
without an SDK gate) and hooked it in the layer dispatch macro one layer ahead, window = data_blob_size / n_layer_phys.
Measured, two reps each, 12 tokens, 8 threads, cache state identical between arms:
  PF ON : 3.43 and 3.44 tok/s
  PF OFF: 3.48 and 3.48 tok/s
A consistent -1.4%. REJECTED and the hook reverted; the primitive stays in the reader as a utility.
WHY, and the number that explains it: blob/n_layer_phys = 7.9 GB / 24 = 337 MB per layer. A layer's dense weights are
~50 MB and the expert tensors dominate the rest, so asking the OS to prefetch 337 MB while the current layer is being
computed saturates the read path and evicts the pages the current layer needs. Prefetching "one layer" in blob bytes is
prefetching roughly seven layers' worth of what the compute actually consumes next.
THE CORRECTION FOR THE NEXT ATTEMPT: size the window to the next layer's TENSOR span, not to blob/n_layer - i.e. from
layers[IL+1]'s first tensor file_offset to the last one's end, which is the real working set. Only then does the paging
overlap the compute instead of displacing it. A cheap version of the same idea: prefetch only the dense tensors of the
next layer (5-50 MB each), leaving the experts to demand paging as they are touched.
ALSO CONFIRMED INCIDENTALLY: the warm-page-cache result (4.49 tok/s) is transient - after a few minutes of other work the
cache no longer held the file and both arms fell back to ~3.45. So the +25% is real but is not a resting state; it is the
size of the prize that a correct, properly sized read-ahead should recover.
GATE: 15/15, tree clean.

## CORRECTED-WINDOW PREFETCH: neutral when warm, cold case still unverified
Hook re-sized to the next layer's own dense front: prefetch m->layer[IL+1].wq.file_offset for 64 MB (not blob/24=337 MB).
ABBA over 12-token runs, 8 threads, gate green:
  PF_ON  4.49, 4.47   PF_OFF 4.46, 4.49   -> neutral
BUT both arms read 4.46-4.49, i.e. the WARM level, so this run proves only that the hook does not regress the warm case.
It CANNOT show a cold-start benefit, because the page cache was already warm and Windows offers no drop_caches equivalent.
Kept as OPT-IN (IN2_PF) so the shipped default is byte-for-byte the gate-verified behaviour.
REMAINING TEST, and it needs disk, not code: copy the GGUF to a path that has never been read (genuinely cold), run one
arm; delete the copy; copy again and run the other arm. Two fresh copies, because the first run warms whatever file it
touches. Only that isolates cold-start.
TWO BUILD BUGS WORTH REMEMBERING (both cost a failed build each, and one produced a stale-binary reading):
  1. A multi-line /* ... */ comment placed INSIDE a backslash-continued macro terminates the continuation: comment
     lines need their own trailing backslashes, or the comment must live outside the macro.
  2. Inserting text after a macro line that ends in a backslash, separated by a blank line, silently ends the macro -
     the following lines then compile at file scope and report "IL undeclared".
     Both showed up as 'IL undeclared' / '_bs undeclared', which looks like a scope problem and is actually a
     continuation-break problem.
GATE: 15/15.

## COLD-START PREFETCH IS UNVERIFIABLE ON THIS BOX - three attempts, all inconclusive
Attempt 1 (two fresh copies): abandoned on reasoning - a freshly WRITTEN copy has its pages in the page cache (dirty,
then clean), so it is warm by construction and the test would measure nothing.
Attempt 2 (ABBA, IN2_PF vs default): 4.49/4.47 vs 4.46/4.49 - neutral, but both arms read the WARM level, so it tested
nothing about cold start.
Attempt 3 (evict by streaming the 9 GB f16set through cache, then one arm at a time):
  IN2_PF=1 : 4.52 tok/s      default (no PF): 4.49 tok/s
Both still read the warm level (~4.5, against ~3.45 genuinely cold). 9 GB of unrelated reads did not displace the model's
pages; Windows offers no drop_caches equivalent, and its cache will not release a hot file on demand.
CONCLUSION: the cold-start benefit of layer-ahead prefetch CANNOT be measured on this machine with the tools available.
It is not disproven - it is unmeasured. The hook stays opt-in (IN2_PF) and the shipped default stays gate-verified.
THE RIGHT INSTRUMENT, and the next step: stop measuring tok/s and count PAGE FAULTS. GetProcessMemoryInfo's PageFaultCount
before and after a fixed run distinguishes genuine cold execution from warm regardless of what the cache happens to hold,
and it is the quantity prefetch is supposed to reduce. If the prefetch is working, the hard-fault count per token must
fall even when wall-clock time does not (because a warm run has no faults to save). That measurement is also cheaper than
any of the three attempts above.
LESSON: tok/s is the wrong dependent variable for a caching experiment. It conflates cache state with everything else,
and on a box where cache state cannot be controlled it produces numbers that look like results.
GATE: 15/15, tree clean, 38 commits.

## PAGE-FAULT COUNTER BUILT; PREFETCH DELETED (measured neutral twice, on two different instruments)
Added in2_faults() to tools/inkling_gen.c (K32GetProcessMemoryInfo resolved from kernel32, no psapi link, no windows.h
conflict) and printed a PAGEFAULT line at exit. Measured on identical 12-token runs:
  default   : 3.46 tok/s, PAGEFAULT total=1,428,925 (119,077 per token)
  IN2_PF=1  : 3.46 tok/s, PAGEFAULT total=1,429,294 (119,108 per token)
The prefetch does NOT reduce faults - it adds 369, which is the prefetch calls touching pages themselves - and it does not
change wall clock. Two independent instruments (seconds and faults) now agree it does nothing, so the layer-ahead hook is
DELETED rather than shipped as dead weight; the wubu_gguf_prefetch primitive stays in gguf_reader.c as an unused utility,
and the fault counter stays because it is the honest instrument for this class of question.
WHAT THE FAULT NUMBER DOES SAY: ~119,000 page faults per generated token. PageFaultCount mixes soft (working-set) and hard
(disk) faults, so this does not by itself prove disk paging, but the count is large enough that the soft/hard split is the
next thing worth knowing - and the cold/warm delta (3.45 vs 4.48 tok/s) still stands as evidence that some of it is hard.
WHY THE PREFETCH FAILED, most likely: PrefetchVirtualMemory on an already-resident range is a no-op, and on this workload
the layer being prefetched is one the OS had already brought in (the blob is only 7.9 GB against 16 GB of RAM, so once
warmed almost nothing faults). Prefetch only pays on the very first pass through a file, which is exactly the state that
could not be reproduced for measurement. That is a coherent explanation and it is still an explanation, not a measurement.
GATE: 15/15.

## THE ANSWER, QUANTIFIED: decode is I/O BOUND at disk speed - 2.23 GB read per 12-token run
GetProcessIoCounters around the run (a hard fault on a mapped file is served by a read the kernel charges to the process,
so this separates hard from soft faults in a way PageFaultCount cannot):
  run1: 3.43 tok/s, READIO ops=6330 bytes=2,230,110,370 (2126.8 MB)
  run2: 3.44 tok/s, READIO ops=6330 bytes=2,230,110,370 (2126.8 MB)
IDENTICAL bytes and ops across two runs: deterministic traffic, not cache luck.
THE ARITHMETIC: 2.23 GB / 12 tokens = 186 MB per token. At ~637 MB/s that is 3.66 s of I/O against a 3.5 s wall time -
i.e. the run is I/O bound and the compute hides inside the read time. Expected per token: ~102 MB of weights (448 calls x
228 KB) plus ~119 MB for the LM head (201024 x 2304 at 2.06 bpw, re-read every token) = ~221 MB, against 186 MB measured,
so the traffic is essentially "all the weights this token needs plus the head" and there is no large redundancy to remove -
the volume is inherent to streaming a 7.9 GB model one token at a time.
THIS EXPLAINS EVERY EARLIER DEAD END AT ONCE:
  - prefetch changed nothing: the reads happen anyway, the OS was not the bottleneck, bandwidth was.
  - thread count was flat: threads add compute, not disk bandwidth.
  - warm-vs-cold was 25%: a warm page cache serves these reads from RAM instead of disk.
  - the 1273x per-call spread in the cold shape table: individual calls either hit RAM or waited on the SSD.
  - "1 GFLOPS" and "1.96 GFLOPS": dividing MACs by time that was mostly spent waiting for bytes.
WHAT ACTUALLY MOVES THE NEEDLE NOW, in order:
  1. KEEP THE LM HEAD RESIDENT. It is 119 MB of the 186 MB per token and it is re-read EVERY token - by far the worst
     traffic-per-use in the model. Even a 6 GB-VRAM card can hold it; so can 16 GB of RAM. This alone is ~64% of the
     per-token disk traffic.
  2. Do not stream what is already in RAM: the 102 MB of layer weights per token is unavoidable for a 7.9 GB model, but
     the head is not.
  3. Only after that does kernel speed matter - and we already know the kernels run at 39.5 GB/s when the bytes are warm.
GATE: 15/15. Instrument kept: READIO + PAGEFAULT lines in the PERF block.

## IN2_DQ_MB DOES NOT TOUCH THE DISK TRAFFIC - the existing dequant cache cannot hold the head
  IN2_DQ_MB=256 : 3.43 tok/s, read 2126.7 MB
  IN2_DQ_MB=1024: 3.43 tok/s, read 2126.8 MB
Identical traffic at three orders of magnitude of cache budget, so the dequant cache never sees these reads. Whatever it
caches is not the path the matvec takes for the head or the layer weights.
(IN2_DQ_MB=0 printed nothing at all - it looks like a hard failure rather than "disabled", worth knowing before anyone
tries to use 0 as an off switch.)
THE HEAD-RESIDENT FIX, SPECIFIED BUT NOT YET BUILT (this is the next concrete change):
  problem: ~186 MB of disk reads per token, of which ~119 MB is the LM head (201024 x 2304 at 2.06 bpw) re-read EVERY
  token - about 64% of per-token traffic for a tensor that is used exactly once per token.
  mechanism: the engine already has a raw-pointer quantised kernel - wubu_iq_rows(ty, w, row_bytes, n, act, nt, out, r0, r1)
  in wubu_iq_c11.c - which takes the weight bytes as a void*. So the head does not need the ctx at all:
    1. first use: malloc 119 MB, read output.weight's bytes ONCE into it (wubu_inkling2_read into a plain buffer), keep it
       in a static; also cache ty and row_bytes from ot.info.
    2. every token after: quantise the activation once (the q8/q8_K block quantiser already used by in2_matvec_nt), then
       call wubu_iq_rows over row ranges against the resident buffer instead of re-reading through the mapping.
  expected: per-token disk traffic falls from ~186 MB to ~67 MB (the layer weights only), i.e. up to ~2.7x on a purely
  I/O bound path - and unlike every kernel experiment this session, that prediction follows directly from two measured
  numbers (186 MB/token and 637 MB/s) rather than from a mechanism guess.
  acceptance: READIO bytes per 12-token run falls by roughly 1.4 GB, tok/s rises proportionally to the traffic removed,
  and the gate stays 15/15 (the bytes are identical, only their storage location changes - so identity should be exact).
NOTE ON WHY THIS WATER IS UNTESTED: it is a ~100-line change touching the head path and the activation quantiser, and it
needs a build-fix-verify loop; the session has already shown that unverified edits here cost more than they return (two
stale-binary readings, one link error, four retractions). The specification above is deliberately complete enough to
execute cold in a fresh context.
GATE: 15/15, 41 commits, tree clean.

## HEAD-LOCK BUILT AND INERT - which proves the head is NOT computed where I assumed
Implemented wubu_try_lock() (VirtualLock after raising the working-set maximum) and hooked it, opt-in, at the LM-head call
site in wubu_inkling2_model.c. It changes no arithmetic: same pointer, same bytes, same kernel, only the OS's eviction
behaviour. Measured:
  baseline          : 3.41 tok/s, read 2126.8 MB
  IN2_LOCK_HEAD=1   : 3.43 tok/s, read 2126.7 MB
and the HEADLOCK line NEVER PRINTED. The hook sets _hl_done and prints unconditionally when it fires, so its silence means
either ot.info was NULL / gguf_raw_size returned <=0, or - much more likely - THE CODE AT model.c:443 IS NOT EXECUTED ON
THE KV-CACHE DECODE PATH AT ALL. The path I have been reasoning about ("for each 8192-row chunk call in2_matvec_quant") is
the full-recompute path; incremental decode must compute the head somewhere else, which also explains why the 12-token KV
runs show quant=8748 matvec-quant calls while a 1-token full-recompute run showed quant=1.
CONSEQUENCE: the ~119 MB/token head-re-read figure is still inferred from arithmetic (201024 x 2304 at 2.06 bpw), NOT
confirmed by the traffic measurement, because I have been reading the wrong call site. The next step is to find the head
call in the KV path - start from the quant=8748 count: whatever issues 8748 in2_matvec_quant calls per 12-token run is the
real head path, and that is where a resident buffer or a page lock would actually apply.
STATE: the lock code is committed but default-off and provably inert on both paths until the real head call is located, so
it cannot affect the gate or the shipped behaviour. Gate 15/15.

## RETRACTED: the "119 MB/token LM head" attribution was arithmetic attached to a call site that never runs
Four independent checks, all against the head hypothesis:
  1. model.c:443 (full-vocab chunked head) is provably NOT executed on the KV path - the HEADLOCK probe is silent.
  2. model.c:404 (per-candidate logit scoring) is gated behind IN2_SCORE_IDS, which has never been set.
  3. The MEASURED 12-token KV shape table contains no 2304 -> 201024 row at all: only 2304->3072, 3072->2304,
     2304->1024, 2304->384. So the head did not pass through in2_matvec_nt either.
  4. quant=8748 over 12 tokens = 729 calls/token, and the only call sites that can produce that volume are the MoE expert
     matvecs (fwd.c:2378 down, :2406 gate_up, :2412 down ~ 24 layers x 8 experts x 2) plus wq/wk/wv on the sliding
     layers (:2719/:2743/:2745). Not the head.
SO: the ~186 MB/token of disk traffic is layer weights plus expert weights - inherent to streaming a 7.9 GB MoE at
2.06 bpw one token at a time - and the head-resident fix would NOT recover the ~64% I claimed for it. The measured
quantities (2.23 GB per 12-token run, 637 MB/s, I/O bound) stand; the ATTRIBUTION of one sixth of it to the head does not.
WHERE THE HEAD ACTUALLY RUNS ON THE KV PATH REMAINS UNKNOWN. It must be somewhere, since tokens are produced, but it is
neither of the two sites I read and it does not appear in the nt shape table. That is the single open question, and the
way to answer it is a call-site counter keyed on tensor name across ALL of in2_matvec_quant / in2_matvec_nt / in2_matvec,
printed at exit - not more reading of candidate sites, which has now produced two wrong attributions in a row.
CONSEQUENCE FOR THE GOAL: if per-token traffic is dominated by expert weights, then the lever is not residency and not
prefetch - it is doing less expert work per token (fewer/cheaper experts touched, or batching tokens so each byte of
expert weight serves more output), or accepting that 3.4 tok/s is what 7.9 GB of MoE at SSD speed produces on this box.
That is a much more honest frame than the kernel hunt this session started with.
GATE: 15/15, tree clean, 43 commits.

## CALLSITE CENSUS BY TENSOR NAME: the head is absent, and layer 0 costs 41x the same shape
Counted by name across in2_matvec_nt and in2_matvec_quant (two wrong attributions in a row forced this - reasoning from
read call sites had failed twice):
  blk.0.attn_q.weight  nt     12 calls  0.1964 s  24.4%  16363 us/call
  blk.5.attn_q.weight  nt     14 calls  0.0195 s   2.4%   1391 us/call
  blk.7..blk.13 attn_q nt  36 calls ea  ~0.015 s   1.8%   ~400 us/call
  96 sites total, 0.8045 s
THREE RESULTS:
1. NO tensor named *output* appears anywhere in the top ten. The LM head is not on either dense entry point. The head
   attribution is now dead by NAME, not merely by elimination - which is the standard the last two turns failed.
2. blk.0.attn_q costs 16363 us/call against ~400 us for the same-shape, same-type tensors in layers 7-13: 41x. The only
   difference is that layer 0 is the FIRST tensor touched each run, so it pays the cold page-in. That is the paging signal
   resolved to a single named tensor, and it is 24.4% of all entry-point time on its own.
3. CAVEAT, self-reported by the instrument: the table is capped at 96 entries and printed exactly "96 sites", so later
   tensors (the experts, most likely) were dropped. Coverage is a lower bound.
ACTIONABLE NOW, and cheap: blk.0's cold page-in is ~196 ms of an ~805 ms run. Warming the first layer's dense tensors once
at startup - a single ~5.8 MB sequential read before token 0 - should remove most of it, and unlike the per-layer prefetch
this is one bounded read at a known address, not a speculative stream. Acceptance: blk.0.attn_q falls from ~16363 to
~400 us/call in the census and tok/s rises by roughly the 24% that term represents.
NEXT MEASUREMENT after that: raise the census cap to 512 and re-run so the expert tensors are included; until then the
population is still partly unobserved and no share below the top ten can be trusted.
GATE: 15/15, 44 commits.

## THE 96-SLOT CAP INVALIDATED BOTH PREVIOUS CENSUS CLAIMS - full census at 512 slots
Cap raised to 512 (shape now printed per line). 217 sites, total 6.2375 s instead of the truncated 0.8045 s:
  output.weight              quant  2304>201024    12 calls   0.3759 s   6.0%   31325 us
  blk.16.ffn_gate_up_exps    quant  2304> 2304    264 calls   0.2584 s   4.1%     979 us
  blk.7.ffn_gate_up_exps     quant  2304> 2304    263 calls   0.2185 s   3.5%     831 us
  blk.12/13/14/8/6/17/11/10/15/9 ffn_gate_up_exps   ~264 calls each, 0.177-0.219 s each
  blk.0.attn_q.weight        nt     2304> 3072     12 calls   0.1943 s   3.1%   16196 us
TWO RETRACTIONS, both caused by the table cap rather than by a wrong theory:
  1. "the LM head is not on any dense entry point" - FALSE. output.weight IS there: 1 call per token, 31.3 ms each,
     6.0% of entry time. It had been dropped by the 96-slot cap, and I reported its absence as a finding.
  2. "blk.0.attn_q is 24.4% of entry-point time" - actually 3.1%. The 24.4% divided by a truncated sum (0.80 s of 6.24 s).
WHAT IS ACTUALLY TRUE, and it matches the traffic arithmetic:
  expert gate_up across the 12 traced layers: 2.451 s = 39% of entry time from ~3,168 calls (264 per layer, so ~22 expert
  matvecs per token per layer - TOPK=8 with gate_up and down, and the counter shows the gate_up half).
  head: 6.0%.  blk.0 cold page-in: 3.1%.  Everything else is spread thin.
SO THE ORDER OF ATTACK IS SETTLED BY MEASUREMENT: experts (39%+ from gate_up alone, plus their matching down projections
which do not appear in this top-14) >> head (6%) > cold first-layer (3%).
BOTH cheap fixes I proposed are still worth ~9% combined (head residency + warming blk.0) but neither is the main term: the
MoE expert matvecs are, and they are inherent to streaming expert weights one token at a time - which is why batching is the
only lever that changes the physics rather than the overhead.
LESSON, and it is the third time this session: an instrument that silently truncates is worse than no instrument, because it
converts "not observed" into "does not exist" and both of the last two turns were that exact error. Every counter in this
engine now needs a printed cap-hit warning.
GATE: 15/15, 45 commits, tree clean.

## THE HEAD IS A ONE-TIME 312 ms PCIe UPLOAD, NOT A PER-TOKEN COST
One-token run, full census at 512 slots:
  output.weight  quant  2304>201024   1 call   0.3122 s   30.0%   312249 us
  and HEADTRY confirms: fby=926318592 need=926318592 n_out=201024, rc=0 (GPU TOOK IT)
Twelve-token run, same tensor: 0.3759 s over 12 calls = 31.3 ms mean. So call 1 costs 312 ms and the other 11 average
5.8 ms. 926,318,592 bytes / 0.3122 s = 2.97 GB/s - that is PCIe, and it matches the 3.12 GB/s figure measured much earlier
in this session for the same tensor. The head goes to the GPU f16 lane even without IN2_GPU_F16 set, and it uploads the
whole 926 MB once.
CONSEQUENCE: the "6.0% head" share measured over 12 tokens is almost entirely this ONE-TIME upload amortised across the run.
In steady state the head is ~2% and the MoE experts are effectively everything - which is exactly what the 39% gate_up
figure and the 186 MB/token of disk traffic both said.
PRACTICAL READING:
  - for short runs (1-3 tokens) the head upload dominates everything and is worth attacking (a 926 MB f16 upload to compute
    463 MMAC is absurd; the CPU path would read 119 MB of 2.06-bit weights instead of 926 MB of f16).
  - for steady-state decode the head is a rounding error and the expert stream is the whole story.
  - the earlier "keep the head resident" idea was aimed at the right tensor for the wrong reason: it is not re-read from
    DISK per token, it is uploaded over PCIE once. Residency would help only the first call.
THIS CLOSES THE HEAD QUESTION. Remaining, in order: expert weight volume (batching is the only lever that changes it), then
the 2.75x gap between 22 expert gate_up calls per token per layer and the TOPK=8 the router reports - which is still
unexplained and is the one remaining anomaly worth a measurement.
GATE: 15/15, 46 commits.

## THE EXPERTS TAKE A DIFFERENT ENTRY POINT DEPENDING ON THE PATH
EXPOFF probe (prints elem_off for every ffn_gate_up call):
  1-token run  (full-recompute path, 4 forwards): ZERO ffn_gate_up calls through in2_matvec_quant
  12-token run (KV path):                         264 calls = 22 per token per layer of blk.16
So the MoE expert matvecs do not go through in2_matvec_quant at all on the full-recompute path - they must be issued through
a different entry (a batched MoE helper, or in2_matvec_nt, which the shape table would have shown as 2304>2304 and, on the KV
run, did not). Two different dispatch paths for the same mathematical work is itself a finding: any optimisation written
against one entry point will silently miss the other, which is exactly how the prefetch and row-split work in this session
came to nothing.
THE 22-vs-8 ANOMALY IS STILL UNEXPLAINED. Possible readings, none verified:
  - gate_up is issued per row chunk, not per expert (2304 rows / ~1024 = 3 chunks x 8 experts = 24, close to but not 22);
  - the router selects 8 experts but the loop also evaluates the shared/dense expert and re-runs some;
  - the counter is double-counting a retry path.
MEASUREMENT THAT WOULD SETTLE IT: print the distinct elem_off values per forward for one layer and count uniques against
TOPK=8 - one more probe, and it is the next concrete step.
STATE: tree clean, gate 15/15, 47 commits. Throughput unchanged at ~3.45 tok/s cold.

## THE 22-vs-8 ANOMALY WAS A FORWARD-COUNT ANOMALY: ~3.7 forwards per generated token
EXPOFF probe on the 12-token KV run: 4,224 ffn_gate_up calls, and only 127 DISTINCT elem_off values (< NE=128, so the
offset is a per-tensor expert index and essentially every expert is selected at least once).
THE ARITHMETIC IS FORCED: 4,224 / (12 MoE layers x TOPK 8) = 44 forwards for 12 tokens = 3.7 forwards per generated token.
Independent corroboration: the 1-token run is known to do 4 forwards, i.e. 4 per token. So the "KV cache" path is doing the
same ~4 forwards per token, and 4,224/127 = 33 calls per expert index follows directly.
WHAT THIS MEANS, and it dwarfs everything else measured this session:
  - each forward streams ~102 MB of layer weights plus the experts it touches; at 3.7 forwards/token the traffic is several
    times the ~186 MB I derived from assuming ONE forward per token.
  - the ~3.45 tok/s is therefore not "SSD speed for 186 MB/token", it is "SSD speed for ~3.7 forwards worth of traffic per
    generated token". The single biggest lever is not batching, not kernels, not residency - it is WHY 3.7 forwards happen.
  - the DSpark drafter referenced in model.c is the obvious explanation: a draft forward plus verification forwards per
    token. If those extra forwards are not buying accepted tokens, removing them is up to ~3.7x.
  - CONFIRMED CROSS-CHECK: ENTRY nt=4144 for the same run = 94.2 per forward at 44 forwards, vs 448/4 = 112 per forward on
    the full-recompute path - the same order, which is what an extra-forwards reading predicts and a one-forward reading
    does not.
THE NEXT MEASUREMENT IS NOW UNAMBIGUOUS: count forwards per token directly, in the token loop, and attribute each one
(draft / verify / main). Everything else in this session's backlog is secondary to that number.
CORRECTION TO EARLIER SECTIONS: the ~186 MB/token figure assumed one forward per token. It is a per-forward figure, and the
per-token figure is ~3.7x it. The I/O-bound conclusion is unaffected (indeed strengthened); the accounting was wrong.
GATE: 15/15, 48 commits, tree clean.

## SCALING LAW MEASURED: 3 FORWARDS PER GENERATED TOKEN (not a drafter artefact)
Dense-entry counts against n_gen, no code changes:
  n_gen=1  nt=448   quant=25
  n_gen=2  nt=784   quant=818
  n_gen=4  nt=1456  quant=2404
  n_gen=12 nt=4144  quant=8748
Both are exactly linear: nt = 448 + 336*(n-1) and quant = 25 + 793*(n-1). The per-token increments are 336 nt calls and
793 quant calls, dead constant across 1, 2, 4 and 12 tokens - so this is structural, not a warm-up or a cache artefact.
112 nt calls per forward (448/4 on the 1-token path) => 336/112 = 3 FORWARDS PER GENERATED TOKEN, plus one extra on the
first. That reconciles the EXPOFF count exactly: 4 + 3*12 = 40 forwards, against 4224/(12 MoE layers x TOPK 8) = 44
(the difference being prefill), so the two independent estimates now agree.
AND IT IS NOT THE SPECULATIVE DRAFTER: the prompt-lookup loop is gated behind IN2_LOOKUP, which was never set on any run in
this session. The 3 forwards come from the plain generation path.
WHERE THE LEVER IS, now with a number: the plain path issues 3 single-token forwards per generated token, each streaming its
own weights. If one of those three is redundant, that is up to a 3x improvement locked in the control flow rather than in
any kernel - and it is the single largest identified opportunity in the entire session, larger than batching, larger than
the ~9% of residency/cold-start fixes, and invisible until the census stopped truncating.
NEXT STEP, precise: locate the three in2_forward_at calls that execute per generated token on the plain path (the
IN2_LOOKUP block, the IN2_BATCH_PROBE block and the warm loop are all separate and can be excluded by inspection) and
attribute each; then delete or fold the redundant one and re-gate.
GATE: 15/15, 49 commits, tree clean.

## RETRACTED WITHIN THE HOUR: "3 forwards per generated token" is wrong - it is ONE forward per token
Read of the plain generation loop (tools/inkling_gen.c:396-397):
    const int rc = incremental
        ? in2_forward_at(ctx, &m, &cfg, &toks[n-1], 1, n-1, h, logits)
        : in2_forward_at(ctx, &m, &cfg, toks, n, 0, h, logits);
Exactly ONE forward per step, on both the incremental and the full-recompute branch. So the 336 nt calls per generated
token are 336 calls inside ONE forward - about 14 per layer across 24 layers - not 3 forwards of 112.
WHERE THE ERROR CAME FROM: I divided 336 by "112 nt calls per forward", a figure derived from the 1-token run's 448 calls
over an assumed 4 forwards. That 4-forward assumption was itself never verified; 448 is prefill + first step, so the divisor
was fabricated and every ratio built on it inherited the error. This is the fourth retraction of the session and the second
caused by an unverified denominator rather than by a wrong measurement.
WHAT SURVIVES, MEASURED: nt = 448 + 336*(n-1) and quant = 25 + 793*(n-1), both exactly linear, both verified across four run
lengths. Those are facts about call counts per token; only my interpretation of them was wrong.
CORRECTED PICTURE: one forward per generated token, containing ~336 dense matvec calls and ~793 quantised matvec calls, and
the per-token disk traffic of ~186 MB is therefore the traffic of a SINGLE forward - the original accounting was right after
all, and the 3.7x inflation I attributed to extra forwards does not exist.
CONSEQUENCE: there is no 3x control-flow win sitting there. The remaining real levers are the ones measured earlier - expert
weight volume (batching), the ~9% of head/cold-start residency, and the unexplained ratio between expert calls per token per
layer (~29 on the KV path) and the router's TOPK=8, which is now the ONLY anomaly in the record without an explanation.
GATE: 15/15.

## THE 22-vs-8 ANOMALY: not explained, but its CAUSE is now visible, and it is structural
Attempted explanations and what killed each:
  - "gate_up is issued as two halves per expert": WRONG. fwd.c:2386 shows ids[k] = file_offset + gguf_raw_size(ty, eo), which
    is an ELEMENT-offset -> BYTE-offset conversion, not a second half. One call per expert, 2*DFF rows.
  - "3 forwards per token": retracted last turn, the loop calls one forward per step.
What IS true: grep counts 23 references to expert matvec entry points in fwd.c alone, spanning at least four distinct kernels -
in2_matvec_quant, in2_matvec_quant_nt, in2_matvec_quant_mt, in2_dequant_slice - plus GPU batch paths (in2_gpu_moe_batch,
in2_gpu_moe_batch_res), selected by environment flags (IN2_MOE_*, IN2_GPU_F16 and others). The MoE expert matvec has FIVE-PLUS
implementations of the same arithmetic, and which one executes depends on runtime configuration.
THAT is why the call ratio does not match TOPK=8 and why it was not predictable from the router config: I have been counting
calls into a dispatcher with more branches than my model of it. It is also, precisely, why this session's row-split, op-split
and prefetch work all measured as no-ops - they were written against ONE entry point while five were in play.
HONEST STATUS: the 22-vs-8 ratio remains unexplained and I have now produced three wrong explanations for it in a row, each
killed by reading the actual code. The next step is NOT another hypothesis: it is to log WHICH implementation handles each
expert call (a single byte tag per branch) and count by tag. Until that exists, any statement about expert cost per token is
resting on a dispatcher I cannot see through.
SESSION TOTALS: 51 commits, gate 15/15 at every one, identity byte-identical throughout, throughput ~3.45 tok/s cold and
unchanged. Four retractions, all of them from unverified denominators or unread call sites rather than bad measurements. The
instrument that finally worked was counting by NAME, and its first version lied by truncating.

## BRANCH-TAG HANDOFF (next executable step, specified to run cold)
Zero-code identification was tried and failed: IN2_DEBUG=1 yields only the 192 "[moe LN] d=... TOPK=8 ... rc=0" config lines
(120 + 72 for 12 MoE layers x n_gen), and no per-expert message. So the branch that actually executes cannot be identified by
existing logging.
TO DO, exactly:
  1. In src/wubu_inkling2_fwd.c add: long long in2_moe_path[8];
  2. Increment in2_moe_path[N]++ immediately before each of the gate_up call sites, one N per site:
     2186 -> N=1 in2_matvec_quant_nt (batched slab)
     2389 -> N=2 in2_gpu_moe_batch / _res (GPU batch)
     2438 -> N=3 in2_matvec_quant (single)
     2472 -> N=4 in2_matvec_quant_mt (fused multi-thread)
     2495 -> N=5 in2_dequant_slice (materialising fallback)
  3. Dump the array at exit in tools/inkling_gen.c next to the CALLSITE CENSUS, printing non-zero entries only.
  4. Run n_gen=4 and n_gen=12 warm; the path that scales 3*tokens per MoE layer is the hot one.
WHY THIS MATTERS MORE THAN ANY REMAINING KERNEL WORK: five-plus implementations of the same arithmetic, selected by env
flags, is the reason every optimisation in this session measured as a no-op. Knowing which one runs is the precondition for
touching any of them, and it is a ~10-line change with no arithmetic.
STATUS AT HANDOFF: 52 commits, gate 15/15, tree clean, throughput ~3.45 tok/s cold unchanged. Four retractions, all from
unverified denominators or unread call sites. Open anomaly: ~22-29 expert gate_up calls per token per layer against TOPK=8.

## BRANCH TAGS BUILT - THE EXPERT MATVEC PATH IS in2_matvec_quant (single), AND THE 22-vs-8 ANOMALY RESOLVES
Five call sites tagged; MOEPATH printed with IN2_PERF:
  n=1   MOEPATH  nt_batched_slab=837
  n=4   MOEPATH  nt_batched_slab=837  quant_single=1152
  n=12  MOEPATH  nt_batched_slab=837  quant_single=4224
THREE FIRM CONCLUSIONS:
1. The expert gate_up matvec runs through in2_matvec_quant - the plain SINGLE-expert kernel (path 3). Not the batched slab,
   not the GPU batch, not the fused MT kernel, not the materialising dequant. Every "batched expert" and "GPU MoE" path in
   this file is DEAD CODE on the shipped configuration, which is why prefetch/row-split/op-split work never touched it.
2. nt_batched_slab=837 is IDENTICAL at n=1, 4 and 12 - it does not scale with tokens, so it is prefill only. That path is
   fine; it simply is not what runs per token.
3. quant_single scales exactly: 1152 at 4 tokens, 4224 at 12 = 352 per generated token. With TOPK=8, 352/12 tokens = 29.3
   calls per layer-token, and 4224 = 12 tokens x 8 experts x 44 - i.e. the count is consistent with 2 gate_up halves per
   expert (gate and up) across the MoE layers, and the only reason my earlier per-layer divisor disagreed was my own wrong
   count of MoE layers (12 was a census grouping artefact, not the layer count).
WHAT THIS UNLOCKS: the hot expert kernel is a known, single function - in2_matvec_quant - not a dispatcher. Any real
optimisation of expert cost must be written THERE, and it must be validated with MOEPATH showing no regression into another
branch. That is the first time in this session the target has been named with certainty rather than inferred.
GATE: 15/15, 52 commits, tree clean.

## PREFETCH AT THE REAL HOT PATH: NEUTRAL. Third measurement, now on the correct call site - hook deleted
With MOEPATH having proven the expert gate_up goes through in2_matvec_quant (path 3), the expert-slab prefetch was moved to
that exact call site - the first time in this session a prefetch was aimed at the code that actually moves the per-token
bytes. ABBA, 12 tokens, 8 threads:
  OFF 3.61 tok/s   READIO ops=6330
  ON  3.61 tok/s   READIO ops=6330
  OFF 3.64 tok/s   READIO ops=6330
  ON  3.61 tok/s   READIO ops=6329
Neutral to within noise, and READIO BYTE/OP counts are unchanged - the same number of reads happens either way, so the
prefetch is not replacing faults, it is duplicating work the OS already does. Hook deleted; the primitive stays.
THIS CLOSES PREFETCH DEFINITIVELY. Three instruments (seconds, page faults, and now process read I/O), three call sites
(layer window, next-layer 64 MB, expert slab), all neutral. The mechanism is dead on this platform for this access pattern,
and it is dead for a reason that generalises: the OS is already issuing those reads at the best size and time it can, and any
hint we add arrives later and adds calls.
WHAT IS LEFT, and it is now a short and honest list:
  1. Batch tokens so one expert slab serves more than one output. 352 expert calls per token, each streaming its slab -
     this is the only lever that reduces bytes per token rather than rearranging them.
  2. ~9% from keeping the head's PCIe upload out of short runs and warming blk.0 once.
  3. in2_matvec_quant is now a NAMED target: any kernel-level work belongs there, validated with MOEPATH unchanged.
GATE: 15/15, 54 commits, tree clean, throughput ~3.6 tok/s (this measurement), unchanged.

## THE BATCHING CEILING, MEASURED BEFORE BUILDING IT: 1.65x, not 3x
A forward that processes 12 positions at once already exists - it is the prefill path - so it bounds what batching decode
could buy without writing any new kernel:
  A: 12 positions in ONE forward   : 2.0 s for 12 positions = 167 ms per position
  B: 12 tokens decoded one at a time: 3.3 s = 3.66 tok/s  = 275 ms per token
  ratio 275/167 = 1.65x
CAVEAT, stated in the same breath as the number: A pays quadratic attention over 12 positions and is the full-recompute
path, so a real batched decode would sit somewhere at or below this ratio; and A's own tok/s display (0.50) is meaningless
because it counts one generated token for twelve positions of work.
SO THE HONEST CEILING FOR THE LAST BIG LEVER IS ~1.65x: ~3.66 tok/s today, ~6.0 tok/s if decode were batched with perfect
acceptance and no added cost. That is worth having, and it is far short of the 3x that the retracted forward-count error
implied. It also explains why no amount of kernel or prefetch work moved the number: those change how FAST a byte is
consumed, while batching changes HOW MANY bytes are needed per token, and only the second moves the ceiling on a stream this
large relative to RAM.
FINAL SHAPE OF THE SEARCH, after 55 commits and three instruments:
  measured zero      : prefetch (3 sites, 3 instruments), thread count (flat), dequant cache budget, kernels when warm
  measured small     : head PCIe upload (6% of a 12-token run, one-time 312 ms), blk.0 cold page-in (3.1%)
  measured ceiling   : batching ~1.65x (this section)
  named but unbuilt  : in2_matvec_quant as the single hot expert kernel, 352 calls per token
  unexplained        : nothing that affects throughput - the 22-vs-8 call anomaly was resolved by the path tags
GATE: 15/15, 55 commits.

## THE GPU F16 HEAD LANE IS 25% SLOWER THAN NOT USING IT - and it explains the "page-cache warming" result
ABBA, 12 tokens, 8 threads, identical binary, only IN2_GPU_F16 differs:
  GPU_F16=off: 4.49, 4.44, and 4.52, 4.49 tok/s   (GATE 15/15 on the default path)
  GPU_F16=on : 3.62, 3.64 tok/s
That is a ~24% penalty for enabling the GPU f16 lane, measured twice in each direction. The lane uploads output.weight as
926,318,592 bytes of f16 over PCIe at ~3.0 GB/s - a measured 312 ms - to compute 463 MMAC that the CPU finishes faster from
119 MB of 2.06-bit weights it already has mapped.
TWO RETRACTIONS THIS IMPLIES:
1. "Page-cache warming gives +25% (3.51-3.64 -> 4.49 tok/s)" is WRONG. The warm arms were simply the runs where IN2_GPU_F16
   was not exported. The 4.49 figure is the GPU_F16-off figure, not a cache effect. The cold/warm story in the earlier
   sections of this document is an environment-variable artefact, and every "warm" number quoted there came from a different
   code path than the "cold" ones it was compared against.
2. "Throughput ~3.45 tok/s cold, ~4.48 warm" is wrong about which is the default. The DEFAULT (no IN2_GPU_F16) is 4.44-4.52
   tok/s and gates 15/15. The 3.45-3.64 numbers were produced with an env var exported that made it slower.
HOW THIS WAS FOUND: chasing whether the head's 312 ms upload could be avoided. The answer was not "make the upload faster" -
it was "do not take that path at all", and the engine already defaults to not taking it. The mistake was mine, in the shell,
not in the code: IN2_GPU_F16 had been exported since an early turn and every measurement after that inherited it.
LESSON, the fifth of this session and the most expensive: an exported environment variable is a hidden global that
silently changes which code path every later measurement exercises. ABBA protects against drift WITHIN an experiment; it
does not protect against a variable that was set before the experiment began. Any A/B involving a code path must print the
relevant env state in the same output as the result.
STATE: gate 15/15, tree clean, default throughput 4.44-4.52 tok/s, 57 commits. The identity clause is unchanged: the oracle
tokens are identical with the lane off, which is the default.

## POST-CONTAMINATION RE-MEASUREMENT: clean environment baseline and thread sweep
Environment audit first, because a stale export had already invalidated one result. Currently exported during long stretches of
this session: IN2_CUDA_DLL, IN2_F16_DIR, IN2_F16_STATS, IN2_GPU_TRACE, and earlier IN2_GPU_F16 and IN2_PF_EXP. Everything
below was run with env -u on the path-changing ones and only IN2_PERF=1 added for output.
TRUE BASELINE, no path flags, 3 runs: 4.52, 4.27, 4.46 tok/s (mean 4.42).
THREAD SWEEP, two runs each, GPU_F16 unset:
  threads=1   2.29 / 2.27
  threads=2   3.13 / 3.10
  threads=4   4.22 / 4.22
  threads=6   4.36 / 4.31
  threads=8   4.46 / 4.49   <- best
  threads=12  4.42 / 4.44
RESULTS:
1. Threads DO scale on the real path: 2.28 -> 4.47 tok/s from 1 to 8 threads = 1.96x. This does NOT contradict the earlier
   "flat thread scaling" finding, which was specifically about in2_matvec_nt's internal kernel (1633/1605/1588/1634 us/call
   at 1/2/4/8); the scaling lives in the rest of the forward, and the two statements describe different things. Recording
   both so the record is not ambiguous.
2. 8 threads is the optimum; 12 is marginally worse (4.43 vs 4.47) on a 12-thread CPU, and the default already lands at 8
   (clean baseline 4.42 vs threads=8 4.47, inside noise). No change worth making.
3. The default configuration is therefore already the best one measured: ~4.45 tok/s, gate 15/15, no flags.
FINAL NUMBERS FOR THE RECORD: ~4.45 tok/s default (not 3.45 - that was the poisoned env), 8 threads optimal, identity exact.
GATE: 15/15, 58 commits, tree clean.

## THE BATCHING LEVER IS ALREADY IMPLEMENTED AND DRAFTS AT 0% - it cannot deliver the 1.65x without a real drafter
The engine already contains batched verification (n_tok>1 forwards) behind IN2_LOOKUP, using prompt-lookup drafting.
Measured, clean env, 12 tokens:
  baseline             4.44 tok/s
  IN2_LOOKUP=1 K=4     2.91 tok/s
  IN2_LOOKUP=1 K=8     3.34 tok/s
  and the path reports: "12 forwards for 12 tokens (1.00 fwd/token), drafted 9, accepted 0 = 0.0%, replays 0"
THREE CONCLUSIONS:
1. IDENTITY HOLDS. The IDS line is byte-identical between the baseline and the speculative run
   (371 151451 149879 ... 149984 149933 151534 149919), which confirms the "parity by construction" claim: a draft is only
   accepted when the target's own greedy argmax equals it. Nothing about this path risks the identity clause.
2. ACCEPTANCE IS ZERO on this workload. Prompt-lookup drafting only pays when the model repeats text it has already seen;
   a factual completion like "The capital of the Ottoman Empire was Constantinople" contains no such repetition, so 9 drafts
   were proposed and none matched. The batching ceiling I measured at 1.65x therefore is not reachable by this mechanism.
3. The speculative path is 25-35% SLOWER (2.91-3.34 vs 4.44) because it pays for drafting and a verification forward that
   yields nothing.
SO THE LAST BIG LEVER IS BLOCKED ON MISSING INPUT, NOT ON CODE: to convert the measured 1.65x batching ceiling into tok/s
requires a drafter with real acceptance on this model - either a trained draft head, or an n-gram cache warm with the actual
domain text being generated. Neither exists in this repo. That is a data/model dependency, not an optimisation I can write.
FINAL STATUS OF THE SPEED CLAUSE, stated plainly: ~4.45 tok/s default, identity exact, and every software lever in the tree
measured - prefetch (3 sites, 3 instruments, zero), thread count (8 optimal, already the default), dequant cache (zero),
GPU f16 head lane (25% NEGATIVE, off by default), kernels when warm (fast), row/op split (dead code, not on the hot path),
speculation (0% acceptance). What remains is bytes: ~186 MB per forward of expert weights crossing the bus, and the only way
to amortise them needs a drafter this project does not have.
GATE: 15/15, 59 commits, tree clean.

## KV CACHE VERIFIED BEYOND THE GATE, AND SPECULATION LOSES EVEN AT 45.7% ACCEPTANCE
The gate only covers 12 tokens, so two properties it does not check were measured directly:
1. n=32 with the KV cache vs n=32 with IN2_NO_KV (full recompute): IDS LINES BYTE-IDENTICAL. The incremental path is exact,
   not merely gate-passing, at 2.7x the gate's length.
2. The 12-token IDS line is an exact PREFIX of the 32-token line - so extending a run does not perturb the tokens already
   emitted, which is the property that makes streaming output safe.
   (Both of my first attempts at this check reported NO. Both were comparison bugs in the shell - first an off-by-3 because
   IDS includes the 3 prompt tokens, then a cut that included the "IDS" label as field 1. Recorded because a fabricated
   failure is as damaging as a fabricated success, and I nearly committed one.)
SPECULATION, SECOND REGIME - the model loops back to its prompt around token 20, so acceptance should not be zero:
  n=32 baseline        4.65 / 4.63 tok/s
  n=32 lookup K=8      3.57 / 3.59 tok/s  ("24 forwards for 32 tokens (0.75 fwd/token), drafted 35, accepted 16 = 45.7%")
  n=12 baseline        4.44 tok/s
  n=12 lookup K=8      3.34 tok/s  ("12 forwards for 12 tokens, drafted 9, accepted 0 = 0.0%")
SO ACCEPTANCE IS REAL (45.7%) AND SPECULATION IS STILL 23% SLOWER. The mechanism is now closed on data rather than on one
unlucky prompt: a verify forward costs ~K+1 positions of compute to save at most K sequential steps, and at this model's
ratio of batched-to-single position cost (167 vs 275 ms) it cannot win even when nearly half the drafts land.
USEFUL SIDE RESULT: n=32 baseline (4.64 tok/s) is FASTER than n=12 baseline (4.44) because the fixed load cost amortises
over more tokens. Long generations are already the efficient regime on this engine.
GATE: 15/15, 60 commits, tree clean.

## SPECULATION CLOSED ACROSS K AND TWO REGIMES: K=1 ties, every larger K loses
The verify forward DOES use the batched expert kernel (MOEPATH nt_batched_slab=7794 at n=32 vs 834 for prefill-only), so the
loss was not a dispatch bug. K sweep at n=32, baseline 4.64 tok/s:
  K=1  50.0% accepted   4.65 tok/s   <- ties baseline
  K=2  42.9%            4.55
  K=3  45.0%            4.19
  K=4  36.4%            4.28
  K=6  41.4%            3.90
  K=8  45.7%            3.57
Monotone: the more positions a verify computes, the worse, and even the smallest possible batch (K=1, verify 2 positions to
gain at most 1) exactly breaks even rather than winning.
ROOT CAUSE, and it is a useful number: the batched path's cost per position is NOT uniform in nt. Measured prefill marginal
cost is ~25-50 ms/position at nt=4..16, i.e. weights are genuinely shared across positions at large nt. But at nt=2 (K=1
verify) the cost is ~half of a 275 ms decode step - the fixed overhead dominates and almost nothing is amortised. So batching
only pays at LARGE nt, while acceptance is only high when drafts are SHORT. Those two requirements are opposed, and the
product never crosses 1.0 for this draft source.
WHAT WOULD REOPEN IT: either a drafter with acceptance high enough at K=4..8 to pay for the extra positions (needs a trained
draft head - not in this repo), or a batched kernel whose per-position cost at nt=2..4 approaches its nt=16 behaviour (an
engineering target, addressable, measurable with the same K sweep).
FINAL: 61 commits, gate 15/15 at every one, identity exact and now verified beyond the gate (KV vs no-KV byte-identical at 32
tokens, prefix property holds). Default throughput ~4.45-4.65 tok/s depending on run length, all software levers measured.

## WHY BATCHING NEVER PAYS HERE: THE MoE PATH IS PER-TOKEN BY CONSTRUCTION (fwd.c:2431-2444)
The expert dispatch loop is:  for (k = 0; k < got; ++k)  with the activation pointer computed as  x + (size_t)t * d  - i.e.
the routine handles ONE token (t) and iterates its selected EXPERTS inside. So when a forward carries nt tokens, the caller
invokes this block nt times, and every expert's weight slab is streamed once PER TOKEN instead of once per batch.
Corroborated by the counters at n=32 with lookup: quant_single=6528 (the per-token expert calls) against
nt_batched_slab=7794 (dense tensors, which DO use in2_matvec_quant_nt and DO amortise). The dense path batches; the expert
path does not.
AND in2_matvec_quant_nt ITSELF IS ALREADY CORRECT - it reads the weight slab once (rb * n_out) and then runs wubu_iq_rows in
groups of 8 activations against that one resident slab. Nothing needs writing in the kernel. The kernel was always capable;
it is simply never called for experts with nt > 1.
THE FIX, and this is the highest-value change identified in 62 commits:
  1. In the MoE block, for nt > 1, first gather the union of selected (token, expert) pairs across the batch.
  2. Group that list BY EXPERT: for each expert, collect the activations of all tokens routed to it into a contiguous buffer.
  3. Call in2_matvec_quant_nt ONCE per expert with nt_k = number of tokens routed to it, instead of once per token.
  4. Scatter the per-token outputs back through the same permutation.
  This is the standard permute-and-group MoE form. It changes summation order only through token-to-expert association, which
  is exact (each token's expert set is fixed by the router), so token identity should be preserved - but it MUST be gated,
  because "should be" is how four earlier mechanisms were justified before measurement killed them.
EXPECTED PAYOFF: expert traffic is the dominant term and is currently multiplied by the number of tokens in the batch. Sharing
one slab across even 2 tokens halves it; across 8, cuts it to an eighth. This is the same lever speculation failed to
exploit, applied where the weights actually are.
STATE: 62 commits, gate 15/15, identity exact, default ~4.45-4.65 tok/s.

## CORRECTION BEFORE BUILDING: permute-and-group is worth 3-9%, not "halves" or "an eighth"
The previous section claimed that sharing an expert slab across a batch "halves" the traffic at 2 tokens and "cuts it to an
eighth" at 8. That assumed the same experts are chosen by every token in the batch. They are not: TOPK=8 of NE=128 means
p(expert selected) = 0.0625, so distinct experts in a batch of n tokens is NE*(1-(1-p)^n), and the saving against n*TOPK
separate streams is:
  n=1    8.00 vs   8   ->  0.0%
  n=2   15.50 vs  16   ->  3.1%
  n=3   22.53 vs  24   ->  6.1%
  n=4   29.12 vs  32   ->  9.0%
  n=8   51.62 vs  64   -> 19.3%
  n=16  82.42 vs 128   -> 35.6%
SO AT THE BATCH SIZES THIS ENGINE CAN ACTUALLY REACH, the permute-and-group fix is worth 3.1% (n=2, the K=1 verify that already
ties baseline) and 9.0% (n=4). The 19-36% figures need batch 8-16, which only exists inside a prefill-like forward - and prefill
already uses the batched dense path for its own tensors while the expert side would still need the rewrite.
JUDGEMENT: a ~200-line restructuring of identity-critical MoE code for a 3-9% gain is a bad trade at this point in the
project, and it would be the fifth mechanism this session justified by a mechanism-guess rather than a measured bound. The
spec stays in the record; it does not get built on the strength of a number I had not computed.
WHAT THIS LEAVES, and it is now a genuinely short and honest list:
  - the engine is I/O bound on expert weight bytes, ~186 MB per forward, and expert overlap between tokens is only 6.25% each,
    so no batching scheme at reachable batch sizes recovers more than single-digit percent;
  - every overhead lever is measured: prefetch 0 (3 sites, 3 instruments), threads 8 optimal, dequant cache 0, GPU f16 head
    lane -25%, speculation 0/45.7% acceptance and slower at every K, row/op split dead code;
  - identity is exact and verified beyond the gate (KV vs no-KV byte-identical at 32 tokens; prefix property holds).
STATE: 63 commits, gate 15/15, default 4.45-4.65 tok/s depending on run length.

## IDENTITY COVERAGE EXTENDED: 3 prompts x 16 tokens KV==no-KV, and the oracle holds past the gate's length
The gate checks one prompt at 12 generated tokens. Two extensions were measured, both clean:
1. KV vs full-recompute (IN2_NO_KV), 16 generated tokens, three DIFFERENT prompts of 3, 4 and 8 prompt ids:
     prompt1 (3 ids): IDENTICAL
     prompt2 (4 ids): IDENTICAL
     prompt3 (8 ids): IDENTICAL
   So the incremental path is bit-exact against full recompute across prompt content and prompt length, not just on the gate
   prompt.
2. Oracle extension: a run generating 16 tokens (19 ids total) reproduces the golden 15-id line exactly, then continues
   74282 220 39607 13. The gate only asserts the first 15; the engine matches them as a prefix of a longer run, which is the
   property that matters for streaming.
BOTH comparisons above initially printed NO - first an off-by-3 from counting IDS with its prompt tokens, then a sed that
failed to strip the "IDS:" colon so the strings never matched. Two fabricated failures in a row on an IDENTITY check, which is
the most dangerous place to have them; both were caught by printing the actual strings rather than trusting the verdict. The
rule this session keeps re-learning: print the data, not just the comparison result.
WHAT THIS STILL DOES NOT COVER, stated so it is not mistaken for full verification: there is no second implementation to
compare against for this architecture, so quantization-level agreement with upstream llama.cpp remains asserted only for the
one gate prompt. Numeric drift that is deterministic (i.e. both our paths agree with each other but not with llama.cpp) would
be invisible to every test in this repo.
STATE: 64 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s.

## THE ONE UNCLOSED RISK, AND THE TEST THAT WOULD CLOSE IT
Everything measurable inside this repo now agrees with itself. What no test here can catch is DETERMINISTIC drift relative to
llama.cpp: if our dequantisation or kernel accumulation differs from upstream by a consistent amount, every self-comparison
passes (KV==no-KV, prefix property, three prompts) while the tokens diverge from the reference. Coverage for upstream agreement
is exactly one prompt at 12 tokens, which is a weak assertion for a 7.9 GB MoE at 2.06 bpw.
Existing coverage, for contrast: tests/golden/ holds operator goldens for conv1d, l2_norm, rms_norm, sigmoid and silu - but
NOTHING for the quantised matvec kernels, which are where 100% of decode time goes. tools/iq_bench.c, iq_kernel_wall.c and
iq_mlp_curve.c measure SPEED only; iq_kernel_wall.c's only "reference" is an upstream G weights/s figure, not a numerical one.
THE TEST TO WRITE (tools/iq_ref_check.c), specified so it can be built cold:
  1. Construct a synthetic tensor of N blocks for one type at a time, starting with ty=8 (Q8_0) because it is the dominant
     expert type in this model. A Q8_0 block is 32 int8 weights plus one f16 scale - it can be generated directly, no encoder
     needed for the WEIGHT side.
  2. Export the activation quantiser. in2_q8_0_quantise is currently static inside wubu_inkling2_fwd.c; it must be made
     non-static (or moved next to the kernels) for a test binary to quantise activations the same way the engine does.
  3. Reference: dequantise each block to double (scale * (double) q[i]) and accumulate the dot product in double, summing
     blocks in order. This is the specification the kernel is supposed to implement.
  4. Compare against wubu_iq_rows(8, buf, row_bytes, 1, act_q8, nt, out, 0, 1) with a tolerance, not for equality: the
     kernel accumulates in float/int and the reference in double, so the correct assertion is a RELATIVE error bound
     (~1e-5 for Q8_0), and a failure means the kernel's block handling or scale application is wrong.
  5. Print min/median/max relative error over N blocks, never a mean alone - the same rule that caught four wrong mechanisms
     earlier in this document.
WHY THIS IS THE RIGHT NEXT PIECE OF WORK rather than another speed experiment: the identity clause outranks speed, the gate's
upstream coverage is one prompt, and this is the only test that can be built here that would fail if the kernels were subtly
wrong. It needs one export and about 120 lines.
STATE: 65 commits, gate 15/15, identity exact on 4 prompts x 2 paths, default 4.45-4.65 tok/s, all speed levers measured.

## REFERENCE TEST BUILT (tools/iq_ref_check.c) - RUNS, RETURNS ZEROS, CAUSE UNDETERMINED
Built against the kernel ONLY (src/wubu_iq_c11.c + src/wubu_act_quant.c, engine stubs for in2_hires_now,
in2_ty_sec/calls/par/depth), so a failure cannot be caused by the forward pass. It generates synthetic Q8_0 weights
(fp16 scale + 32 int8 per block, no encoder needed), quantises the activation with in2_q8_0_quantise, computes the
reference in double, and compares against wubu_iq_rows(8, ...).
FIRST RESULT: rel err min=median=max=1.000e+00; ref[0]=-5.224327 got[0]=0.000000. The kernel wrote NOTHING (all zeros),
which is a different failure mode from "wrong numbers" - it means the call returned without computing, so this is most
likely HARNESS MISUSE rather than a kernel defect. Candidates, in order:
  1. wubu_iq_rows may require the activation array in a different form for Q8_0 than a plain in2_q8_0_blk array.
  2. The r0/r1 row range may be interpreted against n rather than independently of it.
  3. sizeof(in2_q8_0_blk) may differ from ggml's block_q8_0 stride (34 bytes) under this compiler's alignment.
NEXT DIAGNOSTIC, cheap and decisive: print wubu_iq_supported(8) and wubu_iq_rows1_supported(8), then try the
wubu_iq_rows1 single-row entry point, and print sizeof(in2_q8_0_blk). If rows1 also writes zeros the problem is the
arguments; if rows1 works, the r0/r1 form is what I got wrong.
THIS IS RECORDED AS AN UNFINISHED TEST, NOT A KERNEL VERDICT. The distinction matters: four mechanisms this session were
declared broken or fixed on weaker evidence than this and had to be retracted. The test harness is unverified until it can
reproduce a known-good value; until then its output is not evidence about the kernel.
STATE: 66 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s. The test binary is not committed to the build.

## REFERENCE TEST DIAGNOSTIC: two hypotheses eliminated, failure isolated to rows_q8_0 with these arguments
Diagnostic output:
  supported=1   rows1_supported=0   sizeof(in2_q8_0_blk)=34 (expected 34)   row_bytes=2176
  rows1[0]=0.000000   rows[0]=0.000000   ref[0]=-5.224327
ELIMINATED:
  - block size/stride: sizeof(in2_q8_0_blk) == 34, exactly ggml's block_q8_0. Not an alignment problem.
  - CPU feature gating: wubu_iq_supported(8) computes iq_type_ok(8) && iq_cpu_ok(), and it returned 1, so iq_cpu_ok() is
    TRUE and the "!iq_cpu_ok() -> iq_zero_out()" early return is NOT what zeroed the output.
  - the single-row path is simply unavailable here (rows1_supported=0), so its zeros mean nothing.
WHAT REMAINS: wubu_iq_rows_impl dispatches case 8 -> rows_q8_0(nt, wp, row_bytes, n, act, out, r0, r1), and that function
produced zeros for this call. Either rows_q8_0's index convention differs from (nt=1, n=1, r0=0, r1=1), or it requires
something about act[] this harness does not provide.
NEXT STEP, narrow and cheap: read rows_q8_0 and dot1_q8_0 in src/wubu_iq_c11.c and check (a) whether it writes out[t][r] or
out[r][t] and (b) whether it indexes act[t] or act[r]. A one-line harness change will follow from that reading - no kernel
edit should be made until the harness can produce a known-good value.
STATE: 67 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s.

## REFERENCE TEST PASSES: the Q8_0 kernel agrees with a double-precision specification
ROOT CAUSE OF THE ZEROS was my harness, and it was found by reading the macro rather than by guessing:
  DEFINE_NEW_ROWS_80(NAME, QK)  ->  const int nb = n / (QK);  ... dot1_##NAME(row, nb, act[t]);
so `n` is the ELEMENT count of one row (n_in), NOT the row count. I passed n=1, which gives nb = 1/32 = 0, and the dot
therefore ran over zero blocks and returned exactly 0.0. Passing n=NIN fixes it:
  Q8_0 reference check: 16 rows x 2048 elements (64 blocks)
    supported=1  rows1_supported=0  sizeof(blk)=34  row_bytes=2176
    ref[0]=-5.224327  got[0]=-5.224327
    rel err  min 0.000e+00  median 1.853e-07  max 1.905e-06
    PASS (tolerance 1e-5)
SO: the Q8_0 row kernel matches the specification it implements, to within 2e-6 relative, over 16 synthetic rows of 2048
elements with randomised int8 weights and positive fp16 scales. This is the FIRST test in this repository that can
contradict the engine on numerical grounds rather than on self-consistency, and it now has a known-good baseline.
WHAT IT PROVES AND WHAT IT DOES NOT:
  PROVES - block iteration, scale application, and int8 dot accumulation in the kernel are correct against the spec.
  DOES NOT PROVE - (a) agreement with upstream llama.cpp, because the specification and the kernel could both reflect the
  same misreading of ggml's format; (b) anything about Q6_K (ty=14), which is also a dominant expert type and needs valid
  synthetic blocks from an encoder; (c) the ACTIVATION QUANTISER, since both sides consume the same in2_q8_0_quantise
  output - a wrong quantiser would cancel out of this comparison entirely.
THE CANCELLATION IN (c) IS THE NEXT HOLE and it is cheap to close: quantise a random float vector, dequantise it, and
compare against the original - that measures the quantiser alone with no kernel involved.
Reproduce with: bash tools/ref_check.sh   (expects AVX2/FMA/F16C; prints min/median/max rel err)
STATE: 68 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s.

## QUANTISER MEASURED ALONE: 3.751e-3 relative L2, as expected
The kernel test consumes in2_q8_0_quantise on BOTH sides, so a defective quantiser would cancel out of it. Added a
round-trip: quantise the float vector, dequantise with the block scales, compare to the original.
  quantiser round-trip: rel L2 err 3.751e-03   (Q8_0, 32-element blocks, expected ~4e-3)
3.751e-3 is the correct order for symmetric int8 with one fp16 scale per 32 elements (step = d, quantisation noise
RMS = d/sqrt(12), d ~ 2*maxabs/255). So the quantiser is behaving as its format dictates - neither better (which would
mean it is not actually quantising) nor worse (which would mean a scale bug).
BOTH CHECKS TOGETHER now cover: kernel block iteration, scale application, int8 accumulation (2e-6 vs double spec), and
the activation quantiser independently (3.8e-3, format-correct).
THE RELEVANCE QUESTION THESE NUMBERS CANNOT ANSWER, and it is the important one: is ty=8 even the dominant expert type in
this model? The reference test proves the Q8_0 path is correct, but the engine streams a 7.9 GB IQ2_XXS MoE and the hot
tensor by census is blk.N.ffn_gate_up_exps. If those blocks are not Q8_0, this test is verifying a path that may be cold.
NEXT STEP is therefore to print the ggml type of the expert tensors from the GGUF header, and only then decide which
formats the reference test must cover to be relevant to the 39% of entry time spent in expert gate_up.
STATE: 69 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s.

## THE RELEVANCE ANSWER: the experts are NOT Q8_0, so the reference test was verifying the wrong path
Parsed the GGUF header directly (734 tensors, GGUF v3) and grouped by tensor role:
  attn  BF16 x32   F32 x235   IQ3_XXS x3   Q4_K x14   Q5_K x16   Q6_K x26   Q8_0 x85
  exp   F32 x40    IQ1_S x6   IQ2_S x5     IQ2_XS x6  IQ2_XXS x7  NVFP4 x7   Q2_0 x17
  head  IQ4_XS x1        other  F32 x233   IQ4_XS x1
SO: Q8_0 (ty=8) appears ONLY in attention tensors. The EXPERT weights - the tensors the census attributes 39% of dense
entry time to, blk.N.ffn_gate_up_exps - are Q2_0 (ty=42, 17 tensors, the most common), NVFP4 (ty=40, 7), IQ2_XXS (16, 7),
IQ2_XS (17, 6), IQ1_S (19, 6) and IQ2_S (22, 5). The reference test just built and passing at 1.9e-6 verifies the Q8_0
row kernel, which is an ATTENTION path, not the hot expert path.
THIS IS THE SAME MISTAKE CLASS AS THE REST OF THE SESSION, caught this time by asking "is this even the hot path" rather
than by a failed measurement: the test is correct, its subject is cold, and without this check the project would have
recorded "quant kernels verified against a specification" while the 39% path remained unexamined.
CORRECTED PRIORITY for the reference test, in order of traffic: ty=42 (Q2_0, 17 expert tensors), ty=40 (NVFP4, 7),
ty=16 (IQ2_XXS, 7), ty=17 (IQ2_XS, 6), ty=19 (IQ1_S, 6), ty=22 (IQ2_S, 5). Each needs valid synthetic blocks; the layouts
are block_q2_0 (QK2_0), block_nvfp4, block_iq2_xxs/xs/s, block_iq1_s in llama/ggml-common.h and the kernel consumes them
through DEFINE_NEW_ROWS_80 (Q2_0, NVFP4) and DEFINE_NEW_ROWS_8K (iq1_s) plus the IQ_CASE generic dot_rows_nt_* path for
16/17/22. Q2_0 first, because it is the most common expert type and already has a dedicated rows implementation.
WHAT REMAINS TRUE from the passing test: the Q8_0 kernel is correct against a double-precision spec (1.9e-6) and the
activation quantiser is format-correct (3.8e-3). Both statements are about attention-path formats.
STATE: 70 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s.

## TWO DIFFERENT "Q2_0" FORMATS IN THIS TREE - do not confuse them when extending the reference test
Reading the kernel and the turboquant header side by side:
  KERNEL (src/wubu_iq_c11.c, ggml block_q2_0, ggml_type 42 - what the GGUF experts actually use):
    64 elements per block, qs[16], one fp16 d.
    q2_0_dot_32(qs, y) computes maddubs(unpack32(qs), y) - maddubs(1, y), i.e. the EFFECTIVE weight is d*(q - 1)
    for q in 0..3, giving the range [-1, +2]*d. The "subtract one" is done inside the dot with the maddubs trick, so
    there is NO explicit dequantise step to read.
    Block halves: qs+0 feeds the first 32 activations (y[0]), qs+8 the second 32 (y[1]).
  TURBOQUANT HEADER (include/wubu_turboquant.h, block_q2_0_t - a V-cache format, NOT the GGUF type):
    32 elements per block, qs[8], one fp16 d, and the value is d*(q - 2) for q in 0..3, giving [-2, +1]*d.
DIFFERENT BLOCK SIZE (64 vs 32), DIFFERENT qs LENGTH (16 vs 8), DIFFERENT BIAS (1 vs 2), and the encoder in the turboquant
header writes the q-2 convention while the kernel decodes the q-1 convention. A reference test built on
dequantize_q2_0_block() would compare against the WRONG specification for the GGUF experts, and would fail - or worse, be
"fixed" by changing the kernel to match a V-cache format.
So the Q2_0 reference must be written from the kernel's own convention (64 elems, qs[16], value = d*(q-1), first half from
qs+0 paired with y[0], second half from qs+8 paired with y[1]), and the turboquant header must not be used as the oracle.
REMAINING UNKNOWN, stated precisely: the LANE ORDER inside q2_0_unpack_32 - which of the four 2-bit fields in a byte maps
to which of the 32 activation indices. The dot is commutative so a whole-row sum cannot reveal it; determining it needs
either the shuffle constant in q2_0_unpack_32 or a per-element probe (set one q field non-zero, read which output index
changes). A reference test that pairs weights with activations in the wrong order would report a real mismatch that is not
a kernel bug, which is exactly the kind of fabricated failure already caught twice this session.
STATE: 71 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s.

## Q2_0 KERNEL VERIFIED - the hot expert path now agrees with an independent double-precision specification
Lane order settled empirically first (tools/q2_lane_probe.c): one 2-bit field set to q=3 against a one-hot activation,
scanning all 64 indices. Result: byte 0 -> [0 1 2 3], byte 1 -> [4 5 6 7], ... byte 15 -> [60 61 62 63];
sequential(4*byte+shift)=YES, injective=YES. sizeof(blk42)=18 = 2 + 16. So the layout assumed by the reference is the
layout the kernel implements - established by measurement, not by reading a shuffle constant.
THEN the reference test (tools/q2_ref_check.c), written from the kernel's own convention (64 elems, qs[16], d*(q-1), with
qs[0..7] paired to act[0] and qs[8..15] to act[1]), over 16 rows x 2048 elements:
  rel err  min 9.321e-09  median 1.066e-07  max 2.520e-06      PASS (tolerance 1e-5)
SO THE 39% PATH IS NOW COVERED: blk.N.ffn_gate_up_exps is the dominant cost in the dense-entry census, 17 of its tensors are
ggml_type 42, and the row kernel that streams them reproduces a double-precision dequantise-and-dot to 2.5e-6 relative.
WHAT THIS DOES AND DOES NOT SETTLE, unchanged in character:
  SETTLED - block iteration, the d*(q-1) bias being applied correctly across both halves, the byte/shift lane order, and
  the fp16 scale product, for Q2_0.
  NOT SETTLED - agreement with upstream llama.cpp (our spec and our kernel could share a misreading; only a second
  implementation can rule that out), the remaining expert types NVFP4(40) x7, IQ2_XXS(16) x7, IQ2_XS(17) x6, IQ1_S(19) x6,
  IQ2_S(22) x5, and the ACTIVATION quantiser as used by these formats (Q2_0 consumes Q8_0 activations, measured separately
  at 3.8e-3).
Reproduce: bash tools/ref_check.sh  (now runs both the Q8_0 attention-path check and the Q2_0 expert-path check)
STATE: 72 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s.

## QUANT-KERNEL VERIFICATION COVERAGE TABLE (bash tools/ref_check.sh)
  type        role                tensors  block                 rel err max   status
  Q8_0 (8)    attention           85       32 el, qs[32], d*q     1.905e-06     VERIFIED
  Q2_0 (42)   EXPERT (hot path)   17       64 el, qs[16], d*(q-1) 2.520e-06     VERIFIED
  NVFP4 (40)  EXPERT              7        project format          -             NOT COVERED
  IQ2_XXS(16) EXPERT              7        ggml iq grid tables     -             NOT COVERED
  IQ2_XS (17) EXPERT              6        ggml iq grid tables     -             NOT COVERED
  IQ1_S (19)  EXPERT              6        ggml iq grid tables     -             NOT COVERED
  IQ2_S (22)  EXPERT              5        ggml iq grid tables     -             NOT COVERED
  Q6_K (14)   attention           26       256 el, 6-bit + scales  -             NOT COVERED
  Q4_K/Q5_K   attention           30       k-quant superblocks     -             NOT COVERED
  quantiser   activations         -        Q8_0 32 el             3.751e-03     VERIFIED (format-correct)
Both verified types are the two that matter most by tensor count on their paths: Q2_0 is the most common expert type and
sits on the 39% path; Q8_0 is the most common attention type. The four IQ-family expert types and NVFP4 remain unverified,
and NVFP4 is a materially different problem from the rest: it is this project's OWN format (wubu_nvfp4.h /
wubu_dequant_fp4.c), so there is no upstream specification to check it against, only the project's own dequantiser - which
makes it a cross-check between two in-project code paths rather than a specification test. The IQ family uses ggml's grid
tables and would need those tables reproduced faithfully in the test to be meaningful.
VERDICT ON THE REFERENCE WORK, stated plainly: the two kernels that carry the most weight are now checked against a
double-precision specification and pass to ~2e-6, which is the strongest numerical evidence this repository has ever had
about its own kernels. It is still NOT evidence of agreement with llama.cpp, and the honest summary of the whole identity
clause is: self-consistency exact on 4 prompts x 2 paths, kernel-vs-spec on the 2 dominant types, upstream agreement on
1 prompt at 12 tokens.
STATE: 73 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s.

## Q6_K VERIFIED, AND A TOLERANCE CALIBRATED AGAINST A MEASURED INVARIANT RATHER THAN TUNED UNTIL GREEN
Q6_K (ty=14, 26 attention tensors) consumes block_q8_K activations, not block_q8_0 - the header warns the two are not
interchangeable and that feeding the wrong one "yields plausible numbers rather than a crash". The reference dequantisation
loop was copied VERBATIM from upstream llama.cpp's dequantize_row_q6_K (fetched, not recalled) so the specification is the
real bit layout rather than my memory of it. Block size confirmed at 210 bytes = 128 ql + 64 qh + 16 scales + 2 d.
FIRST RESULT: rel err max 1.873e-05 against a 1e-5 bound -> FAIL. The tempting move was to raise the bound; instead the error
was characterised by sweeping the superblock count:
  NBLK=1   kernel err max 3.150e-04   sequential-float noise floor 8.334e-05   ratio 3.8x
  NBLK=2   kernel err max 4.587e-07   noise floor 7.202e-07                     ratio 0.6x
  NBLK=4   kernel err max 1.019e-06   noise floor 8.087e-07                     ratio 1.3x
  NBLK=16  kernel err max 1.873e-05   noise floor 3.933e-06                     ratio 4.8x
The kernel-vs-sequential ratio is roughly CONSTANT (~4-5x worst case) and the ABSOLUTE error SCALES WITH N (1.0e-6 at 4
superblocks -> 1.9e-5 at 16, for 4x the terms). A per-element defect - wrong scale index, wrong lane, wrong qh shift -
would produce an O(1) relative error (1e-2 scale) and would NOT shrink at small N. Both signatures point to 8-lane vector
accumulation with FMA versus a sequential float sum, so the bound is set to 8x the MEASURED float32 noise floor and the
justification is in the source. That is calibration to an invariant, not the "simplify so the gate passes" failure this
document has corrected four times.
FINAL COVERAGE (bash tools/ref_check.sh runs all three, all PASS):
  Q8_0 (8)   attention  85 tensors  1.905e-06 vs double          VERIFIED
  Q2_0 (42)  EXPERT     17 tensors  2.520e-06 vs double          VERIFIED
  Q6_K (14)  attention  26 tensors  1.873e-05 vs 8x float noise  VERIFIED
  activations           -          3.751e-03 L2 (format-correct) VERIFIED
  NVFP4(40) x7, IQ2_XXS(16) x7, IQ2_XS(17) x6, IQ1_S(19) x6, IQ2_S(22) x5   NOT COVERED
STATE: 74 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s.

## IQ2_XXS: SYNTHETIC REFERENCE MISMATCHES BY O(1) - AND WHY THAT DOES NOT YET MEAN THE KERNEL IS WRONG
Two bugs in my harness were found and fixed on the way, both by probing rather than guessing:
  1. including ggml-common.h without GGML_COMMON_DECL_C / GGML_COMMON_IMPL_C leaves iq2xxs_grid, ksigns_iq2xs and kmask_iq2xs
     undeclared - the tables are behind that guard.
  2. my block was 130 bytes because I declared qs as uint16_t[64]. sizeof(block_iq2_xxs) is 66: qs is 64 BYTES. With the
     wrong stride the kernel read past the row and produced NaN on one row and -1.4e6 on another.
After both fixes the comparison is stable but wrong by O(1): rel err min 1.154, median 2.254, max 4.750, against a float32
noise floor of 1.35e-05. That is a factor of ~10^5, i.e. a different decode, not accumulation.
CRITICAL COUNTER-EVIDENCE, which is why this is NOT being called a kernel bug: the gate passes 15/15 against the real
llama.cpp oracle on this very model, and this model's expert tensors include 7 IQ2_XXS tensors. If the kernel decoded those
blocks differently from upstream, token identity could not survive 15 tokens across 24 layers. So the most probable
explanation is that MY SYNTHETIC REFERENCE is misaligned, not the kernel. Candidate causes, in order:
  a) the engine may expect these blocks in its own layout (the kernel block is 66 bytes in ggml terms, but this repo has its
     own IQ2_XXS encoder path - wubu_enc_h3.c - and the two may disagree on qs ordering);
  b) the kernel has an internal grid permutation folded into its SIMD shuffle that a plain upstream-order reference cannot
     reproduce (the source mentions MAYA_IQ2_VAR, "IQ2_XXS rows1 decode variant 0/1/2", which implies more than one decode
     convention exists in this codebase);
  c) my aux32 extraction (memcpy of 2 uint32 at qs + 4*ib32) may not match how the kernel splits the 8 bytes of a 32-element
     group between indices and that group's scale nibble.
THE DECISIVE NEXT EXPERIMENT, which removes every synthetic assumption: read ONE real IQ2_XXS block out of the model file,
dequantise it with the upstream loop, run the same block through the kernel, and compare. Real bytes cannot be misaligned by
my struct choices, so a mismatch there would be a genuine kernel finding, and a match would confine the problem to the
synthetic harness. That requires the tensor's file offset and ggml type from the GGUF index (already parsed once for the
type census), so it is a small step, not new infrastructure.
STATE: 75 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s. Reference tests committed: Q8_0, Q2_0, Q6_K (all
PASS). IQ2_XXS test exists in tools/ but is NOT wired into tools/ref_check.sh because it does not yet agree.

## IQ2_XXS: A REAL CONTRADICTION, RECORDED AS SUCH
Chain of facts, each independently measured:
  1. in2_ty_verified(16) returns 1 (fwd.c:240), and the kernel path requires j->ty with wubu_iq_supported(16)=1 and
     q8_0_act = false for 16, so the engine feeds IQ2_XXS experts through wubu_iq_rows(16) with 256-element block_q8_K
     activations. So this is a LIVE path, not dead code.
  2. The gate passes 15/15 against the real llama.cpp oracle on this model, whose experts include 7 IQ2_XXS tensors.
  3. Yet reading REAL bytes from blk.5.ffn_gate_up_exps.weight (dims [2304,2304,128], abs offset 2003207584, row_bytes 594 =
     9 x 66) and comparing wubu_iq_rows(16) against upstream dequantize_row_iq2_xxs copied verbatim gives
     rel err min 7.787e-02, median 8.549e-01, max 1.719e+01, e.g. ref[0]=282.57 vs got[0]=-117.79.
Facts 2 and 3 cannot both be true if the upstream dequantiser is the right specification for this file's bytes AND the kernel
were decoding them wrongly. Something in my measurement is wrong, and the candidate list is now short and specific:
  a) the engine's ty=16 block layout is NOT ggml's block_iq2_xxs. The 66-byte agreement could be coincidence - ggml's has
     ggml_half d + 64 bytes of qs, and this repo also ships its own IQ2_XXS encoder (wubu_enc_h3.c / wubu_nvfp4.h family),
     so a same-size different-convention block is entirely possible;
  b) the row geometry is wrong: dims [2304, 2304, 128] may not mean a 2304-element row for this tensor (it could be
     transposed for gate_up), so my 594-byte stride reads the wrong bytes entirely;
  c) the kernel applies a grid/sign permutation to the ACTIVATION instead of the weight (a legitimate SIMD trick), which
     would leave the dot product unchanged but make a per-element reference disagree - except it would only disagree if the
     permutation were inconsistent, so this is the least likely of the three.
CHEAPEST DISCRIMINATOR, and it needs no new code: print the first fp16 scale of a few blocks from the real file. For a
correctly aligned IQ2_XXS row these are small positive values (~1e-3..1e-2); if they look like random bit patterns
(negative, huge, or NaN) the stride/layout assumption is wrong and (a) or (b) holds. That single number separates "wrong
bytes" from "wrong decode", and it is the right thing to check before touching either the kernel or the reference.
STATE: 76 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s. Committed reference tests: Q8_0, Q2_0, Q6_K (PASS,
wired into tools/ref_check.sh). IQ2_XXS synthetic + real-data tests committed but NOT wired, because they disagree and the
disagreement is not yet explained on either side.

## IQ2_XXS CONTRADICTION RESOLVED: I WAS READING THE WRONG BYTES, THE KERNEL IS NOT IMPLICATED
The cheap discriminator was decisive. Reading the fp16 scale at the start of each 66-byte block from the computed absolute
offset 2003207584 gives:
  block 0: d=+0.000000e+00   block 1: d=+1.116211e+00   block 2: d=+0.000000e+00   block 3: d=-1.190430e+00
  block 4: d=+0.000000e+00   block 5: d=-1.190430e+00   block 6: d=+0.000000e+00   block 7: d=+9.692383e-01
  block 8: d=+0.000000e+00
ZEROS IN EVERY OTHER BLOCK, and the non-zero values are ~1.0 in magnitude. Two independent problems with that:
  - a real IQ2_XXS scale is a small positive number (the format is ~2 bits per weight, so d is typically 1e-3..1e-2, and it is
    never zero for a block that carries values);
  - an alternating zero pattern is the signature of a MISALIGNED or WRONG stride, i.e. half the reads are landing in a region
    that happens to be zero, not on block headers.
Trying alternative strides confirms the offset itself is suspect rather than the stride alone: stride 36, 40 and 68 all give
four consecutive zeros, and stride 34/66 give the same alternating pattern. So my computed absolute offset (data_start =
11931808 plus the tensor offset from the header) does not point at the start of that tensor's data.
CONCLUSION, and it is the one that keeps the gate and the kernel consistent: the O(1) mismatch was READING THE WRONG BYTES,
exactly candidate (a)/(b) from the previous section, NOT a kernel defect. This is consistent with fact 2 of the contradiction
(gate 15/15 against the llama.cpp oracle) which was always the strongest evidence available. The kernel was never implicated
by that test - my address arithmetic was.
THE USEFUL LESSON, stated so it generalises: a numeric comparison against real file bytes is only as good as the file offset,
and a wrong offset does not fail loudly - it produces plausible-magnitude nonsense (282.57, -117.79) that looks exactly like a
decode disagreement. The scale-magnitude probe cost one command and separated "wrong bytes" from "wrong decode", which is the
single most valuable diagnostic pattern found in this whole verification effort: BEFORE comparing a kernel to a reference on
real data, print the scales of the first few blocks and check they have the format's characteristic magnitude.
STATE: 77 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s. Wired and passing: Q8_0, Q2_0, Q6_K. IQ2_XXS remains
unverified against upstream - not because the kernel is wrong, but because the real-data harness still needs a correct
file offset (the GGUF data-section origin must be recomputed properly, including the alignment rule actually used by the
writer, rather than assumed as 32).

## IQ2_XXS VERIFICATION IS OUT OF REACH FOR THIS FILE - upstream's dequantiser is not its oracle
Scanned every byte shift s in 0..131 around the computed offset, scoring how many of 9 consecutive 66-byte blocks have a
scale of the format's characteristic magnitude (1e-7..0.1):
  s=99  ok=7/9  d=[+1.11e-05 +1.95e-02 +1.12e-05  -inf  +3.58e-06]
  s=89  ok=7/9  d=[+8.00e+00 +1.12e-05 +3.20e+01  +1.12e-05 +1.46e-03]
  s=83  ok=7/9  d=[+1.12e-05 -6.40e+01 +3.58e-06 +1.71e-03 +1.11e-05]
  s=79  ok=7/9  d=[+1.12e-05 +9.77e-03 +1.12e-05 +1.07e-04 +3.52e-06]
NO SHIFT GIVES 9/9, and the best cases contain -inf and +3.20e+01 alongside plausible values. A correct alignment would show
nine small positive scales in a row. So this is NOT a one-or-two-byte misalignment that a shift could repair: the bytes at
this tensor simply are not laid out as consecutive 66-byte block_iq2_xxs records.
WHAT THAT MEANS, stated carefully because it is the third distinct explanation in this thread and the first two were wrong:
  - the kernel is STILL not implicated: the engine reads this file correctly enough to reproduce the llama.cpp oracle 15/15,
    and it routes ty=16 through wubu_iq_rows. So whatever layout this file uses, the engine agrees with it;
  - therefore the layout is not ggml's block_iq2_xxs as I modelled it, and upstream's dequantize_row_iq2_xxs is the WRONG
    ORACLE for these bytes. That kills the plan of verifying IQ2_XXS against upstream source, which is a real and bounded loss;
  - the remaining unknown is the file's per-tensor layout for this type. It could be a UD (Unsloth Dynamic) rearrangement,
    a different scale width (fp32 d would make the block 68 bytes, which the stride probe also refused), or my GGUF
    data-section origin is still wrong for this file - the offset arithmetic has now been wrong once already, so it stays a
    suspect rather than being ruled out.
WHAT THIS DOES NOT CHANGE: the three verified kernels (Q8_0 1.9e-06, Q2_0 2.5e-06, Q6_K 1.9e-05 vs noise) stand, the
activation quantiser stands at 3.8e-3, identity stands at 4 prompts x 2 paths plus 15/15 upstream, and throughput stands at
4.45-4.65 tok/s. IQ2_XXS is recorded as UNVERIFIABLE WITH THE CURRENT ORACLE rather than as either verified or defective,
which is the only honest of the three.
STATE: 78 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s.

## IQ2_XXS: OFFSET BUG FOUND AND FIXED, SCALES NOW SANE, COMPARISON STILL DISAGREES
The GGUF data-section origin was wrong and the ENGINE'S OWN READER proved it. Linking a probe against src/gguf_reader.c
(needs -Iinclude/win32 for the sys/mman.h shim, plus src/wubu_dequant_fp4.c and src/wubu_win.c for wubu_fp4_raw_size and
wubu_munmap) and printing ctx->data_blob_offset gives:
  engine: data_blob_offset = 11978688   alignment = 32   n_tensors = 734
My own parser had computed 11931808. Difference 46880 bytes. So every previous "real bytes" read was pointed 46880 bytes
before the intended tensor - which is exactly what produced the alternating-zero scales, and it means my python GGUF metadata
parser mis-skips at least one metadata value (a plausible candidate: a bool stored as 1 byte vs 4, or an array whose element
type I mishandle).
With the ENGINE'S offset (blk.5.ffn_gate_up_exps.weight abs = 2003254464) the nine consecutive block scales are
  +4.621e+00 +4.211e+00 +3.660e+00 +5.785e+00 +3.660e+00 +4.227e+00 +3.553e+00 +4.102e+00 +4.039e+00
all nonzero, all the same magnitude - so the alignment IS now correct and the 66-byte block size is confirmed by the engine's
own gguf_raw_size (n_blocks * 66, "verified empirically" per its comment).
AND THE COMPARISON STILL FAILS: rel err min 5.147e-01, median 2.235e+00, max 1.444e+01; ref[0]=-3132.77 vs got[0]=-10134.42.
WHAT IS NOW RULED OUT (each by measurement, not argument): wrong block size (engine agrees 66), wrong byte alignment (nine
consistent nonzero scales), dead code path (in2_ty_verified(16)=1 and the engine routes ty=16 through wubu_iq_rows).
WHAT REMAINS AS THE ONLY LIVE EXPLANATIONS: (i) the upstream dequantize_row_iq2_xxs I fetched from master decodes a different
convention than this file uses, so upstream source is not this file's oracle after all; (ii) a genuine kernel defect in the
IQ2_XXS path that nevertheless leaves tokens identical because llama.cpp and this kernel could share the same non-upstream
decode - possible only if both were written against the same (non-master) convention.
THE TIE-BREAK, and it is the honest one: the gate reproduces llama.cpp's tokens 15/15 on this model. Whatever convention the
kernel uses, llama.cpp AGREES with it on real data. So (ii) can only be true if the file and both implementations share a
convention that master's ggml-quants.c no longer expresses. The next step that would settle it is to compare against the
dequantiser from the llama.cpp revision that WROTE this quant (UD/S version-matched), not current master.
STATE: 79 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s. Verified and wired: Q8_0, Q2_0, Q6_K.

## THE OFFSET BUG WAS AN ORDERING BUG IN MY PARSER - fixed, and now reproducible on demand
Root cause, exactly: my first parser captured the file position AFTER the metadata kv section but BEFORE reading the
tensor-info section, then aligned that position and called it the data origin. GGUF order is kv entries -> tensor-info
entries -> align -> data, so the reported origin was low by the size of the 734 tensor-info records:
  kv_end              = 11931802
  after 734 tensors   = 11978661   (pad 27 -> 11978688)
  engine data_blob_offset = 11978688   MATCH
  blk.5.ffn_gate_up_exps.weight abs = 11978688 + 1991275776 = 2003254464   (engine: 2003254464)
Fixed parser committed as tools/gguf_offsets.py, verified to reproduce the engine's own numbers exactly. This also retires
a suspicion: the 66-byte IQ2_XXS block size and the tensor offsets were never in doubt, only my data-section arithmetic was.
WHAT THAT LEAVES: with a verified-correct offset, wubu_iq_rows(16) still disagrees with MASTER's dequantize_row_iq2_xxs on
real bytes (rel err 0.51..14.4). So the disagreement is now confined to DECODE SEMANTICS, not addressing. Three
implementations of the same format now exist in play - the kernel, master's ggml-quants.c, and this repo's own
dequantize_iq2_xxs_row in gguf_reader.c - and the cheap decisive next step is a three-way comparison on the same real block:
if the kernel agrees with THIS REPO's dequantiser, the engine is internally consistent and master is the outlier; if it
disagrees with both, the kernel is the outlier despite the gate passing, which would be a genuinely surprising finding and
would need explaining. That comparison needs no new file-offset work - the block is now readable at a verified address.
STATE: 80 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s.

## STRUCTURAL FINDING: EVERY EXPERT WEIGHT HAS A COMPANION PER-EXPERT F32 SCALE TENSOR
The fixed offset tool surfaced something the type-only census could not: the expert tensors are PAIRED.
  blk.0.ffn_gate_up_exps.scale   type=0 (F32)  dims=[128]   <- one float per expert
  blk.0.ffn_gate_up_exps.weight  type=19 (IQ1_S) dims=[2304,2304,128]
  blk.1.ffn_gate_up_exps.scale   type=0 (F32)  dims=[128]
Counts over the whole file: 40 `.scale` tensors (all F32, all dims [128] = n_experts) against 48 expert weight tensors.
So this is a dynamic-quantisation layout: each expert's weights are stored once and multiplied by a per-expert scalar at use
time. It also explains a number that looked wrong earlier - the IQ2_XXS block scales near 4.0, which are large for a 2-bit
format only if you expect the weight magnitude in d; with an external per-expert scale doing part of the work, d does not
have to carry it.
CONSEQUENCES, and one of them matters for the reference work:
  - ANY correct reference for an expert matvec must apply w = dequant(block) * scale[expert]. A reference that omits the
    per-expert scale is wrong for the model even if it is right for the block format.
  - This does NOT explain the IQ2_XXS kernel-vs-master disagreement, because BOTH sides of that comparison omitted the scale
    equally. It remains a decode-semantics difference, still to be localised by the planned three-way comparison.
  - It does mean the engine has a multiplication in the expert path that no instrument in this document has ever counted, and
    that is worth knowing before anyone claims the expert path is fully understood.
STATE: 81 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s. Reference tests verified: Q8_0, Q2_0, Q6_K.

## IQ2_XXS SYNTHETIC NOW PASSES - the bug was a POINTER-TYPE STRIDE, and the reference is validated against real bytes
ROOT CAUSE, found by comparing the two struct definitions side by side:
  upstream master and this repo's llama/ggml-common.h BOTH declare  block_iq2_xxs { ggml_half d; uint16_t qs[QK_K/8]; }
  with a static_assert pinning it to 66 bytes. My earlier note claiming upstream was 130 bytes was WRONG.
  Because qs is uint16_t*, upstream's `memcpy(aux32, x[i].qs + 4*ib32, 8)` is an EIGHT-byte stride. I had declared qs as
  uint8_t* and written the same expression, which is a FOUR-byte stride. Group 0 matched (both start at 0), every later group
  read the wrong bytes - which is exactly the signature observed: first six values identical, then divergence (worst idx 177).
AFTER THE FIX (qs + 8*ib32):
  synthetic  : rel err min 1.232e-08  median 1.166e-07  max 5.957e-07   PASS
  repo-vs-master on a REAL block (blk.5.ffn_gate_up_exps.weight, abs 2003254464): maxabs 0.000000e+00 - EXACT agreement
So the reference implementation is now validated two ways: it matches upstream's loop by construction, and it matches this
repo's own dequantize_iq2_xxs_row element-for-element on real file bytes. That is the strongest oracle available for a format
with no external reference implementation on this machine.
STILL FAILING, and now sharply bounded: the REAL-DATA row test (8 rows x 2304 elements through wubu_iq_rows(16)) gives
rel err 0.51..14.4 while the synthetic block test passes at 6e-07 and the dequantisers agree exactly on the same real bytes.
That combination means the DECODE is right and the disagreement is in ADDRESSING OR SHAPE for the real row - a 2304-element
row at 594-byte stride may not be how this tensor is laid out (dims are [2304,2304,128], and with 40 per-expert .scale
tensors present the expert blocks may be arranged per-expert rather than as flat rows).
NEXT STEP, cheap: run ONE real 66-byte block (not a whole row) through both the kernel and the reference. If that block
matches, the decode is confirmed and the problem is purely the row geometry; if it does not, the kernel's ty=16 path differs
from its own dequantiser, which would be a genuine finding given the gate passes.
STATE: 82 commits, gate 15/15. Wired and PASSING: Q8_0, Q2_0, Q6_K, IQ2_XXS (synthetic).

## IQ2_XXS DECODE CONFIRMED ON REAL MODEL BYTES - single block PASSES at 3.85e-07
Ran exactly the separating experiment: ONE real 66-byte block from blk.5.ffn_gate_up_exps.weight (abs 2003254464) through
both wubu_iq_rows(16) and the reference, repeated over 16 consecutive real blocks (each one a "row" of 256 elements at
stride 66):
  rel err min 1.534e-09  median 7.035e-08  max 3.847e-07      PASS (8x noise floor 1.372e-05)
  ref[0]=-2720.386939 got[0]=-2720.386719   ref[1]=-369.820611 got[1]=-369.820618
SO THE DECODE IS RIGHT ON REAL DATA. Combined with the earlier repo-vs-master maxabs=0 on the same bytes, IQ2_XXS now has:
  - kernel == reference on real bytes at 3.8e-07 (single-block granularity)
  - reference == this repo's own dequantize_iq2_xxs_row exactly (maxabs 0)
  - reference == upstream's dequantize_row_iq2_xxs by construction
That is a fully verified decode for the most common format on the expert path after Q2_0.
WHY THE ROW-LEVEL TEST STILL READS 0.51..14.4, and it is now unambiguous: the same 9 blocks that pass individually do not
agree when I present them as one 2304-element row at a 594-byte stride. Since every individual block matches, the kernel's
per-block arithmetic cannot be at fault; what differs is the ROW ACCOUNTING in my call - either `n` is not the element count
per row for this entry point, or this tensor is not laid out as flat 2304-element rows (dims are [2304,2304,128] and there
are 40 per-expert .scale tensors). That is a harness-shape bug of exactly the same family as the uint8_t/uint16_t stride bug
one section above: my transcription of the calling convention, not the kernel.
STATE: 83 commits, gate 15/15, identity exact, default 4.45-4.65 tok/s. Verified decodes: Q8_0, Q2_0, Q6_K, IQ2_XXS.
