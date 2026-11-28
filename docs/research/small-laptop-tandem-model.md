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
