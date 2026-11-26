# A — Attention kernels and prefill throughput for long context

**Question.** A pure-C, scalar inference engine for Inkling-2-Mini (48 layers, d=2304,
72 attention visits = 8 full-attention + 64 sliding-window/1024, NH=24, NKV=8, HD=128,
IQ2_XXS GGUF at 7.9 GB) measures 1.66 GMAC/s prefill and ~0.11 s/token decode at f32 KV on a
Ryzen 5 8645HS + RTX 4050 Laptop 6 GB. The owner concludes 262k prefill = 8.0 h and that
"long context is unreachable on this box".

**What this report does.** It measures the RTX 4050 directly (id 18), places that measurement
against the 2024–2026 kernel literature (ids 1–5, 17), and reconstructs the 8-hour figure from
the engine's own numbers to identify which mechanism produces it. Every number below is tagged
**[measured]** or **[claimed]**.

---

## 0. Bottom line

| path | 262k-token prefill | basis |
|---|---|---|
| as reported (decode latency × N) | **8.0 h** | 262,144 × 0.11 s **[claimed, arithmetic on id 18-adjacent engine constants]** |
| same engine, FLOP-honest | **~423 h** | 5.06 PFLOP ÷ 3.32 GFLOP/s measured scalar rate **[derived]** |
| GPU-kernelised, this exact 4050 | **~6 min** (380 s) | measured kernel rates, ids 18 + geometry **[derived from measurement]** |
| factor vs the 8 h claim | **~76×** | |
| factor vs the FLOP-honest scalar path | **~4,000×** | 15.9 TFLOPS ÷ 3.32 GFLOP/s = 4,801× on attention |

The single most decisive number: **causal GQA attention (24 q-heads / 8 kv-heads, D=128) runs at
15.84–16.05 TFLOP/s on this laptop's 4050 at S=16k–65k, and cuBLAS fp16 GEMM at 23.7–24.3
TFLOP/s** (id 18, measured today on this machine). That is **4,800–7,200× the engine's measured
3.32 GFLOP/s**, and it is the whole story. Note this also corrects the brief: the 4050's fp16
tensor peak is 46.7 TFLOPS dense (23.35 with fp32 accumulate) per id 6, not "~15 TFLOPS" — the
hardware was understated, not overstated.

---

## 1. What was actually measured, and where

I installed torch 2.14.1+cu130 and ran two benchmarks on the RTX 4050 Laptop in this machine
(sm_89, 20 SMs, 6.00 GiB, driver 610.91). Raw numbers, id 18:

```
cuBLAS fp16      2048^3            20.91 TFLOP/s
                 4096^3            21.31
                 8192^3            24.01
                 16384x4096x4096   24.32
  prefill shape  M=32768,K=2304,N=9216   23.65   (58.8 ms)
  bf16 8192^3                      24.16
SDPA causal, Hq=24 Hkv=8 D=128, mem-efficient kernel:
                 S=4096   15.40 TFLOP/s (  6.69 ms)
                 S=16384  15.94         (103.47 ms)
                 S=32768  16.05         (411.13 ms)
                 S=65536  15.84         (1666.39 ms)
int8 torch._int_mm 8192^3          20.85 TOP/s
PCIe pinned H2D                    13.42 GB/s
```

Two facts about the software stack matter as much as the numbers:

1. **PyTorch's cu130 wheel ships no sm_89 flash kernel.** Forcing
   `SDPBackend.FLASH_ATTENTION` raises `RuntimeError: No available kernel. Aborting execution.`
   with the internal warning "Torch was not compiled with flash attention". The 15.8–16.1
   TFLOP/s above is therefore the *cutlass memory-efficient* kernel, not FA2 — i.e. **a
   non-flash tiled attention already reaches 66% of cuBLAS GEMM throughput on this GPU**. FA2
   itself supports Ada upstream, but on Windows you either build it yourself (the README still
   says Windows compilation "requires more testing", id 3) or you do not get it.
2. **The achieved rate is 51% of the card's 46.69 TFLOPS fp16-16 tensor peak** (id 6). That is
   the realistic ceiling; the remaining headroom over the numbers above is a 1.2–1.4× tuning
   gain, not an order of magnitude.

---

## 2. The kernel families and their published numbers

| kernel | arch required | published throughput | measured on | status |
|---|---|---|---|---|
| FA2 (2023) | sm_80/86/89/90 | 50–73% of peak A100; 225 TFLOP/s/GPU e2e (id 5) | A100 | **[measured, authors]** |
| FA3 (2024) | **H100/H800 only**, CUDA ≥12.3 (id 3) | 740 TFLOP/s FP16 = 75%; ~1.2 PFLOP/s FP8 (id 4) | H100 SXM | **[measured, authors]** |
| FA4 (2026) | **Hopper + Blackwell** (ids 2, 3) | 1613 TFLOP/s BF16 = 71% of B100 peak; 1.3× cuDNN 9.13, 2.7× Triton (id 1) | B100 180GB SXM6 @1000 W | **[measured, authors]** |
| SageAttention3 (2026) | consumer Blackwell sm_120 | 1038 TOPS FP4 on RTX 5090, 5× the fastest FA on that card (id 17) | RTX 5090 | **[measured, authors]** |

Three things this table is telling you that the marketing does not:

- **The halo numbers are 100× the hardware class.** 1613 TFLOP/s is 67× this 4050's achieved
  24 TFLOP/s. FA4's own framing explains why the gap is not closable by porting: on Blackwell
  tensor throughput doubled while shared-memory bandwidth and the exponential units did not, so
  "shared memory traffic and exponential operations now dominate execution time, exceeding MMA
  compute by 25–60%" (id 1). On a 4050 the ratio is even less favourable — 20 SMs, 12 MB L2,
  192 GB/s — so the FA4-style optimisations (TMEM, 2-CTA MMA, 128×128 tiles) have no substrate
  to run on.
- **FA3 does not work on Ada, and not on consumer Blackwell either.** Upstream states
  "Requirements: H100 / H800 GPU" (id 3); the flash-attn-4 wheel targets "Hopper and Blackwell"
  with no sm_89/sm_120 target (id 2); practitioners hit `no kernel image is available` on an
  RTX 5090 (id 10); and the SageAttention3 authors state flatly that "FlashAttention-3 can only
  run on Hopper GPUs, so FlashAttention-2 is already the fastest on RTX5090" (id 17). **For sm_89
  the ceiling is FA2-class, and in practice the cutlass mem-efficient kernel.**
- **The one kernel family that does beat FA2 on a consumer card is FP4 on Blackwell**
  (SageAttention3, 5× on a 5090, id 17). Ada has FP8 tensor cores (93.4 TFLOPS, id 6) but no
  FP4, and Ada consumer FP8 is blocked in most libraries at the framework level. There is no
  comparable 2026 result for sm_89; anyone claiming one should be asked for the run.

---

## 3. Reconstructing the 8 hours: it is a bandwidth number wearing a compute costume

The engine's 0.11 s/token is a **decode** measurement, and it is internally consistent as a
pure bandwidth figure. At 262k tokens with f32 KV, one step must read

```
8 full-attn layers × 2 (K,V) × 8 kv-heads × 128 dim × 4 B = 65,536 B/token
262,144 tokens × 65,536 B = 17.18 GB
```

17.18 GB ÷ 0.11 s = **156 GB/s = 81% of the card's 192 GB/s** (id 6). Add the 64 sliding-window
visits (64 × 1024 × 8,192 B = 0.54 GB) and it is 161 GB/s = 84% of peak. **The decode step is
already at the memory-bandwidth floor.** That is a real achievement and it is exactly why the
number must not be reused for prefill.

Now apply it to prefill as the report does — one forward pass per token:

```
Σ over i=1..262144 of (i × 65,536 B) ≈ 0.5 · N² · 65,536 = 2.25 PB of KV traffic
  at 192 GB/s (KV in VRAM)       → 11,724 s =  3.3 h
  at  13.4 GB/s (KV over PCIe)   → 167,660 s = 46.6 h   [13.42 GB/s measured, id 18]
  the stated 8.0 h implies         ≈ 78 GB/s effective
```

So the 8 hours is **O(N²) memory traffic generated by serialising prefill into N decode steps**,
with the KV cache partly resident and partly spilled (17.18 GB cannot fit in 6 GB of VRAM — this
is why it lands between the two bounds rather than at either). No tensor core is involved in
that number and no tensor core is needed to beat it. **The dominant fix is batching queries, not
a better attention kernel**: with 8,192-token chunks the same traffic becomes
`0.5·N²·65,536/8192 = 275 GB → 1.4 s` at 192 GB/s. That is an **8,192× traffic reduction from
blocking alone**, before any kernel work.

This also exposes an inconsistency in the brief: 1.66 GMAC/s = 3.32 GFLOP/s cannot produce 262k
prefill in 8 h, because the job is ~5.06 PFLOP of work (below). The stated 8 h corresponds to
only 96 TFLOP at that rate. The two numbers describe different things — one is a decode
bandwidth measurement, the other an extrapolation from it — and neither is a prefill
measurement. **The true scalar path is ~423 h; the 8 h figure understates it by ~50×.**

---

## 4. What 262k actually costs on this 4050

Geometry: attention FLOPs = `2·(N²/2)·HD·NH·2` per full-attention layer (causal halves each of
the two matmuls), `2·(N·W)·HD·NH·2·0.5` per sliding-window visit, linear FLOPs `2·A·N` with
A = 3e9 active parameters. Rates: 15.9 TFLOP/s attention [id 18, measured], 10 TFLOP/s for
quantised linear work (IQ2 must be dequantised or MMQ'd; using 10 of the measured 23.7 TFLOP/s
as the dequant-taxed figure — the tax itself is evidenced by id 15, where IQ2_XXS pp512 on CPU
is 43.5 t/s vs 137.8 t/s for Q8_0, a 3.2× penalty for 2-bit).

| prompt | attn PFLOP | SWA PFLOP | linear PFLOP | time | KV @6-bit |
|---:|---:|---:|---:|---:|---:|
| 32k | 0.05 | 0.013 | 0.20 | 24 s | 0.40 GB |
| 128k | 0.84 | 0.053 | 0.79 | **137 s (2.3 min)** | 1.61 GB |
| 262k | 3.38 | 0.106 | 1.57 | **380 s (6.3 min)** | 3.22 GB |
| 512k | 13.51 | 0.211 | 3.15 | 1,185 s (19.8 min) | 6.44 GB |
| 1M | 54.04 | 0.422 | 6.29 | **4,070 s (68 min)** | 12.88 GB |

Four observations:

1. **8 of 72 attention visits are quadratic; 64 are linear in N.** The sliding-window layers
   contribute 0.106 PFLOP at 262k — **3% of the attention cost**. This model is already
   architecturally cheap at long context. At 1M, SWA is 0.8% of attention FLOPs.
2. **Prefill goes compute-bound as soon as queries are blocked.** Blocked, 262k is 380 s of
   tensor-core work (67% of it attention in the 8 full-attention layers: 212 s of 380 s) versus
   1.4 s of KV traffic. The ratio inverts completely from the serialised case.
3. **The numbers are derivations from a measured kernel rate, not an end-to-end run.** I did not
   run Inkling-2-Mini itself; no public end-to-end 262k prefill on a 6 GB laptop GPU exists to
   compare against (§8). The 15.9 TFLOP/s is measured; the 380 s inherits an assumption about
   active parameter count and dequant efficiency. Treat 5–10 min as the band, not 380 s as a
   fact.
4. **The 4050's tensor cores are the mechanism, at 4,801×.** Flash tiling contributes almost
   nothing to that factor — the non-flash cutlass kernel already hits 66% of GEMM throughput
   (id 18). Flash tiling's job is to *fit* the problem in 6 GB by never materialising the N×N
   score matrix; FA1→FA2's 2× (id 5) is the realistic size of the work-partitioning gain on top.

---

## 5. Wall 1, re-examined with the same arithmetic

The owner's KV model is right: `8 layers × 2 × 8 kv-heads × 128 dim × N × 0.75 B` gives 12.88 GB
at 1M, matching the stated 12.16 GB. But the wall is drawn in the wrong place:

- **At 262k the KV is 3.22 GB.** It fits in 6 GB of VRAM with 2.8 GB left over. The *model* need
  not be resident: streaming 7.9 GB of IQ2 weights over the measured 13.42 GB/s PCIe costs 0.59 s
  per full pass, and with 8k chunks you make 32 passes = **19 s of PCIe traffic** against 380 s
  of compute. **PCIe is not the bottleneck for prefill; it is for decode.** This is precisely the
  "layer-wise prefill" trick KTransformers reports at 7–9× for 16K–64K contexts (id 9, claimed).
- **At 1M the wall is real but it is not prefill.** 12.88 GB of KV cannot live in 6 GB of VRAM,
  so decode must read it every token: 12.88 GB ÷ 192 GB/s = 67 ms/token if resident, ÷13.42 GB/s
  = 0.96 s/token if spilled. **Decode at 1M is bandwidth-doomed on this box at ~1–15 tok/s,
  regardless of how fast prefill is.** Prefill at 1M is ~68 min, which is slow but routine.
- The compression lever belongs to report B (4-bit KV → 6.4 GB at 1M), not here. What matters
  here: **the memory wall is a decode-side wall, and it is a bandwidth wall, so it is attacked by
  KV compression and expert streaming, not by attention kernels.**

---

## 6. Chunked prefill and prefill/decode disaggregation, with real throughput

- **Chunked prefill** is what makes the above runnable: it bounds peak activation memory and lets
  one weight read serve a whole block of queries. vLLM and SGLang both ship it; SGLang's
  production field report runs `chunked_prefill_size=2048` and measures **128K cold prefill at
  6,853 tok/s on 8× RTX PRO 6000** (id 7, measured). Note the counter-intuitive part of that
  same report: **larger chunks were slower** — 4096 gave 6,301 tok/s and 1024 gave 6,012 tok/s
  against 6,853 at 2048 — because at that scale the board was communication-bound (87% GPU util
  at 220 W of 600 W), not compute-bound. Chunk size is a tuning knob against a specific
  bottleneck, not a monotone win.
- **SGLang's 1M-token table is the only clean public 1M prefill curve I found** (id 8, measured
  by authors, 32× H20, Qwen3-235B-A22B-FP8): 128K 10.54 s → 256K 32.68 s → 512K 114.33 s → **1M
  420.91 s**. The growth is super-quadratic (3.1×, 3.5×, 3.7× per doubling) because at 1M the
  attention term dominates and the pipeline is bubble-limited; chunked pipeline parallelism
  (PP8 TP4) is what makes it 7 minutes instead of an hour.
- **P/D disaggregation** exists to stop prefill and decode from interfering. DistServe reports
  **7.4× more requests or 12.6× tighter SLO** within latency for >90% of requests (id 13,
  measured). vLLM's 2026 production run gets **954K tokens/s per GPU** on one GB300 NVL72 rack
  with P/D + Wide-EP, 7.9× its own aggregated single-node result (id 14, measured).
- **None of this transfers to a single 6 GB laptop GPU.** Disaggregation needs two resource pools
  and a fast interconnect; the SGLang and vLLM numbers are per-GPU aggregates over 8–72 GPUs with
  NVLink-class fabric. On one device the relevant pieces are chunked prefill (adoptable, pure
  scheduling) and nothing else from this section.

---

## 7. CUDA graphs and launch overhead

Measured on H100: a dummy kernel costs **2.1 µs launched on a stream and 1.3 µs under a CUDA
graph** (id 11); NVIDIA measured **up to 1.2×** end-to-end for Llama-7B from adding CUDA graphs
to llama.cpp at batch 1, and expects ~10% more from CPU-side work (id 12).

That 1.2× is the ceiling for a *decode* step of a 1B model, where a forward pass is ~100 kernels
of a few microseconds each (id 11). **For prefill it is noise**: at 262k the kernels are hundreds
of milliseconds to seconds long (§4), so 2.1 µs of launch overhead is ~1e-5 of the runtime.
CUDA graphs are worth adopting for the decode loop, and worth ignoring for prefill. The 8-hour
figure contains no launch-overhead component at all.

---

## 8. Real measurements on consumer hardware, and what is missing

What exists, measured by third parties, on hardware in this class:

- **llama.cpp on an actual RTX 4050 Laptop 6 GB**, 35B-A3B IQ-quant at 65,536-token context:
  **PP 208–289 tok/s**, decode 24.7–30.2 tok/s, TTFT 2.0–3.0 s, 15.4–17.5 GB host footprint
  (id 16). Those PP numbers are for short prompts and will fall with context; they are the right
  order of magnitude for "a real GPU path on this exact card". At ~6 GFLOP/token for a 3B-active
  MoE they imply ~1.25–1.73 TFLOP/s of useful work, i.e. **~380–520× the engine's 3.32 GFLOP/s**,
  on a strictly bigger model.
- **8× RTX PRO 6000 (SM120, PCIe, no NVLink)**: 128K prefill 6,853 tok/s, and a 112K prefix-cache
  hit served in 0.34 s = ~325K tok/s, 39× (id 7).
- **SageAttention3 on RTX 5090**: 1038 TOPS FP4 attention, 5× FA2 (id 17).

What does **not** exist, stated plainly:

- **No primary source measures an end-to-end 262k–1M token prefill on a 6 GB laptop GPU.** The
  closest is id 16 (65k context, 35B-A3B, short-prompt PP) and the SGLang 1M curve on 32 H20s
  (id 8). Everything in §4 is therefore a derivation from a measured kernel rate plus the
  model's geometry — labelled as such, not presented as a run.
- **No published FlashAttention-2/3/4 measurement on sm_89 at all.** FA2 supports Ada, but nobody
  in the 2024–2026 literature benchmarks it there; the FA3/FA4 papers are H100/B100 only (ids 1,
  4). My id 18 measurement is the substitute, and it is a cutlass mem-efficient kernel, not FA2.
- **No measured IQ2_XXS GEMM throughput on this GPU.** The dequant tax is bounded from CPU
  evidence (id 15) and from the fact that cuBLAS cannot consume 2-bit weights at all. The 10
  TFLOP/s figure in §4 is an assumption with a stated basis, not a measurement.

A companion sweep in this repo (`docs/research/E-baselines.md`, ledger ids 161–200) collects
llama.cpp/ik_llama/KTransformers prefill numbers on 6–8 GB laptop cards including Inkling-2-Mini
itself; where that report and this one disagree, its numbers are end-to-end runs and should win.

---

## 9. RETROFIT VERDICT

For each mechanism, whether it can be bolted onto a frozen checkpoint + existing C engine.

| mechanism | verdict | why |
|---|---|---|
| **Block queries into prefill chunks (batch the GEMMs)** | **adoptable-post-hoc** | Pure scheduling + a GEMM that takes M>1; divides KV traffic by the chunk size (8,192× at 8k chunks) and is what turns 8 h of traffic into 1.4 s. |
| **Tensor-core fp16/bf16 GEMM path** | **adoptable-post-hoc** | Arithmetic is unchanged by using MMA instead of scalar FMA; measured 23.7–24.3 TFLOP/s here vs 3.32 GFLOP/s, and it is the 7,200× mechanism. |
| **Flash-style tiled attention (never materialise N×N)** | **adoptable-post-hoc** | Exact, no approximation; needed to fit 262k in 6 GB, but worth only ~1.2–2× over an already-tiled kernel (id 5) — it is a memory enabler, not the speedup. |
| **GQA KV broadcast (24 q-heads over 8 kv-heads)** | **adoptable-post-hoc** | Already in the checkpoint; it is a 3× cut in KV bytes read and a kernel-side indexing change, no weights touched. |
| **Sliding-window KV bounding (64 of 72 visits)** | **adoptable-post-hoc** | Already in the architecture; implementing it as a real 1024-token ring buffer instead of a full cache is a bookkeeping change worth 3.2 GB → 0.54 GB at 262k. |
| **Chunked prefill / paged KV** | **adoptable-post-hoc** | Allocation and scheduling only; it is the precondition for streaming IQ2 weights over PCIe at 13.4 GB/s without re-reading them per token. |
| **CUDA graphs** | **adoptable-post-hoc** (decode only) | 2.1 µs → 1.3 µs per launch and ~1.2× e2e at batch-1 decode (ids 11, 12); irrelevant to prefill where kernels are ~1e8× longer. |
| **P/D disaggregation** | **not-applicable** | Needs two GPU pools plus a fast fabric; on one 6 GB device there is nothing to disaggregate onto. |
| **FA3/FA4-class kernels (warp specialisation, TMA, TMEM, 2-CTA MMA)** | **not-applicable** | Require Hopper/Blackwell datacenter features that sm_89 does not have; upstream states H100/H800 for FA3 and Hopper+Blackwell for FA4 (ids 2, 3), and they fail to launch on consumer Blackwell too (id 10). |
| **FP8/FP4 attention (SageAttention-class)** | **approximation-must-be-measured** | Ada has FP8 tensor cores on paper (93.4 TFLOPS, id 6) but consumer FP8 attention is library-blocked and there is no sm_89 result; the one card where it works (5090, 5×) is a different architecture (id 17). Measure perplexity and long-context recall before believing any speedup. |
| **IQ2_XXS blocked-quant GEMM (MMQ) rather than dequant-to-fp16** | **approximation-must-be-measured** | The quantisation is already in the checkpoint so no retraining is implied, but the dequant tax is real — 3.2× on CPU for IQ2_XXS vs Q8_0 (id 15) — and nobody has published its cost on this GPU; measure it against the 23.7 TFLOP/s fp16 ceiling. |
| **Speculative decoding / sparse attention / KV compression** | **needs-retraining** or out of scope | Sparse-attention branches and learned selectors need training (see report C); KV compression is inference-only but belongs to report B's verdict, not to prefill throughput. |
| **Anything that makes 1M decode fast on 6 GB** | **not-applicable** | 12.88 GB of KV must be read every token; at 192 GB/s that is 67 ms/token floor and at PCIe 0.96 s/token. No kernel fixes a bandwidth wall — only KV compression or more VRAM does. |

---

## Sources

Ids match `docs/research/LEDGER.md`, range 1–40.

1. **https://arxiv.org/html/2603.05451v1** — FlashAttention-4 (arXiv:2603.05451v1, 2026-03-05).
   Established: 1613 TFLOP/s BF16 = 71% of B100 peak; 1.3× cuDNN 9.13, 2.7× Triton on B200;
   benched on B100 180GB SXM6 only; the asymmetric-scaling argument that non-MMA units now
   dominate 25–60% of runtime. **[measured by authors, datacenter]**
2. **https://pypi.org/project/flash-attn-4/** — flash-attn-4 4.0.0b41 (2026-11-25). Established:
   the shipped implementation targets "Hopper and Blackwell GPUs"; no sm_89 target.
3. **https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md** — upstream
   README. Established: FA3 "Requirements: H100 / H800 GPU, CUDA >= 12.3"; FA4 "optimized for
   Hopper and Blackwell (H100, B200)"; FA2 Windows compilation caveat.
4. **https://arxiv.org/abs/2407.08608** — FlashAttention-3. Established: 740 TFLOP/s FP16 (75%),
   ~1.2 PFLOP/s FP8 on H100; FA2 was 35% there. **[measured by authors]**
5. **https://arxiv.org/abs/2307.08691** — FlashAttention-2. Established: 50–73% of A100 peak,
   225 TFLOP/s/GPU e2e, FA1 was 25–40% — the size of the tiling/work-partitioning gain.
   **[measured by authors]**
6. **https://hmc-tech.com/gpus/nvidia-geforce-rtx-4050-laptop-100w** — RTX 4050 Laptop specs.
   Established: 20 SMs, 80 tensor cores, 192 GB/s, 12 MB L2, CUDA 8.9; 46.69 TFLOPS FP16-16
   tensor, 23.35 FP16-32, 93.39 FP8, 93.39 INT8. **[theoretical vendor figures]**
7. **https://github.com/sgl-project/sglang/issues/40877** — SGLang field report, 8× RTX PRO 6000
   SM120, 2026-09-23. Established: 128K cold prefill 6,853 tok/s; chunk 2048 > 4096 > 1024;
   112K prefix-cache hit 0.34 s (39×); 87% GPU util at 220/600 W → comms-bound. **[measured]**
8. **https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/** — SGLang chunked pipeline PP blog.
   Established: Table 1 TTFT 128K 10.54 s / 256K 32.68 s / 512K 114.33 s / 1M 420.91 s on 32× H20;
   PP4 TP8 = 3.31× prefill throughput vs TP8. **[measured by authors]**
9. **https://ktransformers.net/talks/gosim-2026/ktransformers-gosim-2026.html** — KTransformers
   GOSIM 2026. Established: layer-wise prefill streams weights per layer to GPU and saturates
   PCIe 5.0 for "7–9× speedup at 16K–64K context"; >4.5× prefill vs llama.cpp. **[vendor claim]**
10. **https://github.com/Edge0-AI/Audio8_TTS/issues/8** — practitioner report, 2026-08-01.
    Established: FA3 raises "no kernel image is available for execution on the device" on sm_120
    (RTX 5090); replaced with FlashInfer/SDPA. **[measured/observed failure]**
11. **https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles** — megakernel blog.
    Established: launch cost 2.1 µs on a stream vs 1.3 µs under CUDA graphs on H100; ~100 kernels
    per forward; 78% vs ≤50% of bandwidth at batch 1. **[measured]**
12. **https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs** —
    NVIDIA, 2024-08-07. Established: CUDA graphs give "up to 1.2×" for Llama-7B on H100, default
    at bs=1. **[measured by NVIDIA]**
13. **https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin** — DistServe, OSDI'24.
    Established: P/D disaggregation = 7.4× more requests or 12.6× tighter SLO at >90% SLO
    attainment. **[measured in paper]**
14. **https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep** — vLLM, 2026-10-23.
    Established: 954K tokens/s per GPU on GB300 NVL72 with P/D + Wide-EP, 7.9× its aggregated
    single-node result; DEP2 instances preferred for prefill. **[measured by authors]**
15. **https://github.com/ikawrakow/ik_llama.cpp/wiki/Jan-2025:-prompt-processing-performance-comparison**
    — ik_llama.cpp, 2025-01-21. Established: LLaMA-3.1-8B pp512 on 7950X, IQ2_XXS 43.54 → 154.17
    t/s (3.54×), Q6_K 3.29×, Q4_0 1.78×; quant-specific kernels alone move prefill 3.5×, and
    IQ2_XXS is ~3.2× slower than Q8_0 in the same engine. **[measured]**
16. **https://llm-bench.io/hardware/rtx-4050-laptop** — community benchmark page for this exact
    GPU. Established: llama.cpp, 35B-A3B IQ-quant, 65,536-token context, PP 208–289 tok/s,
    decode 24.7–30.2 tok/s, TTFT 2.0–3.0 s, 15.4–17.5 GB host RAM. **[measured by third parties]**
17. **https://arxiv.org/html/2505.11594** — SageAttention3 (arXiv:2505.11594v3). Established:
    1038 TOPS FP4 attention on RTX 5090, 5× the fastest FlashAttention there; explicit statement
    that FA3 is Hopper-only so FA2 is fastest on RTX 5090. **[measured by authors]**
18. **Local measurement, this machine** (RTX 4050 Laptop 6 GB, sm_89, 20 SMs, driver 610.91,
    torch 2.14.1+cu130; scripts `bench_gpu.py`, `bench_attn.py` in
    `C:\Users\eman5\AppData\Local\hermes\cache\scratch\`) — Established: cuBLAS fp16 24.01
    TFLOP/s @8192³ and 23.65 TFLOP/s at the model's prefill GEMM shape; causal GQA attention
    (24/8 heads, D=128) 15.40–16.05 TFLOP/s from S=4k to 65k; int8 20.85 TOP/s; PCIe pinned H2D
    13.42 GB/s; **no sm_89 flash kernel in the PyTorch wheel**, so SDPA falls back to cutlass
    mem-efficient. **[measured here, today]**
