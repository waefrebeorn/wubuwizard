# Long-context hop path — what the five fronts established and what to cross next

**Order is dependency, not importance.** A pad that another pad needs goes first even if it is less urgent. Each pad has
one gate; a pad without a passing gate is not done. Sources for every external claim live in the five reports beside this
file (`A-prefill-throughput.md`, `B-kv-compression.md`, `C-sparse-linear.md`, `D-premise.md`, `E-baselines.md`) and are
registered in `LEDGER.md`.

## What the sweep changed about the two declared walls

**The prefill wall was mine, not the machine's.** This card was measured directly, on this host, this session: causal GQA
attention at the model's own shape runs **15.9 TFLOP/s** and cuBLAS fp16 GEMM at the prefill shape **23.7 TFLOP/s**,
against the engine's measured 3.3 GFLOP/s — a **4,800-7,200x** arithmetic gap. At that rate 262k prefill is ~380 s, not 8 h
(A). Real engines agree: a 6 GB laptop GPU lands 15 min-2 h for 262k, with ~45 min the defensible figure for a 12B-class
model that mostly lives in host RAM (E). So the engine is **4-30x off the real baselines, not 200x** — and the 0.11 s/token
is a per-token gemv loop, not an attention cost: KV traffic at 262k is 17.18 GB/step = 81% of the card's bandwidth, and a
per-token loop generates ~2.25 PB of O(N^2) KV traffic where an 8k-chunked pass divides it by 8,192 (A).

**The 1M memory wall is real but the 1M goal is not.** 1M at reference-exact 6 bits is 12.0 GiB of KV; post-hoc, under
8 GB, reference-exact is **not reachable** — the best post-hoc code (TurboQuant-class 3.5 bits) reaches 7.0 GiB and its
quality-neutrality is benchmark-level, not reference-exact (B). And the capability itself is imaginary at this scale:
effective context is 1K-16K for literal association and multi-needle only at 128K on 2026 frontier models; Gemma 3 12B's
effective length is **1K against a 128K claim**, and retrieval beats 128K stuffing by **38.12 points** on Mistral-Nemo-12B
at 128K (D). Utility per unit compute peaks at **~32K** and turns negative at 128K for 12B-class models (D).

## Wave 0 — decide the target (do first, costs nothing, everything downstream depends on it)

### LP-00 — Redefine the product goal as "32K fast + exact retrieval", keep 1M as a crash-free ceiling
32K is 377 MiB of KV at 6 bits, 0.19x of attention FLOPs against weight FLOPs, and the point where utility per unit compute
peaks; 1M costs 187x more time and 32x more memory to deliver a capability the frontier does not have at that scale.
*[D, A]* · **Size S** · Unlocks: stops every downstream pad from being optimised against a number that does not pay.
Gate: a written target with the 32K/1M split and the fidelity floor (reference-exact 6-bit) stated as the acceptance bar.

## Wave 1 — the two in-engine wins that need no new physics

### LP-01 — Wire `quantized_matmul_batched` into the Inkling-2 prefill path  (MEASURED: 1.56x, not the fix)

**Measured on this host before wiring, with the real quant type and the real slab shape** (`tools/pf_batch_probe.c`,
2304x2304 IQ2_XXS, batch-identical weight bytes):

    N= 32   128 gemv  1.57x   batched 10.75 GMAC/s   max|diff| = 0   finite reference
    N=128   128 gemv  1.56x   batched 11.93 GMAC/s   max|diff| = 0
    N=512   512 gemv  1.21x   batched 11.65 GMAC/s   max|diff| = 0

So batching is worth **~1.5x, flat in batch size, and bit-identical** - the weight slab is 1.31 MB and already
cache-resident, so the amortisation the doc comment promises is mostly already being collected. And the engine is
not leaving 100x on the floor here: its 0.11 s/token implies ~13 GMAC/s for the whole model, against this kernel
family's measured 11.9 GMAC/s ceiling. **The CPU is at its limit; batching is a 1.5x, not a rescue.**
It exists (`src/quantized_matmul.c:511`), supports IQ2_XXS/IQ3_XXS/IQ4_XS/Q5_K/Q6_K/Q4_K, and every other model in the
repo uses it; `wubu_inkling2_fwd.c` is the only one issuing per-token gemv at nine sites. The doc comment names the cost
being paid: *weight data read once, shared across all N tokens*.
*[local audit, A]* · **Size M** · Unlocks: amortised weight read + GEMM blocking, the prerequisite for LP-04 being worth
anything. Gate: oracle 12/12 unchanged **and** a measured prefill tok/s before/after at N=120 and N=480. Do not accept a
projected factor — the previous four speedup claims in this thread died exactly there.

### LP-02 — Decode KV once per chunk, not once per read
`in2_kv_k`/`in2_kv_v` decode on every access, so TQ prefill is O(N^2) decodes (measured excess 0.00597xN s/token). An
f32 mirror per chunk of 512 costs 300 MB and cuts decodes ~256x; the 8k-query blocking from (A) divides KV traffic by
8,192 outright.
*[local, A]* · **Size M** · Unlocks: makes the TQ tier usable for prefill at all, and bounds KV traffic independent of N.
Gate: 262k-equivalent prefill time drops superlinearly **and** oracle 12/12 holds with TQ on.

## Wave 2 — sparsity and fidelity, both inference-only

### LP-03 — Quest-style page selection on the 8 full-attention visits only
Per-page key min/max, always keep the first page plus the last 1024 keys, score with `sum_d max(q*kmin, q*kmax)`, take
top-k pages, run exact attention inside them. Parameter-free, no profiling, no index build; selector cost is 0.2% of the
dense attention it replaces, and in a scalar engine a skipped page also skips TQ dequantisation. Full-visit attention MACs
drop 128x at 262k and 512x at 1M, making the curve linear.
*[C]* · **Size M** · Unlocks: exact-looking long context without retraining, and it compounds with LP-02.
Gate: RULER-style retrieval probe at 32k/128k shows <=5 points loss **and** oracle 12/12 holds at short context.

### LP-04 — Own the fidelity number at 3.5 and 4 bits instead of borrowing one
Every 3.5-bit "neutrality" result in the literature is benchmark-level (PPL/NIAH/top-1); the only measured 4-bit
reference-exact claim is identical top-1 over 330 experiments. This engine already knows its gate is stricter: 6 bits
answers reference-exact, 4 bits does not.
*[B]* · **Size S** · Unlocks: decides whether LP-05 is a real pad or a dead end. Gate: measured 3.5/4-bit KV against the
12-token oracle and a long-context probe, with the failure mode named.

## Wave 3 — the device, and the ceiling

### LP-05 — Tensor-core quantised GEMM + flash-style attention on the 4050  (promoted: the only order-of-magnitude lever)
23.7 TFLOP/s GEMM and 15.9 TFLOP/s attention are measured on *this* card at *this* model's shapes. The specific kernels
that produce it: MMQ-class quantised gemm (**5.7-6.3x over dequant-then-cuBLAS on one card, measured**), flash tiling
(3.1x at 32k), and the 20-37x silent penalty when a quantised KV fails the CUDA FA gate and falls back to CPU. `wubu_fast_attn.c`
already exists and is unwired, like the GEMM.
*[A, E]* · **Size XL** · Unlocks: the 4,800x arithmetic gap; nothing else on this list is within two orders of it.
**Promoted to Wave 1** by LP-01's measurement: if the batched CPU kernel is 1.56x and already at the 11.9 GMAC/s
ceiling, no amount of CPU scheduling reaches 23.7 TFLOP/s - the card is the only lever with the required magnitude,
and it is a measured number on this device, not an extrapolation. Gate: prefill tok/s within 2x of the measured llama.cpp baseline for the
same model and card, with oracle parity stated explicitly (parity and batching fight each other here — pick and write down
which one wins).

### LP-06 — 3.5-bit TQ KV for the 1M ceiling
7.0 GiB at 1M including f16 norms and the 56 MiB sliding-window floor, versus 12.0 GiB at 6 bits. Only worth building if
LP-04 shows acceptable fidelity, and only worth *targeting* because LP-00 keeps 1M as a ceiling rather than a product goal.
*[B, D]* · **Size L** · Unlocks: 1M without an OOM; does not unlock 1M usefulness. Gate: 1M allocation succeeds, decode
does not regress past the 67 ms/token VRAM bandwidth floor, and the reference-exact probe at 32K is unchanged.

## What not to break

- **The 12-token oracle is the only truth.** Every pad above is gated on it. Batched GEMM changes accumulation order and
  Quest changes which keys are summed; both are permitted to change *logits* but not silently to change the *gate*. If a
  pad cannot hold 12/12, it ships behind a flag with the parity cost written down, exactly as the MoE batching did.
- **A GPU or sparse path falls back, never fails silently.** The type gate `{8,16,22,23}` and the "return -7, let the
  caller take the slow path" convention stay.
- **Do not quote a speedup measured in isolation as an end-to-end win.** The MoE batching thread produced 2.39x in a
  micro-benchmark and 0.87x in the engine; three intermediate claims were retracted. Measure at the engine, or say
  "unmeasured".
- **The engine stays unaware of the domain.** No retrieval logic inside the forward pass; retrieval belongs beside it.

## LP-05 baseline, measured — the card is not the problem, the kernels are

The existing CUDA IQ2_XXS matvec path (`wubu_cuda_moe_batch`, 8 expert slabs of 2304x2304 IQ2_XXS, tables uploaded,
`rc=0`), against the CPU ceiling measured the same way:

    GPU, 1 batched launch (8 slabs)     30.77 GMAC/s      <- 2.6x the CPU
    GPU, 8 separate launches            12.38 GMAC/s
    CPU, best batched case              11.93 GMAC/s
    this card's tensor-core GEMM        11.82 TMAC/s      <- 384x the IQ2 kernel

Caveat carried forward honestly: the price tool's reference magnitude is 4.03e-37, i.e. the activations are essentially
zero. Values do not change a quantised matvec's cost, so the timing stands as a kernel-throughput number, but **this
particular run is not evidence about accuracy** and must not be quoted as such.

So LP-05 is not "use the GPU". The GPU is already 2.6x the CPU and still 384x off the hardware's own GEMM throughput,
because the kernel everyone has is a dequantise-then-dot matvec, not a tensor-core MMA path. **LP-05 = write an IQ2_XXS
MMA kernel** (2-bit codes unpacked in registers/shared memory, fed to mma.sync, accumulated in fp32), which is the only
construction that reaches the measured 23.7 TFLOP/s, and it is an XL pad that closes the last order of magnitude.

Its gate, in order: (1) bit-identical or oracle-identical output against `wubu_cuda_selftest_iq2_run` on the same bytes,
(2) measured GMAC/s above 300 (10x the current kernel) before any engine wiring, (3) 12/12 oracle after wiring, with the
accumulation-order parity question answered explicitly rather than assumed.
