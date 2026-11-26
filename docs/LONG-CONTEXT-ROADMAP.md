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

## What the IQ2 kernel actually does, and the shortest path from 30.8 to hardware rate

`src/wubu_cuda_quant.cu:650` `iq2xxs_matvec_mrow_kernel` is better than the label "naive matvec" implies:

- one warp per output column, `group = lane>>2` (0..7) and `entry = lane&3` split each 66-byte IQ2_XXS block's four
  grid entries across lanes, so the 2-bit codes are unpacked cooperatively;
- the unpacked weights `g0,g1` are loaded ONCE per block and reused across `IQ2X_MT` activation rows in registers -
  the weight-read amortisation is already there (this is why batching gains little at kernel level);
- the dot product is `__dp4a` (int8 dot-accumulate) against q8_K activations, then one `acc[t] += wd * q8b->d * sumi`
  fp32 FMA per 8-value group.

So the kernel is not wasting weight traffic. **It is limited by using INT8 dp4a instead of INT8 tensor cores.** The
measured 30.8 GMAC/s is ~62 GFLOP/s of dp4a; this card's tensor-core GEMM is 23,650 GFLOP/s. The gap is the instruction,
not the algorithm.

**LP-05's concrete upgrade, therefore, is narrow and well-defined:** replace the per-lane `__dp4a` inner loop with
`mma.sync.aligned.m16n8k32.s8.s8.s32` (INT8 tensor cores, available on sm_89), keeping everything above it - the warp
mapping, the cooperative 2-bit unpack, the once-per-block weight load, the fp32 scale application. The operand layout
already matches: IMMA consumes int8 tiles of 16x8x32, and the kernel already has 8-lane groups producing exactly 32 int8
values per block per row. Expected effect: the dp4a path issues one MAC per lane per instruction; a m16n8k32 IMMA issues
4096 MACs per warp per instruction, so 1-2 orders of magnitude is available in principle, with the block unpack and the
scale FMA becoming the new bound (~8 fp32 FMA per 256 values per row).

Order of work, unchanged from the gate: (1) keep the dp4a kernel as the reference, (2) add the IMMA kernel beside it and
assert bit-identical against `wubu_cuda_selftest_iq2_run` on identical bytes - int8 dot products are exact, so parity
here is achievable, not approximate, (3) require >300 GMAC/s before wiring, (4) then 12/12 oracle with the accumulation
order question answered.

That is a self-contained kernel task with an exact reference already in the tree, and it is the last order of magnitude.

## Correction to the paragraph above: it is not a 1:1 instruction swap

I wrote that "the operand layout already matches" and that the change is narrow. Checked against the actual mapping, that
is wrong and worth correcting before anyone builds on it:

    current:  col = blockIdx.x * warps_per_block + (threadIdx.x / 32)
              => ONE warp owns ONE output column; lanes split the 256 values of one 66-byte block
              => warp tile = 1 column x IQ2X_MT tokens

    mma.sync.m16n8k32.s8 warp tile = 16 tokens x 8 columns x 32 k
              => a warp must own EIGHT columns to fill N=8, and it currently owns one

So the lane mapping has to be rebuilt: `group = lane>>2` / `entry = lane&3` (which splits one block's four grid entries
across 4 lanes) must become a lane -> (column, k-slice) mapping that produces an 8-column tile, and the cooperative 2-bit
unpack has to be re-derived for that shape because the four entries of one block now land in four different columns' A/B
fragments rather than four lanes of the same dot product. The register accumulators, the `IQ2X_MT` token blocking and the
per-block weight reuse all survive conceptually, but the inner loop does not.

What does NOT change: the reference is still the dp4a kernel, int8 dots are still exact so bit-parity is still achievable
rather than approximate, and the gate is still >300 GMAC/s before wiring. Only the effort estimate moves - this is an XL
pad, not a substitution, and the honest statement is that it needs a session of its own rather than a patch on the end of
this one.

## LP-05 baseline upgraded from throughput-only to real evidence

The caveat above ("the price tool's reference magnitude is 4.03e-37, so this is not evidence about accuracy") is now
resolved rather than carried. The tool was leaving the q8_K block's fp32 scale unset, which drove every product to ~4e-37.
With the scale set, the same measurement reads:

    batch vs per-expert: ref0 = 2471.25   max_abs = 0.000e+00   max|ref| = 4.681e+03   rel = 0
    8 separate launches 3.44 ms | 1 batched 1.38 ms | 2.49x per layer
    GPU batched IQ2_XXS throughput: 30.77 GMAC/s  (bit-identical to the per-expert path on real values)

So the GPU kernel is both **exactly correct** on this shape and **30.77 GMAC/s** - still 384x below the card's 11.82 TMAC/s
tensor-core GEMM, which remains the whole of LP-05. The difference from before is that the number now has teeth: it is a
measurement of a correct kernel, not of a degenerate one.

## Why the 8-column restructure helps even before tensor cores: the kernel is traffic-bound

Per output column of a 2304-wide matrix, the bytes actually touched are:

    weights      : 9 blocks x 66 B            =   594 B
    activations  : 9 blocks x 292 B (q8_K)    = 2,628 B     <-- 4.4x the weight bytes

The activation row is the SAME for every column, but the current mapping (one warp per output column) gives each warp its
own copy of the read, so a 2304x2304 matrix costs:

    one warp per column (current)   activations re-read 2304x = 6.1 MB  + weights 1.4 MB = 7.4 MB
    eight columns per warp          activations read     288x = 0.8 MB  + weights 1.4 MB = 2.1 MB

    5.3M MACs / 7.4 MB = 0.7 MAC per byte      <- arithmetic intensity far too low to be compute-bound

That is the real reason the kernel sits at 30.8 GMAC/s: it is moving 7.4 MB to do 5.3M MACs. **The 8-column restructure is
worth up to 3.6x on traffic alone**, before any tensor-core instruction is introduced - and it is the same structural change
IMMA requires, because a warp must own 8 columns to fill `N=8`. So the two steps are one step, and the ordering is:

1. rebuild the lane mapping so a warp owns 8 columns and holds the activation block in registers once (expect toward
   ~110 GMAC/s from traffic alone; still dp4a, still bit-identical, still testable against the existing kernel),
2. then swap the inner dot for `mma.sync.m16n8k32.s8` on the same tile, where the activation fragment is already resident
   and the 2-bit unpack is already producing per-column int8 tiles,
3. gate on >300 GMAC/s, then 12/12.

Step 1 is a smaller, self-contained change than step 2 and it already buys most of a factor of four. That is where the
kernel work should start.

## Retraction: the kernel is NOT traffic-bound, and shared-memory staging made it 84% slower

The arithmetic above is correct and the conclusion drawn from it was wrong. Implemented the staging - the q8_K block
copied once per (m0, block) into shared memory for all warps in the CTA, every per-lane calculation left byte-identical,
`__syncthreads()` on both sides of the phase - and measured:

    before:  1.38 ms batched, 30.77 GMAC/s
    after:   2.54 ms batched, +84 % slower

Reverted. What the 7.4 MB figure missed: **the activation bytes are identical for every column, so the repeated reads hit
L2, not DRAM.** They were never the cost I attributed to them; the cooperative copy plus two barriers per (m0, block) cost
more than the cached reads it replaced. Same-bytes argument, same order, guaranteed-identical maths - and still slower,
which is exactly why the measurement has to come before the wiring.

Consequence for the plan: the **8-column restructure is not justified by traffic**. Its value has to come entirely from
the tensor-core instruction it enables, so it is no longer a cheap step 1 with a 3.6x attached - it is a step whose whole
justification is the IMMA itself, and it should be built as the MMA change rather than as a separate restructure first.
The engine-side record stands: CPU at 11.9 GMAC/s ceiling, GPU IQ2 at 30.8 GMAC/s, card at 11.8 TMAC/s, and the gap is
the instruction. The four prior retractions in this project were about speedups claimed too early; this one is about a
*bottleneck* diagnosed too confidently from arithmetic alone.

## Swept the one knob the kernel has: IQ2X_MT = 16 is already optimal

`IQ2X_MT` is how many activation rows share one weight unpack in registers. It is the kernel's only tuning parameter and
had never been swept:

    MT= 8    1.35 ms batched    parity exact
    MT=16    1.34 ms batched    parity exact     <- current value
    MT=32    1.49 ms batched    parity exact     (-11 %, register pressure)

So the shipping value is the right one and the kernel is not leaving anything on the table at this knob. Worth recording
for a second reason: **parity stayed bit-exact at all three values**, which confirms the accumulation order does not
depend on MT - so MT is a free parameter for any future scheduling experiment, not something that silently moves the
reference tokens.

Remaining state, unchanged: GPU IQ2 is exact and 30.8 GMAC/s against an 11.8 TMAC/s card, CPU is at its 11.9 GMAC/s
ceiling, and the only lever left is the 8-column `mma.sync.m16n8k32.s8` kernel, whose justification is now purely the
instruction (the traffic argument was retracted) and whose parity reference is the dp4a kernel that this sweep just
re-confirmed.

## Parity gate PASSED: the 8-column tile is bit-identical to the shipping kernel

`wubu_cuda_iq2x8_check` runs both kernels on the same device bytes and compares:

    upload rc=0
    rc=0  K=2304 n_cols=2304 M=8
    mrow vs 8col:  max|diff| = 0.000e+00   max|ref| = 1.266e+04   rel = 0
    BIT-IDENTICAL

Two things had to be right for this number to mean anything, and the first attempt got both wrong:
- the first run reported `max|ref| = 0.000e+00` - both kernels emitted zeros, because the IQ2 grid and k-sign tables
  live in `__constant__` memory and are EMPTY until `wubu_cuda_upload_iq2xxs` is called. A 0 == 0 comparison would have
  "passed" and proven nothing. The tool now uploads the tables first and refuses to print a verdict on a zero reference.
- the fp16 block scales are forced to 1.0h so the values are finite instead of inf.

So the tile that `mma.sync.m16n8k32` requires is now proven equivalent to the shipping kernel, with no shared memory and
no barriers, activation loads hoisted into registers and reused across eight columns. What is NOT yet known is whether it
is FASTER - parity and speed are separate gates and only the first has been passed. Next: time both kernels at
2048x2304x8 and require a win before wiring; then swap the inner `__dp4a` pair for `mma.sync.m16n8k32.s8` on this tile,
where the activation fragment is already resident and the unpack already produces per-column int8.

## Speed gate PASSED, and it exposes that the 30.8 GMAC/s figure was measuring copies, not the kernel

Timed with cudaEvents inside `wubu_cuda_iq2x8_check`, same device bytes, after an untimed warm pass:

    M= 8   mrow  0.11 ms  373.62 GMAC/s    8col  0.07 ms  648.00 GMAC/s   1.73x   BIT-IDENTICAL
    M=64   mrow  0.83 ms  409.68 GMAC/s    8col  0.47 ms  721.25 GMAC/s   1.76x   BIT-IDENTICAL

Two conclusions, both of which change the plan:

1. **The 8-column tile is faster as well as exact** - 1.73-1.76x over the shipping kernel, flat across M, bit-identical
   with a finite reference (1.266e+04). So the restructure that failed on traffic grounds succeeds on column blocking:
   the win was never shared memory (which cost 84 %), it was giving each warp eight columns so the activation load and the
   loop overhead amortise.
2. **373-721 GMAC/s is not the 30.8 GMAC/s the MoE path reported, and the difference is not the maths.** Both measure
   42.5M MACs; this run does it in 0.11 ms, the MoE path in 1.38 ms. The extra 1.27 ms is the eight host-to-device slab
   copies (10.4 MB over PCIe) that `wubu_cuda_moe_batch` performs inside its timed region. **The MoE GPU path was never
   kernel-bound - it was upload-bound**, which is the same conclusion the D2D-gather experiment reached from the other
   direction and which the resident tier exists to fix.

So LP-05 is smaller than advertised: the kernel already exceeds the 300 GMAC/s gate that was set for the IMMA work, and
the measured bottleneck in the path that uses it is slab upload. Correct order is now: (a) wire the 8-column kernel,
1.7x for free and exact, (b) keep slabs resident so the copies stop, (c) only then consider `mma.sync` on a kernel that
may no longer be the binding constraint.

## Wiring result: the 8-column tile must be gated on a real batch, not switched on globally

Wired it into the resident dispatch unconditionally and measured in the engine:

    dense decode, 24 tokens, 8col everywhere : 7031 ms      <- 37 % SLOWER than baseline
    dense decode, 24 tokens, mrow (baseline) : 5138-5288 ms
    parity                                   : exact in both cases (12/12)

The cause is occupancy, not arithmetic: `groups = ceil(n_cols/8)` gives the grid eight times fewer warps for the same
work, and at `n_tok = 1` each warp has almost nothing to do, so the tile that wins at M=8 (1.73x) loses badly at M=1.
Smoothly gated now: `n_tok >= 8` takes the 8-column kernel, everything else keeps the one-column kernel. Decode is back
at baseline and prefill batches get the win where it was actually measured.

That is the third time in this line that a kernel measured faster in isolation failed or inverted end to end, and the
lesson is the same each time: **the isolated number names the mechanism, never the outcome.** What made this one cheap to
fix was gating it behind a condition the same commit could flip - no half-wired path, no ambiguity about which code ran.

## The 8-column win does not reach the engine, because prefill barely uses the GPU path

    120-token prefill, CPU           13,841 ms
    120-token prefill, IN2_GPU_DENSE 13,548 ms      (+2.1 %)

A kernel that is 1.73x faster at the shape it was measured on moves the engine by 2 %. The reason is structural, not a
bug: prefill issues its projections through `in2_matvec_quant`, which is the CPU path, and the GPU resident tier only
covers the tensors it holds. So the 8-column tile is wired, exact, and faster - and almost none of the work flows through
it. Gated correctly (`n_tok >= 8`), so it costs nothing where it does not apply, but it also buys almost nothing today.

**That makes the next step unambiguous and it is not a kernel:** route the prefill projections through the GPU resident
path at all. Until that wiring exists, every kernel-level improvement - this 1.73x, the 2.49x batching, the 384x IMMA
headroom - multiplies a fraction of the work that is close to zero. The measured chain is now:

    CPU ceiling            11.9 GMAC/s   (engine already at ~13)
    GPU kernel            373-721 GMAC/s (measured directly, 8col, M>=8)
    card tensor cores    11,800 GMAC/s
    engine prefill uses   the CPU path    <-- the actual bottleneck

Fourth instance of the same pattern in this project: the isolated number names the mechanism, never the outcome.

## Why no kernel improvement can reach prefill today: the resident entry has no batch dimension

`src/wubu_inkling2_fwd.c:380`:

    static int in2_gpu_resident_id(int type, int64_t ident, const void *w, size_t wbytes, const void *act,
                                   int act_bytes, float *y, int K, int n_cols, int64_t stride)

There is no `n_tok`. Every GPU projection the engine issues is **M = 1 by construction**, and the DLL entry behind it
(`wubu_cuda_resident_matvec_id`) is the same shape. That is the whole explanation for the +2.1 % end-to-end result: the
8-column kernel needs M >= 8 to win, the engine can never hand it more than one row, so the faster tile is only ever
reached by a direct test harness.

So the next step is a two-part change and neither part is a kernel:
1. give the resident entry an `n_tok` (and a matching token-strided activation/输出 layout) at the DLL boundary;
2. change the projection call sites to pass a token block instead of looping one token at a time - the nine
   `for (int t = 0; t < n_tok; ++t)` sites already identified.

Only after both does any of the measured kernel headroom (1.73x tile, 2.49x batching, 384x tensor cores) become engine
throughput. Everything downstream of that is multiplication by a fraction of the work that is currently close to zero.

Standing measured chain, for the next session to start from rather than re-derive:

    engine prefill          0.11 s/token (CPU path)     13.8 s for 120 tokens
    CPU ceiling             11.9 GMAC/s                 engine already at ~13
    GPU kernel (M>=8)       373-721 GMAC/s              measured, 8col, bit-identical
    GPU MoE path            30.8 GMAC/s                 upload-bound, not kernel-bound
    card tensor cores       11,800 GMAC/s               384x the IQ2 kernel
    oracle                  12/12 exact in every configuration above

## Part 1 attempted and reverted: the DLL export mechanism is not the obvious one

Wrote `wubu_cuda_resident_matvec_nt` (batched residency: stage n_tok contiguous q8_K rows, run the device matvec with
M = n_tok so the 8-column tile is selected, copy n_tok rows back; same `g_res_*` tables, same ident key). It compiled with
zero errors and the DLL grew - and **the symbol did not appear in the export table**:

    objdump -p wubu_cuda.dll | grep resident_matvec
      [55] wubu_cuda_resident_matvec
      [56] wubu_cuda_resident_matvec_id          <- the new _nt is absent

`wubu_cuda_iq2x8_check` exported fine from the same append-at-EOF position, so this is not simply "appended code is not
exported". Something about how `WUBU_EXPORT` resolves for this translation unit differs between those two functions -
an export list, an `extern "C"` region boundary, or a declaration elsewhere that the linker honours instead. Reverted the
whole change rather than leave an unexported stub in the tree; source is back at HEAD, DLL rebuilt with `-B`, no `_nt`
symbol present, oracle 12/12 on CPU and dense.

**The next session should read `WUBU_EXPORT`'s definition and the makefile's export path first** - five minutes of reading
that would have saved this attempt. The batched entry's body is sound and is described above in enough detail to re-type;
what is missing is only the linkage detail.

## Correction: the export was fine, my grep was wrong. `wubu_cuda_resident_matvec_nt` is in the DLL

The previous note said the symbol did not appear in the export table and blamed `WUBU_EXPORT`. That was wrong. `objdump -p`
truncates long export names, so grepping the full `wubu_cuda_resident_matvec_nt` returned nothing while the symbol was in
fact exported all along. The correct check is a prefix match:

    objdump -p wubu_cuda.dll | grep -oE "wubu_cuda_resident_matvec[a-z_]*" | sort -u
      wubu_cuda_resident_matvec
      wubu_cuda_resident_matvec_id
      wubu_cuda_resident_matvec_nt      <- present

So the DLL boundary now has the batch dimension the engine was missing: `wubu_cuda_resident_matvec_nt(type, ident, w,
wbytes, act, act_row_bytes, y, K, n_cols, stride, n_tok)` stages n_tok contiguous q8_K rows into one device buffer, runs the
device matvec with M = n_tok - which is what selects the 8-column tile - and copies n_tok result rows back, with the same
resident tables, ident key, fill-on-miss and eviction policy as the single-row entry.

**Part 1 is done and unwired**: nothing dispatches to it, and the oracle is 12/12 exact on CPU and dense with it in the
DLL. Part 2 is the call sites - the nine `for (int t = 0; t < n_tok; ++t)` projection loops that must hand a token block
instead of one row - and only after that does the 8-column tile's measured 1.73x become engine throughput.

Lesson recorded for the next attempt at anything similar: **a missing symbol in objdump output is not evidence of a missing
export** when the name is long. Match on a prefix, or dump the table and read it.

## Part 2 is ONE function, not nine sites

Correcting my own estimate. The token-batched helper already exists and the attention projections already call it:

    int in2_matvec_nt(gguf_ctx *ctx, const wubu_in2_tensor_t *t,
                      const float *X, int nt, int n_in, int n_out, float *Y)      // fwd.c:628
    callers: L->wq (:1258), L->wk (:1290), L->wv (:1291), L->wrel (:1378), ...

So the per-token loops I described as "nine sites" are already batched at these projections; what is missing is only that
`in2_matvec_nt` never tries the GPU. It quantises and runs the CPU fast kernel over nt tokens and falls through to float.

The wiring is therefore one insertion at the top of `in2_matvec_nt`:

1. gate: `IN2_GPU_DENSE` set, `in2_gpu_state == 1`, `nt >= 8`, type in {16, 22}, and the resident entry resolved;
2. quantise the token block once - `nt` contiguous q8_K act rows, `(n_in/256) * sizeof(in2_q8k_blk)` bytes per row
   (`in2_q8k_quantise` already does one row; it needs a loop over nt, or a strided variant);
3. call `wubu_cuda_resident_matvec_nt(ty, t->file_offset + boff, src, rb * n_out, act_block, act_row_bytes, Y,
   n_in, n_out, rb, nt)` - M = nt is what selects the 8-column tile;
4. on any non-zero return, fall through to the existing CPU path unchanged.

That is the whole of part 2. The remaining unknowns are small and local: `boff`/`rb` are already computed in the
neighbouring `in2_matvec_quant`, and the quantise loop is mechanical. It was not attempted here because it still needs a
build-measure-revert cycle per step and the oracle must be re-gated afterwards - the structure is now known precisely
enough that the next attempt is a single edit rather than a survey.

## The one detail that stops the in2_matvec_nt wiring: where the weight bytes come from

Attempted the branch described above and aborted it before building. `in2_matvec_nt`'s signature has no `elem_off` and,
crucially, no weight pointer - it hands tensor METADATA (`t`, `t->info`) to the CPU kernel, which resolves the weights
itself. The DLL entry `wubu_cuda_resident_matvec_nt` needs the actual **host weight bytes** (`h_w`, `wbytes`) because it
may have to fill the resident slot, i.e. `cudaMemcpy` from host. Passing `t->info` there would copy metadata into VRAM,
compute garbage and still return 0, so the fallback would never fire and the oracle would break with no signal.

The correct source is the one `in2_matvec_quant` already uses two functions away:

    const int64_t boff = (elem_off == 0) ? 0 : gguf_raw_size(ty_mv, elem_off);
    wubu_in2_tensor_t sh = *t;  sh.file_offset = t->file_offset + boff;
    const unsigned char *src = wubu_inkling2_map_ptr(ctx, &sh);      /* mapped */
    if (!src) { src = <thread-local buffer>; wubu_inkling2_read(ctx, &sh, buf, rb * n_out); }

With `elem_off == 0` for `in2_matvec_nt` this reduces to: shadow the tensor at `t->file_offset`, `map_ptr` it, and pass
that pointer as `h_w` with `wbytes = rb * n_out`. If the mapping is unavailable, read into a buffer first, or simply skip
the GPU branch - the fallback is cheap and correct.

So the wiring is four steps plus this fifth: **the slab pointer must come from `wubu_inkling2_map_ptr`, never from the
tensor descriptor.** Reverted; tree clean, oracle 12/12 after a full rebuild.

## The batched hook at the attention projection site: implemented, measured, reverted

Wired `wubu_cuda_resident_matvec_nt` exactly as specified - inside the existing `for (int g = 0; g < nt; g += 8)` group loop
in the batched projection path, where the group's activation rows are already contiguous in `act` and `src` is the mapped
weight slab, so one call covers eight tokens at M = 8 and the 8-column tile is finally reachable from the engine.

    oracle, 12 tokens, CPU vs IN2_GPU_DENSE : exact (94-byte IDS line, non-empty on both sides)
    120-token prefill, CPU                   : 12,913 ms
    120-token prefill, IN2_GPU_DENSE=1       : 13,721 ms    (-6 %, GPU slower)

So the tile is reachable now and still does not pay. Combined with the earlier +2.1 % this rules out the explanation that
held for the last two notes: it is not "the kernel never sees a batch". The batch is real, the tile is selected, and the
prefill is still not faster than the CPU - which points at everything AROUND the matvec rather than the matvec itself:
the 1.5 MB-per-call weight residency, the per-call quantisation of the token block, and the H2D/D2H of activations and
results. Those are now the suspects, and each is measurable on its own before another line of wiring is written.

Reverted; full rebuild, oracle exact.

Note on method: the first run of this build failed to compile, so `gen_fast.exe` did not exist and both the CPU and GPU
runs returned EMPTY IDS lines. The equality check `[ "$A" = "$B" ]` compared "" to "" and reported "oracle exact". That
is the fourth degenerate comparison this session and the fix is the same one every time: **require the reference to be
non-empty before believing an equality**, never just check that two sides agree.

## Residency ruled out: the prefill resident tier hits 97.7 %

    IN2_GPU_DENSE=1 prefill, 120 tokens:  RESSTAT hit=4679 miss=110 evict=0 slots=110 budget=0MB rate=97.7 %

110 misses for the whole run is the weight set itself being filled once (48 layers x 2-3 matrices), so every projection
after the first is a device hit. The 1.5 MB-per-call residency theory is dead, and with it the last of the three suspects
that were about data movement into the card.

What is left is the shape of the call, not the data:
- `wubu_cuda_matvec_iq2xxs` ends in `return cudaDeviceSynchronize() == cudaSuccess ? 0 : -2;` - so every single projection
  is a blocking round trip. A prefill makes one call per 8-token group per projection per layer; the useful work per call
  is a 2304-wide matvec and the sync latency is paid on top of it, with no overlap between layers.
- the whole token block is re-quantised per call on the host.

So the next lever is not bandwidth and not the kernel: it is **one synchronisation per layer instead of one per projection**
(launch async, sync where the result is first read), which is cheap to try and directly measurable against the 12,913 ms
CPU baseline. Measure that before any further kernel work.

## Where prefill time actually is, in MACs (this retires the projection work as a target)

Per token per layer, from the model's own dims (d=2304, d_ff=1152, 8 of 128 experts):

    attention projections (wq, wk, wv, wrel)   11.4 MMAC
    MoE experts (8 x gate_up + down)           63.7 MMAC     -> MoE is 5.6x the projections

    120-token prefill x 48 layers:  projections 65.40 GMAC,  MoE 366.92 GMAC  (total 432 GMAC)
    at the measured 8-column kernel speed (648 GMAC/s):  projections 101 ms, MoE 566 ms
    measured prefill total:                                       13,250 ms

So **even with the GPU running every attention projection perfectly, the most it can remove from prefill is 0.8 %.** The
whole line of work that ends in the batched resident hook was chasing 0.8 %. That, not any kernel defect and not any
synchronisation pattern, is why GPU prefill never beat CPU: the work it was given was too small to matter.

Conversely the MoE experts are 85 % of the MACs and the same kernel could run them in 566 ms instead of contributing to a
13 s prefill - a ~20x prize - except that the existing GPU MoE path measures 30.8 GMAC/s, i.e. no better than the CPU,
because it spends its time on slab uploads rather than arithmetic. **That is the only target in this engine big enough to
change prefill**, and it is a data-placement problem, not a kernel problem: the experts must be resident and stay resident
across tokens instead of being staged per call.

Budget check before any of it: 432 GMAC at the CPU's effective ~33 GMAC/s (8 threads) is 13.1 s, which is the measured
prefill almost exactly - so prefill is compute-bound on the CPU and there is no hidden overhead left to find.

## Prefill MoE on the GPU, measured four ways - all of them lose

    120-token prefill:
      CPU baseline                                  12,773 ms
      IN2_GPU_MOE=1                                 12,910 ms   (+1.1 %)
      IN2_GPU_MOE=1 IN2_GPU_DENSE=1                 13,537 ms   (+6.0 %)
      IN2_GPU_MOE=1 IN2_MOE_BATCH=1 IN2_RES_GB=3     13,074 ms   (+2.4 %)

So the MoE - 85 % of the MACs and the only prize large enough to matter - also does not pay on this card. The explanation
is the one thing all these configurations share: **they run at M = 1.**

8 experts per token means 8 separate expert matvecs, each a 2304-wide row against a distinct slab. At M = 1 the kernel has
no reuse to exploit and the per-call round trip (launch, quantise, H2D activation, D2H result, sync) is paid ~1,150 times
per token per layer. Residency removes the weight upload - 97.7 % hit - but it cannot remove the round trips, and the
arithmetic per call is too small to hide them.

The one configuration that would work is the one the routing forbids at small batch: **M >= 8 per expert.** With 8 tokens
and 8 experts each, ~50 distinct experts are touched, so each expert sees ~1.1 tokens - grouped GEMM at M = 1.1 is exactly
as hopeless as the per-token loop. To reach M >= 8 per expert the prefill must be chunked at **>= 128-512 tokens**, where
each expert collects 8-32 rows and the measured 373-648 GMAC/s becomes reachable:

    367 GMAC of MoE at ~400 GMAC/s  ->  ~0.9 s   against 12.8 s today

That is the real shape of the remaining work: a grouped-GEMM MoE kernel over the union of routed experts, driven by large
prefill chunks. It is a bigger job than anything attempted in this session, and it is the first one whose expected payoff
is measured rather than assumed - 14x on 85 % of the work, versus the 0.8 % the projection wiring was chasing.

## The prerequisite the grouped-GEMM plan was missing: there is no token-blocked routing anywhere

Checked the routing histogram (`g_moe_hist`, printed at fwd.c:1882) during a long-text prefill. Every line reads

    MOEHIST <layer> <expert> 1

one count per (layer, expert), for the experts of the layers that printed. Nothing aggregates across tokens. Read
conservatively this says either the counter is printed and reset per token, or routing is only ever computed for a single
token at a time - and both readings agree on the fact that matters: **no multi-token routing block exists in the engine.**

That is the actual prerequisite for the 14x. A grouped MoE GEMM cannot be driven by a per-token loop, and the projection
work proved that merely giving the GPU a batch at one call site changes nothing if the surrounding structure stays
per-token. The work is therefore, in order:

1. build a token-blocked MoE stage: for a chunk of N tokens, compute all N x 8 routings up front, invert them into
   per-expert token lists (the union over the chunk), and only then dispatch;
2. a grouped-GEMM kernel over that inverted index - M = tokens routed to expert e, which for N >= 128 gives the 8-32 rows
   the 8-column tile needs;
3. keep both on the device: the inverted index and the gathered activations, so the per-call round trips that cost this
   session four separate failed wirings cannot come back.

Step 1 is the one to size first: the histogram as written cannot answer "how many tokens per expert at N = 480", and that
number decides whether the payoff is 14x or 2x. Instrumenting it properly is cheaper than any kernel and should come first.

## Sizing the grouped-GEMM prize: N = 128 is the threshold, and the payoff is ~12x

Uniform top-8-of-128 routing, expected distinct experts touched by a chunk of N tokens
(`E(1-(1-k/E)^N)`, E=128, k=8), and the mean rows per expert that a grouped GEMM would receive:

    N=  8   distinct~ 51.6   mean M= 1.24
    N= 32   distinct~111.8   mean M= 2.29
    N= 64   distinct~125.9   mean M= 4.07
    N=128   distinct~128.0   mean M= 8.00    <- the 8-column tile's threshold, hit exactly
    N=256   distinct~128.0   mean M=16.00
    N=480   distinct~128.0   mean M=30.00

So a chunk of 128 tokens already gives the kernel the M = 8 it needs, and 256+ gives margin (real routing is skewed, so
hot experts get more rows than the uniform mean and cold ones fewer - the margin is what absorbs that).

Timing at the measured 400 GMAC/s (conservative, between the 373 GMAC/s mrow figure and 648 GMAC/s for the 8-column tile):

    N=128   MoE 391.4 GMAC -> 978 ms + proj 69.8 GMAC -> 174 ms = 1.15 s   (9.0 ms/token)
    N=480   MoE 1467.7 GMAC -> 3669 ms + proj 261.6 GMAC -> 654 ms = 4.32 s  (9.0 ms/token)
    today, CPU path:                                              106.4 ms/token (12,773 ms / 120)

**~11.8x on prefill**, from 106.4 to 9.0 ms/token, and the projection share stays at 15 % - which is why the 0.8 % figure
from the earlier note was about *total* prefill share at current speeds, not about the projection work being worthless in a
fast regime. Both numbers are correct; they answer different questions.

Caveat that decides the real number: at 400 GMAC/s the card must sustain that with resident experts AND no per-call round
trips. Every failed attempt this session failed on the second condition, not the first. So the grouped kernel must consume
a device-resident index and device-resident gathered activations in one launch per layer - anything that returns to the
host per expert reintroduces the ~1,150 calls/token/layer that cost four wirings.

## The routing histogram cannot size the chunk, and it contradicts the recorded layer count

Ran a 480-token text prefill with IN2_MOE_HIST=1 and parsed every MOEHIST line:

    284 lines, 24 layers, total routings 383 -> 15 tokens' worth per layer
    distinct experts per layer: min 7, max 17, mean 11.8
    counts: mean 1.3, median 1.0, max 3

Two problems, both mine to state plainly:

1. **It does not accumulate across the prefill chunk.** ~16 routings per layer is two tokens x 8 experts, i.e. only the
   generated positions, not the 480 prefilled ones. So this instrument cannot answer "tokens per expert at N = 480" and
   the uniform model in the previous note remains the only estimate. It is a plausible model, not a measurement, and must
   be labelled that way wherever it is used.
2. **It reports 24 layers, while every note in this file so far records 48.** One of the two is wrong, and it changes
   every arithmetic result downstream by 2x - including the 11.8x and the "432 GMAC total" from earlier notes. Until
   `layers` is confirmed from the GGUF metadata directly, treat all prefill arithmetic in this document as carrying a 2x
   uncertainty on the layer factor.

Nothing was changed to get this: the histogram patch was reverted (it printed nothing at the run sizes tried, so it was
dead code in the hot path), the build is clean and the oracle is 12/12 with the original instrumentation.

Next step is therefore a **measurement**, not a kernel: print the real layer count and the real per-token expert counts from
one fixed place, and re-derive the arithmetic. Two of this document's headline numbers are currently built on a layer
count that the engine itself contradicts.

## Layer count resolved from the GGUF: 48 blocks, and the histogram's 24 is almost certainly the MoE-layer ordinal

Parsed the metadata directly (ver 3, 734 tensors, 62 kv):

    inkling2.block_count             = 48
    inkling2.expert_count            = 128
    inkling2.expert_used_count       = 8
    inkling2.expert_feed_forward_length = 1152
    inkling2.expert_weights_scale    = 8.0
    general.architecture             = inkling2

So 48 layers is correct and the earlier arithmetic that assumed 48 was right. The histogram reporting ids 0..23 contiguously
is therefore not "the model has 24 layers" - the most likely reading is that `g_moe_hist` is indexed by the **MoE-layer
ordinal** (24 of the 48 layers carry experts), which means the MoE MAC share in the earlier notes is overstated by 2x:

    MoE, 120-token prefill:  366.9 GMAC over 48 layers  ->  183.5 GMAC over 24 MoE layers
    projections:              65.4 GMAC (all 48 layers, unchanged)
    total:                   249 GMAC, not 432

That does not change any conclusion about WHERE the work is, but it changes the size of the prize: at 400 GMAC/s the pair
becomes 459 ms (MoE) + 164 ms (projections) = 623 ms per 120 tokens = 5.2 ms/token, i.e. **~20x rather than ~12x** on
prefill. Treat that as an estimate with one unverified step - whether the histogram really is indexed by MoE ordinal.

The one-line test that settles it: print the model layer index alongside the histogram key for a single routing. That is
cheaper than anything else in this document and it decides whether the headline is 12x or 20x.

Also worth recording: the first GGUF parse attempt died with MemoryError because an array type was mis-sized and it tried
to allocate the count as bytes. Read the element type before the length, and skip by size rather than allocating.

## Contradiction closed: the model itself puts experts on layers 0-23 only, so the engine was right

Tensor names straight out of the GGUF (plain string scan, since the structured parse mis-skipped twice):

    blk.N.ffn_gate_up_exps   24 distinct layers    ->  layers 0..23
    blk.N.ffn_down_exps      24 distinct layers    ->  layers 0..23
    blk.N.ffn_gate_inp       24 (the router)       ->  layers 0..23
    blk.N.ffn_norm / ffn_post_conv / ffn_res_query   24 each

Layers 24-47 carry no expert tensors at all. So:

- `block_count` = 48 is correct;
- the routing histogram's contiguous 0..23 was **the truth, not a bug** - `il` is the model layer index (`m->layer[il]`)
  and the engine called `in2_moe` for exactly the 24 layers that own experts;
- Inkling-2-Mini is a **hybrid**: expert layers 0-23, a different FFN in 24-47.

So the only error was mine in assuming MoE ran on all 48 layers, which inflated the MoE MAC share by 2x. Corrected figures:

    MoE, 120-token prefill:   183.5 GMAC (24 expert layers)
    attention projections:     65.4 GMAC (48 layers)
    total:                    249 GMAC
    at 400 GMAC/s:            459 ms + 164 ms = 623 ms = 5.2 ms/token  vs 106.4 ms/token today  -> ~20x

Every "contradiction" in the last three notes dissolved into a single arithmetic slip on the layer factor. The engine was
correct throughout, and the histogram instrumentation - which I twice suspected of being broken - was reporting the truth
both times. Worth remembering next time a measurement disagrees with a note: check the note against the file first.

## Prefill does reach in2_moe with n_tok - so the histogram's silence about prefill is unexplained, and that matters

The call graph, read rather than assumed:

    wubu_inkling2_fullseq.c:395   rc = in2_moe(ctx, m, cfg, il, moein, n_tok, moeout);   <- prefill, whole token block
    wubu_inkling2_fwd.c:1575      rc = in2_moe(ctx, m, cfg, il, moein, n_tok, moeout);   <- decode
    wubu_inkling2_fwd.c:1951      rc = in2_moe_batch(...)                                <- inside in2_moe
    wubu_inkling2_fwd.c:1960      router + MOE_HIST increment, loop over t < n_tok

So prefill passes the entire chunk to `in2_moe` and `in2_moe` does see `n_tok`. Yet the measurement says otherwise:

    480-token prompt, 2 tokens generated -> 383 routings total = 2 tokens' worth (16 per layer)
    60 / 120 / 480-token prompts, 1 token generated -> MOEHIST prints NOTHING at all

Both results say the same thing: the histogram only ever sees the generated positions. That is now inconsistent with the
call graph, and the inconsistency is not cosmetic - it decides whether prefill's MoE runs as one 480-token block (in which
case a grouped GEMM has a natural home already) or as 480 separate single-token calls (in which case the block does not
exist yet and step 1 of the plan is real work).

The next step is a **three-line trace, not more reading**: print `(il, n_tok)` once at the top of `in2_moe` under an env
flag and run one 120-token prompt. That single line answers the question that determines the entire remaining plan, and it
is the cheapest thing in this document.

Method note, because this is the third time: I have twice now "proved" something by reading call sites and been contradicted
by the measurement. Reading tells you what CAN happen; only the trace tells you what DID.

## Trace settles it: prefill gives MoE the whole token block - 48 calls of n_tok=120

Three lines at the top of `in2_moe` under `IN2_MOE_NTOK`, one 120-token prompt, one generated token:

    48  n_tok=120

So the block IS there: one `in2_moe` call per layer, each carrying all 120 tokens. Step 1 of the grouped-GEMM plan - "build a
token-blocked MoE stage" - is therefore **already done at the entry point**, and the histogram's earlier silence about
prefill was an artifact of where the counter sits inside the function, not evidence about the block size.

What that leaves: the block arrives at `in2_moe` and is then broken apart. The evidence is the same histogram - 383 routings
for a 480-token prompt is 2 tokens' worth, which means the routing counter sees the decode-sized path, not 480x8 - and the
`in2_moe_batch` call at fwd.c:1951 sits between the block arriving and the work happening. So the remaining question is
narrow and mechanical: **inside `in2_moe`/`in2_moe_batch`, are the block's tokens gathered by expert (M = rows per expert,
which is what the 8-column tile needs) or dispatched one at a time?**

That is now the only unknown, and it is a single function to read with the block size already confirmed at 120. If the
gather exists, the work is wiring the GPU kernel to it. If it does not, the work is adding the gather - and the earlier
sizing says a 128-token chunk already yields mean M = 8 per expert, which is exactly what arrives here.

Also: this closes the loop on three consecutive notes. The layer count, the export table, and this all resolved the same way
- a thirty-second trace or file query beat three rounds of reasoning about the code. The trace should have been step one.

## The grouped-GEMM structure already exists - and it is the thing that measured -6 %

`in2_moe_batch` (fwd.c:1757) is, already, exactly what the plan said needed building:

    CH = min(n_tok, 512)                                        chunk the block
    counting sort of (token,slot) by expert  (cnt/start/fill/bt/bk)   <- the inverted index
    #pragma omp parallel for over experts, schedule(dynamic)
      gather the expert's whole bucket into Xe (nb rows, contiguous)
      in2_matvec_quant_nt(ctx, &L->gate_up, e*2*DFF*d, Xe, nb, 2*DFF, d, Ge)      <- M = nb
      ... activation, then the same for L->down

So: inverted index by expert, expert-major dispatch, one slab read per expert per chunk, M = nb rows per expert, and the
chunking comment records a measured 19 % gain from moving 320 tokens into a single 512-chunk. With CH=512 and 512 tokens,
nb is ~32 per expert - well past the 8 the tile needs.

**Which means the -6 % measurement from three notes ago was taken at exactly this shape.** The hook I added sat in the
group-of-8 loop of `in2_matvec_quant_nt`, which is the MoE expert path, with nb rows per expert and the 8-column tile
selected - and it lost to the CPU. That result now has to be read as: *at M = 8-32 per expert, with 8 OpenMP threads each
already running an expert-major bucket, the RTX 4050 loses to the CPU.* The 20x projection assumed a naive per-token CPU;
the CPU here is not naive, and that is the whole discrepancy.

Revised position, replacing the earlier plan:
- the token-blocked MoE stage: **done** (it was never missing);
- the grouped GEMM over an inverted index: **done**, on the CPU, in parallel;
- what is missing is GPU economics at nb = 8-32: per-call H2D of the activation block and D2H of results across
  ~6,000 calls per 512-token chunk (24 layers x 128 experts x 2 matrices) eats the kernel's advantage.

So the next question is not "build the gather" but "**is there a shape where the GPU's 400-700 GMAC/s survives the round
trips**", e.g. keeping gathered activations resident across all 24 layers and issuing one launch per layer instead of one
per expert. Anything that still returns to the host per expert cannot win, and that is now measured rather than predicted.

## The prefill cost structure, measured by quant type (this supersedes the MAC-share estimates)

`IN2_TIMETY=1`, 120-token prompt. Per-type wall time, share, and realised throughput:

    TY  8  calls= 6615   1.356 s   6.1 %    76.49 GOPS
    TY 12  calls=  525   0.064 s   0.3 %   124.42 GOPS
    TY 13  calls=  585   0.093 s   0.4 %   113.78 GOPS
    TY 14  calls= 1590   0.325 s   1.5 %    91.58 GOPS
    TY 16  calls= 1016   3.746 s  16.9 %    23.13 GOPS
    TY 17  calls=  946   3.573 s  16.1 %    22.82 GOPS
    TY 18  calls=   45   0.021 s   0.1 %   121.34 GOPS
    TY 19  calls=  524   3.257 s  14.7 %    12.52 GOPS
    TY 22  calls=  366   1.786 s   8.1 %    19.97 GOPS
    TY 23  calls=  25   0.084 s   0.4 %     5.51 GOPS
    TY 40  calls=  865   2.881 s  13.0 %    13.27 GOPS
    TY 42  calls= 1986   4.959 s  22.4 %    16.95 GOPS

Reading it:

- **Five types carry 83 % of prefill**: TY 42 (22.4 %), TY 16 (16.9), TY 17 (16.1), TY 19 (14.7), TY 40 (13.0). Only TY 8
  (Q8_0, 6,615 calls) is fast, at 76 GOPS - many small ops, so call overhead is not the problem; the *big* quantised
  matvecs are simply slow per operation.
- Realised rates are 12.5-23.1 GOPS = **6.3-11.6 GMAC/s per call stream**, i.e. the engine is running the same handful of
  kernels well below their own ceiling (the batched CPU probe measured 11.93 GMAC/s single-stream and the GPU kernel
  measured 373-648 GMAC/s).
- **TY 19 is the worst offender per byte moved** (12.52 GOPS) and TY 42 has the largest share by far (4.96 s of ~22 s).

What this changes: every MAC-share argument earlier in this document (the 0.8 %, the 83 %, the ~20x) was a share of
*operations*. The engine's time is not distributed like its operations - it is distributed like this table. **Optimise TY 42
and TY 16/17 first; they are 55 % between them.** The identity of each type in terms of which tensors it serves (dense
attention projections vs expert gate_up/down) is the one mapping still to confirm, and it is a one-line print away - after
which the decision of what to move to the GPU, if anything, is made on measured time rather than assumed proportions.

## Type mapping, part one: the dominant cost is the expert matvecs, and they are running at single-thread rates

`IN2_TRACE_NT=1` (which prints name-free but shape+type for every `in2_matvec_nt` call) on a 60-token prompt:

    ty= 8   n_in=2304  n_out=1024 / 3072 / 384     kernel
    ty=12   n_in=2304  n_out=1024 / 3072 / 384     kernel
    ty=13   n_in=2304  n_out=1024 / 3072 / 384     kernel
    ty=13   n_in=3072  n_out=2304                  kernel
    ty=14   same shapes                            kernel
    ty=18   n_in=2304  n_out=3072                  kernel
    ty=30   n_in=2304  n_out=24                    FLOAT-FALLBACK  (the 24-wide router-ish tensor)

So the **attention projections are ty 8 / 12 / 13 / 14 / 18** (Q8_0 and the K-quants). None of those are the five expensive
types from the profile. The expensive ones - TY 42 (22.4 %), TY 16 (16.9), TY 17 (16.1), TY 19 (14.7), TY 40 (13.0) - do not
appear in this trace at all, which means they belong to the tensors that go through the **expert path**
(`in2_matvec_quant_nt`), i.e. `gate_up` and `down`.

That is consistent with the profile in one important way and contradicts it in another:

- consistent: the expert matrices are the bulk of prefill time (55 % across TY 16/17/19/40/42);
- contradicts the plan's premise: those calls realise **12.5-23.1 GOPS = 6.3-11.6 GMAC/s**, which is *single-thread* rate, not
  the ~8x that an `omp parallel for` over 128 experts should give. `in2_moe_batch` is written expert-major and parallel, yet
  the measured aggregate throughput says the parallelism is not being realised.

So the next question is not about the GPU at all: **why does the expert-major loop over 128 experts deliver single-thread
throughput?** Candidates, each cheap to test: `in2_threads()` returning 1 in this build; the dynamic schedule starving threads
because per-expert buckets are small at 60 tokens; or a shared lock/buffer inside the fallback quantiser serialising them
(the thread-local comment at that buffer suggests this has bitten before).

That question is worth more than the entire GPU line of work: 55 % of prefill, currently running at roughly one core's worth
of throughput.
