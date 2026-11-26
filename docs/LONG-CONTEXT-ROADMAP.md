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
