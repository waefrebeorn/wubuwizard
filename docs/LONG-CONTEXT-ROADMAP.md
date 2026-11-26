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

## The engine's real limiter: thread scaling is 2.29x, not 8x

60-token prefill, `IN2_THREADS` swept:

    1 thread   14,700 ms
    2 threads  11,486 ms   1.28x
    4 threads   7,661 ms   1.92x
    8 threads   6,415 ms   2.29x

So the expert-major MoE loop and everything else do parallelise - OpenMP is linked (19 `gomp_parallel`, 110 GOMP symbols)
and the pragmas execute - but 8 threads buy **2.29x, not 8x**. That single number explains every throughput observation in
this document:

    per-thread ceiling      ~11.9 GMAC/s  (measured, batched CPU probe)
    ideal 8 threads          ~95 GMAC/s
    observed (2.29x)         ~27 GMAC/s    <- matches the 12.5-23.1 GOPS per-type profile exactly
    GPU IQ2 kernel           373-648 GMAC/s

The engine is not 4-30x off the GPU because of the kernel; it is 2.3x below its own core count, and then the GPU is another
order of magnitude beyond that. **The first prize is thread scaling, not the GPU: 2.29x -> 6x would take prefill from 6.4 s
to ~2.4 s at 60 tokens, roughly halving it, with no new kernel and no data movement.**

Why scaling is this poor is now the single question worth answering. Candidates in order of likelihood:
- memory bandwidth: every expert matvec streams its slab from RAM, and 8 threads all streaming IQ2 slabs can saturate the
  channel long before the cores are busy - a bandwidth ceiling would look exactly like this;
- four `malloc`/`free` pairs per expert inside the parallel region (128 experts x 24 layers per chunk = ~12,000 allocations);
- the shared fallback quantiser buffer, whose thread-local comment records this exact class of bug before.

Each is testable without changing the maths: bandwidth by comparing aggregate GOPS against measured DRAM rate, malloc by
hoisting the four buffers out of the expert loop, the buffer by checking whether the thread-local guard is still conditional.

Note also that `in2_threads()`'s own comment claims "~3.5x for the same kernels under a persistent pool" - the code was
written expecting better scaling than the 2.29x measured here, so something has regressed since that measurement.

## Allocation traffic ruled out as the scaling limiter

Replaced the four per-expert `malloc`/`free` pairs inside the parallel MoE loop (~12,000 allocations per 512-token chunk,
all through one CRT heap lock) with per-thread reusable scratch, grown once and never freed:

    1 thread   14,727 ms   (was 14,700)
    8 threads   6,411 ms   (was 6,415)   -> scaling 2.30x, was 2.29x
    oracle     12/12 exact

**No effect whatsoever.** So heap contention is not what holds 8 threads to 2.3 cores. Reverted - a flat change stays out
of the hot loop.

That leaves memory bandwidth as the leading explanation, and it is testable arithmetically rather than by another build:
the expert matvecs stream IQ2_4/Q4-class slabs, so at ~27 GMAC/s aggregate the weight traffic is ~8-15 GB/s. On a dual-channel
DDR5 laptop that is well under the ~80 GB/s ceiling, so *if* the arithmetic is right, bandwidth is NOT it either - which
would leave the third candidate: that the work per expert bucket is simply too small to schedule (nb ~ 4 rows at 60 tokens,
so 128 tiny tasks with dynamic scheduling and per-task overhead dominating). That one is testable by the cheapest possible
experiment: run the same sweep at 480 tokens where nb ~ 30, and see whether scaling improves. If it does, the fix is chunk
size, not threading and not the GPU.

## Granularity prediction refuted: scaling gets WORSE with bigger expert buckets

The test said: if 128 tiny dynamic-scheduled tasks were the problem, 480 tokens (nb ~ 30 rows per expert instead of ~4)
should scale better than 60 tokens. Measured:

    60 tokens   1 thread 14,700 ms   8 threads  6,415 ms   -> 2.29x   107 ms/token
    480 tokens  1 thread 124,397 ms  8 threads  66,517 ms  -> 1.87x   138 ms/token

Scaling **falls** from 2.29x to 1.87x as the buckets get larger, and per-token cost rises from 107 to 138 ms. That is the
opposite of the prediction, so per-task scheduling overhead is not the limiter either - it is ruled out the same way
allocation was.

Both observations point at a **shared resource that gets worse with more work in flight**, not at granularity:
- more work per bucket -> worse scaling is the classic signature of a saturating shared resource (memory channel or cache),
  not of scheduling;
- per-token cost rising 107 -> 138 ms at 4x the context is consistent with attention's O(N) KV traffic per token being a
  growing share, so two different effects are mixed inside these two numbers.

Which means this sweep cannot separate them, and nothing further should be concluded from it. The phase counters already
exist for exactly this (`in2_moe_sec`, `in2_lay_sec`, `in2_full_sec`, `in2_slide_sec` under IN2_TIMETY): reporting MoE time
against layer time at both context lengths separates attention growth from MoE scaling in one run, with no new code.

Three hypotheses eliminated by measurement now - allocation (fixed, flat), granularity (predictions reversed), and with them
the assumption that the expert loop's parallelism was the problem. The engine's 2.29x is looking less like a bug and more
like a bandwidth-and-arithmetic-intensity ceiling that this hardware imposes on IQ2 matvecs at batch sizes below 8.

## Phase separation: attention is 78-80 % of layer time, MoE is 29-41 %, and MoE is the part that scales fine

`IN2_TIMETY=1`, same prompt at two lengths, 8 threads:

                        60 tokens        480 tokens     growth
    layers total         5.713 s         64.204 s       11.2x
      full (ar-block)    1.264 s (22 %)  13.067 s (20 %)  10.3x
      sliding            4.449 s (78 %)  51.137 s (80 %)  11.5x
      f_mlp+moe          0.486 s          3.702 s         7.6x
      moe                2.325 s (40.7 % of layers)  18.335 s (28.6 %)  7.9x

Normalised per token:

    moe       38.8 ms/token at 60   ->  38.2 ms/token at 480   **flat** - linear in tokens, exactly as it should be
    sliding   74.2 ms/token at 60   -> 106.5 ms/token at 480   **grows with N**

So, finally, the cost structure of this engine at prefill:

1. **The sliding-window attention path is 78-80 % of all layer time** - 51.1 s of 64.2 s at 480 tokens. Nothing else is close.
2. **MoE is 28.6-40.7 %** and is per-token flat, so the expert path and its threading are healthy; the 2.29x scaling number
   was measuring a minority of the work all along.
3. The dense projections that the whole GPU line of work chased are inside the 22 % `full` term.

MoE also *shrinks* as a share (40.7 % -> 28.6 %) because attention grows faster than it does. Every GPU effort this session -
the batched resident hook, the MoE batching, the grouped-GEMM sizing - was aimed at between 22 % and 41 % of the time while
the 80 % was the KV decode-on-read path identified much earlier: `in2_kv_k`/`in2_kv_v` decode each cached row on every
access, with 64 sliding visits x 1024 rows per visit-group and 576 MB of KV in the 60-token case alone.

**The next step is the sliding attention path, and nothing else.** It is 4 of every 5 seconds of prefill, its per-token cost
rises with context, and the fix (decode each KV row once into an f32 mirror per visit, instead of per access) is already
described in the chunked-prefill note. That work was projected to take 262k prefill from 57,016 h to ~380 s; this profile
says it is also the single biggest win available at 480 tokens, where nobody was looking.

## Sliding-attention fix: exact shape, and the 288 MB trap that would sink it

The accessors return ONE dequantised row per array (`kscr`/`vscr`, header comment says so explicitly), so every
`in2_kv_k(kv, vvis, j)` inside the attention loop re-decodes row j. The query loop then does it again. With 64 sliding
visits x 1024 window rows x 1024 query rows that is ~7.5e7 decodes of a 1024-point TurboQuant vector plus a 1024-point WHT
each - which is the 51.1 s of 64.2 s that the phase profile attributes to sliding attention.

The fix is a mirror, but the SIZE decides whether it exists at all:

    mirror over every row:   72 visits x 1024 pos x 1024 dim x 4 B = 288 MB     <- alongside a 7.9 GB model, no
    mirror for ONE visit:            1024 pos x 1024 dim x 4 B   =   4.0 MB     <- fits trivially
    stamp array for one visit:                                       4.0 KB
    decode reduction:                                                1024x

So it must be **window-sized, not row-total-sized**: `float *kmir` of `window_rows * dim` plus `int *kitag` of
`window_rows`, where `kitag[p]` holds the visit id that populated that slot (-1 = empty). In `in2_kv_k`:
`if (kitag[pos] == visit) return kmir + pos*dim;` else decode once, copy in, stamp it. Same for v. Because the tag carries
the visit id, a new visit invalidates every slot for free - no clearing, no per-visit reallocation.

Why this is the right next change and not another experiment:
- it attacks 78-80 % of prefill time, measured twice, instead of the 22-41 % every GPU attempt targeted;
- it needs 4 MB and no kernel work, no PCIe, no quantisation change;
- its effect is directly observable in the existing `PHASE sliding=` line, so the gate is one number before and two after;
- the trap above is the kind of detail that silently turns a 4 MB change into a 300 MB one, and it is now written down.

Not implemented here: it touches the KV struct in `include/wubu_inkling2_kv.h`, both alloc sites in `in2_kv_init`
(~:151 and ~:159), `in2_kv_free`, and the two accessors - five edit sites with a build-and-gate cycle, which needs room to
verify that this session no longer has. The oracle (12 tokens, exact) is the gate to re-establish before and after.

## IMPLEMENTED: the per-visit KV mirror, and it is a 4.8x win in compressed-KV mode

Five edits as specced (struct in `include/wubu_inkling2_kv.h`, both alloc sites, `in2_kv_free`, both accessors), window-sized
mirror with visit-id tags so invalidation is free. Oracle stays exact in every configuration tested (12/12, 94-byte IDS).

    60-token prefill, PHASE timings:

    f32 KV (default)     layers 5.713 -> 5.661 s     sliding 4.449 -> 4.396 s     (flat, 1.2 %)
    TQ-6 KV              layers ~28.7 s -> 5.932 s                                (4.8x)
                         sliding 4.642 s, moe 2.340 s, oracle exact

The two results together are the important part:

1. **In the default f32 configuration the mirror changes nothing (1.2 %)** - because TurboQuant is off, `kv->tq` is NULL, and
   the decode-on-read path the mirror targets is not active. The 78-80 % sliding cost there is ordinary attention arithmetic
   and f32 KV traffic, not decode. My earlier attribution of that 80 % to TurboQuant decode was wrong for the default config.
2. **In compressed-KV mode it is a 4.8x prefill win** (28.7 s -> 5.93 s at 60 tokens) and TQ-6 is now within 4 % of f32
   (5.932 vs 5.661 s) while holding the 12.06 KB/token footprint. That is the configuration the long-context roadmap needs,
   and until now its prefill cost was prohibitive.

So the honest statement of what this change does: it does not speed up the engine as configured today, and it makes the
memory-efficient configuration viable. The 78-80 % sliding share at f32 remains unexplained and is the next thing to profile,
with `in2_kv_k` now proven not to be its cause in that path.

## The 78-80 % is call overhead, not arithmetic: 3,072 visit-calls per token, 23 us each

Sliding time per token is constant across context, which rules out fixed-window waste:

    28 tokens  sliding 1.987 s  ->  71.0 ms/token
    60 tokens  sliding 4.280 s  ->  71.3 ms/token

So the cost is linear in tokens and the window is not being scanned beyond the live positions. The inefficiency is per CALL:

    layers x sliding visits      48 x 64      = 3,072 visit-calls per token
    71 ms / 3,072                              = 23 us per call
    MACs per call at pos=28      28x1024x2     = 57 K MAC  -> 2.5 GMAC/s effective
    MACs per call at pos=60      60x1024x2     = 123 K MAC -> 5.3 GMAC/s effective

2.5-5.3 GMAC/s is what a call-bound loop looks like: each call moves 57-123 K MACs, which is far too little work to amortise
3,072 of them per token. By comparison the same engine's quantised matvecs run at 12-27 GMAC/s and the GPU kernel at
373-648 GMAC/s, so the sliding path is 5-10x below the engine's own modest ceiling.

**So the fix is batching, not caching:** one call covering all 64 visits (or all tokens) per layer turns 3,072 calls per token
into 48, and the per-call cost disappears into a single larger, better-shaped GEMV/GEMM. That is a different change from the
KV mirror just landed - and unlike the mirror, its benefit applies in the DEFAULT f32 configuration, which is where the
78-80 % actually lives.

This also retroactively explains the earlier chunked-prefill projection: the O(N^2) decode-on-read analysis was right about
the TQ path (where the mirror delivered 4.8x) but the dominant cost in the shipped configuration was always the call count.

## Hardware check closes the loop: not bandwidth, not the kernels - it is fork/join per visit-call

    CPU       Ryzen 5 8645HS, 12 logical processors (6 cores + SMT)
    RAM       2 x 8 GB DDR5-5600, both slots populated -> dual channel, ~90 GB/s theoretical
    GPU       RTX 4050 Laptop 6 GB

Sliding attention traffic at 60 tokens: 64 visits x 60 pos x 1024 dim x 4 B x 2 (K and V) = 31.5 MB per token, which over
71 ms is **443 MB/s** - three orders of magnitude below what dual-channel DDR5-5600 can carry. **Bandwidth is not the limit.**

Thread scaling of the same path:

    1 thread   sliding 11.532 s   full 2.414 s
    8 threads  sliding  4.296 s   full 1.257 s     -> 2.68x and 1.92x

On a 6-core/12-thread part the ceiling is ~6x, so the engine reaches less than half of it. Combined with the per-call figure
(3,072 visit-calls per token, 23 us each, 57-123 K MAC apiece) the only explanation left standing is **OpenMP fork/join
executed per visit-call**: each call is far too small to repay a parallel region, so the pool cost is paid 3,072 times per
token, and that is simultaneously why the per-call rate is 2.5-5.3 GMAC/s and why scaling stalls below the core count.

**So the fix is to hoist the parallel region out of the visit loop** - parallelise over visits (64 independent) or over tokens
at the layer level, so the fork/join happens once per layer instead of 3,072 times per token. That is a restructure of the
attention loop, not a kernel, not a cache, and not the GPU.

This is also the first explanation in the whole session that is consistent with every measurement taken:

    - allocation hoisting      : no effect (the allocs were not the shared cost)
    - granularity/chunk size   : scaling got WORSE with bigger buckets - consistent with fixed per-region overhead
    - MoE per-token flat       : it uses one parallel region per expert-batch, not per call
    - every path ~2-2.7x       : the same fork/join tax appears wherever calls are small

Next step: count the parallel regions entered per token (a counter beside the fork sites) and confirm it is thousands, then
hoist. That counter is a five-line change and it converts the last remaining hypothesis into a number.

## CORRECTION: "sliding" is a layer CLASS, not the attention op - and it has no OpenMP at all

Two facts that together invalidate the last two notes' framing:

1. `grep -n "pragma omp" src/wubu_inkling2_fullseq.c` returns **nothing**. The full-sequence path, which is where the
   `in2_kv_k`/`in2_kv_v` attention loops live (fullseq.c:239, :301, :319), is entirely **serial**.
2. Yet `PHASE ... sliding=` fell from 11.532 s to 4.296 s with 8 threads - a 2.68x drop in a "serial" path.

Both are true only if `sliding` is not measuring attention. And it is not: the phase timer splits LAYERS by
`(il % attn_res_block == attn_res_block - 1)` into `full` and `sliding`, i.e. **`sliding` is the class of sliding-attention
LAYERS, and its timer covers everything those layers do** - projections, MoE, and attention together. The `moe=` line is a
subset of it, not a sibling.

Corrected arithmetic at 60 tokens (8 threads):

    layers total        5.553 s
      of which moe        2.305 s   (41.5 % of layers - a subset, printed separately)
      full-class layers   1.257 s
      sliding-class       4.296 s   of which moe ~2.3 s and matvecs the rest
    attention proper    = sliding-class minus moe minus projections ~ 1.5 s, and it is SERIAL

So three claims from the last two notes were wrong and are hereby retracted:
- **"attention is 78-80 % of layer time"** - false; that number is the share of layers classified as sliding, and most of it
  is MoE plus projections running inside them;
- **"3,072 visit-calls per token at 23 us shows call overhead"** - the 23 us figure divided the whole sliding-class time by
  the visit count, mixing MoE and projections into a per-attention-call number. It measured nothing;
- **"fork/join per visit-call"** - there is no fork/join in that file at all. The hypothesis is dead on inspection.

What survives, and is now the honest position: MoE is 41.5 % of prefill and per-token flat (healthy); the full/sliding split
is a layer classification; attention proper is serial and costs roughly 1.5 s of 5.55 s at 60 tokens; and thread scaling of
about 2.3-2.7x applies to the engine as a whole. The serial, unaccelerated attention loop is a real and bounded target -
it is the only significant work in the engine that has no parallelism at all - but it is 27 % of prefill, not 80 %.

Lesson, fourth time in this session: I read a label (`sliding`) as if it described the work it contains. A phase named after a
layer's attention type does not measure that attention - it measures the layer.

## Parallelising the serial attention loop: the dimension is `t`, and four scratches block it

Structure of the full/attention layer (fullseq.c:273-300 and on):

    for (int t = 0; t < n_tok; ++t) {            <- serial outer loop, the natural parallel dimension
        in2_matvec_nt(&L->wrel, xq + t*d, 1, d, NH*DREL, rs);   <- rs: SHARED scratch row
        in2_l2_clip(rs, ...); in2_rel_bias_table(rs, P, NH, DREL, EXT, bias);   <- bias: SHARED scratch
        in2_matvec_nt(&L->sink, xq + t*d, 1, d, NH, sink + t*NH);              <- sink: per-token, safe
        for (int hh = 0; hh < NH; ++hh) { ... for (int j = jstart; j <= gp; ++j) { ... } }
    }

`#pragma omp parallel for` on `t` is the right move: every token writes its own slice of `attn_out` (indexed `t*qdim`) and
its own `sink + t*NH`, so those are race-free by construction. What blocks a one-line parallelisation is the scratch:

- `rs` - ONE row of `NH*DREL` reused for every token, written then immediately consumed. Needs a per-thread copy
  (`NH*DREL*4` bytes, 24*24*4 = 2.3 KB) or an `rs + t*NH*DREL` layout.
- `bias` - written by `in2_rel_bias_table(rs, P, NH, DREL, EXT, bias)` per token, same problem, and bigger
  (NH*EXT entries).
- `logit`, `alpha`, `attn_out` - `attn_out` is per-token and safe; `logit`/`alpha` must be checked for the same single-buffer
  pattern before the pragma goes in.

So the change is: give `rs` and `bias` per-token or per-thread storage (a `#pragma omp parallel` outermost with
`firstprivate` pointers to thread-local scratch, or index them by `t`), then farm `t`. The arithmetic is identical, so the
12-token oracle is a full check on it - any race shows up as a changed IDS line immediately, which is the cheapest possible
detector and the reason this is safe to attempt.

Why it is worth doing: this loop is the only significant work in the engine with no parallelism at all, it is ~27 % of
prefill (1.5 s of 5.55 s at 60 tokens), and on 6 physical cores it should yield 3-5x on that component - the largest
remaining single win now that MoE is known to be healthy, attention's call structure is not the issue, and the GPU has been
ruled out four separate ways.

Not attempted here: per-thread `rs`/`bias` plumbing plus verifying no other shared scratch exists is a multi-site change
inside the hottest, most parity-sensitive loop in the engine, and the last few turns have shown how quickly a label or a
grep can mislead. It needs a session with room to build, run the oracle, and revert.

## Parallelisation attempt: reverted, and the reason is mechanical

Tried the `omp parallel for` over tokens with per-thread `rs`/`bias`/`logit`/`alpha`, scoping the rewrite to the loop body by
brace counting from `for (int t = 0; t < n_tok; ++t) {`. The build failed:

    fullseq.c:324: 't' undeclared        fullseq.c:324: 'hh' undeclared
    fullseq.c:325: 'n_key' undeclared    fullseq.c:325: 'logit' undeclared

The brace scan stopped early, so the tail of the loop ended up outside the `for` - i.e. the script cut the loop at the wrong
`}`. The cause is that a naive character scan counts braces inside comments and string literals, and this loop is dense with
both (multi-line comments containing `{ }`, and the IN2_FULL_KERNEL note). Any attempt at this rewrite needs a depth
tracker that strips comments and string literals first, or - better - a hand-scoped edit against a printed line range.

Reverted immediately: full rebuild, oracle 94-char IDS exact, `git status` clean.

One thing worth noting: during the failed build `gen_fast.exe` did not exist and the oracle check returned an EMPTY string.
Because the check now prints the length, that was visible as `len=0` rather than passing as a match - the fourth time this
session that requiring a non-empty reference caught what an equality test would have reported as success.

## Upstream PR #33049 (llama.cpp) - what it means for this repo

"spec : fix DFlash2 draft acceptance for Inkling 2 Mini" (n-belokopytov, adapted from renoirvieir4/llama.cpp 0280b55).
Reported: code 51.0 -> 158.0 tok/s, prose 51.2 -> 85.7 tok/s on the same model this repo targets.

Three claims in it, checked against this engine:

1. **token embedding norm** - "When DFlash2 borrows token embeddings from an Inkling 2 or Inkling 2 Loop target, apply RMS
   normalization with the target's epsilon and learned tok_norm weight." This repo already does it:
   `wubu_inkling2_model.c:99` loads `token_embd_norm.weight` and `wubu_inkling2_load.c:65` documents it as "the embedding
   norm the reference applies before layer 0". **No action needed.**
2. **visits** - the PR defines a visit index as "a particular execution of a layer, which matters when a model runs the same
   layer more than once", and Inkling Mini's drafter wants hidden states from visits [5, 23, 41, 48]. This repo already has
   the same model (`n_visit`, the 72-visit rule in `in2_kv_init_pervisit`). **Consistent, no action needed.**
3. **selector softcap order** - "Inkling expects softcap(logit + correction). The default order computes
   softcap(logit) + correction. The function is nonlinear, so the order can change which token the drafter proposes." This
   repo has no logit softcap and no drafter: `in2_softcap` appears only in the SwiGLU FFN (:983). **Not applicable** - the
   issue lives in the speculative-decoding path, not in the model forward.

So nothing in the PR contradicts this engine's forward pass, and two of its three details are already implemented here. But
the PR is strategically important for a different reason:

**The 3x came from draft acceptance, not from a faster kernel.** 51 -> 158 tok/s is an algorithmic win from speculative
decoding that proposes several tokens and verifies them - and greedy verification preserves token-for-token identity, which
is exactly this project's parity invariant. Meanwhile this session spent its effort on kernels and threading and produced
(a) a 1.7x kernel that moved the engine 2 %, and (b) a 4.8x win only in the TQ configuration. A working drafter is a
1.7-3.1x on decode at unchanged outputs, which is a larger lever than anything measured here.

Recorded as the strategic finding of this session: if the goal is "same tokens, faster", the speculative path is worth more
than another kernel, and it does not violate the parity invariant.

## The attention parallelisation landed and measures a real (small) win - and it was mis-bundled

The hand-scoped edit (lines 274-324, per-thread rs/bias/logit/alpha, `omp parallel` + `omp for` over tokens) built clean and
kept the oracle exact (94-char IDS). Measured against the pre-change sweep:

    1 thread   layers 13.946 -> 13.835 s    full 2.414 -> 2.278 s    sliding 11.532 -> 11.557 s
    8 threads  layers  5.553 ->  5.382 s    full 1.257 -> 1.100 s    sliding  4.296 ->  4.282 s
    60-token wall clock            6,415 -> 6,174 ms               (-3.8 %)

So: **full-class layers improved 12.5 %, sliding-class were unchanged, wall clock 3.8 % faster**, oracle exact. The reason
sliding did not move is that this loop is the `full` branch - the parallelisation was applied where the code was, which is
the full-attention/ar-block layer, not the sliding layers. That is consistent with the earlier correction that `sliding` is a
layer class and its cost is mostly MoE, which is already parallel via a different path.

Kept, because it is a measured improvement with an unchanged oracle. But the honest size of it is 3.8 %, not the 3-5x that
would have followed had attention really been 27 % of prefill and fully serial - which means **attention proper is a much
smaller share than the phase arithmetic suggested**, and the earlier "27 % serial attention" estimate should be treated as an
upper bound that this measurement does not support.

**Process error, recorded so it does not repeat:** this code change was committed inside `80e1b16`, whose message is about
assessing a llama.cpp PR - because that commit used `git add -A` while the edit was sitting in the tree. The change is real
and tested, but it is filed under a message that does not describe it. Use explicit paths on `git add` whenever a working
tree may contain an in-flight edit, rather than `-A`.

## Fine-grained breakdown (IN2_TIMETY, 60 tokens, 8 threads) - and `q` is the surprise

    PHASE layers=5.311 s   full=1.090 s  sliding=4.221 s

    mix        0.042 s    0.8%
    q          0.861 s   16.2%   <- single projection, second largest component in the engine
    k+v+conv   0.208 s    3.9%
    rel+sink   0.084 s    1.6%
    attn       0.400 s    7.5%   <- attention proper really is small, as the 3.8 % parallelisation result implied
    gate+wo    0.359 s    6.8%
    resconv    0.011 s    0.2%
    moe        2.226 s   41.9%

Ranked: **MoE 41.9 %, q 16.2 %, attention 7.5 %, gate+wo 6.8 %, kv+conv 3.9 %**. Everything else is noise.

Two things follow:

1. **Attention is 7.5 %, not 27 %.** The parallelisation gained 3.8 % because there was only 7.5 % to gain from, and
   `attn` was never the story. Third correction to that estimate, now with the direct number behind it.
2. **`q` at 16.2 % is anomalous.** `wq` is one projection per layer per token, similar in shape to `gate`/`wo` which together
   cost 6.8 %. If they were all batched the same way, `q` should be in that ballpark, not 2.4x larger on its own. The `q`
   path calls `in2_matvec_nt(..., 1, d, qdim, q)` with nt = 1 - one token at a time - while the MoE and gate paths hand over
   whole buckets. **A per-token nt=1 projection that cannot amortise its own row-split is exactly the shape that measured
   2.5-5.3 GMAC/s in the KV analysis.**

So the next concrete step is cheap and specific: batch the `q` projection across tokens the same way the gate/wo path is
batched, and re-read `q` in this same table. Expected: 0.861 s -> ~0.36 s, i.e. ~9 % off total prefill, with the 12-token
oracle as the gate. If it does not move, the bottleneck inside `q` is something else and the table will say so on the next run.

For completeness, the honest ranked plan this table implies: MoE (41.9 %, already per-token flat and parallel), then `q`
(16.2 %, concrete and cheap), then attention (7.5 %, partly parallel now). The GPU line of work addressed none of the top
three and is closed.

## `q` is single-threaded, to the decimal - and that is 9 % of prefill for free

Batching the `q` projection (hoisting it out of the per-token loop into one `in2_matvec_nt(ctx, &L->wq, xq, n_tok, d, qdim, q)`
call) built clean and kept the oracle exact, then measured **no change at all**: 0.857 s against 0.861 s. Reverted - a flat
change does not stay.

But the arithmetic identifies the real fault precisely:

    q work = 60 tokens x 24 full layers x 3072 qdim x 2304 d = 10.19 GMAC
    measured 0.857 s                                        = 11.9 GMAC/s
    single-thread ceiling (measured separately)              = 11.9 GMAC/s

**Exactly one thread's throughput.** The engine's parallel paths reach ~27 GMAC/s, so `q` is running serial while everything
around it is not. At 27 GMAC/s the same work is 377 ms - **a 480 ms saving, 9 % of total layer time**, from removing a
serial section rather than from any new algorithm.

Why batching did not fix it is the open question, and it is now sharply posed: the row-split helper has an early-out
`if (nt <= 1 || n_out < 2 * nt) return in2_matvec_quant(...)` which a per-token call always takes - so before this change `q`
could never parallelise - yet passing nt = 60 did not engage it either. So either the split is not reached for this tensor's
path (type/`in2_ty_verified` sending it to a float fallback), or the split itself is serial for this shape. Both are one
instrumented print apart, and `q` is the second-largest component in the engine after MoE.

Ranked, measured, and unchanged by anything in the GPU line of work:

    moe  41.9 %   per-token flat, parallel       - healthy
    q    16.2 %   SINGLE-THREADED, 9 % recoverable
    attn  7.5 %   partly parallelised this session (3.8 % overall gain)
    gate+wo 6.8 %

## Fourth attempt at `q`: parallelising the group loop in in2_matvec_nt changed nothing

Added `#pragma omp parallel for schedule(dynamic) reduction(&:done)` to the 8-token group loop, which is independent per
group (each writes only `Y + (g+i)*n_out`, reads the shared already-quantised `act`). Built clean, oracle exact, and:

    before (batched q, reverted)   layers 5.334 s   q 0.857 s
    after  (group loop parallel)   layers 5.516 s   q 0.865 s     -> no gain, 3 % worse

Reverted. The most likely reason, and it is an important one for anything attempted next: **OpenMP does not nest by
default.** `in2_matvec_quant` and the row-split helpers inside this loop already create regions; an outer
`parallel for` therefore runs with one thread and its inner regions are serialised, so the pragma adds fork/join cost and
buys nothing. Any future attempt at this loop has to either set `omp_set_max_active_levels` / `OMP_NESTED` and measure
whether nesting actually helps here, or restructure so there is exactly ONE parallel region and the inner calls are
explicit serial kernels.

Four attempts now at the second-largest component in the engine, all revertible and all reverted:

    1. batch q across tokens (nt = 60)              -> no change
    2. residual-claim: q is call-bound              -> refuted (batching would have moved it)
    3. parallelise the group loop                   -> no change, and nesting is why
    4. (earlier) parallelise the attention loop      -> 3.8 % overall, kept

What `q` needs is a measurement I still do not have: whether `in2_matvec_quant` for `wq`'s type reaches the kernel at all,
or falls to the float path. The `IN2_TRACE_NT` line said `ty=8 ... -> kernel`, but that trace is emitted before the branch,
not by it. Printing inside the taken branch - once per type - would settle it, and that is the next step rather than another
speculative pragma.

## `q` takes the KERNEL path - confirmed by the branch, not by a pre-branch trace

Added an `IN2_PATH_TRACE` print INSIDE the fast path of `in2_matvec_nt`, so it fires only when that branch is the one that
executes (the old `IN2_TRACE_NT` line printed before the branch and therefore never proved anything):

    NT-PATH ty=8  n_in=2304 n_out=3072 nt=60 -> KERNEL path     <- wq, the 16.2 % component
    NT-PATH ty=14 n_in=2304 n_out=3072 nt=60 -> KERNEL path
    NT-PATH ty=13/12/18 ...                  -> KERNEL path
    (and ty=8 n_out=1024 for wk/wv)

So the float-fallback hypothesis is dead: `wq` reaches the quantised kernel. That closes the question the last note posed
and leaves exactly one:

**The work is 8 groups x 8 rows x (3072x2304) = 10.19 GMAC, it measures 0.857 s = 11.9 GMAC/s = one thread, the outer
`parallel for` over those 8 groups was ACCEPTED by the compiler (no pragma warning - checked, for the first time in this
session), and it changed nothing.** Three candidate explanations remain, in order:

1. `omp_get_num_threads()` inside that region is 1 - the region is running inside an already-active outer parallel
   context (nesting disabled), so my `parallel for` became a serialised inner region despite being a separate pragma;
2. the inner `in2_matvec_quant` calls use a private persistent pool that serialises when called from multiple threads;
3. the eight groups are simply not the cost - the 0.857 s is dominated by something in the loop body other than the
   matvec calls (the per-row activation quantisation over `nab` blocks, or the D2H of `op[]`).

Printing `omp_get_num_threads()` once inside that region discriminates (1) from (2)/(3) in a single run, and it is the only
remaining step for this component. It is worth one more turn because `q` is 16.2 % of prefill and the recoverable part was
measured at 480 ms - but nothing further should be tried before that number exists.

Also recorded: this session never checked compiler WARNINGS until now, only `" error"`. That is how a silently-ignored
pragma could have hidden - it did not, but the check should be part of the build loop from here.

## `q` is not a threading problem, and the label is again wider than the thing

Two measurements killed the threading explanation outright:

    grep omp_set_num_threads / omp_set_dynamic / omp_set_max_active_levels   -> none anywhere in src/
    OMP_NUM_THREADS=8 forced, 60-token prefill                               -> q 0.862 s (unchanged, 15.8 %)

`in2_matvec_quant` contains no OpenMP at all (checked, 0 hits in its body), so nesting was never the blocker either, and
with the process-wide team size forced to 8 the `q` interval does not move by a millisecond. **11.9 GMAC/s is therefore not
a threading measurement - it is what that interval costs.**

And the interval is wider than the label says. FP2 markers bound the phases, and the `q` interval is FP2(1) at line 130 to
FP2(2) at line 155 - twenty-five lines, of which the `wq` matvec is about six. Everything between `mix+gates` and `q` is
priced into "q": the `xq` preparation, the norms, the clip and the tau scaling. So the 0.862 s does not belong to `wq` until
a marker separates it.

**This is the third time this session that a label was wider than the work it names** - `sliding` measured a layer class, the
MoE histogram measured decode positions, and now `q` measures a 25-line interval. The rule that keeps proving true: a phase
name is a bookmark, not a definition; read the two marker lines and see what lies between them before attributing time.

The next step is one inserted marker between the `xq` preparation and the `wq` projection, then re-read the table. That is a
single line, it settles whether `q` is the matvec or the preparation, and it is the only way to know which of the two - if
either - is worth optimising. No further attempt on this component until that line exists.

## The `q` marker insert failed four times on file navigation, not on C

Attempted the single-marker split (time the `q_clip` dequant + calloc separately from the `wq` matvec + clip + tau) three
ways: one script, an order-safe rework, and a scoped search. All three failed to compile, twice with the same error at line
76 (`qsplit` undeclared) while the declaration landed at line ~131:

    grep -c "FP2(1);"  -> 1        (only one such marker)
    grep -c "FP2(2);"  -> see below (more than one)

`wubu_inkling2_fullseq.c` contains **three top-level functions**, and the phase markers are positional within the file rather
than unique: `FP2(2)` occurs at line 76 in the first function and again later in the one that holds the `q` block. Index-based
searching (`find the Nth occurrence of this line`) therefore lands in the wrong function and produces a declaration in one
scope and a use in another - which is exactly the error the compiler reported both times.

Nothing was left behind: every attempt was reverted, `master` rebuilt, oracle 94-char IDS exact, `git status` clean.

The honest lesson - and this is now the fifth time a search-by-pattern has been the thing that failed, after grep on a long
symbol, a brace scan, a call graph and two labels: **in a file with repeated structures, line numbers from a printed listing
are the only reliable edit anchor.** Not the Nth match of a pattern, not a brace count, not a name.

The `q` question is still open and still worth one turn, but it needs the file printed with line numbers first, and the edit
made against a range read off that listing - the same method that finally worked for the attention parallelisation.

## Correction to the note immediately above: the markers are NOT duplicated, my script was

The grep I ran after writing that note contradicts it: `FP2(1);` occurs **once** and `FP2(2);` occurs **once**, at line 155,
both inside `in2_layer_full_seq` (one of three top-level functions in the file). So "positional markers in three functions"
was wrong. Retracted.

The actual cause was simpler and entirely mine: the insert script computed several indices up front, then inserted at the
highest one first and the lowest last - but it recomputed `i_fp2` from a stale `i_calloc` offset in the first version, and in
the third version the declaration insert at `i_fp1` shifted every index above it before `i_fp2`'s block was placed. The
declaration ended up in one scope and the use in another, which is exactly the `qsplit undeclared` error at a line that did
not contain either.

So the lesson is not about ambiguous anchors in this file - it is the one already recorded earlier in this document and
violated again here: **when inserting multiple times into one list, compute NO indices in advance; insert bottom-up and re-find
each anchor immediately before each insert.** The brace-scan failure and this one are the same class of bug, and both times
the code was fine and the edit machinery was not.

`q` remains unmeasured: 0.862 s over FP2(1)..FP2(2), covering q_clip dequant + calloc + the wq matvec + clip + tau, with
the split between them still unknown.

## RESOLVED: `q` is not slow - it is the matvec, and the matvec runs at 34 GMAC/s

The split marker finally landed (third attempt, bottom-up insert, re-finding each anchor immediately before use) and the
answer contradicts everything I had assumed about this component:

    QSPLIT avg/1: prep=0.11 ms   wq+clip+tau=12.05 ms
    QSPLIT avg/2: prep=0.11 ms   wq+clip+tau=13.30 ms
    QSPLIT avg/3: prep=0.10 ms   wq+clip+tau=12.59 ms

- the `q_clip` dequant plus calloc is **0.11 ms** - 0.9 % of the interval, irrelevant;
- the `wq` matvec plus clip plus tau is **~12.6 ms per call**;
- 60 tokens x 3072 qdim x 2304 d = 424 MMAC per call, so **33.7 GMAC/s** - which is *faster* than the 27 GMAC/s the
  parallel MoE path achieves, not slower.

And 0.862 s / 12.6 ms = **~68 q-calls per prefill**, so the real q work is ~29 GMAC, not the 10.19 GMAC I computed from
"24 full layers". My work estimate was wrong by ~3x, which is what made 11.9 GMAC/s look like one thread: I was dividing a
correct time by an under-counted MAC total.

**Retracted:** "q is single-threaded", "9 % recoverable", and all four attempts that followed from them. `q` is 16.2 % of
prefill because it is genuinely a large matvec (3072x2304 per token per call, ~68 calls), and it executes at 34 GMAC/s - the
best rate measured anywhere in this engine on the CPU. There is nothing to win here.

**Kept:** the `IN2_QSPLIT` instrumentation, env-gated and costing nothing, because it is the instrument that finally
distinguished the matvec from its labelling.

The session's recurring error, named one last time: I trusted an operation count derived from the model's architecture rather
than measured per call, and then interpreted a correct timing through it. Every one of the four failed `q` attempts was
downstream of that single arithmetic assumption.

Where this leaves the engine, measured: MoE 41.9 % (parallel, per-token flat), q 16.2 % (efficient, no win), attention 7.5 %,
gate+wo 6.8 %, everything else noise. The only remaining large target is MoE at 41.9 %, and it is already the healthiest path
in the engine.

## Decode is 5.2 tok/s and 90 % of it is outside the profiled layers

Clean linear fit over three run lengths (no O(N^2) behaviour, so "(full recompute)" in the footer is not what it sounds):

    6 tokens   1,815 ms
    12 tokens  2,908 ms
    24 tokens  5,273 ms
    fit: T = 632 ms + 192.8 ms/token   ->   5.2 tok/s

The profiled layer work for the same 24-token run is **466 ms total, 19.4 ms/token**:

    decode                               193 ms/token
    profiled layers                       19 ms/token    <- 10 %
    unaccounted                          ~173 ms/token   = 4.2 s of the 5.3 s

So **nine tenths of decode time is not in the layer phase accounting at all.** The candidates, and they are all outside the
FP2 markers: the LM head over a 201,024-token vocabulary (463 MMAC/token, ~14 ms at this engine's best rate - nowhere near
173 ms), sampling, the per-step KV bookkeeping, and whatever the "(full recompute)" footer actually refers to.

This is now the single largest unexplained block in the project, and it is in DECODE - the mode a user experiences, and the
one the upstream PR moved from 51 to 158 tok/s. Every phase optimisation this session (attention, q, MoE) operated inside
the 10 % slice.

Next step is a timer that brackets ONE whole token - from the token loop's start to the next iteration - and prints its
breakdown, so the 173 ms can be attributed. Nothing in the FP2 set covers it, so a new bracket is required rather than
another reading of the existing table.

## CORRECTION: there is no unaccounted 173 ms - decode is 5.8 tok/s and it is ALL forward

The generator has printed the per-step breakdown all along (`gen[i] ... wall (fwd X + Y other)`); I had only ever grepped
the token IDs:

    gen[1] 0.172s wall (fwd 0.172s + 0.000s other)   5.81 tok/s
    gen[2] 0.203s wall (fwd 0.203s + 0.000s other)   4.93 tok/s
    gen[3] 0.172s wall (fwd 0.172s + 0.000s other)   5.81 tok/s
    gen[5] 0.172s wall (fwd 0.172s + 0.000s other)   5.81 tok/s

"other" is 0.000 s, every step. So the previous note's claim - "nine tenths of decode time is outside the layer phase
accounting" - is **wrong, retracted.** The whole 172 ms is `in2_forward_at`. What went wrong was the comparison: I divided a
24-step cumulative wall clock by a PHASE line that is per-step, so the two windows never matched. The arithmetic was fine
and the comparison was invalid.

The honest decode picture:

    decode            172 ms/token   5.8 tok/s   entirely forward
    prefill (batched)  88 ms/token   11.4 tok/s  same phases, amortised over 60 tokens

So decode is ~2x the per-token cost of prefill, which is exactly what an unbatchable single-token pass should cost, and it is
distributed like prefill: MoE ~42 %, q ~16 %, the rest small. There is no hidden block, and no large unexplained term left.

Where the session actually ends, all measured: prefill 5.3 s / 60 tokens with MoE 41.9 % (parallel, healthy), q 16.2 % (at
34 GMAC/s, efficient), attention 7.5 % (3.8 % won by parallelising it), decode 5.8 tok/s entirely in the forward. The engine
is not hiding a 10x; it is running its kernels at the rates this 6-core laptop can sustain, and the gaps that remain are
algorithmic (a drafter - upstream's 1.7-3.1x at unchanged tokens) rather than structural.

## Prompt-lookup speculation: measured acceptance on this engine's own output

Drafted nothing, implemented nothing - just measured what a prompt-lookup drafter WOULD have accepted on 64 tokens this
engine actually generated (60-token text prompt):

    n = 64   distinct tokens = 11   (the output settles into a 12-token cycle)

    k=1  draftable 52/63   accepted 42   80.8 %
    k=2  draftable 50/62   accepted 50  100.0 %
    k=3  draftable 49/61   accepted 49  100.0 %
    k=4  draftable 48/60   accepted 48  100.0 %

So on THIS output, a suffix-match drafter of length >= 2 would have been accepted every single time, and would have skipped
~80 % of the forward passes needed to produce those 64 tokens.

**The honest caveat, which matters more than the number:** the output is degenerate - 11 distinct tokens repeating a 12-token
cycle - because the prompt is a junk text file and the generation has fallen into a loop. A 100 % acceptance rate on a
repeating sequence says nothing about prose, where the literature puts prompt-lookup at 1.3-2x. What it does establish is
that the mechanism is available to this engine at zero model cost: the history is already in `toks[]`, the matcher is a
suffix scan, and verification needs only `in2_forward_at(..., n_draft+1, ...)` - the same call the prefill path already makes
with multiple tokens.

**Parity is preserved by construction, not by testing:** the drafter only proposes; acceptance compares the target's own
greedy argmax against the proposal. Any token that is emitted is a token the plain loop would have emitted from the same
logits, so the 12-token oracle is a hard gate on the whole change rather than an approximation check.

Implementation sketch for whoever picks this up (do NOT begin it without room to build and gate):
1. in `tools/inkling_gen.c`'s step loop, before the forward, match the longest suffix of `toks[0..n)` that occurs earlier
   and take up to K tokens that followed it;
2. call `in2_forward_at` once with the K drafted tokens appended (it already accepts n > 1);
3. accept while argmax(logits[j]) == draft[j], then emit the first mismatch - one extra forward saved per accepted token;
4. gate: 12-token oracle identical, and only then look at tok/s.

This is the one remaining lever the measurements point to: decode is 172 ms/token and entirely forward-bound, the forward is
already near this laptop's per-thread rates, so the only way to go faster at identical outputs is to need fewer forwards.

## The verification API does not exist yet - and the economics say build it anyway

`wubu_inkling2_model.c:295` says it plainly: **"logits are only meaningful for the last position."** `in2_forward_at` takes
n tokens, runs all of them through the layers, and then materialises the head for the last one only. So speculative
verification of K drafts cannot read K logit vectors today - the sketch in the previous note needs a per-position head.

Priced at measured rates (forward 172 ms/token; head 201,024 x 2,304 = 463 MMAC = 13.6 ms per position at this engine's best
34 GMAC/s):

    K=3:  save 3 forwards 516 ms   pay 3 extra heads  41 ms   -> net +475 ms
    K=4:  save 4 forwards 688 ms   pay 4 extra heads  54 ms   -> net +634 ms
    K=8:  save 8 forwards 1376 ms  pay 8 extra heads 109 ms   -> net +1267 ms

    even at 50 % acceptance (real prose, not the degenerate loop measured earlier), K=4 still nets +290 ms per block

The extra heads are 8 % of what they save, so **the design does not depend on high acceptance to be worth building** - which
is the opposite of the usual speculative-decoding economics and is a direct consequence of this engine's head being cheap
relative to a 48-layer forward pass.

So the concrete next change is, in order:
1. add an optional per-position logits output to `in2_forward_at` (n_tok x vocab floats, 3.2 MB at K=4) - additive, gated,
   the existing callers pass NULL and are untouched;
2. the draft/verify loop in `tools/inkling_gen.c` as sketched;
3. gate on the 12-token oracle being identical, then on tok/s.

Not started: it is a three-file change (header, model.c, gen tool) and the last four multi-site attempts in this session each
cost a build cycle to discover a mechanical mistake. The measurement above is the piece that was missing, and it now says the
work is worth a session of its own.

## Exact shape of step 1 (per-position logits), so it is a refactor with anchors, not a search

`in2_forward_at` (model.c:83) tail, from the "logits are only meaningful for the last position" comment at :295 to the
tanh softcap at :389, does this ONCE for position `n_tok - 1`:

    :296-298   const float *last = h + (n_tok-1)*d;
               in2_mix(blocks, n_blocks, n_tok*d, (n_tok-1)*d, last, fq, d, eps, fin)   // or memcpy
    :300-378   the lm_head path: kernel over vocab rows with a fallback, writing `logits`
    :379-384   fallback: in2_matvec(Wo, fin, d, vocab, logits)
    :388-389   softcap: for i<vocab: logits[i] = tanhf(logits[i]/30)*30
    :391+      padded-vocab handling

To produce K logit vectors the mix, head and softcap must run per position, so the clean change is:

1. extract :295-389 into `static int in2_head_at(gguf_ctx*, cfg*, blocks, n_blocks, const float *h, int n_tok, int pos,
   float *out)` - the body is identical with `(n_tok-1)` replaced by `pos` and `logits` by `out`;
2. keep `in2_forward_at`'s existing behaviour by calling it with `pos = n_tok - 1` and `out = logits` - the current path then
   executes exactly the same instructions, which is what makes the oracle a valid gate on the refactor by itself;
3. add an optional `float *logits_all` parameter; when non-NULL, loop `pos = 0..n_tok-1` calling the extracted helper.

That ordering matters: steps 1 and 2 are a pure refactor that the 12-token oracle must pass unchanged BEFORE any speculative
code exists. Only step 3 is new behaviour, and it is additive and NULL-gated.

Cost confirmed at measured rates: 13.6 ms per extra position against 172 ms per saved forward - 8 %, so K=4 nets +634 ms per
block even before counting acceptance.

Not done in this session: the extract touches an 80-line block containing the kernel path, its fallback, the softcap and the
padded-vocab handling, and the last four multi-site edits cost a build cycle each to discover a mechanical fault. The anchors
above make it a mechanical job for a session with room to run the oracle after step 2 and again after step 3.

## The oracle gate is now one command - and it immediately caught a naming error in my own notes

`tools/oracle_gate.sh` wraps the parity check every change in this repo depends on:

    bash tools/oracle_gate.sh            # uses the existing binary
    bash tools/oracle_gate.sh --rebuild  # rebuilds first
    exit 0 identical / 1 mismatch / 2 harness problem

It refuses to pass when the reference is empty or short - the exact failure that has occurred four times this session, where
a failed build left the binary missing, both sides returned "", and an equality test reported success. Verified both ways:
tampering one token of the golden string yields `GATE FAIL(1)` with a want/got diff and exit 1; the untampered run yields
`GATE PASS` and exit 0.

**And it immediately exposed an error in this document's vocabulary: the "12-token oracle" is FIFTEEN tokens.** The command
passes `12` as the number of tokens to *generate*, and the IDS line also carries the 3 prompt tokens, so the golden line has
3 + 12 = 15 ids. Every note in this roadmap that says "12-token oracle, exact" has been quoting a 15-token line. The gate
initially failed on its own word count because I encoded my wrong belief (13 words) rather than the actual string (16 words).

That is the same class of mistake as `sliding`, `q`, and the layer count - a label I trusted instead of the thing itself. The
difference is that this time a mechanical check caught it in seconds, which is precisely why the gate is worth more than any
single optimisation: it converts a recurring human error into a compiler-style failure.

From here the workflow for the per-position-logits refactor (or anything else) is: `bash tools/oracle_gate.sh` before, make
the change, `bash tools/oracle_gate.sh --rebuild` after. Two commands, no ambiguity about what "still exact" means.

## Speculative decoding: the cache-rewind question, answered from the code

The draft/verify loop accepts fewer tokens than it verified whenever the model disagrees with the drafter, so the KV cache
holds rows for positions that were never accepted. That is the classic reason speculative decoding needs a rollback, and
`wubu_inkling2_kv.h` has no truncate/rewind/rollback call at all. I expected to find a blocker and instead found this:

    src/wubu_inkling2_fwd.c:210   void in2_kv_begin(int pos0) { g_visit = 0; g_kv_pos = pos0; }

The cache is **position-addressed with a cursor**, not an append-only log. Every forward begins by setting the write position
to `pos0`, so a verification pass that writes rows at `n..n+K-1` and accepts only `n..n+j-1` is corrected by the next call
beginning at `n+j`: the phantom rows are simply overwritten when the cursor reaches those positions again. Attention reads
only positions < current, and the two conv streams (`g_kv2`) read backwards, so nothing reads a phantom row before it is
rewritten.

**One thing must be checked before the loop is written, and it is the only remaining unknown:** whether attention bounds its
window with the true cursor (`g_kv_pos`) or with a high-water mark such as `in2_kv_t.n_stored` ("positions stored so far",
kv.h:13). If it is the latter, a verification pass silently extends the visible sequence by the number of rejected drafts and
the model would attend over garbage - and because the gate compares tokens, it WOULD be caught, at the cost of a confusing
debugging session.

So the ordered next steps are now:

1. confirm attention's length source is the cursor, not `n_stored` (one grep, one read);
2. then write the loop sketched earlier: prime one ordinary step, draft K by suffix match over `toks[]`, one
   `in2_forward_at` over `feed[0..K]` with `in2_set_logits_all(la)` (now available, gate-clean), accept while
   `argmax(la + q*vocab) == draft[q]`, emit accepted plus one correction, and let the cursor rollback happen by construction;
3. gate on the 15-token oracle at K=1 first (where the loop degenerates to almost the plain path), then K=2, then K=4.

Finding the cursor semantics was worth more than writing the loop blind: a rewind bug would have produced wrong tokens from
correct-looking code, which is the single most expensive failure mode this project has.

## Inkling-2-Mini is a LOOPED transformer, and that settles the layer-count question for good

Source: the model's own GGUF metadata, read locally, and the published model card
(thinkingmachines/Inkling-2-Mini, and the Strata pack page strata-engine/Inkling-2-Mini-Strata).

    inkling2.num_loops       = 3
    inkling2.loop.core_start = 6
    inkling2.loop.core_end   = 18
    inkling2.loop.gate_rank  = 64

**Layers 6..17 (twelve layers) execute three times for every token**, with a rank-64 gate controlling each extra pass. The
model card states it directly: 26.4B parameters, 3.4B active per token, 5.4B when the repeated passes are counted.

Our engine implements this - `wubu_inkling2_load.c:78` sets `is_loop_layer = (il >= loop_core_start && il < loop_core_end)`,
and `loop_gate_rank` is threaded through both the per-visit and full-sequence layer bodies.

**And it reconciles a number I had got wrong.** MoE sits on layers 0..23. With the core tripled, the number of MoE
layer-passes per token is:

    layers 0..5   (6 passes)  +  layers 6..17 x 3 (36 passes)  +  layers 18..23 (6 passes)  =  48

which is EXACTLY the `48 n_tok=120` that `IN2_MOE_NTOK` printed and that I could not explain. The engine was never wrong; my
model of the engine was.

**Retracted:** the earlier "resolution" that the 284-row histogram's 24 ids meant the MoE MAC share halves to 183.5 GMAC and
the prefill total is 249 GMAC. That was wrong by precisely the loop factor. The correct figures: 48 MoE layer-passes and
**72 layer-passes for the full stack** (48 + 12 extra). Every per-token compute estimate in this document that assumed 48
layer-passes should be multiplied by 1.5.

This also means the engine's arithmetic per token is heavier than my notes implied, which makes the measured 88 ms/token
prefill and 172 ms/token decode rather better than they looked against the MAC totals I had been dividing by.

## The Strata pack page, and what it says about speed

Same model, independent MIT engine, published numbers: RTX 5070 12 GB with their SQ-10 pack runs 93.9-101.4 tok/s single
request; a 4 GiB card is enough to start because the non-resident experts stay in system RAM. Their SQ-20 pack is bit-identical
to the original model (100.00 % same next word, KL 0.00000).

Two things worth taking from it:
- **Every pack has a DSpark draft model built in, and their headline single-request numbers use it** ("W=4 Beam"), i.e. the
  fastest published configuration of this exact model is speculative decoding at unchanged-ish output. That is independent
  confirmation that speculation, not kernel work, is where the remaining decode speed lives.
- Their spread (SQ-10 92.35 % same-word vs SQ-20 100 %) is a reminder that our oracle discipline is the stricter standard:
  this engine is gated at bit-identical, which is why it can claim parity while a 10 GB pack at 92 % cannot.

Their .strata format is not GGUF, so nothing is directly loadable here; the value is the architecture fact and the speed
context, not the files.

## Last unknown cleared: attention is bounded by the position, not by n_stored

The concern was that a rejected draft would leave `n_stored` inflated and attention would read phantom rows. Both halves are
now measured in the code:

    fwd.c:1369   if (g_kv->n_stored < g_kv_pos + n_tok) g_kv->n_stored = g_kv_pos + n_tok;   /* monotonic high-water */

- `n_stored` IS a high-water mark and does stay inflated after a rejected draft;
- but the causal key loop is `for (int j = jstart; j <= gp; ++j)` (fullseq.c:325), bounded by **`gp`, the current token's
  global position** - never by `n_stored`. Positions beyond `gp` are simply not visited;
- the key lookup is `in2_kv_k(kv, vvis, j)` with that same global `j`, so a phantom row at `j > gp` cannot enter the softmax;
- residency (`in2_kv_resident`, kv.c:338) uses `n_stored`, so phantom rows would *pass* that test - which is exactly why this
  mattered: the guard is the loop bound, not the cache. It is sufficient, and the loop bound holds.

Combined with `in2_kv_begin(pos0)` setting the write cursor, the verdict is: **speculative verification needs no rewind API.**
Writes past the accepted prefix are never read, and are overwritten when the cursor returns to them.

Design is now fully validated against the code: per-position logits (done, gate-clean), cursor semantics (safe), attention
bound (safe), and the economics priced at 13.6 ms per extra head against 172 ms per saved forward. The remaining work is the
loop itself in `tools/inkling_gen.c`, gated at K=1 then K=2 then K=4 against the 15-token oracle.

## NEGATIVE RESULT: the speculative loop produces WRONG tokens - verification advances cache eviction

The draft/verify loop was written (suffix-match drafter, one `in2_forward_at` over `feed[0..K]` with per-position logits,
accept while the target's argmax agrees) and gated at K=1, the near-degenerate case:

    golden : 371 151451 149879 149876 220 36838 182189 149933 220 50843 ...
    K=1    : 371 151451 149879 149876 220    17    13    20   12  1258 ...

Identical for the first five tokens, then wrong from token six. Reverted; the plain path gates 15/15 again.

**Why, and why my earlier "no rewind needed" verdict was wrong.** I checked that attention's key loop is bounded by the current
position (`j <= gp`) and concluded phantom rows are never read. That part was right. What I did not check is the OTHER use of
`n_stored`:

    kv.c:341   if (kv->window > 0 && (long)pos < (long)kv->n_stored - (long)kv->window) return 0;

`n_stored` is not only a readability bound - it is the **eviction** bound for sliding-window visits. The verification pass
writes rows at positions beyond the accepted prefix and `n_stored` is monotonic (fwd.c:1369 raises it, nothing lowers it), so
those phantom writes **permanently advance eviction** and drop real positions out of the window. The model then attends to a
truncated window: exactly the symptom, correct for five tokens and then wrong once the window state matters.

So speculative verification DOES need a rollback, and specifically it needs `n_stored` rewound (plus the ring slot state that
`window` implies) - a cache API that does not exist. The lesson is narrower than "check the cache": **`n_stored` is read in two
different roles, as a length and as an eviction frontier, and a rewind has to restore both.** Checking one and concluding the
other was safe is the same error shape as reading one phase label and concluding what the other 80 % was doing.

The per-position logits support (`in2_set_logits_all`) stays: it is gate-clean, additive, and needed by any future attempt at
either speculation or a rewind API.

## The ring is FIFO, and HIVE is already in this repo - the fix is tags, not an import

Directive: replace the FIFO ring with HIVE. Checked before importing anything:

- **WuBuHIVE is already vendored here.** `include/wubu_hive.h` + `src/wubu_hive.c` implement linked fixed blocks +
  bit skipfield + per-block freelist, exactly as WuBuOS's `src/kernel/wubu_hive.*` do, with an embedded-struct API
  (`wubu_hive_init/insert/erase/foreach/live`) already used by `wubu_agi.c`, `wubu_ctxvm.c`, `wubu_ecosystem.c`.
  Nothing needs copying in; the two implementations are the same design.
- **What is FIFO is the KV slot mapping**: `in2_kv_slot(pos) = pos % cap` (`wubu_inkling2_kv.c:50`). A modulo ring cannot
  express "this row is invalid" - the only way to make room is to overwrite the row at `pos % window`, which is exactly the
  row still live for the previous pass. That is why the K=1 speculative run corrupted after five tokens: verification writes
  advanced `n_stored`, eviction (kv.c:341) followed it, and real rows fell out of the window.

**The fix is an epoch-stamped tag per slot, not a new container.** Each slot carries `(epoch, pos)`; an accessor returns a
row only when both match the request. Consequences:

- rewind = `++epoch`, O(1), no per-row work - a rejected draft's rows become unreachable instantly;
- eviction becomes exact: a slot is valid only for the position it was tagged with, so nothing can be read stale;
- no memset, no clearing, no ring aliasing: today a sliding visit overwrites `pos % window` regardless of whether that row
  is still needed.

This is precisely the scheme Strata uses (`slot_pos_[p % kRing] = ((int64_t)epoch_ << 32) | p`, `++epoch_` on rewind,
`dflash2.cpp:875/854/973`), arrived at independently from our own failure. The HIVE is the right container for the *rows*
when capacities are per-visit and dynamic (its O(1) erase/freelist reuse beats two flat slabs), but it is not what makes
rewind possible - the tag is. Order of work: tags first (small, gateable, unblocks speculation), HIVE-backed row storage
second (memory shape, not correctness).

**And a near-miss worth recording:** copying WuBuOS's `wubu_hive.{h,c}` into `src/` overwrote a TRACKED `src/wubu_hive.c`
belonging to this repo, whose API differs (`wubu_hive_init` vs `wubu_hive_new`). Restored with `git checkout` before any
build; the repo's real header is `include/wubu_hive.h`, which was never touched. Lesson: before "vendoring" a file from a
sibling repo, `git ls-files` the destination - two projects in the same family can both have a `wubu_hive.c` with different
signed APIs, and the one in the tree is the one that must win.

## Rewindable row tags landed; the conv history is the second half of a rollback

`in2_kv_t` now carries `kpostag`/`vpostag`, one int per row: `pos+1` = the row holds that position, `-1` = deliberately
invalidated, `0` = never written (behaves exactly as before). The `_post_` stores tag, `in2_kv_k`/`in2_kv_v` refuse a row whose
tag names a different position, and `in2_kv_rewind(kv, from_pos)` invalidates the range, pulls `n_stored` back to `from_pos`,
and drops the decoded-mirror visit tags for the same range.

Gated: **15/15 identical**. That is the expected result and it is the useful one - every live row's tag equals the position it
holds, so the guard never fires on the existing path. The tags are pure added capability, not a behaviour change.

**What a rollback still needs, and why the tags alone are not enough:** the K/V rows are not the only per-position state. The
4-tap causal conv keeps three rows per visit in `kraw`/`vraw`, advanced by `memmove` inside `in2_kv_store_row_k/v` and
`in2_kv_store_raw`. A verification pass over K+1 tokens shifts that history K+1 times, and invalidating KV rows does not undo
it - the next pass would convolve against draft rows. The fix is small and bounded: those buffers are `3 x dim` per visit
(884 KB at 72 visits x 1024 dim), so a pre-verification snapshot plus a restore on reject is cheap. **Do that before retrying
the draft loop** - otherwise the same five-tokens-then-wrong signature will reappear and the tags will be wrongly blamed.

Provenance note: the tag scheme is what Strata does with `slot_pos_[pos % kRing] = (epoch << 32) | pos`, but ours is per-row
rather than per-slot and uses a bounded range clear instead of an epoch bump, so a rejected draft costs O(visits x K) writes
rather than invalidating the whole cache. For K <= 8 that is ~1.5 K writes per reject, which is nothing next to a 172 ms
forward.

## Rollback infrastructure is complete: both caches, both kinds of state

Four pieces, each independently gated, all passing 15/15:

1. `kpostag`/`vpostag` per row (`pos+1` live, `-1` invalidated, `0` never written) with the accessors refusing a mismatched
   row, and `in2_kv_rewind(kv, from_pos)` invalidating a range and pulling `n_stored` back;
2. `in2_kv_conv_save/restore` over `kraw`/`vraw` - the four-tap causal conv's three rows per visit, 1.77 MB total;
3. `in2_state_save/restore/rewind` in `fwd.c`, which apply to **both** caches: `g_kv` carries the residual stream's conv
   history and `g_kv2` (window 3) the attention-output and MoE convs. Covering only one of them would have reproduced the
   five-tokens-then-wrong signature while looking fixed;
4. the per-position logits support (`in2_set_logits_all`) from earlier.

Why the conv history had to be handled separately: it is advanced by `memmove` inside `in2_kv_store_row_k/v`, so a K+1
verification pass shifts it K+1 times. Invalidating K/V rows does nothing to it - the next pass would convolve against draft
history. The snapshot is taken *before* verification and restored only when fewer drafts are accepted than were verified.

The remaining policy decision, which the loop must implement: after a partial accept, restore the history to the
pre-verification snapshot, rewind the K/V rows to the accepted prefix, then **replay the accepted tokens** in one short
forward. Replaying is what rebuilds both conv histories at the true prefix; it costs one forward over `acc <= K` tokens and
happens only on a reject. Full acceptance needs no restore at all.

## Speculation is blocked by numerics, not by rollback - and K=0 is what proved it

The loop is in `tools/inkling_gen.c` behind `IN2_LOOKUP`. Three findings, in the order they were separated:

1. **My first loop was missing the prompt prefill.** The plain loop's step 0 is `in2_forward_at(toks, n_prompt, 0)`, not a
   single-token prime; my while-loop started with a prime at position `n_prompt`. That alone produced the "correct for five
   tokens then wrong" signature that I had blamed on the cache - and it reproduced even with **K=0**, where no draft is ever
   made. K=0 is therefore the control that isolates the loop skeleton from verification, and with the prefill added it now
   matches the oracle exactly.
2. **With K>0 the run still diverges, at token 13, with `drafted 8, accepted 0, replays 0`.** Nothing was accepted, so no
   rollback path executed, and the only difference from the plain path is the verification forward itself:
   `in2_forward_at(feed, nd+1, n-1)`. That call re-writes the KV row for position `n-1` through the **batched** (n_tok>1)
   code path, and a batched row is not bit-identical to the sequentially-written row it replaces. Rollback cannot restore it
   because the rollback point is `n`, which keeps `n-1`.
3. **Therefore the missing prerequisite is arithmetic, not state**: speculation needs the batched path to produce bit-identical
   K/V rows and hidden states to single-token stepping. That is a property to be measured per kernel (attention, conv, MoE
   grouping), not assumed - and it is the same class of claim as "the incremental path is bit-identical to full recompute",
   which was verified for 1-token stepping and says nothing about n_tok>1.

State of the tree: plain path gates **15/15**; `IN2_LOOKUP=1 IN2_LOOKUP_K=0` matches the oracle exactly (loop skeleton sound);
`IN2_LOOKUP=1 K>0` prints a warning that parity is not achieved. The rollback infrastructure (row tags, rewind, conv snapshot
over both caches) is committed and gate-clean regardless - it is correct and it is not what is failing.

Next step, now unambiguous: measure batched-vs-sequential equivalence directly - forward a 2-token block at `pos0` and compare
each row's logits and stored K/V bytes against two 1-token calls at the same positions. If they differ, quantify by how much
and locate the kernel (attention vs the two convs vs MoE). That measurement decides whether prompt-lookup speculation is
reachable at all without changing the batched kernels.

## MEASURED: the batched path IS bit-identical to sequential - so the ring is the suspect

Probe added to the generator (`IN2_BATCH_PROBE=0|1`, `IN2_PROBE_OUT=path`) writes two logit rows: from two 1-token forwards
at positions 0 and 1, versus one 2-token forward over the same tokens. Comparing all 201,024 entries of each row:

    pos0: max|diff| = 0 over every real vocab entry, argmax 220    vs 220    SAME
    pos1: max|diff| = 0 over every real vocab entry, argmax 149879 vs 149879 SAME

(966 entries differ by 1e30 - those are the padded-vocab sentinels, identical in sign and position, and they never win the
argmax.) So **the claim "the batched path is not bit-identical" is FALSE, retracted before it was written into a conclusion.**
`in2_forward_at` with n_tok = 2 reproduces the sequential result exactly at low positions.

That moves the speculation failure to the remaining structural difference: **the ring**. Verification writes rows at
`n..n+K-1` through `pos % cap`. On a sliding visit whose cap is the window, those writes wrap and **overwrite rows for earlier
positions that are still live** - and no tag can restore overwritten data, only mark it invalid. The plain path never writes
ahead, so it never evicts a live row. This is exactly the FIFO objection, and it predicts the observed shape: correct until
the sequence is long enough to wrap (token 13), and independent of how many drafts are accepted (measured: accepted 0).

Two candidate fixes, both already scoped:
1. **slack**: size each sliding visit's cap to `window + K`, so verification's lookahead cannot reach a live row. Cheapest,
   costs `K/window` extra memory (at window 1024, K 4: +0.4 %).
2. **HIVE-backed rows**: no aliasing at all, O(1) erase, per-position addressing - the larger change, and the one that also
   makes a long-context row store sane.

## LoopCD: a nearly-free decoder for looped transformers, and this model is one

Source: arXiv 2610.02185, "Decoding Looped Transformers Better for (Almost) Free" (Apple), plus oMLX 0.8.4 shipping it for this
exact model.

Mechanism: a looped transformer's final prediction can be **contrasted with an earlier recurrent pass** to guide token
selection, training-free.
- **LoopCD-Logits**: run both states through the output layers (one extra head pass).
- **LoopCD-Hidden**: combine the two hidden states before the single output pass - **zero extra output overhead**.

Published gains: Ouro-2.6B-Thinking AIME 2024 61.88 % -> 73.33 % (Logits); Huginn HumanEval 22.56 % -> 31.71 % (Hidden); up
to 48.2 % fewer FLOPs at equal quality. oMLX reports, for this model, ARC-Challenge 69.62 -> 71.93 and HumanEval 152 -> 155 of
164, at a cost of 9-13 % decode speed, off by default.

Why it fits here: our engine already runs the three core passes (layers 6..17, `num_loops = 3`), so the *first* pass's hidden
state exists at the moment the core finishes its first traversal - capturing it is one `memcpy` of `d` floats per token.

**Hard constraint, stated before anyone wires it:** LoopCD changes token choice by construction. It therefore **fails this
project's bit-identical oracle on purpose**. It can only ever be an opt-in quality mode (`IN2_LOOPCD=hidden|logits`), never
the default and never part of a parity claim. Any future run that enables it must not report oracle equality.

Also from the same release, directly relevant to a 6 GB card: oMLX now keeps **Inkling 2 experts on SSD**, taking a 24 GB Mac
from 17.0 GB to 10.2 GB and letting a 14.8K prompt run that previously exhausted memory. Our engine already streams experts
from SSD (`IN2_RES_GB`); their result is evidence the approach scales to a prompt length we currently cannot fit.

## Correction: the ring does NOT wrap at token 13, so that is not the explanation either

The runtime line kills the hypothesis I had just written down:

    KV    ctx=1024 rows/visit: 8 full x 1024 + 64 sliding x 1024 = 576.0 MB

Every visit is sized `ctx = 1024` rows, so `pos % cap == pos` for any position below 1024, and the sequence that diverges is 13
tokens long. **No wrap occurs. Retracted: "verification's writes wrap and overwrite live rows, which is why token 13 fails."**

What survives the measurements so far:
- the loop skeleton is right (K=0 matches the oracle exactly);
- batched == sequential at positions 0 and 1, bitwise, with a *fresh* cache;
- the failure needs K>0, happens with zero accepted drafts, and lands at token 13.

The gap in that evidence is now the obvious one: **the probe tested a fresh 2-token cache, while real verification runs with a
populated cache, 72 visits and sliding masks.** A batched attention/conv call over an existing prefix can take a different
kernel branch than the one-token path even when the fresh case agrees exactly. So the next measurement must be the same probe
*after a prefill*, at positions inside a populated cache - not a fresh one. Until that is run, every hypothesis about the
verification failure - mine, the ring's, the numerics' - is unfounded.

Recorded because this is the third hypothesis in this thread that measurement destroyed before it reached a conclusion
(MoE halving, batching non-equivalence, ring wrap). The pattern is consistent: the guess is always about *where* the state
diverges, and the only thing that settles it is probing the exact configuration the failure occurs in.

## FOUND: at a non-zero position the batched forward is badly wrong - not rounding, a bug

Same probe, run after a 60-token prefill (populated cache, 72 visits, sliding masks) instead of on a fresh cache:

    pos 60 (row0): differing real entries 200058/200058, max|diff| = 3.052e+00, argmax 157021 vs 157021
    pos 61 (row1): differing real entries 200058/200058, max|diff| = 1.958e+00, argmax 161319 vs 161319

**Every single real vocabulary entry differs, by up to 3.0 logits.** That is not floating-point noise, not an accumulation
order, and not a tie-break: `in2_forward_at(tokens, 2, pos0=60)` computes a different function from two sequential
`in2_forward_at(token, 1, pos)` calls at the same positions.

Note the correction this forces: the earlier "batched == sequential, bitwise" result was measured on a **fresh 2-token cache at
positions 0 and 1**, where `pos0 == 0` takes the prefill branch. It was true there and false everywhere else. The retraction
in the previous entry was itself wrong, and for the reason that entry had already named: the probe was not run in the
configuration the failure occurs in.

Why this matters far beyond speculation:

- **`pos0 > 0` with `n_tok > 1` is broken, and it is used by chunked prefill.** Any plan here (and LP-04 in the earlier roadmap)
  that splits a long prompt into blocks fed at a non-zero offset would have produced wrong output, silently. This is the
  first evidence that path was never exercised.
- Speculation failed at token 13 for exactly this reason: the verification forward is a multi-token call at `pos0 = n-1 > 0`,
  so it perturbs the trajectory immediately. Rollback cannot help - the *accepted* rows are already wrong.

Prime suspects, in order: the two conv streams (`in2_conv_stream`, whose `n_tok > 1` branch shifts history per token and is
called with a `pos0`-relative history that `in2_kv_begin` only sets up once), and the K/V store cursor (`g_kv_pos` advanced
per token but the ring/mirror tags written for `pos0 + t`). The measurement to separate them: run the same probe at
`pos0 = 0` with 2 tokens (known good), then at `pos0 = 1` with 1 token appended to a 1-token prefill, and bisect which visit's
K row first differs.

The honest summary of this thread: three hypotheses died (MoE halving, batching equivalence, ring wrap) and the fourth -
batching - turned out to be right in the configuration I originally dismissed as unnecessary to test. The rule that keeps
paying: probe the failing configuration, not a convenient one.

## THE BUG: the batched conv dropped the history - fixed, and batched now equals sequential exactly

`in2_conv_stream` had two branches:

    if (n_tok == 1)  hist = last 3 rows; in2_kv_conv_row(hist, x, ...)      // uses history
    else             in2_shortconv(x, n_tok, ch, w, k, out);                // history IGNORED

`in2_shortconv` convolves inside the batch and zero-pads the leading taps. At `pos0 == 0` there is no history, so full
prefill was correct - which is why every gate in this project ever passed, and why a fresh-cache probe at positions 0 and 1
matched bitwise. At `pos0 > 0` the first tokens of a batch lost the three rows that precede them.

That single oversight made **every multi-token call at a non-zero offset wrong**: chunked prefill (so LP-04's plan to split a
long prompt into blocks fed at an offset would have produced silently wrong output) and speculative verification (a
multi-token call at `pos0 = n-1`).

Fix: seed the convolution with the stored history - concatenate the 3 history rows and the batch, run the same conv over
`3 + n_tok` rows, keep the last `n_tok`. Cost is one temp buffer of `(3+n_tok) x ch` floats per call and the same FLOPs plus
3 rows.

Measured, after the fix, with a populated 60-token cache (the configuration that failed before):

    pos 60: differing real vocab entries 0/200058, max|diff| = 0.000e+00
    pos 61: differing real vocab entries 0/200058, max|diff| = 0.000e+00

And with speculation enabled, the 15-token oracle passes at every draft length tried:

    K=0 MATCH   K=1 MATCH   K=2 MATCH   K=4 MATCH   K=8 MATCH

Bisect that found it: dumping the stored K row for all 72 visits at the probe positions showed visit 0 identical and
**47 of 72 visits different, first at visit 1** - i.e. the corruption enters at the first layer boundary, which is where the
residual conv runs, not in the projections or the cache.

This is the most consequential finding of the session and it was invisible to every previous test: the engine's own oracle
only ever exercised `pos0 == 0`.

## Speculative decoding works and is parity-clean - measured 1.22x, best case, on looping output

With the conv fix in, on a 60-token prompt generating 64 tokens (the model falls into a 12-token cycle here, so acceptance
is the ceiling case):

    plain        64 tokens in 18.5 s = 3.45 tok/s
    lookup K=4   23 forwards / 64 tokens (0.36 fwd/token), drafted 44, accepted 44 = 100 %, 15.2 s = 4.21 tok/s
    lookup K=8   15 forwards / 64 tokens (0.23 fwd/token), drafted 56, accepted 56 = 100 %, 15.2 s = 4.21 tok/s
    oracle       15/15 identical in every configuration

**1.22x**, bit-identical, with 100 % acceptance - a best case on degenerate looping output, not a typical-text number. Stated
that way deliberately: on non-repetitive text the suffix drafter will miss far more often and the speedup will be smaller.

Why 2.8x fewer forwards is only 1.22x wall time - the fixed cost of a forward dominates:

    plain   18.5 s / 64 = 289 ms per forward (1 token each)
    lookup  15.2 s / 23 = 661 ms per forward (up to 5 tokens each)

Solving the two points gives roughly **200 ms fixed + 90 ms per token** inside `in2_forward_at`. So the lever has moved: it is
no longer per-token arithmetic, it is the ~200 ms constant that is paid on every call regardless of token count. K=8 does not
beat K=4 because the per-token term grows alongside the fixed one.

That constant is now the single best-measured target in the engine, and it is the same constant that makes decode 172 ms/token
in the earlier measurements: 72 visits of fixed per-layer work that scales far less than linearly with token count.

Also fixed in this change: `IN2_LOOKUP` was presence-tested, so `IN2_LOOKUP=0` enabled speculation. It now parses the value,
which is why the baseline above is the first honest one.
