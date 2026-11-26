# SOTA 2026-10/11 → WuBu lilypads

> A **lilypad** is one crossable deliverable with a gate on the far side. Order is
> **dependency**, not importance: a pad that another pad stands on goes first.
> Every pad ships a real gate: no gate, no pad.
> Source window: 2026-10-01 → 2026-11-20, from eight cited sweep reports.
> Status 2026-11-20: all 16 pads LP-01..LP-16 have a gated C11 module and a
> passing `make test_<gate>` target; gates are force-added past the `tools/test_*` ignore rule.

## Standing rules

- Read upstream, reimplement from scratch in C11. Never vendor a third-party library.[179][180]
- Opaque structs at module seams, minimal includes, no stubs, runtime kernel dispatch stays runtime.
- The engine stays ignorant of the goal it serves. Goal/state/slerm logic lives in a host module.
- "Slerming" in this roadmap means the slermes-style persistent-goal/continuation loop, not a new inference kernel.[33][133]
- No primary source was found for a distinct 2026-10/11 "slerming paradigm"; do not invent one.

## Wave 1 — goals and slerming (host layer)

### LP-01 — Persistent goal state + evidence-first judge
Opaque `wubu_goal_t` with text, status, turn budget, and a `.goal` sidecar that survives restart. The judge is deterministic: it reads a **separate evidence string** (never the agent's own completion claim), requires a verifier callback to pass, and treats an explicit blocked reason as done-with-reason.[147][148] Real user messages preempt; budget exhaustion pauses. Gate: `make test_goal` (continue / report-only rejected / pass / blocked / preempt / budget / sidecar round-trip). **Size M**. Unlocks LP-02..LP-04 and LP-14.
Shipped as `src/wubu_goal.c` + `include/wubu_goal.h` + `tools/test_goal.c`; wired into both Makefiles.

### LP-02 — Fast/slow goal-drift steering
One scalar drift score plus a tunable threshold τ: the fast path monitors every turn, the slow path rewrites the subgoal only when drift exceeds τ, and the existing judge veto stays authoritative.[91][99] Gate: `make test_goal_steer` (below τ no rewrite, above τ rewrite, veto preserved). **Size S**. Depends on LP-01.

### LP-03 — Bounded verifier-guided resampling with rollback
Cap candidate trajectories, use the LP-01 verifier to reject early, stop as soon as one valid plan exists, and restore a snapshot on rejection.[38][39] Gate: `make test_goal_resample` (bounded count, early stop, snapshot restored). **Size M**. Depends on LP-01/02.

### LP-04 — Failure-pattern harness bandit
Turn hive failure cells into arms scored by severity, fixability, breadth and side-effect risk; allocate revision effort with a bandit and keep a fixed-capacity role pool.[189][193][199] Gate: `make test_harness_bandit` (arm scoring, exploration cap, rollback on unsafe revision). **Size L**. Depends on LP-03.

## Wave 2 — KV spine (engine layer)

### LP-05 — All-or-nothing paged KV claim
Add `alloc_exact`/`can_alloc_exact`: claim all requested block IDs in caller order or leave state untouched, checking logical quota and physical page capacity together.[108][176] Gate: `make test_kv_alloc_exact` (all-or-nothing, quota, capacity, no partial mutation). **Size M**. Independent spine pad.

### LP-06 — Shared-cell SWA reclamation invariant
A cell may be reused only when it is masked for every owner under each owner's position bound; record overwritten positions per owner.[200][215] Gate: `make test_swa_reclaim` (two owners, divergent bounds, unsafe reuse refused). **Size M**. Depends on LP-05.

### LP-07 — Confidence-budgeted speculation policy
Choose draft/verification length from cumulative draft confidence and global rank; keep the target verifier unchanged.[103][104][207] Gate: `make test_spec_budget` (budget allocation, length selection, acceptance accounting). **Size M**. Depends on LP-05.

### LP-08 — Deterministic seeded speculation
Seed rejection-sampling proposals and make acceptance reproducible across runs; expose the seed in the spec context.[207][180] Gate: `make test_spec_determinism` (same seed → identical accepted prefix, different seed → different). **Size S**. Depends on LP-07.

### LP-09 — Constrained decoding survives speculation
Build masked multi-row windows so draft tokens cannot break grammar/JSON/tool constraints; advance along the draft prefix, invalidate at the first illegal draft, commit at divergence.[206] Gate: `make test_spec_constrained` (illegal draft rejected, mask intact across rows). **Size L**. Depends on LP-07/08.

## Wave 3 — kernels and formats

### LP-10 — Compile-time family specialization
Express model-family differences as compiled rows (attention variant, routing, stop tokens) instead of runtime family branches; keep the kernel bodies byte-identical.[179] Gate: `make test_kernel_identity` (1837/1843-style identity check on the local kernel set, serving output unchanged). **Size L**. Independent.

### LP-11 — KV reconstruction in compute format
Reconstruct KV directly in MX/NVFP4 with a small deterministic anchor set so attention never dequantizes to FP16 just to read the cache.[10][16] Gate: `make test_kv_qfmt` (finite dot products, max error bound, no FP16 intermediate). **Size L**. Depends on LP-05.

### LP-12 — Tolerance-parameterised KV fetch for the 9P layer
Add an `(eps, delta)` fetch gate: sketch + pivotal tokens on the fast side, error estimate decides whether to host-fetch the full page.[12] Gate: `make test_kv_fetch_eps` (error under eps avoids fetch, over delta forces it). **Size M**. Depends on LP-05/11.

### LP-13 — Fused diffusion logits/sampler
Fuse entropy, stopping and Gumbel/bit-exact sampling in one canvas×vocab pass; after W4A16 experts this is the decode bottleneck on bandwidth-bound devices.[23] Gate: `make test_diffusion_sampler` (bit-exact vs unfused reference, one pass). **Size M**. Independent.

### LP-14 — Measured-cost test-time compute budget
Charge wall-clock and KV cost to candidate counts, cap when ranking bias plateaus, and audit end-to-end speedup instead of FLOP savings.[27][30] Gate: `make test_ttc_budget` (budget respects measured cost, plateau stops early). **Size M**. Depends on LP-01/03.

### LP-15 — Per-shape SIMD quant decode + W4A streams
Autotune the native SIMD dequant path per tensor shape and stream E4M3/NVFP4 W4 activations into the existing matmul spine.[201] Gate: `make test_simd_quant` (per-shape dispatch, bit-identity on NaN/Inf, no regression on IQ2_S). **Size L**. Depends on LP-11.

### LP-16 — State/ABI semver boundary
Version the opaque state files and session serialisation the way llama.cpp/ggml v1.0 and slermes' goal state do; refuse stale versions with a clear reason.[76][110][133] Gate: `make test_state_version` (current loads, stale rejected, migration named). **Size S**. Independent.

## Accepted risk

Paged-KV fault attacks surfaced in this sweep only as an untitled stub with no extractable primary source. WuBuOS is an agentic OS insecure by design; users are told not to bank on it. Recorded once, not turned into a pad.

## What not to break

- Opaque structs and status-as-value at every module seam.
- No third-party linking — slerm means read and reimplement, never vendor.[179][180]
- Runtime kernel dispatch via `wubu_kernel_register()`; LP-10 adds compiled rows, it does not hardcode a backend.
- KV namespace remains a 9P-exportable filesystem; LP-05/12 add gates around it, not a replacement.
- The engine core stays ignorant of goals, hive cells and orchestration; LP-01..04 live in the host layer.
- `make test_all` and the colony gate remain the release truth. No pad ships without its own gate.

## Sources

[10] https://arxiv.org/abs/2611.21216
[12] https://arxiv.org/abs/2611.11319
[16] https://arxiv.org/abs/2611.10511
[23] https://arxiv.org/abs/2611.21852
[27] https://arxiv.org/abs/2610.25519
[30] https://arxiv.org/abs/2610.41970
[33] https://github.com/waefrebeorn/slermes — slermes - C11 Hermes Agent (waefrebeorn)
[38] https://arxiv.org/pdf/2611.06606 — Verifier-Guided Self-Consistency (VGSC)
[39] https://arxiv.org/pdf/2610.11223v1 — SafeInferCom: Safe Inference-Time Compute via Verifier-Guided Mid-Generation Intervention
[76] https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0 — llama.cpp v1.0.0 GitHub release
[91] https://arxiv.org/html/2610.22316 — Learning to Steer LLM Agents and Multi-Agent Systems (steering agent, goal drift)
[99] https://learnaivisually.com/ai-explained/learned-steering-fast-path-monitor-slow-path-intervention — Learned steering supervisor: fast-path monitoring vs slow-path intervention (explainer)
[103] https://github.com/deepseek-ai/DeepInfer
[104] https://github.com/RueClaw/public-data/blob/main/reviews/deepinfer.md
[108] https://github.com/ovg-project/kvcached
[110] https://github.com/ggml-org/ggml/releases
[133] https://github.com/waefrebeorn/slermes/commit/f95fb59d456894691c153665d5712bc19605c028 — slermes docs: Persistent Goals feature page
[147] https://arxiv.org/html/2610.17724v1 — Evidence-First Agentic Verification (Sauron verifier)
[148] https://arxiv.org/html/2610.29609 — Who Verifies the Agent's Success Claim? Self-Report Leakage in GUI agents
[176] https://github.com/ovg-project/kvcached/releases/tag/v0.1.6
[179] https://github.com/incoai/splash/pull/573
[180] https://github.com/synchopate/llama-modchip
[189] https://arxiv.org/abs/2610.00906
[193] https://arxiv.org/abs/2610.05300
[199] https://autosaddler-projectpage.github.io/activesaddler
[200] https://github.com/ggml-org/llama.cpp/pull/32309 — llama.cpp PR 32309: reuse SWA cells shared by several sequences
[201] https://github.com/deepseek-ai/TileKernels — deepseek-ai/TileKernels repo README
[206] https://github.com/Niko1221/Strata/pull/4222 — Strata PR 4222 tools+JSON token masks
[207] https://github.com/sgl-project/sglang/releases/tag/v0.5.23 — SGLang v0.5.23 primary GitHub release notes
[215] https://github.com/ggml-org/llama.cpp/issues/32258 — llama.cpp issue 32258 unified SWA shared-cell slot bug

## Wave 8 — long context (1M) on one 16 GB box

Model facts that set the problem, read from the GGUF rather than assumed: `inkling2.context_length = 1048576` and
`inkling2.rope.dimension_count = 0` with a `relative_length = 16` scheme, so this model is natively 1M and needs no
positional extension at all - no YaRN, no NTK, no continued training. `inkling2.attention.sliding_window = 1024` with
pattern `[T,T,T,T,T,F]`, confirmed by IN2_KV_TRACE (full attention only at visits 5,11,17,23,29,35,41,47), means 8 of
72 KV visits ever read past 1024 positions. Full-attention KV at 1M is 8 x 1M x 1024 dims x 2 x 4 B = 68.7 GB f32.
The 12-slot reading in the briefs assumes the loop passes duplicate attention visits; the trace says they do not, so
the budget here is the 8-slot one.

### Measured: the kernel is at parity, and prefill is memory-bound because the MoE runs per token

Three measurements settle what "prefill is slow" actually means, and two of them contradict what I first wrote.

1. **The IQ2_XXS kernel is not the problem.** `tools/iq_kernel_wall.c` drives the real kernel over 4096 rows x 2304
   and gets **19.08 G weights/s single-threaded = 38.2 GFLOPS, 13.41 ns per 256-weight block against upstream's
   14.94** - i.e. 1.1x the reference. There is no 4-5x kernel headroom; that hypothesis is dead. (Note the bench must
   be L3-resident to measure this; 2.32 MB of weights fits, and at that size it measures compute, not memory.)
2. **Prefill re-reads the weights every token.** The engine's per-type timer shows TY16 calls scaling exactly with
   token count (4348 / 13035 / 43440 for 16 / 48 / 160 tokens) with throughput flat at ~8.7 GOPS. That is the
   signature of nt=1 per call. The cause is in `in2_moe`: `for (t < n_tok) { #pragma omp parallel for ... }` - a fresh
   parallel region per token inside the token loop, so the 8 experts' weight slabs are streamed once per token. The
   attention projections DO batch (`in2_matvec_nt(ctx, &L->wq, xn, n_tok, ...)`), which is why my earlier "batching is
   already there, it buys 11%" correction was right for the projections and wrong for the MoE - and the MoE is 72% of
   the FLOPs (63.70 of 87.88 M MAC per layer).
3. **Prefill is therefore memory-bound, not FLOP-bound.** Thread sweep on a 48-token prompt: 1 thread 29.7 s, 2 ->
   25.5, 4 -> 16.9, 6 -> 15.2, 8 -> 13.25 s. That is 2.24x over 8 threads with saturation past 4, which is a
   bandwidth ceiling. `MAYA_IQ2XXS_1T=1` changes nothing (47.547 s vs 47.563 s on a 320-token prompt), so the cheaper
   IQ2_XXS decode step is not the lever either.

Consequence: batching the MoE across tokens turns prefill from memory-bound into compute-bound and is worth up to ~5x
(13.25 s -> ~2.6 s on the 48-token probe; ~60 h -> ~12 h for a 1M prompt), which is the real content of LP-17. The
work is: route all tokens first (that matvec batches), bucket token indices by expert, run one nt-sized matvec per
expert, scatter back with the routing weights.

- **LP-17 - prefill is at the hardware FLOP ceiling, so this pad is about fewer FLOPs, not faster batching.**
  Measured and corrected from an earlier guess: prefill costs 0.195 s per position, the decode step costs 0.22 s per
  token, and `in2_matvec_nt` already takes `n_tok` - so batching at 320 tokens buys ~11%, not 3x, because weight
  traffic is already amortised and the limit is arithmetic. Active compute is ~90M MAC/layer x 48 layers = 8.6 GFLOP
  per token, which at 0.195 s/position is 44 GFLOPS sustained against a measured machine ceiling of 26 GFLOPS fp32
  (the IQ kernels use integer MACs, which are wider, so 36-44 is peak rather than a shortfall). Consequence, stated
  plainly because it bounds every other pad: **1M prefill is ~62 hours and 512k ~31 hours on this CPU at hardware
  peak.** No amount of batching, prefetch or cache work changes that; only fewer FLOPs per token does. Gates: (a)
  a measured FLOP/token reduction from sequence pooling (LP-18) or from attending fewer keys, and (b) a documented
  decision about ingest - offline prefill, a GPU, or a shorter working context.
- **LP-18 - quantised KV for the full visits, dequant fused into the score dot.** Gate: 128k context resident with the
  5.2 GB model on this 16 GB box, and identical argmax over 50 decode steps at 4k versus the f32 cache. KV quant is
  NOT bit-exact, so the gate is token equality and the commit must say so. Budget: 1M is 68.7 GB f32, 34.4 q8, 17.2
  q4, 8.6 at ~2 bits. Unpack cost measured on this box per value: f32 100.7 ps, q8 19.7, q4 13.7, q2 scalar 48.3, q2
  with AVX-512 `vpmultishiftqb` 7.9 - the 2-bit path is only viable vectorised. The 60 sliding rings stay f32 and
  exact; they cost 0.5 GB at any context.
- **LP-19 - selectivity, because residency alone is not enough.** Gate: decode at 128k >= 2x dense with needle
  retrieval >= 90% at 128k. Mechanism: block/page selection with page >= 256 - Quest's own numbers are page 16 = 6.44
  GB scanned per step = 184 ms against page 256 = 402 MB = 12 ms, i.e. the index can cost more than the attention it
  saves - and one indexer per KV *group*, not per query head, since GQA heads sharing a KV slot must take the union of
  their selections. Eviction is a hard cap only: Quest's 100k passkey table kills H2O/TOVA/StreamingLLM, and their
  RULER slope is ~8 points per doubling.
- **LP-20 - KV persisted to disk, so a long prefill is paid once.** Gate: restore a 100k-token KV in <= 5 s and
  continue decoding with the same tokens. This is the only shape that makes 1M a product on this box: prefill once,
  persist 12.9 GB at q4, then answer at ~4 tok/s with LP-19 selectivity. Restore cost from the literature: 100k tokens
  = 1.7 s with GDS (4.74 GB/s) to 5.0 s with O_DIRECT (1.97 GB/s). Use NO_BUFFERING so KV streaming cannot evict the
  5.2 GB model, and preallocate the file (append-extend measured 537 MB/s against 1206 MB/s preallocated).

### Reading: DeepSeek-V4.1-Flash (arXiv 2609.19969), and what does not retrofit

Anchor: 890 B/token of global KV at 1M for a 552B MoE in 40 layers, ~1/4 of DeepSeek-V4-Flash, plus SWA Bounded
Replay taking the *persistent* cache to ~1/8. Ours is 65,536 B/token (8 full visits x 1024 dims x 2 x 4 B) = 68.7 GB
at 1M. The gap is not one trick; it is three dimensions multiplied.

- **Sequence-dimension compression is the lever we are missing.** CSA2 folds every m tokens into one KV entry
  (ratio 1 = no compression). We have none: every full-visit KV row is one token. This is the only dimension that
  reduces the per-token READ as well as the footprint, and therefore the only one that moves the ~4.3 tok/s ceiling
  that all-resident f32/q4 KV is stuck at. At m=8 the 1M budget is 8.6 GB with no precision loss at all; m=4 plus q4
  is 4.3 GB; m=8 plus q4/fp4 is 2.1 GB, which reads in 47 ms and puts the ceiling near 21 tok/s instead of 4.3.
  DeepSeek learned the compressor; a mean/max-pooled retrofit is an approximation and has to be measured before it
  is believed (needle retrieval at 128k is the gate, and it cannot be measured until prefill is affordable at all).
- **`SWA Bounded Replay` transfers directly and is free.** Their finding: exact reconstruction of the decoder's SWA
  state needs L x n_win replayed tokens, but replaying only n_win is negligibly worse, so SWA KV is never persisted.
  For us the 64 sliding visits are 0.5 GB and never grow, so this changes nothing in RAM - but it is exactly what
  LP-20 needs: persist only the 8 full visits and replay the last 1024 tokens on resume, instead of checkpointing
  0.5 GB of sliding state on every save.
- **FP4 global KV is trained, not post-hoc.** They quantise during training and call the degradation marginal. Our
  LP-18 must therefore measure post-hoc q8/q4/FP4 against the f32 cache rather than cite their result.
- **CED (decoder global KV projected from H_{L/2}) halves prefill but cannot be retrofitted** - it needs a trained
  per-layer W^KV_l, and ours cannot be retrofitted from it either.
- **CSA2's Full/Reindex/Reuse modes cannot be retrofitted either**: which layers share KV and indices is a static
  trained assignment. The transferable half is Reuse of *Top-K indices* across neighbouring layers, which is a
  runtime approximation (cheaper indexing, identical outputs only if the reused index set happens to match) and
  belongs to LP-19 as an optimisation, gated on retrieval quality.
- **Their decode-FLOPs curve confirms LP-19's ordering**: 4K -> 1M grows single-token decode FLOPs by only 1/4
  because the indexer plus Top-K bounds the work. That is the same conclusion the Windows measurements reached from
  the other direction - selectivity, not residency, is what makes long context usable.

### The measured 1M wall: 0.13 tok/s from the score loop alone

`tools/attn_wall.c` measures this engine's attention score loop directly, because the bandwidth arithmetic above
(4.3 tok/s for any all-resident KV) assumes the maths is free and it is not:

    1,048,576 keys x 128 dims, single thread: 0.060 s per (head,visit) pass = 57 ns/key = 2.24 GMAC/s
    8 full visits x 8 kv heads = 64 passes, x2 for QK and PV = 7.67 s/token => 0.13 tok/s

So at 1M the binding wall is the NUMBER OF KEYS ATTENDED, not the bytes stored, and it binds about 30x harder than
bandwidth does. Compressing the sequence dimension by m is therefore the master lever rather than an optimisation:
m=8 projects to ~1 tok/s and m=32 to ~4 tok/s with the heads spread over 6 cores. The loop does vectorise (AVX-512
vcvtps2pd/vmulpd) but a double reduction cannot be reassociated under -std=c11, so it sits at ~2 cycles/element;
breaking that chain would change the summation order and void the token-parity contract, which is why the answer is
fewer keys and not a faster dot product.

Revised Wave 8 order, on the evidence rather than on intuition: **LP-17 still gates everything**, because a needle
test at 128k-1M is how m, quantisation and selection get validated and prefill at 3.5 tok/s makes such a test
unaffordable. Then **LP-18 becomes sequence-first**: fold every m tokens into one entry, measure m = 2/4/8/32 on
retrieval, and only then spend the precision dimension (q8 -> q4/fp4) on what is left. Selection (LP-19) is the
second lever on the same axis - it reduces the key count per query without changing what is stored - and the two
compose multiplicatively with quantisation.

Sharpened LP-18: compress along the sequence dimension FIRST (m = 2/4/8, pooled, measured), then precision (q8 ->
q4/fp4), then consider index reuse. Footprint and read bandwidth both scale with m, and m is the difference between
a 4.3 tok/s ceiling and a usable one.

### The box is on the Balanced power plan, which is a measurement hazard

Every throughput number in this register was taken on the **Balanced** scheme (`381b4222`), which was the only scheme
present on the machine until now. Two settings matter and both were read directly:

    powercfg /query SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN
        Current AC = 0x50 (80 %)      Current DC = 0x05 (5 %)

So on battery the CPU is allowed to sit at 5 % minimum processor state, which would gut sustained inference, and on AC
it is held no lower than 80 % but never asked to run at 100 %. `MaxClockSpeed` reports 4301 MHz and
`\Processor Information(_Total)\% Processor Performance` read 94.5-99.6 % during load, i.e. the part was **not**
boosting at all - it was sitting just under nominal.

Created but not yet adopted: `High performance` (fda6462b-bd6b-479e-8540-6f39d95cf766) and `Ultimate Performance`
(83161a31-a738-495a-aef5-79af342790c8), both via `powercfg -duplicatescheme`.

**Pending A/B, and the reason it is pending matters:** three consecutive 320-token prefills measured 55.8 s, 79.6 s
and 70.8 s against 37.4 s for the same work earlier. That looks like a 2x sustained-load fall-off and it is NOT one
yet - five research subagents were running local tool calls at the same time, which is exactly how a `-march` A/B got
polluted earlier in this project (base swung 2.65 -> 1.02 tok/s and the numbers were discarded). The clock counters
said 94-99 % throughout, so it was not thermal throttling either.

To run when the box is quiet: Balanced vs High performance vs Ultimate Performance, 320-token prefill, three
interleaved reps each, recording prefill seconds and `% Processor Performance` per rep. Until that exists, treat all
absolute GFLOPS figures as "Balanced scheme, idle box" and do not compare them against measurements taken while any
of these research fan-outs is running.

Evidence base, all local, each with its own verified citation ledger:
`C:/Users/eman5/kv-1m-brief.md` (KV representation), `C:/Users/eman5/longcontext-kv-offload-brief.md` (engines),
`C:/Users/eman5/kvbrief/KV-MEMORY-BRIEF.md` (Windows memory, measured on this box, with kvbench.c),
`C:/Users/eman5/AppData/Local/hermes/cache/scratch/pe/positional-extension-512k-brief.md` (positional; moot here).

Two results worth carrying although they are not pads: KTransformers is the only engine measured at 1M on a single
device (16 tok/s on a 24 GB GPU) and it gets there by attending ~16k of blocks, not by shrinking KV to fit; and
llama.cpp hit a hard bug at `n_kv * n_ubatch = 2^32`, so mask or index arithmetic added for long context must be
64-bit. This engine has no masks and its KV offsets are already size_t, so that bug class does not apply as it stands.

## Wave 9 — tandem GPU (CPU and GPU busy on the same token)

Three engines already do this on hardware close to this box, and they converge on one design. Strata (Niko1221) is
written for exactly this problem - a 125B model on 12 GB of VRAM, 64 GB of RAM and a six-core desktop - and its own
description puts the split as follows: the GPU holds attention and the DeltaNet mixers, the gated-residual weights,
the routers, the shared experts, the output head, the MTP draft layer, part of the KV cache, and an **expert cache**
that fills the rest of VRAM with the most-used experts and adapts during the conversation; the CPU computes the
experts the GPU does not hold **in place, at the same time as the GPU works on the cached ones**. The concurrency is
explicit and worth stealing: for each layer the GPU writes the chosen expert ids into a small block of pinned host
memory (a "doorbell") and the CPU **spins on the doorbell instead of waiting on the driver**, immediately splitting
the experts into cache hits (GPU), copies to stream (copy engine, concurrently) and CPU work. Strata reports copying
55% of the misses for i-quant files but only 20% for Q2_0, because "whose CPU needs the RAM bandwidth itself" - i.e.
the fetch/compute split is a function of what else the CPU is doing, not a constant.

Avalon (a Ryzen 7 3800XT + RX 7900 XT 20 GB, PCIe 3.0 x16, measured 33-35 GB/s of DDR4) runs the same shape and is
the best-measured datapoint for the payoff: **66.1 t/s on a short prompt / 98.4 with MTP**, **57.3 / 81.1 t/s at 256k
context**, 1,854 t/s prefill. Its warning is the one that matters for us: refusing the split and keeping all experts
on the CPU "throws away the VRAM tier entirely and drops you back to baseline speed".

project-maya is where this repo's CPU kernels came from, and it contributes the four details that make a split engine
actually fast: (a) a **static per-layer placement plan** ("1-2 PCIe fetches/layer, up to 7 CPU", PR #234) rather than
per-token optimisation; (b) **multi-token expert rows** so several tokens sharing an expert decode the weights once,
bit-identically - measured 455/681/889/1794/3554 us against 333/375/434/864/1684 us for 1/2/3/4/8/16 tokens, i.e.
~2.1x at 4 and 16 tokens (PR #154), which is the same change this repo landed in d17b51f; (c) an **AVX-512 one-token
IQ2_XXS row that reads neither the codebook nor the sign table** (PR #227), because ggml's AVX2 dot "spends most of a
block on the codebook: 32 scalar grid loads and 32 sign-table loads per 256 values", worth +6-8% decode; (d)
**Aged-LFU VRAM eviction with a promotion floor** (PROMOTE_MIN) to stop expert-promotion thrash.

### The budget for THIS box, from the file

    Inkling-2-Mini-UD-IQ2_XXS.gguf   12,444 MiB of tensors
      experts              11,360 MiB   91.3%
      attention               587 MiB    4.7%
      head+embed              469 MiB    3.8%
      norms/scales/small       28 MiB    0.2%
    non-expert total        1,084 MiB   -> fits VRAM outright
    VRAM free (6141 total)   5,920 MiB
    left for expert cache + KV  4,836 MiB  = 43% of the expert population
      (KV costs 64 MiB at 1k ctx for the 8 full visits, 1,024 MiB at 16k)

So the split is not a compromise, it is arithmetic that happens to be favourable: **every non-expert tensor is 1,084
MiB and can live in VRAM permanently**, and 43% of the expert population fits on top of it. Per token the engine reads
dense 696 MB and experts 369 MB; moving the non-expert tier to VRAM puts ~930 MB of the 1,535 MB step on a 192 GB/s
link and leaves the CPU only the expert misses.

**Correction, because this figure was wrong in two earlier drafts of this note:** the output head is **234.7 MiB**, not
469. `output.weight` is `[2304, 201024]` in IQ4_XS, which is 463.2 M weights at 136 bytes per 256 = 234.7 MiB; 469 MiB
is head **plus** `token_embd.weight`, which is a separate tensor of the same shape (the head is not tied). So the head is
~15% of a decode step, not 31%.

One measurement decides the fetch/compute ratio, and it comes out in favour of fetching: PCIe 4.0 **x8** on this
laptop is ~10-13 GB/s real (nvidia-smi reports link width 8), so an 0.83 MB expert slab costs ~70-85 us to copy,
against ~119 us to compute on the CPU at this engine's measured 7 GB/s. Fetching a miss is therefore faster than
computing it whenever the link is idle - which is exactly why Strata streams 55% of i-quant misses rather than
computing them, and why its share drops when the CPU needs the bandwidth.

### What already exists here, and what does not

Already in the tree, unused by the Inkling path: `src/wubu_cuda_quant.cu` (device-side IQ2_S block dot with the grid
and sign tables resident as `__device__` symbols, plus a generic type-dispatched `matvec_kernel`),
`src/gpu_wubu.cu` (a VRAM weight cache: `wc_get`, `gpu_wubu_matmul{,_nt,_tx}`, `gpu_wubu_mark_weights_dirty` and a
generation counter), `src/wubu_model_gpu.cu` (1,559 lines), `src/flash_attn_*.cu`, `src/kv_arena.cu`. CUDA 13.5
builds; the card is compute capability 8.9 (Ada). So the device side is further along than the CPU side was, and what
is missing is exactly the part the three engines above are built around: a **tandem scheduler** - a placement plan,
the doorbell, and the CPU/GPU split - not more kernels.

### What the checkpoint gives for free, read out of the GGUF

- **No MTP, no draft head.** 734 tensors, none matching `nextn`/`draft`/`mtp`/`eh_proj`/`enorm`/`hnorm`, and nothing
  outside `blk.*` except `loop_kv_res_query`, `loop_res_query`, `output`, `output_norm`, `output_res_query`,
  `token_embd`, `token_embd_norm`. So every multi-token-per-pass scheme has to be **draft-free** (prompt lookup,
  lookahead/Jacobi, self-speculative layer skip) - there is nothing to hang a Medusa/EAGLE head on.
- **The head is untied and re-quantisable.** `output.weight` is IQ4_XS at 234.7 MiB per token. At IQ2_XXS it would be
  113.9 MiB, saving **120.8 MiB = 8.2%** of the measured 1,535.5 MB step - the largest single arithmetic saving
  identified so far that needs no new kernel, only a re-quantised tensor. It is not free: greedy output moves unless
  the top logit survives 2 bits, which is a measurable question and not an assumption.
- **The sparse pattern is in the metadata, not inferred.** `attention.sliding_window_pattern` is an explicit array of
  48, alongside `attention.relative_extent 1024`, `attention.relative_length 16`, `attention.qk_clip 22.63`,
  `attention.log_scale_floor 8192` and `expert_weights_scale 8.0`. The 8-full/64-sliding split this project derived by
  tracing is therefore a declared property of the file, and any code that recomputes it from behaviour can read it
  instead. **Verified, not assumed:** the array decodes to full attention on exactly ,
  8 full and 40 sliding out of 48 layers, which is precisely what  traced - so the engine and the file
  agree on the one structural fact the whole KV plan rests on.  with  at
   likewise produces the 3072-wide  and 1024-wide  tensors actually present.
- **The dense tier is mostly Q8_0** (`attn_output` and friends are type 8, 1.06 bytes/weight), which is why the
  non-expert tier costs 587 MiB for only 24.18 M MAC/layer. Re-quantising attention to IQ4_XS is a second 8%-class
  arithmetic saving, at a larger quality risk than the head.

### Measured: the GPU library builds here, but the tier is not ready

`wubu_cuda.dll` builds on this box (657 KB, sm_89, CUDA 13.5, MSVC 14.44 via `-ccbin`), and
`tools/gpu_iq_wall.c` drives the host-callable entry `wubu_cuda_quant_matvec_host` to find out what it can do. Three
results, the first of which matters most:

- **FIXED: IQ2_XXS (type 16) now computes on the card.** `iq2xxs_matvec_mrow_kernel` + `wubu_cuda_upload_iq2xxs` +
  type-16 dispatch, verified against the exact dequantised reference in `tools/gpu_vs_cpu_iq.c`: max |ref-gpu| =
  **2.86e-06** on a reference magnitude of 35.9, max relative 1.07e-3 with 1 of 2304 rows marginally over (float
  reduction order - the warp sums 32 lanes, the reference sums sequentially in double). The tables had to be uploaded
  for this: `c_ksigns_iq2xs` was declared on the device and **never populated**, which nothing noticed because the
  IQ2_S kernel reads its signs out of `qs[32+...]` instead of the table.
- **Was:** IQ2_XXS (type 16) returns zeros while reporting success. `rc=0`, `y[0]=0`, `y[n-1]=0`. That is the exact silent
  no-op class this engine's own comments warn about ("a half-finished type returning 1 silently changed the logit from
  29.67 to 22.52"). 25.5% of this model's weight bytes are IQ2_XXS, so the dominant format for the VRAM tier has **no
  working device path yet** - only IQ2_S, Q2_0 and Q8_0 do.
- **End-to-end host-call throughput is launch/copy-bound at these sizes.** 2304x2304 slab (1.31 MB):
  IQ2_S 7.33, Q2_0 9.58, Q8_0 9.15 G weights/s, against 19.08 G weights/s for one CPU thread. The copy measures
  2.4 GB/s against a PCIe 4.0 x8 expectation of ~10-13, so at 1.3 MB per call this is synchronisation and kernel
  launch, not transfer - the measurement says nothing yet about a *resident* slab, which is the regime the design
  actually needs.
- **Environment trap, worth fixing in the tools:** `cuda-env.sh` adds `$CUDA_ROOT/bin` but the runtime lives in
  `bin/x64`, so any nvcc-built DLL fails to load with a bare exit 127 that names no missing file. And a harness linked
  without `-static` picks up `api-ms-win-crt-private-l1-1-0.dll`, which this Windows build does not provide - the same
  silent-start failure. Both cost a build cycle each to find.

So LP-21's first task is not the doorbell but a device-resident IQ2_XXS dot that agrees with the CPU kernel, because
without it 45% of the per-token bytes (dense attention projections are partly IQ2_XXS/IQ2_XS) cannot live on the card
at all. The ordering changes accordingly: **LP-21 becomes "a bit-exact IQ2_XXS device row kernel, verified against
`tools/iq_kernel_wall.c`'s CPU reference"**, and the doorbell moves to LP-22.

### RESIDENT IQ2_XXS on this card: 219 G weights/s, 11.5x the CPU kernel

`tools/gpu_resident_bench.c` uploads a slab once and times only kernel launches (`wubu_cuda_iq2xxs_bench_host`), which
is the regime a tandem engine keeps the dense tier in:

    slab MB      rows     kernel us    G weights/s    GB/s read
        1.3      2294          33.8         156.41        40.32
       16.0     28244         303.1         214.69        55.35
       64.0    112977        1206.3         215.79        55.63
      256.0    451911        4762.6         218.62        56.36
     1024.0   1807646       19046.4         218.67        56.38
     4096.0   7230584       64193.5         259.52        66.91

So a resident IQ2_XXS slab streams at **~219 G weights/s / 56 GB/s**, against **19.08 G weights/s for one CPU
thread** and an aggregate CPU rate of ~43 G weights/s. That is **11.5x**, and it is the first measurement in this
project that makes the tandem split an arithmetic conclusion rather than an analogy with Strata.

It also prices the two regimes exactly. The per-call host path (`wubu_cuda_quant_matvec_host`, which copies first)
measured **8.93 G weights/s** on the same 1.31 MB slab - **25x slower than resident** - because a 1.31 MB copy at
2.4 GB/s is 550 us of PCIe against a 34 us kernel. That is why the dense tier has to be *resident* rather than
streamed per call, and why Strata's doorbell and copy-engine split exists at all.

### What that is worth per token, on measured numbers

    resident non-expert tier        1,084 MiB (attention 696 MB read per token + head 235 MB)
      at 56.4 GB/s                  ~16.5 ms
    experts, all from CPU            369 MB at the engine's measured ~7 GB/s  = ~53 ms   -> 18 tok/s
    experts with a 43% VRAM cache    ~210 MB miss at ~7 GB/s                 = ~30 ms   -> 31 tok/s

Both against **4.5 tok/s today**, because the CPU alone re-reads all 1,535 MB. The tandem split is therefore worth
**4x at zero cache hit rate and ~7x if the 4.8 GB expert cache behaves**, before any kernel work on the CPU lane.
The binding constraint is now the CPU lane's 7 GB/s, which is the number LP-24 and the thread-scaling front attack.

### Pads

- **LP-21 - the doorbell and the split loop.** Gate: one layer executes with the GPU doing the attention projections
  and the CPU doing the expert misses concurrently, and the token is bit-identical to the CPU-only path. No timing
  claim until the tokens match.
- **LP-22 - resident non-expert tier.** Gate: the 1,084 MiB of attention + head + routers + shared experts live in
  VRAM and the CPU never reads them again; measured as a decode step that issues no host reads for those tensors.
- **LP-23 - expert cache with Aged-LFU + promotion floor.** Gate: cache hit rate measured over a long generation, no
  promotion thrash (hit rate must not fall while the cache is full), and a documented fetch-versus-compute decision
  rule using the ~70-85 us copy against ~119 us compute figures above.
- **LP-24 - the AVX-512 one-token IQ2_XXS row without the codebook** (maya PR #227's trick): the CPU lane is the
  long pole for misses, and this is +6-8% on it in their measurement.

### LP-25 killed by arithmetic before it was written: this engine does not amortise a batch

Draft-free multi-token decode (prompt-lookup, lookahead, Jacobi, Medusa-style) rests on one premise: a pass over K
candidate positions costs about one position's worth of work, because the weights are read once and positions are
nearly free. That premise holds for a bandwidth-bound engine. **It does not hold here**, and the numbers to see it were
already in this register:

    320-token prefill    37.4 s   = 0.117 s per position
     48-token batch     10.31 s   = 0.215 s per position
    single-token decode  0.22  s  = 0.220 s per position

Per-position cost barely moves between a 48-wide batch and a single token; only the 320-wide pass is 1.9x cheaper per
position, and that is the pass where a 7.9 GB fault-in is amortised over 320 positions rather than 48. So this engine
charges roughly **one position's arithmetic for each position in the batch** - which is what a compute-bound engine
does, and it is consistent with decode measuring 4.22 GMAC / 0.22 s = 19.2 GMAC/s = 38 GFLOPS aggregate against a
38 GFLOPS single-thread kernel rate, i.e. the box is delivering one core's kernel throughput for the whole step.

The consequence is a hard kill on speculation at the current per-position cost. A 9-position proposal pass costs
~9 x 0.18 = 1.6 s. If 40% of proposals match (a generous rate for prose), an average pass emits 1/(1-0.4) = 1.67
tokens, which sequentially costs 1.67 x 0.22 = 0.37 s. **Speculation would be 4x slower than plain greedy.** For it to
break even the pass would have to cost under ~0.37 s, i.e. under 0.04 s per position - 5x cheaper than today's
0.22 s.

So the lever is not "more tokens per pass", it is **one position made cheaper**, and every remaining path runs through
that: the device IQ2_XXS row kernel (LP-21), the thread-scaling wall, and the 66-byte block access pattern. This is the
same conclusion `flop_count.py` reached from the other direction when it showed prefill sitting at 43 GFLOPS against a
~228 GFLOPS ceiling.

Recorded as LP-25 = **do not build**, with the reopen condition stated: if a decode step falls below ~0.05 s
(bandwidth-bound regime, e.g. once a working device tier carries the dense weights), re-run this arithmetic, because
the conclusion inverts exactly there.
### LP-26 - the attention projections still run on float32 materialised weights, and that is bigger than the token

Hooking the resident tier into `in2_matvec_nt` engaged nothing, and the reason is worth more than the hook.
`attn_q`, `attn_k` and `attn_v` never reach a quantised kernel at all: in `wubu_inkling2_fullseq.c` those three
projections call `in2_matvec` on a **float32 matrix** (lines 139, 171, 172) with no `IN2_FULL_KERNEL` alternative,
while `wrel`, `sink`, `wgate` and `wo` all have one. So q/k/v materialise their weights to floats, and the arithmetic
is stark:

    attn_q  2304 x 2304 float32 = 20.2 MB per layer
    attn_k  2304 x 1024          =  9.0 MB
    attn_v  2304 x 1024          =  9.0 MB
    --------------------------------------------
    per layer                      38.2 MB  x 48 layers = 1,836 MB

    the same weights as Q8_0: attn_q 5.4 MB + k/v 4.8 MB = 10.2 MB/layer x 48 = 488 MB

**1,836 MB against 488 MB.** That is larger than the entire measured 1,535 MB of quantised traffic per token, and it
is invisible to the per-type instrumentation, because float materialisation never enters a typed kernel - the type
table still reads 16/17/19/22/40/42 and says nothing about it. It also explains two things this register already
records without connecting them: the 4.9 GB resident set on a model whose quantised weights are 12.4 GB but whose
hot floats are gigabytes, and why the dequant cache had to be sized in the hundreds of MB.

**LP-26 correction - the per-token figure above is wrong and the measurement says so.** `in2_tensor_to_f32` does
cache (`in2_dq_tab`): it dequantises once and hands out either a copy or a borrowed view. The engine already counts
both, and over a 4-token decode the counters read:

    DQ  f32 produced 557.5 MB, copied 65.6 MB, borrow miss 235 hit 1793, to_f32 calls 2959

That is **~16 MB copied per token, not 1,836 MB**, and 1,793 of 2,028 lookups were borrows (no copy at all). The
arithmetic agrees: 1,535 MB of quantised traffic plus 1,836 MB of float traffic at the measured ~7 GB/s would be
~0.48 s per token, against 0.22 s on record. Both cannot be true, and the counter is the one with evidence behind it.

So the 38.2 MB/layer figure is a **residency** fact, not a traffic fact: the q/k/v of the float path occupy about
557 MB of f32 in the dequant cache and are mostly borrowed rather than copied. It also means the incremental decode
path does not run float q/k/v per token at all - the type table's silence about type 8 is because those projections
are quantised on that path, and `in2_layer_full`'s float projections are the prefill/`n_tok>1` path.

What survives of LP-26: **1,836 MB against 488 MB is still the right comparison for the fullseq path**, and it is
still worth routing q/k/v through `in2_matvec_nt` there, but it is a prefill cost and a residency cost, not a
decode-per-token cost. What does not survive: quoting it as per-token. Recorded here rather than quietly edited,
because the wrong version was carried for one commit.

## Wave 10 - what five deep briefs changed, and where they were wrong about us

### The roofline correction: our kernel is not at the memory wall, and the brief that said so assumed our format wrong

The toolchain brief measured this exact silicon class and concluded the quantised GEMV is "indistinguishable from a
pure sequential read" - 53.44 GB/s against 55.95 GB/s for an AVX2 byte-sum over the same 64 MiB - and therefore that
no codegen flag can help. That measurement is sound and it is the right way to test the claim. But its roofline check
assumes **1 byte per weight** (int8). This model is IQ2_XXS at **66 bytes per 256 weights = 0.258 B/weight**, so:

    19.08 G weights/s single thread = 4.92 GB/s of weight bytes   (not 19 GB/s)
    43.5  G weights/s aggregate     = 11.2 GB/s of weight bytes   (not 43.5 GB/s)
    measured stream ceiling, same box: 32-40 GB/s single thread, 42-55 GB/s aggregate

So the IQ2_XXS kernel runs at **~15 % of single-thread stream bandwidth and ~25 % of aggregate**, i.e. it is
**dequant-bound, not bandwidth-bound**, and the 2.28x thread ceiling is *not* DRAM saturation. That single arithmetic
error inverts the brief's headline conclusion for our case: codegen and ISA work *does* have room here, and the gain
is on the decode side of the block, not the load side. It also explains why maya's codebook-avoiding AVX-512 row was
only worth 6-8 % to them: their dot is already past the LUT, ours is not.

### What is real, measured on our box or at our shape

- **`__builtin_prefetch` is negative, consistently.** 128 B ahead -0.9 %, 512 B -2.8 %, 1 KiB -6.0 %, 2 KiB -3.2 %,
  and -5..-12 % in earlier trials; the literature agrees on regular streams. We have `iq_env_pf_rows` prefetch in the
  IQ row kernels. It is worth deleting rather than tuning - and it is consistent with the flat `IN2_IQ_PF_ROWS=0/8`
  result already in this register.
- **mingw `target("avx2")` costs 16 vector registers on Zen4.** Under `-march=native` gcc keeps the activation vector in
  ymm16-22 (EVEX); inside a `target("avx2")` function only ymm0-15 exist, so the vector spills and every
  `vpmaddubsw` gains a stack operand - measured 0 to -25 % against a globally `-mavx2` build. Our kernels are built
  from `target` attributes. Also: `target` + `always_inline` intrinsics is a hard compile error in mingw gcc (a
  13-year-old open bug), which is a build hazard we have presumably avoided by luck of structure.
- **Theatre on PE, confirmed by byte-identical objects:** `-fno-plt`, `-fno-semantic-interposition`, alignment
  tuning, unroll pragmas, LTO-for-speed, `-static`/`-static-libgcc` (hot loop byte-identical). Huge pages fail with
  error 1314 (no `SeLockMemoryPrivilege`) and measured neutral-to-worse for streaming weights anyway. BOLT is ELF-only
  with no COFF plan. clang's +4-7 % exists only in the L2-resident regime.

### Expert cache: LRU is already near the causal ceiling, and Strata's prefetch brought zero

Measured on **Qwen3-Coder-30B-A3B, 48 layers / 128 experts / top-8** - our exact shape - at 43 % residency:

    LRU 80.7-82.4 %  |  LFRU 82.0 %  |  shared global pool +1.5 pp  |  Belady 90.1 %
    ARC / LRU-2 / W-TinyLFU: never more than 2.4 pp above shared LRU, W-TinyLFU up to 13 pp below

So the plan in Wave 9 should use **Aged-LFU (halve the counter every 64 tokens) with a global pool over all slots**,
not per-layer caches, and should not bother with ARC/TinyLFU. 84.3-96.6 % of the gap to Belady is *knowing the future
victim*, which eviction policy cannot supply.

The future is available one layer ahead, from the router we already have: applying `ffn_gate_inp[L+1]` to the residual
entering layer L recovers **84 % of cache misses** at 87-90 % top-8 agreement, for 0.5 % of one expert's FLOPs and no
training. Two layers ahead costs ~14 pp of miss coverage.

And the null result that should govern expectations: **Strata built exactly this and measured zero decode gain**
while removing 21-39 % of CPU misses, because "below one miss per layer-window the cost is per layer, not per expert -
a layer that still hands one expert to the CPU waits for the CPU round trip either way". Their extra copies also took
7-13 % of the expert cache. So the metric to optimise is **fraction of layers with zero CPU fallback**, not average
miss count, and prefetch slots must come from a separate small ring.

### Speculation: our own LP-25 kill survives, and the brief supplies the test that would overturn it

Dense self-speculation costs **rho(B) = 0.92 + 0.038*B** per pass on an 8-thread CPU (measured: rho(9)=1.27,
rho(17)=1.57), which is why n-gram drafting with K=2-3 gives 1.4-1.6x on this MoE and K=8+ loses. Our engine measured
rho ~= B instead, which the brief names precisely: "if your verify loop calls the single-token GEMV B times,
rho(B)=B and everything loses". That is our batched path. So LP-25 stands as written, with one explicit reopen test:
measure rho(B) for B=1,2,4,8 directly. If it comes out near the dense curve, revisit; if it comes out at B, the kill is
confirmed by construction.

### The head: independently confirmed at 246 MB, and the ceiling is 1.19x

The head brief derived **246 MB from 463.16 M params at IQ4_XS (4.25 bpw)** - the same correction this register made
from the GGUF, and it flags the 469 MB figure as implying 8.1 bpw. With the head at 246 MB the Amdahl ceiling for
eliminating it entirely is **1.19x**. Its recommended shape beats everything else it measured: a router shortlist of
2048 rows at **F16** plus per-row residual norms for an exactness bound is 14.9 MB total, gives 1.42x on the full
1,535 MB step and *raises* head precision while cutting its traffic 31x. It also confirms the precision cliff is
already behind us (IQ4_XS 90.4 % top-1 agreement, Q3_K 82.8 %), so "drop the head's precision" is the worst available
lever.

### 512k: compute-bound, and a real 1M-on-16GB datapoint exists

- **What actually breaks 512k is not memory.** llama.cpp's CUDA/HIP and CPU-decode attention kernels accumulate
  softmax*V in FP16: relative error vs float64 goes 0.23 @32K -> 0.76 @128K -> 0.87 @256K, and the bug report names
  NoPE global-attention layers as worst-hit - which is exactly our 8 full layers. A patch exists (9/9 needles vs 8/9)
  but is unmerged and unvalidated past 128k.
- **q8_0 is free at 32K, not at 256K.** Same-binary KLD q8/q8 0.000587 vs q4_0/q4_0 0.00245, but vLLM MRCR AUC at
  256k reads BF16 45.8, FP8 43.1, 4-bit 42.3, 3-bit 31.2. NIAH saturates and hides all of it.
- **The answer for our shape:** q8_0 K + q4_1 V on the 8 full layers + a 4k f32 tail + 4 sinks = **7.25 GB**
  (8.02 GB with the 0.5 GB sliding part). With 5.2 GB of weights that is 13.2 GB resident on a 16 GB box. Cost is
  **1.5-2.5 tok/s and it is FLOP-bound**: 8 layers x 4*T*d = 17.18 GFLOP/token at 26-43 GFLOPS, while the 8.02 GB at
  52 GB/s would allow 6.5 tok/s. Compression buys fit, not speed.
- **llama.cpp has no disk KV** - `-nkvo` moves KV to host RAM only; there is no KV-file paging. Disk paging exists in
  LMCache, Dual-Blade (page-cache hit ratio collapses 42 % -> <1 % the moment host memory drops below the working set,
  a cliff not a slope) and HiFC.
- **A 1M-on-16GB engine exists:** llama.cpp + Nemotron-3-Nano-30B-A3B (52 layers, 6 KV layers) at 1M uses
  **14,104 MiB with 2,200 free**, q4_0 KV, and its 1M blocker is the 4,038 MiB compute buffer, not the cache.
- Our nominal bit figures understate real llama.cpp blocks by +6.2 % (q8_0), +12.5 % (q4_0), +25 % (q4_1).

### LP-25's reopen test, run: rho(9) = 7.83, so the kill is now measured rather than inferred

`IN2_NO_KV=1` recomputes the whole prefix on every step, so step k costs one batched pass over n = 3+k positions.
That measures rho(n) = T(n positions)/T(1 position) directly. Measured on this box, 8 tokens from a 3-token prompt:

    n=4  1.000 s      n=7  1.703 s
    n=5  1.313 s      n=8  1.969 s
    n=6  1.578 s      n=9  2.281 s
                      n=10 2.438 s

Fitting t(n) = a + b*n over n=4..10 gives **a = 0.041 s, b = 0.240 s per position**, i.e. a 41 ms fixed cost per
pass and 240 ms of marginal cost for each extra position. Against a standalone single-token step of 0.281 s (3.61 tok/s
in the same window), that is

    rho(9) = (0.041 + 9*0.240) / 0.281 = 7.83

So our batched path costs **0.87 of a full step per additional position** - it amortises almost nothing. The briefs'
dense curve (rho(9) = 1.27) does not apply to this engine, and the brief named the reason exactly: "if your verify loop
calls the single-token GEMV B times, rho(B) = B and everything loses". At rho(9) = 7.83, a 9-position proposal pass
emitting 1.67 tokens would be ~4.5x slower than plain greedy. **LP-25 stays killed, now on measurement.**

The corollary is the useful part: per-position cost of 0.240 s against 4.22 GMAC is 17.6 GMAC/s = **35 GFLOPS
aggregate**, essentially one core's kernel rate, which is the same wall the 2.28x thread scaling and the 43 GFLOPS
prefill figure have been showing from three different directions.

### The prefetch A/B is inconclusive on this box, and saying so is the result

Interleaved, 3 rounds, 6 tokens, pf=0 against pf=8 (`IN2_IQ_PF_ROWS`):

    round 1   pf=0 3.15 tok/s   pf=8 1.16 tok/s
    round 2   pf=0 1.37         pf=8 1.38
    round 3   pf=0 1.41         pf=8 1.38

Round-to-round variance is 2.7x, an order of magnitude larger than the effect the brief measured. pf=0 holds the best
single value, which is the direction the brief predicts (-0.9 % to -12 %), but that is not evidence. **No claim, no
code change**: removing the prefetch on this basis would be acting on noise. It needs a quiet box and medians of many
passes, which is the method the brief itself prescribes and this window could not provide.

### What this changes in the plan

LP-21's device IQ2_XXS kernel is now the highest-value item in the project, because it is 11.5x the CPU's
dequant-bound rate - not because of the byte count. LP-22's residency should target **q/k/v first** (they are the
tensors that still have no quantised path on the fullseq side). The expert cache becomes Aged-LFU + global pool +
router-ahead prefetch, gated on the dirty-layer metric. And the CPU lane's 4-7x dequant headroom is a real target that
the toolchain brief's null result does not apply to, because its roofline assumed 1 byte per weight.


### q is Q6_K, not Q8_0 - and the resident hook still does not fire there

Read from the GGUF rather than assumed: `blk.N.attn_q.weight` is **[2304, 3072] type 14 (Q6_K)**, `attn_k` and
`attn_v` are [2304, 1024] type 8 (Q8_0), `attn_gate` is type 14, `attn_output` and `attn_rel` are type 8. So the
"dense tier is mostly Q8_0" note earlier in this register is right for k/v/output/rel but **wrong for q and gate**,
which are Q6_K at 6.56 bits. That changes the residency budget: q and attn_gate are the two fattest attention
tensors (3072-wide) and they are at nearly triple the bit width of the rest.

`IN2_TRACE_TY` only prints from `in2_matvec_quant`, so it never showed the q/k/v path at all - the trace in that run
was the head being chunked into 8192-row slabs (`matvec ty=23 n_in=2304 n_out=8192`). Absence of a q/k/v line was
therefore never evidence about q/k/v.

What is established: the q/k/v quantised path is live (the float production fell 180 MB and the tokens are unchanged),
but the resident GPU hook inside `in2_matvec_nt` **still does not fire for them** - only `output.weight` engages. So
either that function's fast-path guard is not satisfied for types 14/8 at this call site, or the hook's own entry
refuses and the refusal trace is not reached. Both are checkable in one instrumented run next; neither is assumed
here.

### Most of the device type table computes nothing, and the throughput numbers I quoted for it were measuring copies

`tools/gpu_q8_check.c` asks the device for a Q8_0 matvec and compares against the exact dequantised reference in
double. Result:

    rc=0   non-zero rows 0 / 512
    y[0]=0  ref[0]=7.62655     y[last]=0  ref[last]=-19.6473
    max |ref-gpu| = 4.766e+01 on magnitude 47.6593 ; rows over 1e-3 rel = 512 / 512

**Q8_0 returns success and writes nothing**, which is the third instance of this exact failure in this file (after
IQ2_XXS and IQ4_XS), and it has a direct consequence for numbers already in this register: the host-entry figures of
9.23 G weights/s for type 8, 4.21 for type 14 and 4.41 for type 23 were measing *the host->device copy and a kernel
that did nothing*. They are not compute rates and must not be compared with anything. Only type 16 has a verified
number (2.86e-06 against the exact reference) and only type 23 works at all, because it has a dedicated kernel plus
the `c_iq4nl` upload.

So `wubu_cuda_quant_matvec`, the generic dispatcher, is not a working fallback for our formats: it returns true
without computing. Any resident dispatch added on top of it would silently produce zeros on the GPU path - the tokens
would still be "produced", and the only sign would be wrong output.

That makes the ordering clear and non-negotiable: **a device kernel exists for IQ2_XXS and IQ4_XS only.** Q8_0 (k, v,
attn_output, attn_rel) and Q6_K (q, attn_gate) each need their own kernel and their own reference check before any of
them is allowed near the resident tier, and the check must assert non-zero and a numeric error, not just a return code.

### The hook was in the wrong function - and q/k/v were never on the float path

`in2_matvec_nt` is defined at **fwd.c:560**. The function I instrumented earlier, the one whose guard I printed and
whose call site I hooked, is `in2_matvec_quant_mt` at **fwd.c:1526** - the expert path. That single mistake explains
several rounds of confusion: the "NT" trace showing only expert types, the resident tier never engaging for q/k/v, and
the conclusion that the dense projections had no typed path.

The real function already carries its own trace (`IN2_TRACE_NT`), which prints `kernel` or `FLOAT-FALLBACK` per type.
Run on a decode:

    NT ty=14 n_in=2304 n_out=3072 raw=1 supported=1 verified=1 -> kernel      (attn_q, Q6_K)
    NT ty=8  n_in=2304 n_out=1024 raw=1 supported=1 verified=1 -> kernel      (attn_k, Q8_0)
    NT ty=8  n_in=2304 n_out=1024 raw=1 supported=1 verified=1 -> kernel      (attn_v, Q8_0)
    NT ty=8  n_in=2304 n_out=384  raw=1 supported=1 verified=1 -> kernel      (attn_rel, Q8_0)
    NT ty=30 n_in=2304 n_out=24   raw=1 supported=0 verified=0 -> FLOAT-FALLBACK (sink)
    NT ty=8  n_in=3072 n_out=2304 raw=1 supported=1 verified=1 -> kernel      (attn_output, Q8_0)

**Every dense attention projection takes the quantised kernel path, with or without `IN2_FULL_KERNEL`.** The earlier
statement in this register that q/k/v run on float32 and never enter a typed kernel is therefore **wrong**, and so was
the 1,836 MB-per-token figure derived from it: that was the size of the borrowed float views, not a traffic measure.
The only float tensor in this block is `sink` (type 30), which is 24 wide and irrelevant.

What follows is simple and was always available: put the resident hook in `in2_matvec_nt` at **fwd.c:560**, next to its
`wubu_iq_rows` call, instead of in `in2_matvec_quant_mt`. The destination is ready - Q8_0 is verified at 3.8e-06 and
Q6_K is the only dense type still lacking a device kernel.

### attn_q is not one format: it changes per layer, so "add a Q6_K kernel" was wrong

With the hook finally in `in2_matvec_nt` and a refusal trace that prints once per type, a 4-token decode says:

    NT-GPU refused ty=14 rc=-1 actb=2628 wb=5806080 (blk.0.attn_q.weight)    Q6_K
    NT-GPU refused ty=13 rc=-1 actb=2628 wb=4866048 (blk.3.attn_q.weight)    Q5_K
    NT-GPU refused ty=12 rc=-1 actb=2628 wb=3981312 (blk.18.attn_q.weight)   Q4_K
    NT-GPU refused ty=18 rc=-1 actb=2628 wb=2709504 (blk.20.attn_q.weight)   IQ3_XXS

**`attn_q` is Q6_K on layer 0, Q5_K on layer 3, Q4_K on layer 18 and IQ3_XXS on layer 20**, and the tensor byte
sizes step down with them (5.81 MB -> 4.87 -> 3.98 -> 2.71 for the same 2304x3072 shape). Reading type 14 from
`blk.0` and concluding "q is Q6_K" was a sample-of-one error, and it means the dense tier needs **four** more device
kernels, not one, before the whole attention block can be resident. Each is a K-quant with its own block layout and
super-block scales, so this is not one afternoon's work; the reference harness exists for each of them now.

Good news in the same output: the type gate works exactly as intended. Every unverified type is refused with `rc=-1`
and falls through to the CPU kernel, and the tokens stay identical:

    149876 220 36838 182189 149933 220

Also still unexplained and worth saying plainly: type 8 (attn_k, attn_v, attn_output, attn_rel) neither engages nor is
refused - nothing is printed for it. So either those calls do not reach this hook at all, or they succeed without the
one-shot message firing because another hook incremented the counter first. One print of the counter inside the hook
distinguishes those two, and it is the next edit rather than another guess.

### The resident tier works and is worth nothing yet: per-call sync costs more than the CPU's share

The success trace settles the last question - type 8 IS resident:

    NT-GPU ok ty=8 (blk.0.attn_k.weight)
    NT-GPU refused ty=14 rc=-1 (blk.0.attn_q.weight)  ... 13, 12, 18 on other layers

So attn_k, attn_v, attn_output and attn_rel - **617 MB of the 1,535 MB step, 40 %** - are being computed on the card,
along with the head. And the measurement says zero gain (3.71 / 3.41 / 3.51 CPU against 3.71 / 3.53 / 3.53 GPU).

The reason is arithmetic, not mystery. Each resident call is synchronous: memcpy activation in, launch, sync, memcpy
out, which costs ~35 us before the kernel does anything. Against the CPU's eight threads:

    tensor          MB    GPU us   CPU 8-thread us   winner
    attn_k        2.39        51               28    CPU
    attn_v        2.39        51               28    CPU
    attn_output   7.17        83               84    GPU (by 1 us)
    attn_rel      0.90        41               10    CPU

192 calls per token at ~35 us is 7 ms of pure round trip, and every tensor under about 4 MB loses to a CPU that is
already spreading the same work over eight threads. Net effect: nothing, exactly as measured.

Two real fixes, both ordinary engineering rather than insight: **stop synchronising per call** (issue asynchronously and
sync once per layer or per token, which would make the card win everywhere), or **gate on size** and only offload
tensors above ~5 MB, which today would mean attn_output alone. Until one of those lands, the honest summary of the
resident tier is: correct, verified, covering 40 % of the token's bytes, and slower per call than the CPU it is meant
to relieve.

### The next change, quantified: 124 ms of the step is not bytes at all

Every group of this token's traffic, costed at the two measured rates (11.2 GB/s CPU aggregate, 56 GB/s resident on
the card):

    group                                              MB   CPU ms   GPU ms
    q  (Q6_K/Q5_K/Q4_K/IQ3_XXS - 4 kernels needed)    580     51.8     10.4
    k/v/attn_output/rel  (Q8_0 - resident)            617     55.1     11.0
    head  (IQ4_XS - resident)                         235     21.0      4.2
    experts  (IQ2/IQ1/Q2_0)                           369     32.9      6.6

All of it on the CPU in series is **160.8 ms**, against a **measured 285 ms step**. So **124 ms - 43 % of every
token - is not traffic at all**: it is dequantisation and serialisation, the exact thing the card does 11.5x better
because its dequant is a different instruction mix, not a different bandwidth.

That reframes the remaining work. The plan is not "move more bytes to the card" - the bytes are only 57 % of the
step. It is: dense tier on the GPU (all of q, k, v, output, rel, head = 1,432 MB), experts on the CPU (369 MB), and
the two running concurrently.

    dense on GPU, experts on CPU, serial:      25.6 + 32.9 = 58.5 ms
    the same, overlapped:                      max(25.6, 32.9) = 32.9 ms
    if the 124 ms of dequant moves with it:    ~33 ms = 8.7x the present step

The 8.7x is a ceiling and is stated as one. What makes it credible rather than arithmetic theatre is that the GPU
path is already at 11.5x on the dequant-heavy part of exactly this format, and the sync economics are the only thing
standing between the two halves - measured directly as ~35 us per call against 28 us of 8-thread CPU work on the same
tensor.

Order of work, each gated on tokens before any timing:
1. batch the resident submission (one sync per layer, not per tensor) - no new kernel needed;
2. Q4_K, Q5_K, Q6_K, IQ3_XXS device kernels, each verified against an exact reference, which completes `attn_q`;
3. run the dense tier on one stream and the expert path on the CPU concurrently, which is Strata's doorbell;
4. only then the expert cache (Aged-LFU, global pool, router-ahead prefetch), gated on dirty-layer fraction.

## Wave 10 - the five research fronts, synthesised (and one of them changes the CPU model)

### Decode is concurrency-bound, not bandwidth-bound: ~8 of 24 miss slots in flight

The most useful result in the batch, and it inverts the roofline I had been quoting. 43 GFLOPS at the IQ kernels'
3.88 FLOP/B is **11.1 GB/s = 15 %** of what this part can stream, so bandwidth was never the wall. Per-thread
memory-level parallelism is the wall: at 1 thread we have ~8 lines in flight at ~100 ns and at 8 threads still ~8 lines
each but at ~400 ns of inflated latency, and
`8 threads x 8 lines x 64 B / 400 ns = 10.2 GB/s` predicts the 11.1 measured. Zen 4 documents **24 outstanding misses
per core** in the Miss Address Buffer (AMD Zen 4 SOG), corroborated by Chips and Cheese (DDR5-5600: 82 ns idle ->
>400 ns with five bandwidth threads). We are using a third of the hardware's concurrency.

Correction to the same brief, filed rather than edited away: its "4.5 tok/s x 7.9 GB = 35.6 GB/s, 67 % of the ceiling"
multiplies the whole FILE by the token rate. Per-token traffic is 1,535 MB, so the engine streams ~7 GB/s, not 35, and
it sits at 15 % of the roof - which is what the MLP argument says independently.

The lever, in payoff order, with the source's own measurements:
1. **Row tiling R=4** (R=8 when the activation fits), 6-deep unroll: four independent row streams in flight per thread
   turns 8 outstanding misses into ~24. ik_llama's 4-row-interleaved formats measured **248.3 vs 188.7 t/s PP-512 on
   Zen 4 (1.32x)** and 2.5-3.4x over mainline on Q4_K/Q5_K. This is the single biggest CPU-side change available and
   it applies to single-token decode, not just prefill.
2. **SoA scales with 64 B-aligned payloads.** A scale sharing a line with nothing, in a different line from its quants,
   adds a SECOND dependent miss per block and halves MLP; ~half of 32 B loads also straddle a line on Zen 4. The
   "66-byte blocks defeat the prefetcher" folklore is wrong - a dense 66-byte run is still a sequential stream.
3. **Prefetch distance computed, not guessed:** D ~ latency x per-thread BW (16 lines at 10 GB/s/100 ns, 31 at
   20 GB/s/200 ns), targeting the next row block 1.5-2 KB ahead, PREFETCHNTA for streaming weights.
4. Contiguous row ranges per thread, one thread per physical core, 2 MB pages.
5. Batch tokens T=4-8 so the matvec becomes a GEMM (AI 3.9 -> 15-31 FLOP/B) and leaves the latency regime entirely -
   but rule 64 says a 9-position pass does not pay here, so this is prefill-only.

### Affinity: +61 % in the brief, exactly zero on this engine (measured, not argued)

The brief measured 6 threads pinned one-per-core at **263 vs 162 GFLOPS** (+61 %) on this same laptop, with both SMT
siblings of core 0 at 95 % while the rest idled - i.e. Windows packing a partially-occupied team. Tested here directly
on the engine, 8 tokens, `IN2_THREADS` x `OMP_PROC_BIND`:

    6 threads: none 3.63 | close 3.63 | spread 3.61
    8 threads: none 3.74 | close 3.71 | spread 3.66
   12 threads: none 3.68 | close 3.71 | spread 3.68

Nothing beats 8 threads unbound, and binding is flat or worse. The +61 % is real for a dense SGEMM whose team leaves
six cores idle for Windows to mis-pack; an engine already at 8 threads on 6 cores has no such packing to fix. Keep the
finding, do not port the fix.

### Sustained-load ceiling is the 45 W STAPM limit, not the OS

30 minutes of continuous 12-thread matmul on this box: 270.0 GFLOPS (5-15 min) -> 240.5 (15-30) -> **229.8 (25-30),
-15 %**, peak window 280.4 vs worst 211.9 (-24 %), 97-100 C throughout, 4.12 GHz average. Every OS knob measured flat
for all-core work (PERFBOOSTMODE -3 %, PROCTHROTTLEMAX 99 -0 %, CPMINCORES 100 + DISTRIBUTEUTIL 0 +0.5 %,
PROCTHROTTLEMIN noise); the ceiling lives in the SMU. Single-thread cap is 4.30 GHz mean / 4.44 max, identical hot and
idle, so "up to 5 GHz" is unreachable on this platform. No undervolting: AMD lists Curve Optimizer as **No** for this
part.

### Long context: the exact, not merely accurate, stack

For the 512k-1M goal the batch surfaced three methods that preserve bits or bound the error rather than approximating
it, which is what this engine's parity contract needs:
- **AnchorKV** - reconstructs KV **bit-exactly** from layer-sparse residual-stream anchors, O(L) cascade replay (16.3x
  faster than naive at 4K), exact DP anchor placement for interleaved sliding-window models, break-even spacing
  computable from the config alone, verified on CPU. This composes directly with our 8-full-of-48 split.
- **GAAC** - lossless exponent codec, 1.405-1.407x on BF16, ~26 % resident-KV cut, 1.19-1.27x decode attention at 64K,
  explicitly aimed at personal devices.
- **LoomKV** - bit-exact tiering, BF16 at 0.71x raw, 2.7-3.6x goodput at the LRU cliff.
- Cheapest quality-preserving win without any of the above: **HeadWiseKV** - per-KV-head static windowing on the global
  layers only, training-free, 68.95 % KV retention costs RULER 0.35 points and extends context 114K->161K.
- Design warning for us: **NLL-guided layer selection** says 1/4 full-attention layers match the 1/2-periodic baseline
  with ~15 min of calibration, i.e. our 8-of-48 is likely over-provisioned on attention and under-provisioned nowhere.

### Lineage, and the one negative result that saves a month

Vec-LUT (MobiSys 2026) unified LUT across parallel tokens (1->N lookup per index) reports **4.2x over SOTA on 5 edge
devices**; T-MAC does 20 tok/s on one core for a 3B BitNet (4-5x llama.cpp); midge uses the OS page cache AS the expert
cache via MADV_RANDOM (3.98 vs 0.48 GB/s/core). Strata's published 6-thread desktop-Zen table says it is at **100 % of
RAM bandwidth at one token per expert** - i.e. no kernel headroom in decode for it, only weight reuse across 3-4 tokens
(KQ256, bit-exact, 11-20 %). That contradicts our measured 15 %-of-roof, and the contradiction is unresolved: different
format, different thread count, possibly different accounting. **Do not resolve it by assumption** - if Strata is right
for its setup, R=4 tiling buys nothing and the win is elsewhere; rule 75 says recompute before committing, so the R=4
implementation should land with a bytes/s-vs-threads curve at 512 KB / 8 MB / 2 GB that shows where the wall actually is.

Negative result worth acting on immediately: with 7.9 GB in 16 GB, the expert-streaming tier is pure pessimisation -
skip it. Two-level NUMA barriers are NUMA-only. llama.cpp's weight-replication PR is 22-59 % slower than its own
`--numa distribute`.

## Wave 10 addendum - the serial remainder is the dense/attention path

the serial remainder is the dense/attention path: MoE scales 3.25x, everything else 1.13x

Attributed the 47.8% serial fraction by summing the per-layer MoE timers against the run total, same 8-token prompt,
one thread then eight (IN2_TIME=1 IN2_THREADS=<n>):

            total     MoE            rest
   1 thread  4.10 s   2.47 s (60%)   1.63 s
   8 threads 2.20 s   0.76 s (34%)   1.44 s
   scaling    1.86x     3.25x          1.13x

The MoE phase is healthy - 3.25x on 8 threads, which is what an `omp parallel for` over 8 selected experts should give
(the 8th thread matches n_expert_used exactly). Everything that is NOT the MoE scales at 1.13x, i.e. it is effectively
serial, and it is now 65% of the eight-thread step and 40% of the one-thread step.

That inverts where the work has been going. The MoE has been the optimisation target all session (expert-major
restructure, dequant fusion, borrow conversions, per-visit KV) because it is the largest single phase by name, and it
is the one part that was already parallel. The dense/attention path - q/k/v/output projections, the attention itself,
norms, the router, the embedding row read and the head - has been assumed to be parallel because it calls
in2_rows_parallel, and it converts 8 threads into 1.13x.

Two candidate causes, and they need distinguishing rather than guessing:
  * ~100 small matvec calls per token, each opening its own `omp parallel for`: at ~3 us of libgomp barrier per region
    (measured earlier at ~3.2 us per extra thread, rule 49) that is ~0.3 s per token of pure barrier against regions
    that may be shorter than the barrier itself;
  * or the path is genuinely latency-bound per call and threads add contention without adding work in flight.
The tell is the same one that found the MoE: sweep IN2_THREADS over just that path, or time a single projection in
isolation at 1 and 8 threads. A region that measures identical at 1 and 8 is a barrier-dominated region, and the fix is
to persist the thread pool instead of opening a parallel region per tensor.
