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

