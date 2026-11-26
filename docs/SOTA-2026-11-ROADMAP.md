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
  is believed (needle retrieval at 128k is the gate, and it cannot be measured until LP-17 lands).
- **`SWA Bounded Replay` transfers directly and is free.** Their finding: exact reconstruction of the decoder's SWA
  state needs L x n_win replayed tokens, but replaying only n_win is negligibly worse, so SWA KV is never persisted.
  For us the 64 sliding visits are 0.5 GB and never grow, so this changes nothing in RAM - but it is exactly what
  LP-20 needs: persist only the 8 full visits and replay the last 1024 tokens on resume, instead of checkpointing
  0.5 GB of sliding state on every save.
- **FP4 global KV is trained, not post-hoc.** They quantise during training and call the degradation marginal. Our
  LP-18 must therefore measure post-hoc q8/q4/FP4 against the f32 cache rather than cite their result.
- **CED (decoder global KV projected from H_{L/2}) halves prefill but cannot be retrofitted** - it needs a trained
  per-layer W^KV_l. Ours must come from LP-17 batching, not from architecture.
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

Evidence base, all local, each with its own verified citation ledger:
`C:/Users/eman5/kv-1m-brief.md` (KV representation), `C:/Users/eman5/longcontext-kv-offload-brief.md` (engines),
`C:/Users/eman5/kvbrief/KV-MEMORY-BRIEF.md` (Windows memory, measured on this box, with kvbench.c),
`C:/Users/eman5/AppData/Local/hermes/cache/scratch/pe/positional-extension-512k-brief.md` (positional; moot here).

Two results worth carrying although they are not pads: KTransformers is the only engine measured at 1M on a single
device (16 tok/s on a 24 GB GPU) and it gets there by attending ~16k of blocks, not by shrinking KV to fit; and
llama.cpp hit a hard bug at `n_kv * n_ubatch = 2^32`, so mask or index arithmetic added for long context must be
64-bit. This engine has no masks and its KV offsets are already size_t, so that bug class does not apply as it stands.
