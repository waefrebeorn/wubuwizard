# WORKLIST — SOTA 2026-10/11 pads

> Execution view of `docs/SOTA-2026-11-ROADMAP.md`. Pads are dependency-ordered.
> Status reflects a **real gated test**, not intent: SHIPPED / PARTIAL / TODO.
> Engine-absorption TODO lives in `docs/ENGINE-GAP-REGISTER.md`; upstream reading
> pointers live in `docs/ENGINE-SOURCE-INDEX.md`.

| Pad | Size | Depends on | Best source | Gate | Status |
|---|---|---|---|---|---|
| LP-01 Persistent goal + evidence-first judge | M | — | slermes `/goal` docs + Sauron + self-report leakage | `make test_goal` | SHIPPED |
| LP-02 Fast/slow goal-drift steering | S | LP-01 | goal-drift steering paper | `make test_goal_steer` | SHIPPED |
| LP-03 Bounded verifier-guided resampling | M | LP-01/02 | VGSC + SafeInferCom | `make test_goal_resample` | SHIPPED |
| LP-04 Failure-pattern harness bandit | L | LP-03 | ActiveSaddler / MESH-Harness | `make test_harness_bandit` | SHIPPED |
| LP-05 All-or-nothing paged KV claim | M | — | kvcached v0.1.6 | `make test_kv_alloc_exact` | SHIPPED |
| LP-06 Shared-cell SWA reclamation | M | LP-05 | llama.cpp PR #32309 | `make test_swa_reclaim` | SHIPPED |
| LP-07 Confidence-budgeted speculation | M | LP-05 | DeepInfer DSpark + SGLang v0.5.23 | `make test_spec_budget` | SHIPPED |
| LP-08 Deterministic seeded speculation | S | LP-07 | SGLang v0.5.23 | `make test_spec_determinism` | SHIPPED |
| LP-09 Constrained decode + speculation | L | LP-07/08 | Strata PR #4222 | `make test_spec_constrained` | SHIPPED |
| LP-10 Compile-time family specialization | L | — | Splash PR #573 | `make test_kernel_identity` | SHIPPED |
| LP-11 KV reconstruction in compute format | L | LP-05 | MX/NVFP4 KV + CALFA-KV | `make test_kv_qfmt` | SHIPPED |
| LP-12 Tolerance-parameterised 9P KV fetch | M | LP-05/11 | VerKV | `make test_kv_fetch_eps` | SHIPPED |
| LP-13 Fused diffusion logits/sampler | M | — | W4A16 diffusion sampler paper | `make test_diffusion_sampler` | SHIPPED |
| LP-14 Measured-cost test-time compute | M | LP-01/03 | TTC cost audit papers | `make test_ttc_budget` | SHIPPED |
| LP-15 Per-shape SIMD quant decode + W4A | L | LP-11 | TileKernels v2.1.0 | `make test_simd_quant` | SHIPPED |
| LP-16 State/ABI semver boundary | S | — | llama.cpp v1.0 / ggml v1.0 / slermes goals | `make test_state_version` | SHIPPED |

## Immediate frontier

LP-01 is the first crossing: nothing in the goal/slerming wave can be gated until a
persistent goal object and a report-isolated judge exist. LP-05 is the independent
spine fork and can be worked in parallel; LP-06..LP-09 stand on it.

## Dependency forks

- Goal fork: LP-01 → LP-02 → LP-03 → LP-04, with LP-14 joining after LP-03.
- KV/spec fork: LP-05 → {LP-06, LP-07 → LP-08 → LP-09}, and LP-05 → LP-11 → LP-12 → LP-15.
- Independent: LP-10, LP-13, LP-16.

## Engine absorption gates (evidence-first)

These techniques were already implemented in the tree but had source-only tests with no
`make` gate. They are now real gates in both `Makefile` and `Makefile.win`, verified on
the Windows T1 host with `mingw32-make -f Makefile.win <gate>`.

| Technique | Source family | Gate | Measured evidence |
|-----------|---------------|------|-------------------|
| Runtime per-type `vec_dot` dispatch + multi-row GEMV | llama.cpp `ggml-cpu.c` type traits | `make test_eg01_vecdot` | 7-type table, unknown type refused, 4 rows `err=0.000` vs scalar reference, `rows=4 lookups=1`; gate exposed and fixed a 4× AVX2 Q8_0 reduction bug |
| GGUF tokenizer metadata completeness | llama.cpp pre-tokenizer registry lineage | `make test_eg02_tok_meta` | `pre=llama3`, `token_type_n=3`, `add_bos=1`, full `chat_template`; generated 9-KV fixture, exit 0 |
| Metadata-first VRAM planner (80% rule) | Ollama VRAM prediction + vLLM UMA sizing | `make test_eg03_vram_plan` | 8.49 GB on 6 GB → REFUSE (STREAM when allowed), UMA-corrected 5 GB → FIT vs raw 3 GB → REFUSE, large KV → SHRINK to 2867, wait=100ms; exit 0 |
| Attention-sink KV retention | StreamingLLM lineage | `make test_eg04_attention_sink` | 2 sink cells survive reclamation at window bound 6, non-sink cell reused, sinks=0 restores old behavior, snapshot deterministic; LP-06 gate still passes |
| Neuron-cluster hot/cold predictor | PowerInfer / PowerInfer-2 | `make test_eg05_neuron` | 616 observations, 10% hot set = 8 neurons, runtime cluster size 16→4, predicted cluster reads 140 vs 160 without predictor; exit 0 |
| Offload placement solver + zigzag schedule | FlexGen ICML 2023 §4 | `make test_eg06_plan` | greedy 0.070s vs exhaustive optimum 0.070s (ratio 1.000), all-SSD 3.500s, 24-entry alternating schedule; exit 0 |
| SSD sliding-window row reuse | LLM in a Flash §2–3 | `make test_eg07_row_reuse` | 8 reads / 61440 B vs naive 64 reads / 262144 B over 8 windows, 7 incremental deltas, 32 KiB coalescing; exit 0 |
| Iteration-level continuous batching | vLLM / ORCA lineage | `make test_contbatch` | 8 requests, 35 tokens in 7 steps vs 35 serial = **5.00x**; overlap prefill/decode consumes all 104 tokens in 200 iterations with zero starvation |
| Cross-request prefix KV reuse | SGLang RadixAttention lineage | `make test_prefix_reuse` | 512-token prefix: first miss, second full 512 hit, different prefix misses, partial 256/512 hit; hits=1 misses=2, hashes distinct |

Both gates are linked against the owning modules directly, so they cannot pass on a
stubbed or unlinked build. `test_contbatch` depends on `wubu_scheduler`,
`wubu_continuous_batching`, `wubu_contbatch`, `wubu_paged_kv` and `wubu_prefix_cache`;
`test_prefix_reuse` depends on `wubu_prefix_cache`.

## Definition of SHIPPED

The named gate exists in the Makefile, compiles with the project's C11 flags, runs
without stubs, and its assertions cover the mechanism named in the pad. A documentation
edit or a partial struct is PARTIAL at best.
