# MODULES.md — the module registry (2026-08-09)

> Doctrine: NO code is deleted. Every module is either WIRED (compiled into
> the build), SUPERSEDED (marked, replaced by the live version), or BROKEN
> (fixed in place). This registry is the source of truth.

## WIRED — the 41 modules integrated into EXTRA_OBJ (2026-08-09)

These compiled clean and are collision-free vs CORE_OBJ. They were sitting
in src/ unreferenced by the Makefile (the "toolbox"). Now they build as part
of `make all` via EXTRA_OBJ — verified by `make module-registry`:

| Module | Role |
|--------|------|
| gaad_nesting_llm | GAAD phi-recursive nesting LLM |
| lfm2_{attn,conv,ffn,forward,load,math} | LFM2 (linear-feedback MoE v2) engine |
| thread_pool | the parallel-for thread pool |
| wubu_agi | the AGI supervisor interface |
| wubu_ambig | ambiguity scoring |
| wubu_colonel | the Colonel/HolyC runtime bridge |
| wubu_credit_sft | credit-assignment SFT |
| wubu_dbstate | database-state verifier |
| wubu_dedup | corpus dedup |
| wubu_dense_ffn | dense FFN variant |
| wubu_dsv4_layer | DeltaNet v4 layer |
| wubu_epcap | EPCap (evidence-based capability) |
| wubu_eval | the eval runner (wubu_eval_run) |
| wubu_fmt | formatting |
| wubu_gaad | GAAD generator |
| wubu_h3_norm | H3 normalization |
| wubu_kernel_budget | kernel budget accounting |
| wubu_masked_ce | masked cross-entropy |
| wubu_mix | the corpus mixer |
| wubu_mmrope | multimodal RoPE |
| wubu_ngram_cascade | SUPERSEDED variant (see below) — kept as EXTRA, not core |
| wubu_passk | pass@k |
| wubu_patch_embed | patch embedding |
| wubu_priority | priority scheduling |
| wubu_recency | recency weighting |
| wubu_rollout | rollout buffer alloc |
| wubu_seed | deterministic seeding |
| wubu_spawn_win | Windows spawn |
| wubu_traj_grpo | trajectory-level GRPO |
| wubu_traj_sft | trajectory -> masked-observation SFT converter |
| wubu_user_sim | tau-bench style user simulator |
| wubu_uuid | UUID generation |
| wubu_vision_moondream | MoonDream vision tool-calling |
| wubu_width | width growth |
| wubu_win | Windows mmap helpers |
| wubu_mobius_new | SUPERSEDED variant (see below) — kept as EXTRA, not core |

## SUPERSEDED (kept, marked — NOT deleted)
| File | Superseded by | Note |
|------|---------------|------|
| src/wubu_mobius_new.c | src/wubu_mobius_gyrate.c (in core) | old gyrate variant; same symbol |
| src/wubu_ngram_cascade.c | src/wubu_spec_cascade.c (in core) | old standalone cascade; the spec-cascade is live |
| src/wubu_model_ckpt.c | src/wubu_model.c (rollback state) | struct drift (ssm_v_heads gone); superseded |
| src/kv_paged_attention.c | src/wubu_paged_kv.c (in core) | CUDA bench fragment; used only by tools/build_bench.sh |

## BROKEN — fixed in place (2026-08-09)
| File | Fix |
|------|-----|
| src/quantized_matmul_fixed.c | +stdint/stddef/gguf_reader/immintrin, QK_K define — WIRED |
| src/wubu_bear_mlp.c | +stddef, +M_PI define — WIRED |
| include/wubu_turboquant.h | CUDA `__device__` guarded for C compiles |
