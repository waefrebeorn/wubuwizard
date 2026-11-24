# Front O — Atomic Engine and Unsloth Inkling 2 Mini (2026-11-23)

Two new sources from the user, read the same evening. Both are llama.cpp-derived and
both target the exact constraint wubuwizard is built for: a model larger than VRAM,
run from SSD/host memory under a budget.

Study clone (read-only, never vendored, never linked):
`C:/Users/eman5/WuBu/research/engine-sources/atomic-engine`, MIT
(`Copyright (c) 2023-2026 The ggml authors`), branch `dev`, commit
`fa190d7b5080195b05fdf85c15868086afec7f4f` (2026-11-24).

## Atomic Engine

A llama.cpp fork that adds a *budgeted* engine: model, KV/state, compute and expert
buffers are admitted under one budget with an explicit runtime reserve, plus SSD expert
streaming and a backend MoE cache. Their claim set (their numbers, Apple M4 Max /
multi-GPU): Qwen3.5 MoE prefill/decode matches resident logits under a 5 GiB engine
budget; live shrink/grow releases backend memory without dropping the session; MoE
expert cache gains +2% to +81% decode on CPU-resident experts.

| Mechanism | Exact upstream location | wubuwizard today |
|---|---|---|
| Engine budget + runtime reserve | `--engine-budget-mib` (README.md:39,48) → `cparams.moe_cache_budget_mib` (src/llama-context.cpp:127) | `wubu_mem_budget.c` exists; no admission at load |
| MoE cache provider interface | `ggml/src/ggml-backend-moe-cache.h`; `ggml_moe_cache_active()`, `ggml_moe_cache_get()`, `ggml_moe_cache_config` (src/llama-context.cpp:903-924); eligibility `llama_model_has_cacheable_moe_weights()` (:896) | `wubu_expert_cache.c` (EG-36b) is a contract only |
| Cache eligibility gates | `docs/backend/MOE-CACHE.md` — compute capability floor (7.0/8.0), `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB`, 1 GiB auto slab floor, `GGML_CUDA_MOE_CACHE_RESERVE_MB` (default 3072 MiB) | none |
| Inventory-then-fit load | no-allocation model load inventories exact expert shapes/aliases + model/context/compute memory before choosing placement (MOE-CACHE.md) | `wubu_residency.c` (EG-110) plans bytes but nothing inventories tensors first |
| KV stream config | `llama_kv_stream_config_validate()`, `llama_kv_stream_phase_plan_make()` (src/llama-kv-stream-config.cpp:5,41) | `wubu_kv_tier.c`, no phase plan |
| Loop/MoE arch dispatch | `LLM_ARCH_INKLING` (src/llama-arch.h:160) + `LLM_KV_INKLING_D_REL`, `LLM_KV_INKLING_DENSE_BLOCK_COUNT`, `LLM_KV_INKLING_LOG_SCALING_*` (:417-425) | **no Inkling support at all** (`rg -i inkling src include` → 0) |
| DSpark draft | `--spec-type draft-dspark` = "DFlash backbone + semi-autoregressive Markov head" (docs/speculative.md:387); weights `dspark-Inkling-2-Mini-Q8_0.gguf` | `wubu_markov_w4.c`, `wubu_zero_logits.c` (both contract-only) |
| MTP / NextN | docs/MTP.md, docs/NEXTN.md, `draft-mtp-adaptive` | `src/wubu_mtp.c` exists and is linked, but there is no spec-type switch and no acceptance accounting |

## Unsloth Inkling 2 Mini

`https://x.com/unslothai/status/2124898356770263382` (23 Nov 2026): "~1.9x faster
locally", 125 tok/s on an RTX Pro 6000, claimed no accuracy loss; the model is Thinking
Machines' Apache-2.0 **looped-transformer MoE, 26.4B, "runs on 16GB RAM"**, and the
GGUF download is the same family this host already holds
(`Inkling-2-Mini-UD-IQ2_XXS.gguf`, 8.49 GB; `UD-Q4_K_XL`, 17.6 GB).

Their speedup is a llama.cpp-side change, not a new format: the relevant upstream surface
is the looped-arch graph in `src/models/` plus quantization kernels. Treat the 1.9x as
their measurement on their hardware; the transferable part is that looped MoE decode is
kernel/placement bound, not math bound.

## Shortlist for wubuwizard (ranked for 6 GB / 16 GB)

1. **Inventory-then-fit before residency planning.** EG-110 plans bytes that the caller
   supplies; atomic's rule is that placement decisions come from a no-allocation load
   that has already enumerated every tensor shape, alias and the runtime reserve. Action:
   extend the GGUF loader to emit a tensor inventory (name, type, bytes, alias) and feed
   `wubu_residency_plan` from it instead of the hard-coded attn-hot / ffn-stream split now
   in `g4_apply_residency_plan()`.
2. **MoE cache eligibility + working-set policy**, gated exactly as atomic gates it
   (device capability, minimum expert size, slab floor, reserve). Our `wubu_expert_cache`
   has the class policy but no eligibility probe and no per-device budget arithmetic.
3. **Inkling loop arch + DSpark Markov head.** Without `LLM_ARCH_INKLING` equivalent the
   engine cannot even open the strongest model on this disk. This is the largest item and
   the one that decides whether "everyman inference" includes the 26B looped MoE.

## Honest limits

- Atomic qualifies specific model/config pairs; their numbers do not transfer to a 4050.
- Nothing here was linked or vendored; the clone is study-only.
- The unsloth 125 tok/s figure is RTX Pro 6000, not this host.
