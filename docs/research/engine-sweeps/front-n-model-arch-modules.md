# Front N — Model-agnostic architecture dispatch and reusable module organisation

Date: 2026-11-23. Scope: how eight upstream engines dispatch per-architecture behaviour without
forking the runtime, and what that implies for wubuwizard's layer loop. Standing target unchanged:
6 GB VRAM / 16 GB RAM / SSD, Windows 11, C11 core. Rule unchanged: read upstream, extract the
mechanism, reimplement in C11 behind a named gate; never vendor, link, or copy.

All upstream identifiers below were read from the local study clones in
`C:/Users/eman5/WuBu/research/engine-sources/`, pinned at these commits:

- vLLM `ef687f6` [14][88]
- llama.cpp `94f3af3` [89][90]
- SGLang `1918da6` [22]
- KTransformers `3a78b9a` [28][94]
- MNN `a5c3dd9` [43][95]
- mlx-lm `1bf8da0` [96][97]

ONNX Runtime GenAI is documentation-only: `github.com/microsoft/onnxruntime-genai` returned HTTP
404 on 2026-11-23 (verified by direct request), so only the live docs site is cited [99][100]. No
source file, Makefile, or gap-register row was modified by this report.

## 1. What wubuwizard already has, and exactly where it stops

An architecture registry exists and is already called on the load path: `wubu_arch_probe()` at
`src/wubu_model.c:44`, implemented at `src/wubu_model_arch.c:144`, with a 16-row table at
`src/wubu_model_arch.c:11-28` keyed on `general.architecture` (`llama`, `gemma3`, `gemma4`,
`qwen3moe`, `lfm2`, …). The record it returns (`wubu_arch_spec_t`, `include/wubu_model_arch.h:61`)
carries `id`, `family`, `layout`, `runner`, `n_layers`, `d_model`, `d_ff`, `n_heads`,
`n_kv_heads`, `head_dim`, `rotary_dim`, `sliding_window`.

The problem is that the record is advisory. Only `n_layers`, `layout`, `n_heads`, `n_kv_heads`,
`rotary_dim` and `d_ff` are consumed (`src/wubu_model.c:58-133`); everything else is printed. Four
concrete leaks:

| Symptom | Location | Consequence |
|---|---|---|
| Layer kind is a modulo pattern, not data | `wubu_is_ssm_layer()` = `(layer_idx + 1) % 4 != 0`, `src/wubu_ops.c:29-33`; assigned at `src/wubu_model.c:143` | Every model is forced into Qwen3.5's 3:1 SSM:GQA cadence; Gemma 3/4 and Qwen3-MoE (all-GQA) mis-load |
| `wubu_layer_t.block_type` is dead | declared at `include/wubu_model.h` with the comment "tensor-driven at load — NOT the hardcoded (l+1)%4 pattern"; grep finds no assignment in `src/*.c` | The intended per-layer kind field never got wired; the forward loop keys on `bool is_ssm` |
| Expert counts are compile-time | `model->n_experts = N_EXPERTS` (`256`), `model->n_active_experts = N_ACTIVE_EXPTS` (`8`), `src/wubu_model.c:131-132` | Qwen3-30B-A3B (128 experts, 8 activated [106]) and Gemma 4's in-block MoE are mis-sized before a single weight is read |
| `sliding_window` parsed, never used | `src/wubu_model_arch.c:194`, printed at `:248`; no read in `src/*.c` | Gemma 3/4 KV is allocated full-length on every layer; the SWA memory win is left on the table |

One head dimension for the whole model (`model->gqa_head_dim`) is the fifth constraint: Gemma 4
carries a distinct `head_dim` per layer (`hparams.n_embd_head_k(i)` in `src/models/gemma4.cpp:77`),
which a single scalar cannot express [101].

So the engine does not lack an architecture registry; it lacks a **per-layer geometry record** and
a **dispatch seam** that the forward loop actually consults. That is the same pair every engine
below solved, in one of three ways.

## 2. Eight engines, eight dispatch mechanisms

### 2.1 vLLM — lazy module table + capability Protocols

`_VLLM_MODELS` is not one dict but a merge of nine task-partitioned dicts
(`_TEXT_GENERATION_MODELS` 122 entries, `_MULTIMODAL_MODELS` 135, `_SPECULATIVE_DECODING_MODELS`
73, `_EMBEDDING_MODELS` 28, …; 430 architecture keys total) [14]. Each value is a
`(module_name, class_name)` pair, e.g. `"Gemma4ForCausalLM": ("gemma4", "Gemma4ForCausalLM")` and
`"Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM")` [14]. `_LazyRegisteredModel` stores
only those two strings and `importlib`-loads on first use, caching a `_ModelInfo` keyed by a hash of
the module file [14]. `_ModelInfo` is the capability record: `architecture`, `is_hybrid`,
`is_attention_free`, `supports_lora`, `supports_pp`, `has_inner_state`, `supports_kv_cache_grouping`
[14]. Behaviours the engine needs to branch on are `@runtime_checkable Protocol`s with
`isinstance`-style predicate functions — `MixtureOfExperts` exposing `num_routed_experts`,
`num_shared_experts`, `num_moe_layers`, checked by `is_mixture_of_experts(model)` [87]. The runtime
never names a model class; `_ModelRegistry.resolve_model_cls()` walks the architecture list and
returns `(cls, arch)` [14].

### 2.2 llama.cpp — one arch enum, one factory, six virtuals

`enum llm_arch` holds 181 entries [90]; names live in `LLM_ARCH_NAMES` and resolve through
`llm_arch_from_string()` [90]. GGUF keys are formatted, not hard-coded: `LLM_KV_BLOCK_COUNT` is
`"%s.block_count"`, `LLM_TENSOR_ATTN_NORM` is `"blk.%d.attn_norm"` [90]. Dispatch is a single
factory: `llama_model_mapping(llm_arch arch, const llama_model_params & params)` with
`case LLM_ARCH_GEMMA4: return new llama_model_gemma4(params);` [91]. The runtime touches only the
abstract `llama_model` interface — `load_hparams`, `load_arch_hparams`, `load_tensors`,
`load_arch_tensors`, `post_arch_tensors`, `build_arch_graph` are pure virtuals [91] — so a new
architecture is one new file under `src/models/` (187 such files) and one `case`.

Three details matter more than the factory:

- **Per-layer arrays, not scalars**: `n_head_arr`, `n_head_kv_arr`, `n_ff_arr`, `n_ff_exp_arr`,
  `n_expert_used_arr`, `is_swa_impl`, indexed `[LLAMA_MAX_LAYERS]`, with accessors
  `n_embd_head_k(il)`, `is_swa(il)`, `n_expert_used(il)` [91]. Gemma 3 fills them from
  `load_swa_pattern(ml, 6)` — a period-6 local:global pattern — and `LLAMA_SWA_TYPE_STANDARD` [103];
  Gemma 4 additionally reads `LLM_KV_ATTENTION_KEY_LENGTH_SWA` into `n_embd_head_k_swa` and derives
  KV-sharing with `hparams.kv_source_impl[il] = n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1)`
  [101].
- **State selection is arch-keyed but centralised**: `llama_model::create_memory()` is one
  `switch (arch)` returning `nullptr` for embedding/BERT/Dream arches and `new llama_kv_cache_msa(...)`
  for `LLM_ARCH_MINIMAX_M3` [91].
- **One MoE entry point**: `llm_graph_context::build_moe_ffn(cur, gate_inp, up_exps, gate_exps,
  down_exps, exp_probs_b, n_expert, n_expert_used, type_op, norm_w, w_scale, gating_op, il, …)`
  serves every MoE arch; per-arch differences are arguments, not code paths [91]. Qwen3-MoE only
  supplies `n_ff_exp_arr` and maps 48 layers to `LLM_TYPE_30B_A3B`, 94 to `LLM_TYPE_235B_A22B` [105].

Inkling-2 is the loop reference. `LLM_KV_LOOP_START/END/COUNT/GATE_RANK/VARYING` (`"%s.loop.start"`
…) [90] are read into `core_start`, `core_end`, `n_passes`, `gate_rank`;
`n_slot = n_phys + (n_passes - 1) * (core_end - core_start)` and `hparams.n_layer_all = n_slot`
expands the layer count *before* allocation, copying every per-layer array from the physical layer
[92]. `post_arch_tensors()` then aliases weights: `layers[il] = layers[layer_phys(il)]`, so each
slot gets its own KV/conv state but no duplicate weights [92]. The graph runs the prelude once,
then for `pass in 0..n_passes` replays the core, feeding a cross-pass message computed by
`control()` from `ctrl_down/ctrl_gain/ctrl_forget` with `mix(bank, bank_out, loop.msg_query)` [92].

### 2.3 SGLang — self-registering modules, no central dict

`import_model_classes(package_name)` walks `pkgutil.iter_modules(package.__path__, …)` and reads a
module-level `EntryClass` (single class or list) from each file; `_ModelRegistry.resolve_model_cls`
then resolves by architecture string [22]. The registry file states it is "Adapted from
https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/models/registry.py"
[22]. Fallback is explicit: `_normalize_archs` appends `"TransformersForCausalLM"` when nothing
matched, so an unknown model degrades instead of failing [22].

### 2.4 KTransformers — model from SGLang, kernels injected by rule

The 2026 tree no longer defines models: `third_party/sglang` is a submodule and only the operator
layer is KTransformers' own [28]. Injection is a YAML rule engine — each rule has `match`
(name regex and/or `class`) and `replace` (`class:` dotted path plus kwargs), with
`recursive: False` to stop the walker descending into an `nn.ModuleList` of experts [94].
`inject()` iterates `module._modules`, imports the replacement class, constructs it with
`orig_module=child`, and calls `set_module(module, name, inject_module)` [94]. Kernel selection is
a `__new__` factory: `KTMoEWrapper(...)` takes `method="AMXINT4" | "AMXINT8" | "LLAMAFILE"` and
returns the backend-specific class, so the call site names one type and the method string picks the
implementation [28].

### 2.5 MNN-LLM — architecture resolved at export, runtime sees a graph

`transformers/llm/export/utils/model_mapper.py` holds `ModelMapper.regist_<arch>()` methods
(`regist_gemma3`, `regist_gemma4`, `regist_qwen3_moe`, …) that map a `model_type` to an export-time
model class [95]. The C++ runtime in `transformers/llm/engine/src/llm.cpp` then loads `llm.mnn` /
`llm.mnn.weight` and passes `mMeta` (a `KVMeta`) to `MNN::Interpreter` via the
`Interpreter::KVCACHE_INFO` hint [43]. Almost no architecture survives to runtime: `llmconfig.hpp`
exposes only scalars such as `sliding_window()` and `is_mrope()` [43]. This is the "compile the
architecture away" end of the spectrum, and the reason MNN can serve dozens of model types with one
engine —
at the cost of a Python exporter in the loop.

### 2.6 mlx-lm — module per arch, and architecture-specific *state* isolated in cache classes

`model_package(config)` maps `config["model_type"]` through `MODEL_REMAPPING` and a
`MODEL_ARCHITECTURE_REMAPPING` pair-keyed table, then `_get_classes` does
`importlib.import_module(f"mlx_lm.models.{model_type}")` and returns `arch.Model, arch.ModelArgs`
[98]. The sharp idea is cache: `make_prompt_cache(model)` defers to `model.make_cache()` if it
exists, else builds a flat list of `KVCache` [97]. Gemma 4's `make_cache` chooses per layer —
`KVCache()` for `layer_types[i] == "full_attention"`, `RotatingKVCache(max_size=args.sliding_window,
keep=0)` otherwise — and stops at `first_kv_shared = args.num_hidden_layers -
args.num_kv_shared_layers`, because the tail layers reuse earlier caches [104]. Architecture-specific
KV policy therefore lives in a small per-arch function, not in the generation loop.

### 2.7 ONNX Runtime GenAI — `model.type` plus graph geometry

`genai_config.json` carries `model.type` (documented values include `gemma`, `gemma2`,
`gemma3_text`, `qwen2`, `qwen3`) and the decoder geometry `num_hidden_layers`, `head_size`,
`num_key_value_heads`, with KV bound by name patterns `past_key_names:
"past_key_values.%d.key"` [100]. One decoder implementation consumes that config; the C API
exposes `OgaCreateModel` and `OgaModelGetType` and nothing architecture-specific [99]. Because the
GitHub repository is currently unreachable, the source-level claim that `CreateModel()` is an
if/else chain over `model.type` cannot be verified here and is **not** relied on
[unverified]; the config surface is [100].

### 2.8 ggml backends — capability vtables instead of type tests

`ggml_backend_i` is a function-pointer table (`graph_compute`, `graph_plan_compute`, async tensor
ops) [93]. The device table `ggml_backend_device_i` adds the capability queries that make
heterogeneous placement possible without `#ifdef`s: `supports_op(dev, op)`, `supports_buft(dev,
buft)`, `offload_op(dev, op)` [93]. Registries are dynamic: `ggml_backend_reg_i` exposes
`get_device_count`/`get_device`/`get_proc_address`, `ggml_backend_load_all()` scans libraries, and
a backend publishes an integer `ggml_backend_score()` symbol used to pick the best build [93].
Operation dispatch itself is one 228-case `switch` in `ggml_compute_forward()` [102].

### 2.9 Comparative table

| Engine | Dispatch key | Per-arch artefact | Capability query | How the runtime avoids a fork |
|---|---|---|---|---|
| vLLM | first entry of `architectures` | `(module, class)` + `_ModelInfo` | `Protocol` + `is_*()` predicates [87] | lazy import, one `resolve_model_cls()` [14] |
| llama.cpp | `general.architecture` → `llm_arch` | `src/models/<arch>.cpp` subclass | per-layer `*_arr` + `is_swa(il)` [91] | factory + 6 virtuals [91] |
| SGLang | `EntryClass.__name__` | module with `EntryClass` | arch string in registry [22] | pkgutil scan, `TransformersForCausalLM` fallback |
| KTransformers | YAML `match` rule | operator replacement class | `method` string → `__new__` [28] | module substitution after construction [94] |
| MNN-LLM | export-time `model_type` | exporter mapper entry [95] | `sliding_window()`, `is_mrope()` [43] | architecture absent at runtime |
| mlx-lm | `model_type` | `models/<arch>/` package | `make_cache()` presence [97] | module import + per-arch cache |
| ORT GenAI | `model.type` | graph + `genai_config.json` [100] | none documented | one decoder over config geometry |
| ggml | backend registry | backend `.so` | `supports_op`/`offload_op` [93] | vtable, `ggml_backend_score` |

## 3. Proposal — `wubu_arch_plan`, four files and three hooks

The proposal keeps the existing 16-row registry as the *name → descriptor* map and adds what the
forward loop actually needs: a resolved per-layer plan plus an ops table. Flat `src/` files, one
`CORE_OBJ +=` line each, matching `Makefile.win`'s existing style.

| File | Owns | Gate |
|---|---|---|
| `include/wubu_arch_plan.h` | `wubu_layer_geom_t`, `wubu_arch_ops_t`, `wubu_arch_plan_t`, plan API | — |
| `src/wubu_arch_plan.c` | GGUF hparams → `plan->layer[n_slots]`; loop slot expansion; validation | `test_eg111_arch_plan` (proposed id, unreserved) |
| `src/wubu_arch_ops.c` | descriptor table (`wubu_arch_desc_t[]`) + generic ops = today's SSM/GQA/dense path unchanged | `test_eg111_arch_plan` |
| `src/wubu_arch_gemma.c` | gemma2/3/4: SWA pattern, qk-norm, `attn_scale`, per-layer `head_dim`, KV-shared tail | `test_eg112_swa_plan` (proposed) |
| `src/wubu_arch_qwen_moe.c` | qwen2moe/qwen3moe/deepseek2: per-layer `n_expert`, `n_expert_used`, `n_ff_exp`, `norm_topk_prob` | `test_eg113_moe_dims` (proposed) |
| `src/wubu_arch_inkling.c` | loop hparams, `slot_phys()`, cross-pass `mix()` | `test_eg114_loop_slots` (proposed) |

### 3.1 Seam types (C11)

```c
/* include/wubu_arch_plan.h */
typedef enum { WUBU_LK_GQA = 0, WUBU_LK_GQA_SWA, WUBU_LK_SSM_DELTA, WUBU_LK_CONV } wubu_layer_kind_t;
typedef enum { WUBU_FFN_DENSE = 0, WUBU_FFN_MOE, WUBU_FFN_MOE_SHARED }        wubu_ffn_kind_t;

typedef struct {
    uint8_t  kind;        /* wubu_layer_kind_t */
    uint8_t  ffn;         /* wubu_ffn_kind_t   */
    uint8_t  rope;        /* 0 neox, 1 normal, 2 partial */
    uint8_t  kv_shared;   /* 1 = reuse an earlier slot's KV */
    int32_t  n_head, n_head_kv, head_dim, rotary_dim;
    int32_t  n_ff, n_expert, n_expert_used, n_ff_exp;
    int32_t  swa_window;  /* 0 = full attention */
    int32_t  kv_slot;     /* index into the KV arena; loop passes get their own */
    float    rope_base, rope_base_swa, attn_scale;
} wubu_layer_geom_t;

typedef struct wubu_arch_ops {
    int (*hparams)(gguf_ctx *ctx, const char *arch, wubu_layer_geom_t *g, int n_phys);
    int (*tensors)(wubu_model_t *m, gguf_ctx *ctx, int il);
    int (*attn)  (wubu_model_t *m, const wubu_layer_geom_t *g, const float *cur, float *out, int B, int T);
    int (*ffn)   (wubu_model_t *m, const wubu_layer_geom_t *g, const float *cur, float *out, int B, int T);
    int (*mix)   (wubu_model_t *m, const wubu_layer_geom_t *g, int slot, int pass, float *cur); /* NULL unless looped */
} wubu_arch_ops_t;

typedef struct {
    wubu_arch_id_t         id;
    const wubu_arch_ops_t *ops;     /* NULL → generic transformer ops */
    int  n_phys, n_slots;           /* n_slots = n_phys + (passes-1)*(loop.end-loop.start) */
    struct { int start, end, passes, gate_rank; } loop;   /* end==start → no loop */
    wubu_layer_geom_t *layer;       /* [n_slots]; slot >= n_phys aliases layer[slot_phys(slot)] */
} wubu_arch_plan_t;

int  wubu_arch_plan_build(gguf_ctx *ctx, wubu_arch_plan_t *out);
int  wubu_arch_slot_phys(const wubu_arch_plan_t *p, int slot);   /* Inkling: llama_model_inkling2_loop::layer_phys [92] */
void wubu_arch_plan_free(wubu_arch_plan_t *p);
```

`hparams` is pure (no allocation, no I/O past the open `gguf_ctx`), `tensors` binds weights once per
*physical* layer, and `attn`/`ffn`/`mix` are the only per-token hooks. That mirrors llama.cpp's
`load_arch_hparams` / `load_arch_tensors` / `post_arch_tensors` / `build_arch_graph` split [91]
without adopting its one-class-per-arch tax.

### 3.2 Where each hook attaches to the existing engine

| Hook | Current call site | Change |
|---|---|---|
| plan build | `wubu_arch_probe(ctx, &arch)` at `src/wubu_model.c:44` | `wubu_arch_plan_build()`; `wubu_arch_probe` stays as the metadata-only path used by `tools/model_arch_probe.c` |
| layer kind | `layer->is_ssm = wubu_is_ssm_layer(l)` at `src/wubu_model.c:143`; `(layer_idx + 1) % 4` at `src/wubu_ops.c:29-33` | `plan->layer[l].kind`; the modulo function is deleted, not defaulted |
| attention | `if (layer->is_ssm) { … } else { … }` inside `wubu_model_forward_from_embd()` (`src/wubu_model.c:718` onward) | `plan->ops->attn(model, &plan->layer[l], normed, attn_out, B, T)` |
| FFN / MoE | the `if (layer->moe.loaded && model->enable_moe)` … `else if (layer->dense_ffn …)` chain at `src/wubu_model.c:929-966` | `plan->ops->ffn(...)`; the chain becomes the generic ops' body |
| MoE dims | `model->n_experts = N_EXPERTS` (`256`), `src/wubu_model.c:131` | `plan->layer[l].n_expert` / `n_expert_used` / `n_ff_exp`; `wubu_moe_forward(x,B,T,w,out,sel,n_active,n_experts,d_model,d_ff)` keeps its signature (`include/wubu_moe.h:110`) |
| loop | none — `for (int l = 0; l < model->n_layers; l++)` at `src/wubu_model.c:718` | outer `for (pass = 0; pass < plan->loop.passes; pass++)` over `[loop.start, loop.end)`, `plan->ops->mix()` before each replayed slot |
| KV | `model->ssm_states + l * …` and `kv_cache_alloc_size()` (`include/wubu_model.h:228`) | size by `plan->n_slots`, index by `plan->layer[l].kv_slot` |

`wubu_loop_place_plan()` (EG-105, `include/wubu_loop_place.h`) already models the loop core's
placement; it reads `loop_start/loop_end/passes` from a synthetic config and would take the same
three fields from `plan->loop` instead of duplicating them.

### 3.3 Per-architecture descriptor rows

| `gguf_arch` | ops | flags | Keys that must be read instead of hard-coded |
|---|---|---|---|
| `gemma3`, `gemma3_text` | gemma | `SWA_PATTERN` (period 6), `QK_NORM` | `attention.sliding_window`, `rope.freq_base_swa`, per-layer head count [103] |
| `gemma4`, `gemma4_text` | gemma | `SWA_PATTERN`, `QK_NORM`, `LAYER_MOE`, `KV_SHARE` | `attention.key_length_swa`, `attention.value_length_swa`, `attention.sliding_window_pattern`, `n_embd_head_k(i)` [101] |
| `qwen2moe`, `qwen3moe` | qwen_moe | `LAYER_MOE`, `NORM_TOPK` | `expert_feed_forward_length` as an array, `expert_used_count`, `expert_count` [105] |
| `inkling2_loop` | inkling | `LOOPED`, `LAYER_MOE` | `loop.start`, `loop.end`, `loop.count`, `loop.gate_rank`, `loop.varying` [90][92] |

Gemma 4's `n_embd_head_k != n_embd_head_v` case throws upstream (`"Gemma 4 requires n_embd_head_k
== n_embd_head_v"`) [101]; plan validation should fail closed the same way rather than clamp.

## 4. Three highest-value adoptions, ranked

| # | Adoption | Upstream precedent | What it unblocks on 6 GB / 16 GB | Cost | Risk |
|---|---|---|---|---|---|
| 1 | **Per-layer geometry plan + ops table** (kills `(l+1)%4`, per-layer `head_dim`, SWA windows, `kv_slot`) | llama.cpp `*_arr` + `is_swa(il)` [91][103]; mlx per-layer `make_cache` [104] | Gemma 3/4 and Qwen3-MoE become loadable in the *same* binary; SWA cuts KV from ~32 MiB/layer at 8K to ~4 MiB/layer for patterned layers (illustrative: 8 KV heads × 128 dims × F16 × 2 tensors = 4 KiB/token) | ~400 LOC est. + 2 gates; no new kernels | Medium: every KV allocation site must switch to slot indexing at once |
| 2 | **One MoE entry with per-layer expert dims** (`n_expert`, `n_expert_used`, `n_ff_exp`, `norm_topk`) | `build_moe_ffn(..., n_expert, n_expert_used, gating_op, …)` [91]; `Qwen3MoeSparseMoeBlock` with `renormalize=config.norm_topk_prob` [88] | Qwen3-30B-A3B (48 layers, 30.5B total / 3.3B activated, `LLM_TYPE_30B_A3B` [105][106]) runs as a streamed-expert model: 8 of 128 experts touched per token per layer, so VRAM holds attention + shared experts and DRAM bandwidth, not capacity, is the binding constraint | ~250 LOC est., reuses `wubu_moe_forward` | Low: existing 8-expert path is the `n_expert==8` special case |
| 3 | **Loop slot expansion with shared weights and per-slot KV** | `n_slot = n_phys + (n_passes-1)*(core_end-core_start)`, `layers[il] = layers[layer_phys(il)]`, `mix()`/`control()` [92] | Inkling-2 looped MoE cannot run at all without it; with it, the loop core is read once and replayed, so the working set is the loop core plus one streaming buffer, not the whole file (`docs/ENGINE-AGGREGATION.md` records Inkling 2 Mini at ~20 GB) | ~300 LOC est. + 1 gate | High: depends on #1 for slot KV and #2 for MoE dims; `mix()` needs three gate matrices per attention/FFN site |

Ranking rationale: #1 is the only adoption that changes *which* families load, fixes three of the
five leaks in §1, and is a prerequisite for both others. #2 is a small delta on an existing
function and is the difference between "Qwen3-MoE loads and is mis-sized" and "runs". #3 is
mandatory for one family only, has the largest design surface, and is worthless until #1 and #2
land.

## 5. What not to adopt

- **One source file per architecture.** llama.cpp carries 187 model `.cpp` files [89]; SGLang 272
  model modules [22]. That pays off where each arch can express itself as a graph class. Where the
  forward pass is one function over one layer struct, it multiplies near-identical layer code; four
  arch files plus a shared ops body is the right granularity here.
- **KTransformers' YAML rule engine.** `inject()` walks `module._modules` and rewrites class
  instances at runtime [94]; C11 has no module tree to walk, and the mechanism only exists because
  the model definition belongs to a foreign Python runtime.
- **Export-time architecture resolution (MNN, ORT GenAI).** `ModelMapper.regist_*` [95] and
  `genai_config.json` [100] move architecture knowledge into a converter. That is defensible when
  the goal is running graphs exported elsewhere; it contradicts a single-binary GGUF runtime and
  would put a Python exporter between the user and every new checkpoint.
- **A central registry dict alone.** vLLM's 430-entry dict works because `_ModelInfo` extraction and
  lazy import carry the behaviour [14]; a table of names with no per-layer plan is exactly the state
  wubuwizard is in today.

## 6. Verification and next actions

1. `test_eg111_arch_plan` — build a plan from a synthetic GGUF fixture with 6 layers, assert
   `kind[]` matches a Gemma-3-shaped period-6 SWA pattern and that a Qwen-shaped pure-GQA model gets
   zero `WUBU_LK_SSM_DELTA` layers. This is the gate that retires `wubu_is_ssm_layer()`.
2. `test_eg113_moe_dims` — the Qwen3-30B-A3B shape (128 experts, top-8, 48 layers [106]) must survive plan build and reach
   `wubu_moe_forward` unchanged; a 256/8 model must not be silently coerced.
3. `test_eg114_loop_slots` — 18 physical layers with core `[6,18)` and 3 passes must yield
   `n_slots = 30`, `slot_phys(29) = 17`, and reject `core_end % block != 0` the way
   `inkling2-loop.cpp` does [92].

Proposed gate ids (EG-111…114) are **not** reserved: the register was not edited by this report, and
`docs/ENGINE-GAP-REGISTER.md` currently ends at EG-110. Re-run the dedupe check before claiming any
of them.

## Sources

[14] https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py — vLLM models/registry.py
[22] https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py — SGLang models/registry.py
[28] https://github.com/kvcache-ai/ktransformers — KTransformers
[43] https://github.com/alibaba/MNN — MNN
[87] https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/interfaces.py — vLLM models/interfaces.py
[88] https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_moe.py — vLLM models/qwen3_moe.py
[89] https://github.com/ggml-org/llama.cpp — llama.cpp
[90] https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp — llama.cpp src/llama-arch.cpp
[91] https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp — llama.cpp src/llama-model.cpp
[92] https://github.com/ggml-org/llama.cpp/blob/master/src/models/inkling2-loop.cpp — llama.cpp src/models/inkling2-loop.cpp
[93] https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend-impl.h — ggml src/ggml-backend-impl.h
[94] https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/deepseek-v2-injection.md — KTransformers injection tutorial
[95] https://github.com/alibaba/MNN/blob/master/transformers/llm/export/utils/model_mapper.py — MNN llm/export model_mapper.py
[96] https://github.com/ml-explore/mlx-lm — mlx-lm
[97] https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py — mlx-lm models/cache.py
[98] https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py — mlx-lm utils.py
[99] https://onnxruntime.ai/docs/genai/api/c.html — ONNX Runtime GenAI C API
[100] https://onnxruntime.ai/docs/genai/reference/config.html — ONNX Runtime GenAI config reference
[101] https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma4.cpp — llama.cpp src/models/gemma4.cpp
[102] https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c — ggml src/ggml-cpu/ggml-cpu.c
[103] https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma3.cpp — llama.cpp src/models/gemma3.cpp
[104] https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text/model.py — mlx-lm gemma4_text/model.py
[105] https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen3moe.cpp — llama.cpp src/models/qwen3moe.cpp
[106] https://huggingface.co/Qwen/Qwen3-30B-A3B — Qwen3-30B-A3B model card
