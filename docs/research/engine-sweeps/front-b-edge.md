# Front B — Local/Edge engine sweep: mechanisms wubuwizard has not absorbed

**Scope.** Seven local/edge engine families (llama.cpp, mistral.rs, llama.swift wrappers, candle, WebGPU/WebLLM+MLC, MLX/mlx-lm, Ollama). Purpose: find *concrete, named* mechanisms worth reimplementing in wubuwizard's own C11 — never copy or vendor source. Target hardware throughout: **6 GB VRAM / 16 GB RAM / SSD**, single user, no server room.

**Method.** 14 `web_search` calls, 20 `web_extract` calls, plus a local audit of `C:/Users/eman5/WuBu/repos/wubuwizard` (source tree, `src/`, `include/`, `Makefile` gates, symbol greps). Every external claim carries a ledger id; the Sources block at the end was generated from the ledger, not retyped.

## Local truth: what is already there (do NOT re-report as gaps)

The audit confirmed these are real code, not stubs:

- **GGUF/Safetensors readers with zero-copy mapping** — `src/gguf_reader.c:1626-1695` maps the whole data blob with `mmap(MAP_PRIVATE)` + `madvise(MADV_WILLNEED)` + `MADV_HUGEPAGE`, page-aligned offset fix-up, and malloc+fread fallback; `src/safetensors_reader.c:145-228` maps the whole shard. GGUF IQ1_S/IQ2 grid tables are in-tree (`NGRID_IQ1S`, `iq2s_grid.inc`, `iq2xxs_grid_data.inc`).
- **Quantized dot kernels** — `src/quantized_dot_generic.c` has `ggml_vec_dot_q4_K_q8_K_generic/_sse/_avx2`, Q5_K, Q6_K, NEON variants; `src/quantized_matmul.c` quantizes activations to Q8_K per column and dispatches per weight type; `src/wubu_simd_quant.c`, `src/wubu_hwcaps.c` (CPUID), `include/wubu_portable_simd.h`, `src/wubu_gemv_dispatch.c` exist.
- **Tokenizer** — `src/wubu_tokenizer.c` has byte-level BPE (Qwen3.6/GPT2 byte maps, FNV-1a+Murmur merge hash), `src/wubu_gguf_tokenizer.c` extracts `tokens/merges/bos/eos/pad`, `src/wubu_tokenizer_hf.c` handles `tokenizer.json`.
- **Constrained decode** — `src/wubu_spec_constrained.c` is a hand-rolled state machine for one tool-call JSON shape.
- **KV/scheduling** — continuous batching, paged KV, prefix reuse, chunked prefill, flash attention/decode, KV filesystem, weight streaming, CUDA graphs, LM-head cache, spec decode families, 16 LP pads (per the standing context).
- **Sampling** — `tools/infer_text.c:597` `sample()` covers temperature, top-k, top-p, repetition penalty; `src/wubu.c:757` has temperature + multinomial.

Everything below is **absent or materially different** in that tree.

---

## 1. llama.cpp — the reference C/C++ engine

- **Repo / language / license:** `https://github.com/ggml-org/llama.cpp`; C++17 core + C ggml; MIT [35][13].
- **Why study it:** it defines the GGUF-format ground truth and the CPU kernel/dispatch conventions that every other local engine copies.

### 1a. Quantized dot kernels with packed per-block scales (`ggml/src/ggml-cpu/quants.c`)

Concrete symbols: `quantize_row_q4_K`, `ggml_vec_dot_q4_K_q8_K_generic`, `ggml_vec_dot_q4_K_q8_K` (AVX2/AVX512/NEON specializations), `block_q4_K` (QK_K = 256), `block_q8_K` with `int16_t bsums[16]`, and per-quant epsilon constants `GROUP_MAX_EPS_IQ2_S`, `GROUP_MAX_EPS_IQ1_S`, `GROUP_MAX_EPS_IQ3_XXS`. Included headers are `simd-mappings.h`, `vec.h`, `arch-fallback.h` — i.e. one ISA-neutral loop body plus per-ISA macro remapping [14]. The dot kernel unpacks nibbles into `int8_t aux8[QK_K]`, accumulates in an eight-lane `int32_t aux32` array, then folds 8 super-block scales in one pass; the Q8_K activation side carries precomputed sub-block sums (`bsums`) so per-block scale multiply is deferred [14].

**6 GB relevance:** Q4_K_M (~4.5 bpw) is the practical 3–8 B fit point; a dot kernel that never materializes f32 weights keeps the resident set inside 6 GB and reduces DRAM traffic on the 16 GB side [14].

**Not in wubuwizard:** the *multi-row* form of that kernel (below). The single-row Q4_K/Q5_K/Q6_K dots themselves are already native — do not re-report them.

### 1b. Runtime per-type `vec_dot` dispatch table + multi-row GEMV (`ggml/src/ggml-cpu/ggml-cpu.c`)

Concrete symbols: `static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT]` with fields `.vec_dot`, `.vec_dot_type`, `.from_float`, `.from_float_to_mat`, `.nrows`, `.ncols`, `.gemv`, `.gemm`; `ggml_compute_forward_mul_mat_one_chunk`; `ggml_vec_dot_t const vec_dot = type_traits_cpu[type].vec_dot`; `num_rows_per_vec_dot`, `src1_col_stride`, `blck_1 = 16`, `GGML_VEC_DOT_UNROLL 2`; capability probes `ggml_cpu_has_avx/avx2/avx512_vnni/avx512_bf16/wasm_simd` [31][13]. In the multi-row path the kernel emits several output rows per activation load (`vec_dot(..., num_rows_per_vec_dot)`), writing through a `tmp[32]` staging buffer with `memcpy` — an explicit trade of registers for fewer activation reads [31].

**6 GB relevance:** decode is batch-1 in VRAM but *output-row-heavy*; fusing 4–16 output rows per Q8_K activation load cuts the bandwidth that dominates a 6 GB card. The `.nrows/.ncols` traits let a build choose the fusion factor per quant type without recompiling.

**Not in wubuwizard:** wubu has CPUID detection (`src/wubu_hwcaps.c`) but `src/quantized_matmul.c` selects kernels with compile-time `#if defined(__AVX2__)`; its comment says "for each output column, quantizes the F32 input to Q8_K then calls" a dot function. There is no `type_traits_cpu`-style runtime table, no `num_rows_per_vec_dot > 1` path, no `.from_float_to_mat`/`.nrows`/`.ncols` metadata.

### 1c. GGUF reader discipline: metadata-first sizing, alignment, shard validation

Concrete behaviour: fixed open order (magic, version, counts, KV, tensor table, padding, weights); default `ALIGNMENT = 32` when `general.alignment` absent, power-of-two validation when present; every tensor offset checked to be the padded sum of preceding sizes; `split.no` / `split.count` / `split.tensors.count` used to open sibling shards and reject mismatched sets; `general.file_type` only overrides a majority-type guess [18]. Vocabulary is read from `tokenizer.ggml.tokens/.merges/.token_type/.pre/.chat_template`; `tokenizer.ggml.pre` selects the pre-tokenizer family and a missing value emits "GENERATION QUALITY WILL BE DEGRADED!" [18][15].

**6 GB relevance:** metadata-first sizing means the scheduler can decide "this quant will not fit" *before* touching 4 GB of SSD pages.

**Not in wubuwizard:** `src/wubu_gguf_tokenizer.c` extracts only `tokens/merges/bos/eos/pad` — no `tokenizer.ggml.pre`, no `token_type`, no `add_bos`, no `chat_template`. `src/gguf_reader.c` does not implement the sibling-shard count/offset cross-check described in [18].

### 1d. mmap prefetch policy (`src/llama-mmap.cpp`, issue #91)

`llama_prefetch` calls `madvise(addr, size, MADV_WILLNEED)`; the loader can additionally use `MAP_POPULATE` at map time [32][18]. The upstream issue records the deliberate choice **against** `MADV_SEQUENTIAL` (it evicts pages just read, and a decode pass rereads all weights every token) in favour of `MADV_WILLNEED` on the whole model [19].

**6 GB relevance:** on an SSD, time-to-first-token is a page-fault problem; async prefetch off the timed path is free throughput.

**Not in wubuwizard:** already largely absorbed — `gguf_buffer_data()` maps and issues `MADV_WILLNEED`/`MADV_HUGEPAGE`. The only unabsorbed nuance is the *explicit rejection of MADV_SEQUENTIAL* rationale and `MAP_POPULATE` opt-in. Study, do not re-implement as a new subsystem.

### 1e. Tokenizer families and pre-tokenizer registry (`src/llama-vocab.cpp`, `src/llama-vocab.h`, `src/unicode.cpp`)

Five families: `llm_tokenizer_bpe`, SPM, WPM, UGM (unigram lattice/Viterbi), RWKV trie; selected by `llama_vocab_type` (`SPM/BPE/WPM/UGM/RWKV`). Per-model regex variants live in `llama_vocab_pre_type` (`..._LLAMA3`, `..._GPT2`, `..._QWEN2`, `..._DEEPSEEK_LLM`, ...) and are chosen by hashing the upstream `tokenizer.json` regex in the conversion script, then storing the *name* in GGUF [15][16]. `src/unicode.cpp` supplies NFC normalization and byte-fallback for invalid UTF-8; detokenization is `llama_token_to_piece` / `llama_detokenize` with `lstrip`/`special` flags [15].

**Not in wubuwizard:** one hardcoded byte map in `src/wubu_tokenizer.c`; no family enum, no pre-tokenizer registry keyed on `tokenizer.ggml.pre`, no Unicode normalization tables, no WordPiece/Unigram/RWKV tokenizers, no `lstrip`/special-token detokenization rules.

### 1f. GBNF grammar engine and sampler chain (`src/llama-grammar.cpp/.h`, `common/json-schema-to-grammar.cpp`)

`llama_grammar` compiles GBNF to a stack machine (`llama_grammar_stack`); `llama_grammar_accept_impl` advances state; `llama_sampler_init_grammar` / `llama_sampler_init_grammar_lazy_patterns` mask illegal tokens at sample time. Lazy patterns activate only after a trigger (e.g. `<tool_call>`). JSON Schema → GBNF conversion is a C++ converter used by `llama-cli`/`llama-server` response_format, plus a Python script for `$ref` schemas [17]. The sampler chain composes named filters in a fixed order — `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature` (CLI default) — with mirostat and dynatemp as alternatives [33].

**6 GB relevance:** grammar masking is pure CPU and avoids regenerating long invalid JSON — every wasted token is a full 6 GB weight sweep.

**Not in wubuwizard:** `src/wubu_spec_constrained.c` is a single hardcoded JSON tool-call shape (states `ST_START…ST_DONE`, `WUBU_JTOK_*`), not a parseable grammar engine; no GBNF reader, no JSON-Schema converter, no composable sampler chain (min-p, top-n-sigma, XTC, DRY, mirostat, dynatemp all absent).

### 1g. minja chat templates (`common/jinja/`, `src/llama-chat.cpp`)

`common_chat_templates_apply` renders a Jinja subset with a lexer/parser/runtime in `common/jinja/`; `llama_model_chat_template()` returns the GGUF `tokenizer.chat_template` string (≈10 KB in modern models) [16][18].

**Not in wubuwizard:** no C-level Jinja/minja renderer; prompt assembly is hardcoded per tool (`tools/gen_text*.c`, `tools/serve.py` delegates to Python tokenizers).

---

## 2. mistral.rs — Rust, hardware-aware quant placement

- **Repo / language / license:** `https://github.com/EricLBuehler/mistral.rs`; Rust; MIT [20].
- **Crate map:** `mistralrs-core/` (`src/pipeline/mod.rs` `Pipeline` trait, `src/loader.rs`, `src/device_map.rs`), `mistralrs-quant/`, `mistralrs-paged-attn/`, `mistralrs-cli/`, `mistralrs-pyo3/` [21].

### 2a. Hardware-aware `mistralrs tune` + topology

`mistralrs tune` recommends a quantization *and* device mapping from the model config and detected hardware; GGUF artifacts are selected by `--quant`, otherwise a prebuilt UQFF is used, otherwise ISQ is applied [20]. A YAML "Topology" configures ISQ and device mapping *per layer* [21][34]. Automatic device mapping places some layers on GPU and the rest on CPU [20].

**6 GB relevance:** this is exactly the 6 GB fit problem expressed as a planner: choose bpw and the GPU/CPU layer cut from detected VRAM before loading.

**Not in wubuwizard:** `src/wubu_resource.c` uses a coarse `VRAM*quant_factor` heuristic; `src/wubu_mem_budget.c` budgets **host RAM** (`MemAvailable`, 10% headroom), not VRAM from GGUF metadata. No tune-style recommender, no per-layer topology YAML, no ISQ-vs-prebuilt selection policy.

### 2b. In-situ quantization (ISQ) with spike avoidance

ISQ quantizes HF `.safetensors` weights in place at load; importantly, ISQ-able weights are loaded on CPU, quantized there, then moved to device **to avoid load-time memory spikes** [34][20].

**6 GB relevance:** a 6 GB card cannot hold both the f16 source and the q4 result during conversion; CPU-side conversion makes tiny-VRAM loads possible.

**Not in wubuwizard:** no quantize-on-load path from safetensors; quantization tools (`wubu_gptq.c`, `wubu_smoothquant.c`, `wubu_q4k_m.c`) are dequant/quant routines, not an ISQ loader.

### 2c. PagedAttention as a separate crate

`mistralrs-paged-attn/` is a distinct implementation used with continuous batching on CUDA and Metal; prefix caching including multimodal is integrated [20][21].

**6 GB relevance:** already-absorbed territory in wubu — study only for the *crate-boundary discipline* (KV allocator isolated from model code), not as a new feature.

---

## 3. llama.swift — Swift wrappers around llama.cpp

- **Repo / language / license:** `https://github.com/profclaw/swift-llama` (representative); Swift 6; MIT [30]. Other variants: `mattt/llama.swift` re-exports the llama.cpp XCFramework, `a-aigner/LlamaSwift` uses `llama_bridge.h/.mm` + `LlamaModel.swift`.
- **Concrete mechanism:** `LlamaActor` isolates inference behind Swift 6 concurrency and streams tokens via `AsyncThrowingStream`; `LlamaSampling` wraps temp/top-p/top-k; a **streaming tool-call parser detects function calls as tokens arrive**; GGUF loads through a prebuilt llama.cpp XCFramework with Metal [30].
- **6 GB relevance:** almost none for a Windows/C11 target — this is a thin binding layer, not an engine. The one transferable idea is the streaming tool-call parser matching *partial* token pieces to a call grammar.
- **Not in wubuwizard:** `src/wubu_tooluse.c` dispatches complete calls; there is no incremental token-piece → tool-call state machine (the nearest code, `wubu_spec_constrained.c`, masks token ids rather than parsing streamed text).

---

## 4. candle — Rust tensor crate with native GGUF reading

- **Repo / language / license:** `https://github.com/huggingface/candle`; Rust; **MIT OR Apache-2.0** [22][23].
- **Concrete paths/symbols:** `candle-core/src/quantized/gguf_file.rs:85` `pub struct Content`, `:396` `Content::read`; `candle-core/src/quantized/mod.rs:626 dequantize`, `:696 pub enum QMatMul` (`QTensor`/`Tensor`/`TensorF16`), `QuantizedType::matmul_t`, `k_quants::matmul`, `GgmlDType` mirroring ggml's enum, `CANDLE_DEQUANTIZE_ALL` env switch; `candle-core/src/safetensors.rs:423 MmapedSafetensors`, `candle-nn/src/var_builder.rs:642 from_mmaped_safetensors` [22][23].

### 4a. Packed-weight QMatMul with an explicit per-process dequantize-all escape hatch

`QMatMul::forward` multiplies activations against *packed* quantized blocks, dequantizing per block on the fly rather than materializing f32 weights; the `DEQUANTIZE_ALL` switch deliberately converts to a plain f32 `Tensor` when asked [22][23]. `VarBuilder` binds named weights through a path-prefix stack, so a model file can be re-bound to different implementations without touching layer code [22].

**6 GB relevance:** two explicit residency modes for the same checkpoint — packed for the 6 GB path, dequantized for debugging/precision checks — is a useful policy pattern.

**Not in wubuwizard:** wubu has a precision ladder (`wubu_precision_plan.c`, `wubu_gemv_dispatch.c`) but no single `QMatMul`-style abstraction with a *whole-model* dequantize-all override bound to a named-weight loader; the switch is per-process and env-driven in candle.

---

## 5. WebGPU engines — WebLLM / MLC LLM

- **Repo / language / license:** `https://github.com/mlc-ai/web-llm`; TypeScript + WGSL + WASM; license not captured in this sweep [unverified]. MLC LLM: `https://github.com/mlc-ai/mlc-llm`, Apache TVM-based compiler.
- **Concrete mechanism:** three-part architecture — lightweight `ServiceWorkerMLCEngine` at the app end, `MLCEngine` inside a **Web Worker** (message-passing OpenAI-style requests/responses), and **AOT-compiled WGSL WebGPU kernels** in a WASM library; kernel named `batch_decode_paged_kv_kernel`. A model library is specified by architecture + quantization (e.g. `q4f16_1`) + metadata (`context_window_size`, `sliding_window_size`, `prefill-chunk-size`) + platform [28][29]. MLC combines continuous batching, speculative decoding, paged KV, common-prefix caching and cascade inference behind one engine, with `server`/`local`/`interactive` modes trading concurrency against GPU memory [29]. Graph-level fusion and operator-level GEMM tiling come from the TVM compiler [28].

**6 GB relevance:** `q4f16_1` (4-bit weights, f16 compute) and small `prefill-chunk-size` are exactly the knobs for a tight VRAM budget; the `local`/`interactive` modes cap concurrency instead of OOM-ing.

**Not in wubuwizard:** the compiler-generated-kernel idea has no equivalent — wubu hand-writes CUDA/HLSL-style kernels. Do **not** import WASM/Web Workers (irrelevant to a native C11 binary). The transferable item is *metadata-parameterized kernel variants* (`prefill-chunk-size` compiled into the library): wubu's chunked prefill is runtime-sized, not a compile-time model-library parameter. Structured generation across all platforms is also named as a first-class engine capability [29] — see the grammar gap (1f).

---

## 6. MLX / mlx-lm — Apple silicon, lazy evaluation

- **Repo / language / license:** `https://github.com/ml-explore/mlx-lm`; Python + C++ MLX; MIT [36][24].
- **Concrete paths/symbols:** `mlx_lm/sample_utils.py:10-69 make_sampler` (argmax only when `temp == 0`, otherwise a chain); `mlx_lm/models/cache.py make_prompt_cache`, `RotatingKVCache`, `QuantizedKVCache`; `mlx_lm/generate.py` `maybe_quantize_kv_cache` (per-layer, lazily converts a layer's cache once `c.offset >= quantized_kv_start`), flags `--kv-bits`, `--kv-group-size` (default 64), `--quantized-kv-start` (default 5000), `--max-kv-size`, `--prefill-step-size` (default 2048), `--prompt-cache-file`; `mlx_lm.cache_prompt` writes a `.safetensors` prompt cache; `mx.async_eval` and lazy evaluation keep intermediates alive until `eval` [25][24].

### 6a. Sampler chain completeness

`make_sampler` composes temp → top-p → **min-p** → top-k → **XTC** (`--xtc-probability`, `--xtc-threshold`) with `min-tokens-to-keep` floor; llama.cpp adds top-n-sigma, DRY, typical-p, mirostat, dynatemp, adaptive-p [25][33].

**6 GB relevance:** sampling is free; repetition-control filters avoid regenerating full 6 GB sweeps after degenerate loops.

**Not in wubuwizard:** `tools/infer_text.c:597 sample()` implements temperature/top-k/top-p/repetition only; none of min-p, top-n-sigma, XTC, DRY, mirostat, dynatemp, typical-p, adaptive-p exist.

### 6b. Lazy-conversion quantized KV *with a documented anti-pattern*

`maybe_quantize_kv_cache` switches a layer to packed `uint32` + per-group scales/biases only after a threshold; the handbook explicitly warns that a naive Python blockwise/online-softmax quantized attention is **worse** because lazy evaluation keeps every block's intermediates alive [25]. `SERVER.md` warns that attention on a quantized cache is unfused and keeps a `prefill_step_size × context_length` score matrix — lower the step size or the matrix exceeds the memory saved [25].

**6 GB relevance:** this is a genuine capacity-vs-throughput trap on a 6 GB card; a reimplementation needs an explicit fused path or a strict step-size cap.

**Not in wubuwizard:** wubu has KV quantization (`wubu_kvcache_quant.c`, `wubu_kv_qfmt.c`) and chunked prefill, but no *lazy per-layer threshold conversion* and no score-matrix budget check tying prefill chunk size to KV savings.

---

## 7. Ollama — Go scheduler around a vendored llama-server

- **Repo / language / license:** `https://github.com/ollama/ollama`; Go (daemon/scheduler) + C++ (pinned llama-server subprocess); MIT [unverified — repo README not extracted in this sweep].
- **Concrete symbols** (from the handbook's source index): `server/sched.go:61 Scheduler`, `:492 Scheduler.load`, `:165 getRunner`, `:1061 bestSingleGPUFit` (80% rule at `:1065`), `:758 reduceAutoNumCtxForLoadOOM`, `:1449 waitForVRAMRecovery`, `:1339 runnerRef`, `:786 explicitPartialGPUOffload`; `llm/llama_server.go:2474 PredictServerVRAM`, `:2535 offloadedLayersRegex`; `fs/ggml/ggml.go:573 GGML.Decode`, `:56 BlockCount`, `:93 HeadCountKV`; `manifest/paths.go`, `manifest/manifest.go:16`, `manifest/layer.go:26`; `template/template.go:257 Template.Execute` [26][27].

### 7a. Metadata-only VRAM prediction + fit/evict policy

The scheduler reads GGUF KV metadata (`BlockCount`, `HeadCountKV`) to size a model **without loading it**, predicts weights+KV VRAM (`PredictServerVRAM`), places only if `predictedVRAM ≤ available × 0.8` (20% headroom), evicts LRU runners, waits for VRAM to actually be reclaimed (`waitForVRAMRecovery`), and on OOM shrinks `num_ctx`/`num_batch` and retries (`reduceAutoNumCtxForLoadOOM`) [26][27]. The Go layer picks the device; llama-server auto-detects the offloaded layer count [26].

**6 GB relevance:** this is the single most directly useful scheduler pattern here — never start a 4.7 GB load on a 6 GB card with only 1.2 GB free, and retry smaller rather than crash.

**Not in wubuwizard:** `src/wubu_mem_budget.c` measures **host RAM** and reserves 10% headroom; `src/wubu_resource.c` uses `VRAM*quant_factor` rather than GGUF metadata. No 80%-of-free-VRAM rule, no OOM context shrink, no post-eviction VRAM recovery wait.

### 7b. Content-addressed blob store + manifests

Models are OCI-style: `manifests/<name>` points to typed layers whose blobs live at `blobs/sha256-<hex>`; digest validity is regex-checked (`sha256[:-][64 hex]`); downloads are chunked/resumable with sha256 verification before manifest write; identical content is stored once across models [26][27]. Chat templates are rendered by a Go `text/template` from the GGUF `tokenizer.chat_template` [26].

**6 GB relevance:** on a SSD with finite space, deduplicated shards and resumable pulls matter more than raw speed; a failed download never corrupts the manifest.

**Not in wubuwizard:** `src/wubu_sha1.c`, `src/wubu_torrent.c`, `src/wubu_cid.c` exist, but there is no OCI-style manifest + sha256 blob store with digest validation, layer typing, dedup and resumable chunk writes. The Jinja/Go-template rendering side is also absent (see 1g).

---

## Cross-cutting gap pads (dependency order — a hop path, not a schedule)

Every pad below needs a real make gate; "read the reference, reimplement in C11, never vendor."

**LP-B1 — GGUF tokenizer metadata completeness.** Read `tokenizer.ggml.pre`, `token_type`, `add_bos`, `chat_template` in `src/wubu_gguf_tokenizer.c`; expose a pre-tokenizer-id field [15][18]. *Unlocks every text-fidelity pad; depends on nothing.*

**LP-B2 — Pre-tokenizer registry + Unicode normalization.** Port the *idea* of a hash-keyed family registry keyed on `tokenizer.ggml.pre` (llama3/gpt2/qwen2/deepseek) with NFC byte fallback, in new C11 files [15][16]. *Depends on B1; fixes "GENERATION QUALITY WILL BE DEGRADED" class bugs.*

**LP-B3 — Runtime per-type vec_dot table + multi-row GEMV.** Replace compile-time `#if __AVX2__` selection with a `type_traits`-style table carrying `vec_dot`/`vec_dot_type`/`nrows`, plus a `num_rows_per_vec_dot > 1` path [31][14]. *Depends on nothing; biggest decode-bandwidth win on 6 GB.*

**LP-B4 — Chunked sampler chain with min-p/top-n-sigma/DRY/XT** (no grammar yet) [25][33]. *Depends on B2 for correct piece boundaries; small, pure CPU.*

**LP-B5 — General GBNF stack machine + JSON-Schema→GBNF converter + lazy patterns** [17]. *Depends on B4 for the sampler hook; unlocks valid structured output without regen.*

**LP-B6 — minja-style Jinja-subset chat template renderer** [16][29]. *Depends on B1 (reads the GGUF field); unlocks tool-call templates.*

**LP-B7 — Metadata-first VRAM planner: 80% rule, OOM ctx shrink, recovery wait** [26][27]. *Depends on nothing; makes every larger quant choice safe on 6 GB.*

**LP-B8 — ISQ-style quantize-on-CPU-before-GPU load with per-layer topology** [34][20]. *Depends on B7; enables f16→q4 conversion without a second VRAM copy.*

**LP-B9 — Content-addressed sha256 blob+manifest store with dedup/resumable writes** [26][27]. *Depends on nothing; SSD hygiene.*

**LP-B10 — Lazy per-layer quantized-KV conversion with score-matrix budget check** [25]. *Depends on B7; capacity, not throughput.*

## What not to break

- Keep the mmap-zero-copy weight law; `MADV_SEQUENTIAL` must not be added (it evicts pages before the next token rereads them) [19].
- Keep the engine ignorant of domain concepts: grammar/tool-call semantics stay in sampler/prompt layers, not in GEMV or KV code.
- Keep every pad gated by `make <gate>`; do not mark a pad done on a self-report.
- Never vendor llama.cpp/ggml, candle, mistral.rs, MLX or Ollama sources — read the named files, reimplement in wubuwizard's C11.

## Sources

[13] https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/quants.c — llama.cpp ggml-cpu/quants.c
[14] https://fossies.org/linux/llama.cpp/ggml/src/ggml-cpu/quants.c — Fossies llama.cpp quants.c
[15] https://factory.com/open-source-wikis/llama-cpp?page=systems/vocab-and-tokenizer.md — llama.cpp Tokenization and Vocabulary wiki
[16] https://deepwiki.com/qualcomm/llama.cpp/2.5-tokenization-and-vocabulary — llama.cpp DeepWiki tokenization
[17] https://factory.com/open-source-wikis/llama-cpp?page=systems/grammar.md — llama.cpp Grammar wiki
[18] https://research.strata2signal.com/inside-a-gguf-file/index.html — Inside a GGUF file (strata2signal)
[19] https://github.com/ggml-org/llama.cpp/issues/91 — llama.cpp issue 91 mmap
[20] https://github.com/EricLBuehler/mistral.rs?tab=readme-ov-file — mistral.rs README
[21] https://github.com/glaziermag/mistral.rs/blob/master/CLAUDE.md — mistral.rs CLAUDE.md crate layout
[22] https://rumitx.com/books/candle/handbook — candle Internals Handbook
[23] https://docs.rs/candle-core/0.8.4/i686-unknown-linux-gnu/src/candle_core/quantized/mod.rs.html — candle-core quantized/mod.rs docs.rs
[24] https://github.com/ml-explore/mlx-lm/blob/main/README.md — mlx-lm README
[25] https://github.com/hbmartin/Foundation-Models-and-Core-AI-and-MLX-skills/blob/main/guides/part-12-mlx-python/references/04-mlx-lm-cli-generation-and-caching.md — mlx-lm generation and caching guide
[26] https://rumitx.com/books/ollama/glossary — Ollama Internals Handbook glossary
[27] https://codevetter.com/unpack/ollama — Ollama codebase unpack
[28] http://www.arxiv.org/pdf/2412.15803 — WebLLM paper arXiv 2412.15803
[29] https://blog.mlc.ai/2024/06/07/universal-LLM-deployment-engine-with-ML-compilation.html — MLC LLM universal deployment engine blog
[30] https://github.com/profclaw/swift-llama/blob/main/README.md — swift-llama README
[31] https://huggingface.co/spaces/Xenobd/whisper.cpp/raw/f08388797e47c4b45187288027c26eeb194c6325/ggml/src/ggml-cpu/ggml-cpu.c — ggml-cpu.c raw (vec_dot dispatch)
[32] https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-mmap.cpp — llama-mmap.cpp raw
[33] https://android.googlesource.com/platform/external/ggml-org/llama.cpp/+/refs/tags/studio-2026.1.3-rc01/tools/completion/README.md — llama.cpp tools/completion README sampler defaults
[34] https://github.com/adam-odin/mistral-rust — mistral.rs README mirror ISQ device-load detail
[35] https://github.com/ggml-org/llama.cpp/blob/7992aa7c8e21ea2eb7a5e4802da56eec7b376036/LICENSE — llama.cpp LICENSE MIT
[36] https://github.com/ml-explore/mlx-lm/blob/main/LICENSE — mlx-lm LICENSE MIT
