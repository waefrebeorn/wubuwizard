# Engine Gap Register — ordered hops across the pond

This is the **TODO** side of engine absorption. Each row is one crossable technique with a
gate on the far side. Order is **dependency, not importance** — do not re-sort by urgency.
Status is real gated code only: a research note claiming "port complete" is a claim, not
evidence. Full citations are in `docs/ENGINE-SOURCE-INDEX.md` and
`docs/research/engine-sweeps/`.

## Rules

- One technique, one gate, one evidence row. **No gate means not absorbed.**
- Gate names are reserved here as `test_egNN_<slug>`; they are free in both Makefiles as
  of 2026-11-22. Add the target to **both** `Makefile` and `Makefile.win`, with source
  files listed directly (the two Makefiles do not share an implicit object rule).
- Verify with `mingw32-make -f Makefile.win <gate> && ./<gate>.exe`; capture measured
  numbers and update `docs/WORKLIST.md` before marking the row DONE.
- Read the upstream reference, reimplement in C11. **Never vendor or link third-party code.**

Status legend: TODO / PARTIAL / DONE(gate, evidence).

## Wave 0 — foundations (no dependencies)

| ID | Technique | Local evidence | Reference | Gate | Size | Next action |
|---|---|---|---|---|---|---|
| EG-01 | Runtime per-type `vec_dot` dispatch table + multi-row GEMV | `src/quantized_dot_generic.c` selects kernels with compile-time `#if __AVX2__`, one output column at a time | llama.cpp `ggml-cpu.c` `type_traits_cpu[].vec_dot/nrows/ncols` | `test_eg01_vecdot` | M | Add dispatch table + `num_rows_per_vec_dot>1`; gate on identical dequant output across types |
| EG-02 | GGUF tokenizer metadata completeness | `src/wubu_gguf_tokenizer.c` reads tokens/merges/bos/eos/pad; `tokenizer.ggml.pre`, `token_type`, `add_bos`, `chat_template` unused | llama.cpp `llama-vocab.cpp`, GGUF spec | `test_eg02_tok_meta` | S | Read and expose the four fields; gate asserts values from a fixture GGUF |
| EG-03 | Metadata-first VRAM planner (80% rule, OOM ctx shrink, recovery wait) | `src/wubu_mem_budget.c` budgets host RAM; `src/wubu_resource.c` uses a coarse VRAM×quant_factor heuristic | Ollama `PredictServerVRAM` + `reduceAutoNumCtxForLoadOOM` | `test_eg03_vram_plan` | M | Plan from GGUF metadata before load; gate on refusing an 8.49 GB quant on 6 GB |
| EG-04 | StreamingLLM attention sinks | 0 hits for `attention_sink\|attn_sink\|sink_token\|streaming_llm` in src/include/tools | StreamingLLM / attention-sink literature | `test_eg04_sink` | S | Keep first-position KV as persistent sinks; gate long-context perplexity vs sliding window |
| EG-05 | Neuron-cluster abstraction + activation-sparsity predictor | 0 hits for `activation.predictor\|hot.neuron\|neuron.cluster` in inference path | PowerInfer SOSP 2024; PowerInfer-2 §1/§3 | `test_eg05_neuron` | L | C11 cluster descriptors, runtime sizing, hot/cold split; gate on cluster I/O count |
| EG-06 | Offload-placement solver + zigzag block schedule | `docs/ssd_moe.md` is a per-layer LRU slot bank; no policy solver | FlexGen ICML 2023 §4 Algorithm 1 | `test_eg06_plan` | L | Solve placement from bandwidth/capacity; gate within 2× of cost model |
| EG-07 | SSD sliding-window row reuse + ≥32 KiB coalesced reads | weight streaming reads per tensor on demand; no incremental `s_agg(k+1)-s_agg(k)` reuse | LLM in a Flash §2–3 | `test_eg07_row_reuse` | M | Bundle rows into coalesced reads; gate on read syscall/byte count |
| EG-08 | RadixAttention tree | 0 hits for `radix\|prefix_tree` in src/include (only unrelated audio/codec/kvfs hits); prefix cache is hash-based | SGLang compressed FSM + RadixAttention | `test_eg08_radix` | M | Tree with LRU eviction and incremental branch match; gate sibling-prefix sharing |

## Wave 1 — scheduling and control

| ID | Technique | Local evidence | Reference | Gate | Size | Next action |
|---|---|---|---|---|---|---|
| EG-09 | Overlapping scheduler with deferred result queue | `src/wubu_scheduler.c` steps CPU and decode in the same loop; no step pipelining | SGLang `event_loop_overlap`, mini-sglang `overlap_loop` | `test_eg09_overlap` | M | Defer result drain one step; gate on no starvation + identical tokens |
| EG-10 | Single shared admission validator + token budget | admission scattered across request paths; no queue timeout | TGI shared validator; TensorRT capacity admission | `test_eg10_admit` | S | One validator, one budget; gate rejection of over-budget requests |
| EG-11 | Structured decoding: grammar → automaton → batched bitmask + jump-forward | `src/wubu_spec_constrained.c` is a hardcoded single-shape tool-call mask; no grammar engine | SGLang xgrammar backend; vLLM `StructuredOutputManager`; llama.cpp `llama-grammar.cpp` | `test_eg11_grammar` | L | GBNF stack machine + bitmask, then jump-forward; gate invalid-token rejection |
| EG-12 | Composable sampler chain (min-p, top-n-sigma, XTC, DRY, mirostat, dynatemp, typical-p, adaptive-p) | `src/wubu_token.c` has temperature/top-k/top-p/repetition only | MLX sampler chain; llama.cpp samplers | `test_eg12_sampler` | M | Ordered chain with per-stage RNG; gate distribution invariants |
| EG-13 | Pre-tokenizer registry + Unicode NFC fallback | depends on EG-02; no family registry keyed on `tokenizer.ggml.pre` | llama.cpp `llama-vocab.cpp`, `unicode.cpp` | `test_eg13_pretok` | M | Hash-keyed registry for llama3/gpt2/qwen2/deepseek; gate token-ID parity |
| EG-14 | Jinja-subset chat template renderer | prompts are hardcoded per tool; GGUF `chat_template` unused | llama.cpp minja `common/jinja/`; TGI | `test_eg14_template` | M | Read template from GGUF, render minimal subset; gate exact prompt string |
| EG-15 | Disaggregated PD gate (already implemented, source-only) | `src/wubu_pd_serve.c`, `wubu_pd_split.c`, `wubu_kv_transfer.c` exist; `tools/test_disagg_prefill_decode.c` has no gate | in-tree | `test_eg15_pd` | S | Wire the existing test as a gate against its owning modules |

## Wave 2 — memory and tiering

| ID | Technique | Local evidence | Reference | Gate | Size | Next action |
|---|---|---|---|---|---|---|
| EG-16 | Tagged CUDA sleep/wake allocator with partial restore | no sleep lifecycle or tagged restore in allocator | vLLM `gpu_worker.py` `Worker.sleep/wake_up`, `CuMemAllocator.sleep(offload_tags=)` | `test_eg16_sleep` | M | Tag allocations, sleep to host, restore subset; gate VRAM after wake |
| EG-17 | Multi-adapter LoRA slots + grouped apply | one adapter merged at load time | Punica `bgmv`/`sgmv`, `LoRALRUCache` | `test_eg17_lora_slots` | L | depends EG-16 host paging; gate per-request adapter isolation |
| EG-18 | HiCache L1/L2/L3 with async prefetch/write-back + deadline policy | flat file-backed `src/wubu_lmcache.c`; `wubu_tier_advice()` is a policy function, not a cache engine | Mooncake HiCache design, SGLang blog | `test_eg18_hicache` | L | New module beside lmcache; gate hit/miss counts and transfer overlap |
| EG-19 | KV restore-vs-recompute scheduler | `src/wubu_kv_tier.c` moves KV but has no recompute decision | ALISA §3, KVPR, mzCache | `test_eg19_recompute` | L | Cost compare restore vs recompute; gate bytes moved vs tokens recomputed |
| EG-20 | Compute-centric MoE offload + expert activation traces | `src/wubu_moe.c`, `docs/ssd_moe.md` have no trace cache or latency model | MoE-Infinity iEAM/pEAM; Fiddler AVX512_BF16 | `test_eg20_expert` | L | Activation trace cache + layer-decay eviction; gate expert hit rate |
| EG-21 | Full Medusa tree verification | `src/wubu_medusa.c` is 39 lines: init + EMA adapt only, no draft-head forward or tree verify | Medusa self-draft heads | `test_eg21_medusa_tree` | L | Tree attention + verification path; gate acceptance length |
| EG-22 | Auto-configured pipelined offload at 6 GB | weight streaming exists; no per-iteration adaptive policy or proven 6 GB pipeline | PIPO (RTX 3060 6 GB, <40%→>90% util, 3.1×), TightLLM | `test_eg22_pipo` | L | depends EG-06; gate measured GPU util and throughput |

## Wave 3 — precision and decode acceleration

| ID | Technique | Local evidence | Reference | Gate | Size | Next action |
|---|---|---|---|---|---|---|
| EG-23 | Skip-Softmax / BLASST calibrated block pruning | flash attention exists; no per-length calibrated threshold | NVIDIA Skip Softmax blog, TensorRT sparse-attention docs | `test_eg23_skipsoftmax` | M | Calibration table + block skip; gate accuracy delta AND speedup |
| EG-24 | EAGLE-3 direct token prediction + tri-layer fusion | `src/wubu_eagle.c` implements EAGLE-1 truncated-layer drafting only | EAGLE-3 arXiv 2503.01840 | `test_eg24_eagle3` | L | Direct prediction head + dynamic tree; gate acceptance rate |
| EG-25 | Speculative family additions (Sequoia DP sizing, SpecTr optimal transport, DAG verify, token recycling, relaxed verification) | `src/wubu_more_spec.c` has REST/tree-restructure/contrastive/distil/spec-MoE/cascade/swap/early-exit only | speculative decoding survey 2502.19732v3 | `test_eg25_specfam` | L | One mechanism per gate, starting with Sequoia sizing |
| EG-26 | Per-token dynamic activation quantization | `act_quant` hits scattered; no named module or gate | W4A16 / FP8 literature | `test_eg26_dynact` | M | Scale per token, gate max absolute error vs reference |
| EG-27 | ISQ quantize-on-CPU before GPU load | quant routines exist; no safetensors quantize-on-load path | mistral.rs ISQ + topology | `test_eg27_isq` | M | depends EG-03; gate peak VRAM during load |
| EG-28 | Lazy per-layer quantized-KV conversion with budget guard | KV quantization + chunked prefill exist; no threshold conversion score | mlx-lm `maybe_quantize_kv_cache` | `test_eg28_kvquant` | M | depends EG-03; gate quality score vs memory saved |
| EG-29 | Content-addressed sha256 blob store + manifests | sha1/torrent/CID primitives exist; no blob store | Ollama manifest/blob layout | `test_eg29_blob` | M | Digest validation, dedup, resumable writes; gate round-trip + corrupt blob rejection |

## Wave 4 — tier-gated / deferred (do not start before T1 gates pass)

| ID | Technique | Why deferred | Gate | Size |
|---|---|---|---|---|
| EG-30 | FP8 attention | T3/T4 precision ladder; FP8 quant/dequant already present but not in attention | `test_eg30_fp8attn` | L |
| EG-31 | Pipeline parallelism | 0 hits for `pipeline_parallel\|stage_id\|pp_rank`; multi-GPU only, out of scope for a single 6 GB box | `test_eg31_pp` | XL |
| EG-32 | Capacity + micro-batch scheduler and persistent compact session | Architectural, medium fit; waits for EG-09/EG-10 | `test_eg32_capacity` | L |
| EG-33 | Disaggregated connector protocol | In-tree file-backed handoff exists; network protocol only if a multi-node use case appears | `test_eg33_connector` | L |
| EG-34 | Watermarking / typical-p logits processors | LOW fit for the target; no use case yet | `test_eg34_logits` | S |

## What not to break

- Engine stays ignorant of domain concepts. Grammar and tool schemas arrive as request
  metadata; do not teach GEMV or KV code an application schema.
- Keep the mmap zero-copy weight law. Do **not** add `MADV_SEQUENTIAL` — it evicts pages
  before the next token rereads them.
- Keep `wubu_ctx_manage.c:wubu_tier_advice()` and `wubu_spec_variants.c:wubu_offload_decision()`
  as per-step policy functions. HiCache-style threading belongs in a new module beside
  `wubu_lmcache.c`, not inside those functions.
- Do not regress the file-backed prefill/decode handoff in `wubu_lmcache.c`; it is the
  low-RAM compatibility path.
- Preserve single-process, no-third-party-dependency. No vendored Punica, XGrammar,
  sgl-kernel, TensorRT, llama.cpp, ggml, candle, MLX or Ollama source.
- Never mark a gap DONE on an agent self-report or a research note.

## Definition of done

A gap is DONE only when: the gate target exists in **both** Makefiles with direct source
dependencies, `mingw32-make -f Makefile.win <gate>` builds it, `./<gate>.exe` exits 0 with
its mechanism assertions, measured numbers are quoted in `docs/WORKLIST.md`, and the change
is committed. The next session starts at the lowest-numbered TODO in the earliest wave
whose dependencies are DONE.
