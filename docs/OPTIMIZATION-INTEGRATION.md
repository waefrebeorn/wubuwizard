# Optimization integration & tuning — 2026-11-20

## What is integrated into a real model run

`tools/gen_text_gemma4.c` now uses three of the shipped pads in the actual
Gemma 4 generation path. All three are **env-gated**: with no new variables set,
the tool behaves exactly as before and keeps llama.cpp parity.

| Pad | Env knob | Effect |
|---|---|---|
| LP-01 persistent goals | `WUBU_GOAL_STATE=<path>` | loads a `.goal` sidecar, appends the continuation instruction to the prompt tokens, judges the result at the end |
| LP-01 evidence judge | `WUBU_GOAL_EVIDENCE=<text>` | completion is accepted only when the receipt contains `[PASS]` and the verifier callback agrees; sidecar is saved with the new status |
| LP-14 TTC budget | `WUBU_TTC_MAX_MS`, `WUBU_TTC_MAX_KV_BYTES`, `WUBU_TTC_MAX_CANDIDATES`, `WUBU_TTC_PLATEAU_WINDOW`, `WUBU_TTC_MIN_GAIN` | charges measured decode wall time + approximate KV bytes per candidate and stops early |
| LP-16 versioned state | `WUBU_STATE_OUT=<path>` | writes a versioned, CRC-checked run-state file (prompt tokens, generated tokens, first token, prefill/decode seconds) |

The remaining pads are linked into `CORE_OBJ` in both Makefiles and have their
own gates; they are library surfaces for the engine and host layers, not
silent changes to decode semantics.

## Measured tuning (Gemma 4 12B IQ2_M, 21-token prompt, 1 generated token)

| OMP threads | Prefill | Decode (1 token) |
|---|---|---|
| 4 | 46.19 s | 15.22 s |
| 8 | 41.43 s | 15.22 s |
| **12** | **38.24 s** | **15.11 s** |
| 20 | 43.56 s | 15.20 s |

12 threads is the prefill sweet spot on the i7-14700HX (8P+12E): ~17% faster
than 4 threads. Decode is memory-bandwidth-bound and essentially flat, so
thread count is not a decode lever. Use `OMP_NUM_THREADS=12` for batch-1 runs.

## LM-head cache (shipped, opt-in)

`G4_CACHE_LM_HEAD=1` dequantizes the 3840×262144 tied embedding once (3840 MiB F32)
and replaces the per-token re-dequant with a dense matvec. Measured on the same
prompt with 12 threads:

| path | Prefill | Decode |
|---|---|---|
| streaming (default) | 38.31 s | 15.12 s/token |
| LM-head cached | 23.91–29.31 s | ~15.0 s/token |

First token stays `100`, second `45518` — llama.cpp parity is preserved.
Enable it for repeated runs where the one-time 4 GB cost is amortized.

## Rejected experiment: full F32 layer-weight cache

Caching all layer weights in F32 needs ~43 GB and immediately pages on Windows:
prefill 135 s, decode 271 s/token (vs 38 s / 15 s streaming). It was removed;
do not re-add without a machine that can keep the whole working set resident.

## GGUF weight streaming (shipped, opt-in)

`G4_STREAM_WEIGHTS=1` skips the 4 GB blob buffer entirely. Each weight is read
from the open GGUF file on demand (`gguf_read_raw_tensor`), dequantized to a
temporary F32 buffer, used for that layer's matmul, and freed. Token embedding
rows are read individually from the same file handle. Measured on Gemma 4 12B:

| path | Resident blob | Prefill | Decode |
|---|---|---|---|
| default (mmap/blob) | 4003 MB | 38.24 s | 15.11 s |
| streamed weights | 0 MB | 25.85 s | 17.67 s |

Top5 and first token are byte-identical to llama.cpp (`100, 101, 9079, 20535,
236829`). Combine with `G4_CACHE_LM_HEAD=1` to avoid re-streaming the tied LM
head every token. Two upstream bugs were fixed to make this work: 32-bit
`fseek` on tensor offsets past 2 GB in `gguf_reader.c`, and a missing
`token_embd.weight` tensor base offset (16384) in the embedding row reader.

## Tiered weight residency (SSD → RAM → compute)

`G4_STREAM_WEIGHTS=1 G4_WEIGHT_CACHE_MB=5120` adds a bounded LRU RAM tier that
caches **quantized bytes** (not dequantized F32). This keeps the tier ~4×
smaller than an F32 cache and holds the whole 4 GB model with room to spare.
Measured with Gemma 4 12B on this machine:

| path | Prefill | Decode | cache |
|---|---|---|---|
| no blob, no RAM cache | 25.85 s | 17.67 s/tok | 0 |
| no blob, 5120 MiB raw cache | 27.88 s | **11.90 s/tok** | 656 hits / 329 misses |
| no blob, 5120 MiB + 32-slot prefetch | 25.76 s | **11.48 s/tok** | **874 hits / 111 misses** |
| same + native quantized matmul | **23.13 s** | **9.99 s/tok** | 874 hits / 111 misses |
| full mmap/blob | 38.24 s | 15.11 s/tok | n/a |

An F32 cache was tested first and rejected: it needs ~16 GiB, thrashes even at
24 GiB (0 hits), and slows decode to ~70 s/token.

`G4_PREFETCH=1` (default when streaming + cache cap > 0) starts a worker with a
second `FILE*` and a 32-slot queue. Since layer weights are static, the first
32 layers are requested at model load; the forward loop then keeps the queue
32 layers ahead. The worker never touches LRU state — the main thread adopts
ready buffers at the layer boundary (`g4_prefetch_request` /
`g4_prefetch_adopt` in `src/wubu_gemma4_model.c`), and stale slots are released
before reuse. The worker is joined in `g4_model_destroy()` before the GGUF
handle closes. On the cold 21-token prefill this raised hits from 656 to 874 and
cut misses from 329 to 111; a 16-slot queue gave 765/220, and a one-layer
lookahead gave no reliable gain, so the load-time queue is what ships.

`g4_batched_qmatmul()` now passes cached raw bytes directly to
`quantized_matmul_batched()` for natively supported types (IQ2_S/IQ3_S,
IQ3_XXS, IQ4_XS, K-quants, Q8_0), with an F32 fallback for the rest. This
removes the 4× F32 scratch allocation and separate dequant pass. It also made
the streamed logits exactly match the unstreamed reference:
`top5 (100,27.3) (101,11.88) (20535,11.52) (9079,11.3) (236829,10.16)`,
first token `100`. Decode fell from 11.48 to 9.99 s/token and prefill from
25.76 to 23.13 s.

## Token streaming in the API server (shipped)

`POST /v1/chat/completions` (or `/v1/completions`) with `"stream":true` now
returns `text/event-stream` with OpenAI-style `chat.completion.chunk` events,
a terminal `finish_reason:"stop"` chunk, and `data: [DONE]`. The inference
subprocess is unbuffered and each generated line is emitted immediately;
sandbox mode emits word-sized chunks. Windows portability fixes: Winsock
startup, `recv/send` instead of `read/write`, no-fork accept loop, optional
OpenSSL via `__has_include`. Gate: `make test_sse` → `test_sse: PASS (8 stream events)`.

Real end-to-end test (`SSE_REAL=1 ./test_sse.exe --sse-selftest`) streams actual
Gemma 4 tokens with a `<<<STREAM_START>>>` marker gating out loader logs:
`<|channel>`, `thought`, `The`, ` user`, ` said`, ... followed by `[DONE]`.
Windows `popen` uses `cmd.exe`, so the CLI path is converted to 8.3 short form
and engine settings are inherited via `_putenv` instead of `VAR=value` prefixes.

## Architecture registry (not one hard-coded model)

Borrowing WuBuOS's manifest/capability pattern, `src/wubu_model_arch.c` keeps a
registry of GGUF architecture names (`llama`, `gemma4`, `qwen3moe`,
`deepseek2`, `lfm2`, ...) mapped to a kernel family, tensor layout, and
capabilities. `wubu_arch_probe()` reads the architecture-prefixed GGUF keys
(`gemma4.block_count`, `gemma4.rope.freq_base`, ...) and produces a concrete
`wubu_arch_spec_t`; unknown architectures get a generic transformer/SSM/MoE
description instead of being forced through a Qwen/Gemma path.

`wubu_model_init()` now uses this spec for layer count, tensor naming,
`d_model`, `d_ff`, head counts, head/rotary dims, and prints the detected
architecture. Tensor-name and shape probing remains a fallback for models
with sparse metadata. `model_arch_probe <file.gguf>` prints the registry's
view; `test_arch` generates tiny metadata-only GGUF fixtures for llama,
qwen3moe, and an unknown future architecture and checks dispatch.
`infer_any <file.gguf>` uses the spec to select the runner: Gemma 4 goes to
the dedicated G4 engine; MoE and hybrid-SSM models are explicitly refused
instead of being mis-executed by the dense generic path; unknown architectures
are also refused. Only registered transformer architectures use the generic
engine. Runner is part of `wubu_arch_spec_t` and is printed by
`model_arch_probe`.

The generic loader now binds `ffn_gate/up/down.weight` into `wubu_dense_ffn`
and calls `wubu_dense_ffn_forward()` in the FFN slot, so dense transformer
blocks no longer silently pass their FFN through unchanged. A synthetic F32
llama fixture reaches a full forward with finite logits; its teardown still
has a heap issue in the generic path, so that fixture is a diagnostic tool,
not a passing gate.

`wubu_model_init`'s failure path also had a double-cleanup bug: it closed and
nulled `gguf_ctx` before `wubu_model_free()`, which owns that context. It now
calls `wubu_model_free()` directly; a metadata-only fixture exits cleanly with
status 1 instead of corrupting the heap.

The default (non-streamed) G4 path had a heap-corruption crash at teardown:
the model took ownership of an mmap'd GGUF blob but freed it with `free()`.
The mmap base/length are now transferred to `g4_model_t` and released with
`wubu_munmap()`; verified `PS_EXIT=0` for both default and streamed runs.

Verified spec for Gemma 4 12B: `layers=48 d_model=3840 d_ff=15360 heads=16
kv_heads=8 head_dim=512 rotary=512 vocab=262144 rope_base=1000000.0 swa=1024
metadata=complete`.

## What the Oct–Nov 2026 research actually says (and what applies here)

Fan-out reviewed Edge0/edge0-strata, Tideway, StrataFlux, SGLang Paged Experts,
CECAR, Bevy, TensorRT-LLM weight streaming, vLLM layerwise offload, FlexGen,
PowerInfer and DeepNVMe. Sources: [Edge0](https://github.com/Edge0-AI/Edge0),
[Tideway](https://arxiv.org/html/2611.14848v1),
[StrataFlux](https://arxiv.org/abs/2611.08363),
[SGLang Paged Experts](https://docs.sglang.io/docs/advanced_features/paged_experts),
[CECAR](https://arxiv.org/html/2611.20119), [Bevy](https://arxiv.org/html/2611.04794v1),
[TRT-LLM weight streaming](https://nvidia.github.io/TensorRT-LLM/advanced/weight-streaming.html),
[vLLM layerwise offload](https://vllm-project.github.io/2026/08/17/distributed-layerwise-offload.html),
[FlexGen](https://arxiv.org/html/2303.06865v1),
[DeepNVMe](https://pytorch.org/blog/deepnvme-affordable-i-o-scaling-for-deep-learning-applications).

Key conclusions for wubuwizard:

- Expert-level residency is the winning granularity for MoE (Edge0 prerouter
  +59% decode; Tideway 1.97× e2e; CECAR 12.5 tok/s from SSD; SGLang K-slot
  pinned-RAM/disk tier). **Our Gemma 4 12B is dense**, so this does not apply
  directly — do not build an expert cache for a dense model.
- For a dense 12B on 96 GB RAM, the whole model belongs in RAM; SSD should be
  an overflow/KV tier, not the decode hot path. The 5120 MiB raw LRU above is
  exactly that RAM tier and already beats full mmap on decode.
- Overlap is mandatory: Edge0 predicts next-layer routing one token ahead,
  Tideway uses phase-aware layouts and streams layer *l+1* while *l* computes,
  DeepNVMe/FlexGen use pinned staging on a separate copy stream. Our current
  CPU path still reads layer *i* synchronously — next concrete step is a
  prefetch worker with a duplicated `FILE*` and a one-layer lookahead.
- Windows traps already hit: 32-bit `fseek` past 2 GB (fixed in
  `gguf_reader.c` and `gpu_gemma4_forward.cu`), pinned `cudaMallocHost` above
  ~4 GB fails (split buffers), and mmap random page faults are reactive —
  explicit `_fseeki64`/`fread` into a bounded slot pool is more predictable.
- GPU tier remains unimplemented: `g4_layer_vram_t` has no LRU/eviction and
  `d_weight_pool` uploads every layer once. Fixing the stream-mode file
  offsets was a correctness prerequisite; VRAM LRU + async `cudaMemcpyAsync`
  from pinned RAM is the next tier to build.

## Verification of the integrated binary

- Parity preserved: first token `100 <|channel>`, second token `45518 thought`,
  the same sequence as llama.cpp.
- TTC test with `WUBU_TTC_MAX_MS=1000` stopped after one candidate and wrote a
  version-2 run-state file.
- Goal sidecar test appended the continuation instruction, judged the receipt,
  and saved the updated goal state.

## Build notes

- `mingw32-make -f Makefile.win gen_text_gemma4` compiles all objects cleanly;
  the final link in this environment hits a pre-existing CodeIntegrity `e=5` on
  the CUDA library path. Linking the same object set without `-lcudart -lcublas`
  succeeds (`LINK_EXIT=0`), so this is not caused by the new modules.
- Gate targets in both Makefiles: `make test_<gate>` for all 16 pads.
