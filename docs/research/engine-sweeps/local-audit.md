# Local audit — wubuwizard engine absorption (2026-11-22)

Repo: C:/Users/eman5/WuBu/repos/wubuwizard, HEAD 44aba50, main clean/pushed.
Build reality: 460 C/CU sources, 462 test sources, 95 unique named gates across
Makefile (78) and Makefile.win (36). Gates are NOT equivalent across platforms.

## Method
Grepped src/ and include/ for the *mechanism symbols* an absent feature would
have to contain, not for marketing words. Read thin modules (<150 lines) that
sound like full techniques to distinguish real implementations from bookkeeping.

## Confirmed present (do not re-report as gaps)
continuous batching (contbatch + continuous_batching + scheduler), prefix KV
reuse (hash-prefix cache), paged KV, chunked prefill, flash attention/prefill/
decode, CUDA graph, KV tier/offload, SSD MoE, expert allreduce, GGUF+safetensors
readers, 16 LP pads, 5 reference ports, Medusa/EAGLE/lookahead/n-gram speculative
helpers, FP8 quant/dequant, attention gating, mmap weights.

## Confirmed absences / thin implementations

| Mechanism | Evidence | Why it matters for 6GB/16GB/SSD |
|---|---|---|
| StreamingLLM attention sinks | 0 hits for `attention_sink|attn_sink|sink_token|streaming_llm` in src/include/tools | Long-context decode degrades sharply once the sliding window drops the first tokens; cheap sink tokens are a low-cost fix on T1 |
| RadixAttention tree | 0 hits for `radix|prefix_tree` in src/include (only unrelated audio/codec/kvfs hits); prefix cache is hash-based | No branch-sharing prefix tree with LRU and incremental match across sibling continuations; SGLang's signature mechanism |
| FP8 attention | 0 hits for `fp8_*attn|attn_*fp8|q8*k8`; FP8 quant/dequant exists but not in attention | Needed for T3/T4 precision ladder; lower priority on T1 |
| Pipeline parallelism | 0 hits for `pipeline_parallel|stage_id|pp_rank` | Multi-GPU only; explicitly out of scope for T1, record as tier-gated not a T1 gap |
| Full Medusa tree verification | `src/wubu_medusa.c` is 39 lines: init + EMA adapt only, no draft-head forward or tree attention verify | Present as bookkeeping, not a full self-speculation path |
| Per-token dynamic activation quant | `act_quant` hits are scattered; no named module/gate for per-token dynamic quant | Relevant to W4A16/FP8 paths on higher tiers |

## Note on "master-engine-compendium"
`research/master-engine-compendium/` is a TV/writing-room archive, not an
inference engine corpus. Do not mine it for engine work.

## Tooling gotcha
`make` is not on PATH; use `mingw32-make -f Makefile.win <gate>` from the repo.
