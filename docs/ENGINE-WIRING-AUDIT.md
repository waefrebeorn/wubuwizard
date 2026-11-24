# Engine wiring audit (2026-11-23)

Scope: does the 119-gate EG suite actually drive the inference engine, or does it only
drive test binaries? Answer recorded here with the measurement method, the numbers, and
the wiring order that replaces adding more gates.

Method (reproducible, symbol level, two independent passes):

1. `rg -o 'wubu_[A-Za-z0-9_]+' src tools` over every `.c` in `src/` and `tools/`, building
   a symbol -> referencing-files map.
2. For every `tools/test_eg*.c`, take the `wubu_*.h` it includes to get its gate module
   (`src/wubu_<module>.c`), extract the `wubu_*` function definitions from that `.c`, and
   count references from any file that is **not** the module itself and **not** a
   `test_*` translation unit.
3. Spot check by hand: `wubu_admission_`, `wubu_hadamard_`/`wubu_had`,
   `wubu_residency_`, `wubu_grammar_fsm_`, `wubu_ev_depth_`, `wubu_strata_` all resolve
   to their own `.c` only (`wubu_had*` also matches `wubu_rotate.c`, `wubu_enc_h3.c`).

## Numbers

| Measure | Value |
|---|---|
| EG gate tests (`tools/test_eg*.c`) | 119 |
| Distinct gate modules reachable from those tests | 123 |
| Gate modules called from engine source (non-test) | 6 |
| Gate modules with zero engine call sites | 117 |
| ...of which are linked into `CORE_OBJ` (dead weight in every binary) | 114 |
| ...of which are not linked at all | 3 (`kv_transfer`, `pd_split`, `residency`) |

The six modules with real call sites:

| Module | Called from |
|---|---|
| `wubu_fp8` | `wubu_fp8_attn.c`, `wubu_fp8_kv.c`, `wubu_enc_h3.c`, `wubu_nvfp4.c` |
| `wubu_lora` | `wubu_lora_slots.c`, `wubu_model_safetensors_bridge.c` |
| `wubu_unified_arena` | `wubu_arena.c`, `wubu_kernel_budget.c` |
| `wubu_pd_serve` | `wubu_early_exit.c`, `wubu_pd_split.c` |
| `wubu_blob_store` | `wubu_agent_state.c` |
| `wubu_gguf_tokenizer` | `wubu_tokenizer.c` |

## What this means

- The suite proves *contracts* (byte-exact codecs, planner arithmetic, fail-closed
  rejection). It does not prove the engine uses them. `PASS=119 FAIL=0` is true and
  means "123 contracts hold", not "119 techniques are deployed".
- The whole 6 GB fit story (`vram_plan`, `offload_plan`, `residency`, `ssd_expert`,
  `ssd_row_reuse`, `leanstream`, `memspec`, `hostkv`, `strata`, `ev_depth`) lives in
  test binaries. `src/wubu_model.c` (1672 lines) and `src/wubu_generate.c` (411 lines)
  contain no placement, no streaming and no residency decision.
- Offload exists only in the **safetensors** path:
  `src/wubu_model_safetensors_bridge.c:154` calls
  `wubu_ssd_moe_open_from_shards(sc, ssd_slots)` for paged experts. The **GGUF** path
  (`tools/gen_text.c` -> `wubu_model.c`, the IQ2/IQ-quant path this machine actually
  runs) has no hook to attach a residency plan to.
- KV virtualisation is contract-only too: `virt_kv`, `paged_kv`, `page_radix`,
  `radix_cache`, `kv_connector` have no engine callers; `wubu_model.c` manages KV
  inline.
- 114 unused gate objects are linked into `CORE_OBJ`, so every engine binary carries
  them. `blockverify`, `kv8`, `fp8_shift`, `residency` etc. are payload-free today.

## Wiring order (replace the gate treadmill)

Acceptance criteria are measurable, on this host, with the GGUF model the machine
already runs; each item must show a number, not a `PASS` line.

| # | Wiring | Hook | Acceptance |
|---|---|---|---|
| 1 | `residency` + `offload_plan` + `ssd_expert` into the GGUF loader: decide per-layer resident vs streamed at load, report the plan | `src/wubu_model.c` load path; `tools/gen_text.c` flag (`--residency-budget-mb`) | loads a model whose resident set previously OOM'd; peak RSS via `GetProcessMemoryInfo` <= budget; token agreement vs resident run on 20 fixed prompts |
| 2 | `virt_kv` + `paged_kv` into the model KV allocator: one virtual range, pages on demand | `src/wubu_model.c` KV alloc | same logits on a 4K-context prompt; `--kv-pages` report shows mapped < reserved |
| 3 | `sampler_chain` + `grammar_fsm` into `gen_text` sampling (replace inline top-p/temp) | `src/wubu_generate.c` sample step; `tools/gen_text.c` | identical output for plain sampling on fixed seed; JSON grammar output 100% parseable over 50 generations |
| 4 | `admission` + `scheduler` + `capacity_sched` into `api_server` request path | `tools/api_server.c` | N concurrent requests admitted/queued per token budget; no over-admit under 8x load |
| 5 | spec-decode family (`eagle3`, `medusa`-side `dynamic_tree`, `tree_verify`, `accept_table`, `keyed_sample`) into `wubu_generate.c` | `src/wubu_spec_decode.c` -> generate loop | measured tokens/s vs no-spec on the same prompt set; accept-rate reported |

Rule until item 1 lands: no new `EG-*` gate. A gate whose module has zero engine call
sites is documentation, not capability.

## Related

- `docs/ENGINE-GAP-REGISTER.md` — per-gap contracts and evidence (all DONE rows stay valid).
- `docs/ENGINE-SOURCE-INDEX.md` — upstream pointers for each mechanism.
- `docs/ENGINE-AGGREGATION.md` — ownership map.
