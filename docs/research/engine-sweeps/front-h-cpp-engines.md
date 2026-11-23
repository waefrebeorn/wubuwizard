# Front H — C/C++ local-inference engines: source-mined absorption candidates

Scope: clones under `C:/Users/eman5/WuBu/research/engine-sources/`.
Read first: `C:/Users/eman5/WuBu/repos/wubuwizard/docs/ENGINE-GAP-REGISTER.md` (EG-01..EG-70)
and `docs/ENGINE-SOURCE-INDEX.md`. Method: search/read actual source, tests and benchmark
artifacts; marketing claims were used only as pointers to code and to author-reported numbers,
never as evidence of mechanism. Standing target T1 = **6 GB VRAM / 16 GB RAM / SSD,
model- and backend-agnostic, single process, no third-party code linked**. Every mechanism
below is a *read-and-reimplement-in-C11* candidate; nothing here is vendored.

All numbers are **authors' claims from the checked-in source/docs**, not local measurements.
A reported speedup is a reason to write a gate, not evidence that wubuwizard has the feature.

## Result summary

| # | Repo | Mechanism (symbol) | T1 fit | Register status |
|---|---|---|---|---|
| H1 | h3.c | Two-slot DiT SSD block streaming with cyclic first-block prefetch (`read_stream_layer_thread`) | MED-HIGH | genuinely new |
| H2 | h3.c | Multi-worker layer ring prefetch for streamed text encoder (`layer_prefetch_main`, `text_prefetch_depth`) | MEDIUM | genuinely new |
| H3 | basalt | Content-addressed conversation/KV state files, per-1 MiB hashes, turn checkpoints (`ConversationStore`) | HIGH | new; adjacent EG-29/EG-39/EG-66 |
| H4 | basalt | Programmatic dependent launch with pre-`pdl_wait` weight staging (`PdlScope`, `pdl_launch_dependents`) | LOW (SM90+) | genuinely new, tier-gated |
| H5 | Strata | Batched expert H2D uploads (`cudaMemcpyBatchAsync`, `copy_expert_blobs`) | HIGH | genuinely new |
| H6 | BigMoeOnEdge | Real-layer microbenchmark probe for device-tier pricing (`layer_probe.cpp`, `probe_layer`) | HIGH | new; refines EG-06 by measurement |
| H7 | BigMoeOnEdge | Persisted measured knob verdicts with interleaved A/B and noise guards (`KnobTrialConfig`) | HIGH | genuinely new |
| H8 | BigMoeOnEdge | Memory margin derived from kernel overflow class (`margin_frac`, `reclaim_is_cheap`) | HIGH | new; distinct from EG-03/EG-35 |
| H9 | MACaQ | MXFP4 W4A4 Hessian-diagonal pre-scale + greedy flip refinement (`flip_refine_levels`, `balance_pre_scale_from_diag`) | HIGH | genuinely new |
| H10 | MACaQ | In-place refinement of existing GGUF block scales (`llamacpp/macaq.py refine`, MACaQ-R) | HIGH | new; distinct from EG-27 |
| H11 | MemVanta | Feedback-controlled prefetch depth with bounded probes and rejected-transition memory (`AdaptivePrefetchController::observe`) | HIGH | genuinely new |
| H12 | frink | Address-identity residency proof for small f32 planes (`resident_f32_shared`) + bulk f16 KV row push | HIGH | genuinely new |
| H13 | turbo-fieldfare | Cross-layer early expert read scored by next layer's router, adopted on miss (`NextLayerExpertPrefetcher`) | HIGH | new; adjacent EG-51, lossless |
| H14 | splash | CRC-32C checked persistent KV/state slot files with adopt/label (`SlotFile`, `Crc32c`) | MED-HIGH | new; adjacent EG-66/EG-47 |
| H15 | splash | ANE FFN split: 2560-channel segments and 1024-channel output blocks (`AneFfn.cpp`) | LOW (Apple ANE) | genuinely new, hardware-gated |
| H16 | lucebox | Stream-side wait on a host atomic flag (`HostStreamFlag::enqueue_wait`) | MEDIUM | genuinely new, small |

`gpt4all` (last commit 2025-05-27, so outside the 2026 Q3/Q4 window) contributed **no** new
mechanism: `gpt4all-backend/src/llamamodel.cpp` is a wrapper over llama.cpp (`use_mmap`,
fp16 KV size estimate, `llama_kv_cache_seq_rm`) already covered by EG-01/EG-02/EG-03 and the
source index rows for llama.cpp. It is excluded rather than padded out.

---

## H1 — h3.c: double-slot DiT SSD streaming with cross-step cyclic prefetch

* Repo/path: `h3.c/h3_dit.c` (`dit->ssd_streaming`, `stream_slots[2]`,
  `read_stream_layer_thread`, `h3_dit_stream_job`), generation loop near lines 2210-2295;
  `h3.c/README.md` "Weight residency and streamed prompt encoding" (lines 533-556),
  last commit `2026-08-11` "Clarify SSD streaming memory and speed tradeoff".
* What it does: only per-block norm weights stay resident. Two full BF16 matrix slots
  alternate. On block *b* the current slot is used for compute while a background
  `pthread` reads the next active block into the other slot, explicitly wrapping with
  `if (future == H3_DIT_BLOCKS) future = first_active_block(dit)` — so the first block is
  re-fetched during the final block and is resident for the *next denoiser evaluation*.
  The GPU command buffer is submitted before `pthread_join`, hiding the read behind GPU
  work; `H3_PROFILE=1` reports total bytes, read throughput, and the *unhidden* part of the
  read wait. Darwin uncached reads avoid a second filesystem-cache copy. This is a
  two-slot double buffer whose reuse unit is the denoising step, not the token.
* Numbers claimed: "about 13--14.6 GiB/s from the internal SSD" (README:552-554).
* T1 fit: MED-HIGH — same shape as EG-07 (overlapped SSD reads) but the reuse axis is a
  diffusion/denoising pass, which the register does not cover. The C11 port is a generic
  double-slot block reader keyed by an external "next block" oracle; no model-specific code.
* Gap: **genuinely new**. Do not re-report EG-07's sliding-window row reuse.

## H2 — h3.c: worker-lane layer ring prefetch for the text encoder

* Repo/path: `h3.c/h3_text_encoder.c`: `text_prefetch_slot`, `text_layer_prefetch`,
  `layer_prefetch_main` (line 303), `prefetch_slot_start/take/retire`,
  `text_prefetch_threads` (124), `text_prefetch_depth` (135), caller at 648-659;
  README lines 540-545.
* What it does: preallocates a ring of future-layer weight buffers (`text_prefetch_slot
  slots[6]`) and fills them on up to eight I/O worker threads while Metal runs the current
  layer; threads split each layer by *lane* (`layer_weights_read_lane(..., lane, lanes)`).
  Depth defaults 2 on M3 / 3 on M5; `H3_QWEN_PREFETCH_DEPTH=1..6`, `H3_QWEN_PREFETCH=0`
  restores the synchronous path. Adjacent mechanism in the same file: on M5 weights are
  mapped zero-copy from safetensors shards (`H3_ZERO_COPY_WEIGHTS`) instead of copied into
  anonymous shared buffers, keeping a 37 GiB file-backed/reclaimable.
* Numbers claimed: default 2/3 layers, 8 workers, ring capped at 6; 37 GiB model kept
  file-backed (README:536). README:465 reports avoiding two slice blits and 18.8 MiB of
  scratch at 512x512 (29.4 MiB total) for a separate fused-tile change — do not conflate.
* T1 fit: MEDIUM — the lane-split ring is useful for any CPU weight reader; depth 2-3 on
  16 GB RAM is realistic.
* Gap: **genuinely new** as a *feedback-free bounded ring with lane-parallel reads*.

## H3 — basalt: content-addressed conversation/KV state store

* Repo/path: `basalt/src/core/conversation_store.cpp` (header comment lines 1-13),
  constants at 44-64, `conversation_name` (146), `hash_srcs` (219),
  `write_container` (237), `ConversationStore::write_job` (1262),
  `keep_stored_checkpoints` (1050), `conversation_store_hash_file` (1451).
* What it does: three file kinds in one aligned container (4096 B alignment): a
  conversation file (live tokens, chat state, K/V shapes, checkpoint keys), K/V **block
  files** (`kBlockCells = 4096` K/V cells each) and state files. Every 1 MiB block of every
  section has a hash; block and state files are keyed by *the token prefix they belong to
  and by their bytes*, so rewriting a conversation a few turns later — or a different
  conversation with the same prefix and bytes — writes only new files. Unreferenced files
  are GC'd after `kOrphanAge = 60 s`. Checkpoints are kept only for the root (end of system
  prompt) and `kTurnStates = 3` turn ends; a live end within `kLiveSlack = 4096` tokens of
  a checkpoint is stored as of it. Writes leave `kSpaceFloor = 4 GiB` free; reads use
  `kReaders = 8` in flight, `kPiece = 4 MiB`, writes `kStage = 8 MiB`.
* Numbers claimed: structural constants above; no throughput figure checked in.
* T1 fit: HIGH — this is the "resume a chat without re-prefill, on SSD" mechanism, and it
  is deduplicating across sessions, which EG-08's in-RAM radix cache is not.
* Gap: **new**. Adjacent to EG-29 (sha256 weight blobs), EG-39 (persistent agent state),
  EG-66 (disk prefix snapshots) — but none of those specifies prefix+byte keying with
  1 MiB block hashes and bounded turn checkpoints. Gate suggestion: two conversations with
  a shared prefix must share block files, and a resumed conversation must produce
  byte-identical logits with zero re-prefill.

## H4 — basalt: programmatic dependent launch (PDL) with staged weight reads

* Repo/path: `basalt/include/basalt/kernels/pdl.hpp` (full header comment),
  `PdlScope`, `pdl_launch_wanted`, `pdl_kernel_ok`, `pdl_wait`,
  `pdl_launch_dependents`; used in `basalt/src/models/qwen4exp/kernels/qsa_select.cu`,
  `hc_fused.cu`, `fused_gr.cu`.
* What it does: a dependent kernel is launched with CUDA's programmatic-dependency
  attribute so it can *start* before its predecessor finishes. Every kernel calls
  `pdl_wait()` (`griddepcontrol.wait`) before reading a predecessor's writes or writing
  anything; only constant weights may be read before the wait. The decode chain stages
  its first weight reads before `pdl_wait`, so each stage's weight reads overlap the
  previous stage's tail. `pdl_launch_dependents()` (`griddepcontrol.launch_dependents`)
  lets a whole grid release its dependents once all blocks have called it. Per-device,
  per-function capability cache refuses SM < 9.0 / PTX < 90; the whole thing degrades to
  ordinary `<<<>>>` launches before CUDA 12.3, and `pdl_wait()` is a no-op on older cards.
* Numbers claimed: none quoted in source; the design is explicitly "only the launch overlaps".
* T1 fit: LOW on a 6 GB consumer card (SM90+ required) — **tier-gated T3/T4**, but the
  *pattern* (pre-stage constant reads before the dependency fence, fall back cleanly) is
  portable to CPU/GPU command-buffer submission.
* Gap: **genuinely new**; no register row covers launch-dependency attributes.

## H5 — Strata: batched expert H2D uploads

* Repo/path: `Strata/engine/src/core/verify.cpp:4396` (`Verifier::fetch_dma`),
  `Strata/engine/src/program/generate.cpp:15329,24652`,
  `Strata/docs/BATCHED_DMA.md`.
* What it does: `STRATA_DMA_BATCH=1/2` groups a verifier round's independent pinned-host
  expert uploads into one `cudaMemcpyBatchAsync` (mode 2 adds CUDA 13.4's
  `cudaMemcpyFlagPreferOverlapWithCompute`); mode 0 keeps the per-expert loop. Group is
  capped at 128 and sources must be immutable pinned blobs with disjoint destinations.
  Windows registered host pointers are resolved to device-visible aliases before
  submission. A submission failure **aborts** rather than raising flag B, so consumers
  never see stale weights. Copy and kernel stay on their existing streams.
* Numbers claimed (RTX 4070 Laptop 8 GB, Ryzen 9 7940HS, 16 x 2 MiB uploads):
  host submission 0.0936 -> 0.0162 ms (**-82.7%**), total transfer 2.5589 -> 2.5248 ms
  (-1.3%); second run 449,824 -> 67,939 cycles (-84.9%). Full-model screen did **not**
  establish a t/s gain (30.3 with individual DMA vs 30.0 batched), so default stays 0.
  63-case parity matrix (0/1/4/17/64/128/129 uploads x 37 B/4 KiB/2 MiB) passed on CUDA
  13.4, 13.0, and 12.6 fallback.
* T1 fit: HIGH — host submission overhead is exactly the tax a 6 GB/SSD box pays on
  per-expert staged reads; the parity matrix is a ready gate shape.
* Gap: **genuinely new**. EG-50 covers concurrent/async *reads*, not batched H2D
  submission. Note the honest negative t/s result: gate submission latency, not tok/s.

## H6 — BigMoeOnEdge: measured real-layer probe drives device tier selection

* Repo/path: `BigMoeOnEdge/core/src/plan/layer_probe.cpp` (design comment, lines 1-36),
  `core/src/plan/planner.cpp` (`choose_device_tier` 814, `device_tier_device` 649,
  `fitter_decode_ceiling` 681/703, `device_cache_decode_seconds` 713,
  `host_expert_flash_seconds` 739, `prompt_gather_seconds` 766,
  `price_layers_home` 1158, `device_tier_split` 494);
  `docs/measured-planning.md`.
* What it does: builds **one real layer from the file's tensors** with ggml's public API
  and times two halves separately: the expert block (`mul_mat_id` over the layer's own
  expert shapes at k-per-token) and the dense matrices (attention projections, router,
  shared experts as plain `mul_mat`). Two widths: decode width holds enough experts to
  outrun LLC, prefill width routes just enough tokens to reproduce prompt arithmetic
  intensity. A device whose outputs disagree with the host is refused (bounds
  `k_max_nmse_device = 1e-3` at 1 token, `1e-2` at 128). Missing kernels are left out and
  priced at the host rate for that half. The resulting rates feed a cost model combining
  device-cache decode ceiling, host-expert flash time and prompt gather time.
* Numbers claimed: on one 48-core host the synthetic GEMV read 77 GiB/s while the real
  decode experts ran at ~12 (layer_probe.cpp:2-9); cache/decode examples:
  15.7 GB cache held 83% of routed experts over 1024 tokens at 11.6 tok/s, 22 GB held 96%
  at 20.7 (27.5 over the last 256); first-load cache 75-90% device hits; probe cost
  "15-139 ms on a desktop core" per archived plan (measured-planning.md:120-131, 389).
  A per-layer cache policy hit its simulated hit rate but was 30% slower — hence speed is
  measured, never inferred from a hit curve.
* T1 fit: HIGH — hardware-agnostic by construction ("this layer took 3.1 ms on device 1"),
  works with one device or a host-only split.
* Gap: **new**. EG-06 is a placement LP solver over synthetic weights; this is a
  *correctness-checked measurement on the model's own layer shapes* that prices devices.

## H7 — BigMoeOnEdge: persisted, measured knob verdicts

* Repo/path: `BigMoeOnEdge/core/src/plan/planner.cpp` (`arm_thread_trial` 4124,
  overlap trial 4011-4021, `KnobTrialConfig::answer_device`, `answer_prompt_per_token`
  2249-2272, `width_across_loads` 1445, `narrowed_width` 1389),
  `docs/measured-planning.md` (steps 2/5), `docs/benchmarks.md`.
* What it does: for policy knobs that cannot be settled by a rule (`--overlap`, thread
  count, device tier, narrow compute width), the first run runs an interleaved A/B trial,
  keeps a verdict keyed by model/hardware/config, and later loads start from that verdict
  instead of re-deciding. Guardrails: equal arms under 10% noise never win over 50 seeds;
  25% always wins, 5% is a tie, 12% wins; a stalled window does not overturn the median;
  a candidate 20x slower stops after one window. Trials are symmetric across defaults and
  are refused in states where the switch cannot be exercised (e.g. overlap needs every
  layer on the host). Every run prints `predicted 0.048 s/token, measured 0.055` and the
  credited vs measured hit rate.
* Numbers claimed: 10%/5%/12%/25% noise thresholds, 50 seeds, "answer_prompt_per_token"
  cap; example 74.9/76.8/32.4% reproduced hit rates (measured-planning.md:104, 192-195).
* T1 fit: HIGH — turns wubuwizard's static policy functions into *measured, persistent*
  decisions without threading state into `wubu_tier_advice()` (keep that a pure function;
  persist verdicts beside it, per the register's "what not to break").
* Gap: **genuinely new**; EG-22 is a one-off auto-configuration, not persisted A/B verdicts.

## H8 — BigMoeOnEdge: memory margin from overflow class

* Repo/path: `BigMoeOnEdge/core/src/plan/planner.cpp:41-96` (`budget_bytes`, `margin_frac`,
  `margin_bytes`, `reclaim_is_cheap`), `core/include/bmoe/hardware_profile.h`
  (`enum class Overflow { Compress, Swap, Kill, None, Refault, Unknown }`).
* What it does: the fraction of the residency budget left free is chosen by *what losing
  memory costs on this kernel*, not by a fixed percentage: compress -> `margin_when_compressed`,
  swap -> `margin_when_swapped`, kill -> `margin_when_killed`, unknown -> `margin_when_unknown`,
  floored by `margin_min_bytes`. `reclaim_is_cheap()` is deliberately narrow — only
  `Compress` and `Refault` — because those are the cases where the kernel routinely takes
  pages back and a model with no slack is reclaimed continuously. `Unknown` keeps the
  default: an unprofiled machine does not inherit a measured conclusion.
* Numbers claimed: the failed runtime cache governor moved gpt-oss from 0.517 tok/s
  (cache off) to 0.156 tok/s governed — "a budget the device cannot concede starts a
  reclaim war the loop itself feeds" (measured-planning.md:59-62).
* T1 fit: HIGH — directly addresses the 16 GB RAM half of T1, where a naive 80% rule can
  put the process in swap/compress churn.
* Gap: **new**. EG-03/EG-35 use an 80%-of-free / UMA-corrected rule; this classifies the
  *reclaim semantics* and refuses to shrink against the kernel.

## H9 — MACaQ: MXFP4 W4A4 with Hessian-aware flip refinement

* Repo/path (Python research source, port the algorithm to C11): `MACaQ/quantize/macaq_mxfp4.py`
  (`OCPDenseW4A4Linear.forward`, `mxfp4_rtn_signed_index` 106,
  `screen_block_exponents` 141, `pack_mxfp4_from_signed_index` 172,
  `_flip_sweep_fast` 193, `_flip_sweep_dense` 228, `flip_refine_levels` 255,
  `mxfp4_realquant_h` 359); `MACaQ/quantize/macaq_balance.py:4`
  (`balance_pre_scale_from_diag(W, sum_x2, n_tokens, alpha=0.75)`);
  `MACaQ/quantize/hybrid_text.py:55` (calibration 16,384 tokens, chunk 512, flips 512).
* What it does: MXFP4 (e8m0 block scales, packed nibbles) is refined against calibration
  activations. Per input column it computes `pre_scale = (w_std^alpha / act_rms^(1-alpha))`
  clamped to [1e-4, 1e4], screens candidate block exponents (offsets +1/0/-1), then greedily
  flips weight indices up/down one level at a time using the second-moment matrix H and the
  current error (`XTyfp - XTyq`), with separate fast (active-set) and dense sweeps. The
  result runs as W4A4, not weight-only.
* Numbers claimed (README:139-146, 15-16): Qwen3.8-27B BF16 PPL 2.8463 -> MACaQ W4A4
  2.8994, prefill **2.33x**, decode **2.58x**; W4A4 matches published Llama-3-8B PPL with
  "11x less memory", using "just 32.5 GB of device memory".
* T1 fit: HIGH for the *offline quantizer* (a 6 GB card benefits more from 4-bit weights),
  but note W4A4 runtime needs a matching C11 kernel; start with the weight-only path.
* Gap: **genuinely new**. EG-27 is quantize-on-load, EG-26 per-token activation scale,
  EG-63 multi-precision variants — none is Hessian-block-exponent + flip refinement.

## H10 — MACaQ-R: refine an already-quantized GGUF in place

* Repo/path: `MACaQ/README.md:90-116`, `MACaQ/llamacpp/macaq.py refine`
  (`--model model-q4.gguf --hf original --teacher original-f16.gguf`), plus
  `MACaQ/quantize/source_provenance.py`.
* What it does: takes a quantized MoE GGUF plus its original source and an F16 teacher,
  and refines the codes **while preserving native GGUF formats and block scales**; the
  output runs under a standard llama.cpp installation with no custom kernel. This is a
  post-quantization optimisation over an existing artifact, not a new quant format.
* Numbers claimed: Qwen3.6-35B-A3B, KL divergence to FP16 at identical 22.36 GB file size:
  Unsloth UD-Q4_K_XL 0.015731 -> MACaQ-R Q4_K_XL **0.013367** (README:99-104).
* T1 fit: HIGH — improves quality at zero VRAM/size cost for an existing GGUF; ideal for a
  wubuwizard offline tool that rewrites block scales.
* Gap: **new** vs EG-27 (quantize before load) — this refines after quantization with
  the file layout unchanged.

## H11 — MemVanta: adaptive prefetch depth with bounded probes and memory

* Repo/path: `MemVanta/src/prefetch_policy.cpp` (whole `AdaptivePrefetchController::observe`),
  `include/memvanta/prefetch_policy.hpp` (`min_depth=1`, `max_depth=4`,
  `low_useful_ratio=0.60`, `high_useful_ratio=0.90`), `src/prefetcher.cpp` (`Prefetcher::loop`,
  `pending_bytes/peak_pending_bytes`, no background `MADV_DONTNEED`),
  `tests/prefetch_policy_tests.cpp`, `benchmarks/issue12_real_prefetch_ab.cpp`.
* What it does: a closed-loop read-ahead controller fed by per-window telemetry
  (`consumed`, `useful`, `late`, `evictions`, `budget_skips`, `peak_inflight_bytes`).
  Depth drops on >5% latency regression, budget skips, low usefulness (<0.60) or eviction
  pressure that is not explained by high usefulness. Depth rises only when at least one
  request arrived late, two consecutive windows are stable and highly useful (>=0.90), and
  >=25% of the inflight byte budget is free. The increase is a **bounded probe**: the
  deeper depth is kept if latency stays within 0.5% of the reference, otherwise the exact
  transition is recorded as rejected with a 2-window cooldown, re-opened only when the
  lower-depth baseline moves >5% (phase change). Sequential-stream evictions alone are
  not treated as pressure.
* Numbers claimed: thresholds above (0.5%, 5%, 1.01/1.05, 25%, 2 windows,
  `min/max depth 1/4`). Kernel A/B numbers in `BENCHMARKS.md` (e.g. Q4_0 4096x4096/5-thread
  8.228 -> 29.576 GFLOP/s, 3.59x) are separate — do not attribute them to prefetch.
* T1 fit: HIGH — the controller is pure policy over measured windows, directly portable to
  wubuwizard's SSD weight/expert read path and consistent with the "no MADV_SEQUENTIAL"
  law (`prefetcher.cpp` explicitly avoids WILLNEED/DONTNEED churn from the worker).
* Gap: **genuinely new**. EG-07 fixes a reuse window; EG-19 is restore-vs-recompute;
  neither adapts read-ahead depth from usefulness/late/byte-pressure telemetry.

## H12 — frink: address-identity residency proof for small f32 planes

* Repo/path: `frink/crates/frink-metal/src/attn.rs` (`resident_f32_shared`, many call
  sites at 2973-3001, 3387-3388, 3673-3674), `decode_dense.rs:499-514`,
  plus `KvCache::push_rows` / `push_rows_from_metal` / `MetalKvBuffers::tokens_f16`
  from commit `2026-11-14` "#643".
* What it does: previously every lookup of a borrowed f32 plane (norms, RoPE divisors,
  sinks, conv taps, router bias) proved residency by comparing every byte of the slice —
  read twice per lookup. Now planes are shared `Arc<[f32]>` and residency is proven by
  **address identity** (`resident_f32_shared`); the byte-comparing entry point is removed
  so callers cannot accidentally use it. Entries held only by the cache are dropped on the
  next miss (`Resident::orphaned`) so model swaps do not retain old planes. Separately,
  `push_rows` appends n KV rows in one pool-accounted operation, and the Metal->host copy
  writes f16 into f16 bit-for-bit instead of widening then rounding back.
* Numbers claimed (commit message): idle encode 0.6 ms, 2.9 ms contended against the
  early-commit GPU traffic; 35 MB/token on Mamba-1.4B and 94 MB on FalconMamba-7B;
  per-decode plane bytes 1.1 MB (Llama-3.1-8B) to 10.6 MB (Bonsai-2-27B), up to 53 MB
  per prefill batch (Qwen3-30B-A3B router). M5 Max: Mamba-1.4B tg128 108/206 -> **324**
  tok/s (llama.cpp 221); FalconMamba-7B 56 -> **105** (llama.cpp 83); LFM2 pp512
  15301 -> 15813. Steady-state decode held to zero uploads across 71 fixtures.
* T1 fit: HIGH — a CPU-side launch-encode optimization independent of GPU vendor; the C11
  analogue is "prove an upload is current by pointer identity / owner token, not memcmp".
* Gap: **genuinely new**; no register row addresses constant-tensor upload validation.

## H13 — turbo-fieldfare: cross-layer early expert read, lossless

* Repo/path: `turbo-fieldfare/Sources/TurboFieldfare/Runtime/Inference/NextLayerExpertPrefetcher.swift`
  (header 1-40, `EarlyExpertReadStats`, `readsPerLayer`, `beginForward`, `adopt`),
  `RealForwardRunner.swift:1637-1697,1830-1842`,
  `Infrastructure/Streaming/PreadExpertStreamer.swift:56,199-247`
  (`ExpertStagingBuffer`, `adoptStagedExpert`), `Runtime/Configuration/RuntimeConfiguration.swift:22-24`.
* What it does: after layer L's routed-expert reads finish, storage is idle while the GPU
  finishes L and starts L+1. It scores L+1's experts with **L+1's own router applied to
  L's router input** in the same command buffer, and preads the highest-scoring uncached
  ones into staging buffers *outside* the expert slot pool. If L+1's real plan misses one
  of them, `adoptStagedExpert` swaps the staging buffer into the planned slot and the
  normal read is skipped; if it hits, the read is simply wasted. The cache plan, hit rate
  and generated text are unchanged by construction.
* Numbers claimed: one **3.36 MB** staging buffer per read; 1-8 reads/layer
  (`TURBO_FIELDFARE_EARLY_EXPERT_READS`, default 1); off by default in the library config,
  on by default via the app; CLI prints `[early-read ...]` stats (issued, used, hit-rate).
* T1 fit: HIGH — a single-expert, output-neutral lookahead for SSD-resident MoE on 6 GB.
* Gap: **new**, adjacent to EG-51 (SPICE) but lossless (no low-rank surrogate) and keyed
  on next-layer router scoring rather than confidence thresholds.

## H14 — splash: CRC-32C persistent KV/state slot files

* Repo/path: `splash/runtime/model/SlotFile.cpp:397,426,446,467,1121`
  (`crc32c` on header, per-record checksum, verification on read, running CRC over pieces),
  `splash/runtime/model/Crc32c.hpp` (Castagnoli CRC-32C, "which a persistent slot file keeps
  for each slot's bytes"), `splash/runtime/engine/KvPageTier.hpp:27-36` (page extents to
  slot file, APFS-aligned uncached reads), `splash/runtime/engine/KvTier.hpp:51-58`
  (quota, writeability, copies outliving the process, `label` for writes started before a
  checkpoint), `splash/runtime/Checked.hpp:62-66` (16 KiB VM-page alignment).
* What it does: a KV/state backing store whose slots carry CRC-32C records, a quota, a
  write-in-flight label, and `adopt` semantics — a page whose file copy is still being
  written can be adopted by a later request instead of rewritten. Copies survive process
  restart. This is the on-disk half of splash's checkpoint story; the generic tiering itself
  is not new here.
* Numbers claimed: structural constants; README changelog reports 1.7.1 "a state at a
  shared prompt head, faster weight loads and starts, int8 MoE prefill experts".
* T1 fit: MED-HIGH — the CRC+label+adopt protocol is portable; file alignment/ANE bits are not.
* Gap: **new** as a protocol. EG-66 is a prefix snapshot with an exactness gate; EG-47 is
  an INT8 host record. Neither specifies CRC-checked, quota-limited, adopt-able persistent
  slots. Gate: corrupt a slot and require refusal; adopt an in-flight slot and require
  byte-identical KV.

## H15 — splash: ANE FFN segmented matmul with output blocking

* Repo/path: `splash/runtime/ops/AneFfn.cpp` (header comment): `kSegment = 2560`,
  `kOutputBlock = 1024`, `gateUpInBlocks(appleGpuFamily) { return family >= 11; }`,
  `kHandoffBound = 2 s`, `kProgramLimit = 210 s`.
* What it does: the Apple Neural Engine program splits a 5120-channel FFN into two 2560
  channel segments (each matmul runs its own scales/SiLU/product before the next), and on
  Apple11+ further blocks gate/up output into 1024-channel chunks. Weights are staged in
  whole rotation blocks and inputs in whole packing tiles. Family gating exists because
  the same split is a win on M6 but a loss on dual M5 Pro; the ANE/GPU handoff is bounded
  so a timed-out event cannot poison the Metal queue.
* Numbers claimed: two 2560 segments run a 5120-channel FFN ~**14% faster** than one on
  M5 Pro; 1024-channel blocking cost 7.6% less at 26/34 units (26.4 vs 28.6 ms) and 2.5%
  less at 10 on M6, but 2.5-10% *more* on M5 Pro and 30-45% longer compiles everywhere.
* T1 fit: LOW — Apple ANE only; record as a hardware-gated pattern ("split a big FFN into
  cache/rotation-aligned segments, gate by family, bound the accelerator handoff").
* Gap: **genuinely new**, hardware-gated.

## H16 — lucebox: stream-side wait on a host atomic flag

* Repo/path: `lucebox/server/src/common/host_stream_flag.cpp` (`HostStreamFlag::init`,
  `enqueue_wait`, `signal`, `copy_to_device`), header `host_stream_flag.h`.
* What it does: allocates 64 B of host-pinned memory, publishes its device pointer, and
  enqueues `hipStreamWaitValue32(stream, dev, value, hipStreamWaitValueEq, 0xffffffff)`;
  the host sets the value with a release atomic. A device stream therefore waits on a host
  decision without a host-side `hipStreamSynchronize` round trip. Cleanly disabled on
  non-HIP builds.
* Numbers claimed: none checked in.
* T1 fit: MEDIUM — small, but the pattern ("producer signals, stream waits, host never
  blocks") is useful for staged weight handoff on any runtime with stream-wait-value.
* Gap: **genuinely new**; no register row covers host-flag stream synchronization.
  The rest of lucebox's distinct code (`common/dflash_*`, `dspark_head`,
  `gpu_page_pool.cpp`, `cluster/`) maps to EG-56/EG-62/EG-35 and deferred cluster rows.

---

## Deliberately NOT re-reported (already in the register)

* basalt `native_mmvq.cu`, `iq_kernels.cu` (MMVQ / IQ2 / IQ4 codebook grids, DP4A,
  `vec_dot_*_q8_1`) are transcribed from llama.cpp and are EG-01's dispatch/quant family —
  do not port as-is (licence note in those files).
* basalt `src/models/qwen4exp/kernels/qsa_select.cu` radix-select top-k (IDX_HEADS=4,
  R=4, `block_scores_kernel`, `block_topk_kernel`, dead-block `+1e9` sentinel) is the
  QSA/DSA indexer family already represented by EG-44 and the source-index watch row.
* basalt `core/expert_tier.cpp` (`kMaxPending=128`, ranked/held slots, peer moves,
  `fetch/promote/recall/shrink`), BigMoeOnEdge `moe/expert_stream_source.cpp` and
  `router_hook.cpp`, Strata `core/expert_source.cpp`, splash `ExpertStore.cpp`:
  expert tiering/streaming = EG-20/EG-50/EG-52/EG-67.
* Strata `zero-allocation host`, prefill launch-count amortization and 4-way split-KV
  decode = EG-68 (do not re-report even though the clone is large).
* lucebox `common/dflash_*`/`pflash/qwen35_drafter.cpp` = EG-56; `dspark_head.cpp` = EG-62;
  `gpu_page_pool.cpp` (UMA page correction) = EG-35; `cluster/` = deferred T2/T3.
* splash `KvPageTier`/`KvCache` generic tiering = EG-18/EG-47; `verify_budget` =
  EG-61/EG-62; `ragged_scheduler` = EG-09/EG-32.
* MACaQ `quantize/moe_*` grouping and multi-precision variants = EG-63/EG-20 lineage.
* Everything in `gpt4all` = llama.cpp rows of the source index.

## Recommended new gate names (ranked by T1 value)

1. `test_h11_adaptive_prefetch` — MemVanta H11: deterministic window feed, assert
   depth-up only on late+stable+headroom, rejected transition not retried, depth-down on
   >5% regression; byte cap enforced. Highest fit/portability.
2. `test_h13_early_expert` — turbo-fieldfare H13: score L+1 from L's activation, adopt on
   miss, output KV/logits byte-identical with early reads on/off; count wasted reads.
3. `test_h08_reclaim_margin` — BigMoeOnEdge H8: for each Overflow class assert the chosen
   margin and that no loop shrinks against active reclaim.
4. `test_h05_batch_h2d` — Strata H5: 0/1/4/17/64/128/129 groups, fail-closed on submit
   error, measure submission latency (not tok/s).
5. `test_h03_conversation_blocks` — basalt H3: shared-prefix dedup, root+3 turn
   checkpoints, 60 s orphan rule, byte-identical resume with zero re-prefill.
6. `test_h06_layer_probe` — BigMoeOnEdge H6: probe a tiny synthetic "real layer", assert
   device refusal on a wrong-output backend, assert priced rate feeds tier choice.

Lower priority / tier-gated: H09+H10 (quantizer, needs an accuracy gate), H07 (needs a
stable model/hardware fingerprint), H12 (backend-specific), H04/H15 (SM90 / Apple ANE),
H16 (HIP-only), H01/H02 (diffusion/text-encoder paths).

### Caveats

* These are cloned references for study. The basalt and Strata CUDA files carry llama.cpp
  MIT provenance headers; reimplement the *mechanism* in C11, never copy the code or link
  ggml/llama.cpp.
* Several repos (MACaQ, frink, turbo-fieldfare) are Python/Rust/Swift, not C/C++. They are
  included because the task named them and the mechanisms are language-independent;
  portability is noted per row.
* No gate was written or run in this sweep — this is a reading report, not evidence of
  absorption.
