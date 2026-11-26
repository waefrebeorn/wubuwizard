# What other inference engines do for MoE on a small card - and what we got wrong

Research note, 2026-11-26. Sources: llama.cpp PRs #29887 (Oct 7), #30112 (Oct 8, multi-GPU), #30637 (LRU -> SLRU),
#30594 (hybrid + profile-prefill), ik_llama.cpp #2644 (port) and **the ik_llama.cpp implementation that is already on
this disk** at `research/engine-sources/ik_llama.cpp/src/llama-moe-cache{,-policy,-share}.{cpp,h}` (1425 lines total).

## The short version

The exact feature this project spent a session designing - a GPU LRU cache for host-resident MoE experts - **shipped
upstream a month ago**, and it is faster than what we were building. Two independent measurements agree:

    llama.cpp, PR #30112, single card:            our measurement, this engine:
      master --fit          34.78 t/s              LRU @ 32 experts/layer = 79.0 % hit
      --moe-cache-mib 6544  60.38 t/s  (1.74x, 85.9 % hit)   LRU @ 48 experts/layer = 85.7 % hit
      -cmoe + 15000 MiB     64.39 t/s  (1.85x, 95.8 % hit)

Our 85.7 % and their 85.9 % at a comparable cache fraction is not a coincidence - it is the same routing skew being
measured on two different models. That part of the analysis was right.

## Why ours crashed and theirs does not

Reading their implementation, our device path is missing four structural pieces, and the absence of the third is
exactly the segfault:

1. **An id -> slot map per layer.** `plan(il, ids, n_ids, ...)` builds `gpu_map` and `cpu_map` (expert id -> slot, -1 for
   absent) for every layer, every step. Our engine called a *dense* matvec entry with a byte offset into a shared tensor
   and hoped the slab happened to hold the right expert. There is no such mapping anywhere in our path.
2. **Upload only misses, never a whole slab.** Their `fills` list is exactly the experts that are not resident, capped
   per layer per step (`--moe-expert-cache-inserts` in the original PR). We uploaded whole expert tensors on first touch.
3. **A guard row.** `bank_geometry` reserves `guard_slots = 1` beside the real slots: a requested expert that is not
   resident indexes a valid row instead of running off the end of the bank. That is the crash insurance we lacked, and a
   runaway index in a `MUL_MAT_ID`-shaped gather is exactly a SIGSEGV.
4. **A policy that tolerates scans.** Master started with LRU and moved to **SLRU/CLASSLRU** (#30637). The ik port keeps
   both: `kind::lru` and `kind::classlru` with `n_class = 8`, `half_life = 128` tokens. A plain LRU loses its hot set to
   a burst of one-off experts; our own trace showed the same effect from the other direction - a static hot set learned
   on the first half hit only 63.2 % later.

## The other thing they do that we did not consider

`--moe-cache-hybrid` (PR #30594) does not upload misses at all: cache hits are computed on the GPU, and **misses are
computed on the CPU in parallel**, with a "zero slot" arrangement (`share_q16`, `zero_slots` up to 64 per layer) so the
GPU contributes zeros for the missing experts and the two results are summed. That removes the PCIe transfer from the
miss path entirely, which is precisely the cost our all-streaming experiment measured at 14 % - and it explains why that
experiment lost: streaming was the only option we had implemented.

They also ship `--moe-cache-profile <file>`: pre-populate the cache from an **offline routing profile**. Our static-set
experiment (63.2 % later hit rate) argued *against* a static set being sufficient; upstream uses it only as a seed and
keeps the LRU on top, which is consistent with our finding rather than contradicting it.

## Constraints they encode that we should copy verbatim

    batch        <= 32 tokens (max_batch) - the cache is a decode-time feature; prefill bypasses it
    cache size   ~10 % of total expert bytes is "good" (their guidance, 32 GB card)
    misses       uploaded with a per-layer per-step insert cap, or computed on CPU (hybrid)
    accounting   n_hit and n_eviction are counted and reported, not inferred

## What that means for this engine on 6 GB

Our experts are 7.25 GB and the card is 6 GB, so 10 % of expert size (725 MB) buys only ~55-63 % hit rate here - their
guidance assumes a cache 30 % of the model. At 2.72 GB (37 % of the expert bytes) we measured 85.7 %. Two consequences:

  - **A uniform partial cache is the wrong shape for this card.** Caching 37 % of every layer means every layer pays
    upload churn on 14 % of its routes. The small-card recommendation upstream is layer-wise: keep some layers' experts
    fully resident and the rest fully on CPU (`--n-cpu-moe N`), which has no churn at all.
  - **Either way the GPU work must be a gather, not a dense matvec.** MUL_MAT_ID shape: rows selected by expert id,
    indices indirections, guard row for misses. That is a kernel and an entry point, not a flag.

## Ordered work, revised by this reading

1. **Expert gather entry point on the device** (`wubu_cuda_expert_matvec(ids, n_ids, slot_map, guard)`), with the guard
   row from the start. This is the piece whose absence caused the crash; nothing else can be measured until it exists.
2. **Layer-wise residency first** (`IN2_GPU_MOE_LAYERS=N`): whole expert layers resident, no eviction, no churn -
   trivially correct, and it gives a clean measurement of "how much does having N/24 layers' experts on the device buy".
3. **LRU/CLASSLRU cache only if step 2 shows the remaining layers are worth it**, with hit/eviction counters reported
   live (never inferred), and the ≤32-token restriction honoured.
4. Hybrid CPU-share for misses is the last step, and only if the upload path proves to be the bottleneck.

Steps 1-2 are hours, not days, and they answer the question the last three attempts kept failing to reach: what does the
device actually buy for MoE when the code is shaped like the problem.

## Confirmation of the guard-row diagnosis, from the crash itself

The same configuration built `-g -O1` instead of `-O3` does **not** fault:

    -O3 : Segmentation fault, no tokens
    -O1 : exits normally, IDS 371 151451 149879 149876 220 34361 149933

Same input, same flags otherwise, one build level apart: a memory error that faults in an optimised build and silently
returns wrong numbers in an unoptimised one. That is the signature of reading outside the intended expert region - the
read lands in mapped memory at `-O1` and off the end of it at `-O3`. It is not a numerical-precision difference, which is
what the token-5 divergence was initially filed as, and it is the failure mode the upstream `guard_slots = 1` reserve
exists to make impossible: a requested expert that is not resident must index a valid zero row, never a computed offset.
