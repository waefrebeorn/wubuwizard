# Local audit round 2 — landing zones for EG-35..EG-46

## Landing modules (line counts)
wubu_arena.c, wubu_mem_budget.c, wubu_kv_exact.c, wubu_kv_tier.c,
wubu_lmcache.c, wubu_agentic_mem.c, wubu_agentic_kv.c, wubu_prefix_cache.c,
wubu_paged_kv.c, wubu_swa_reclaim.c, wubu_kv_cacheline.c, wubu_kv_transfer.c
were checked for size and API; see terminal output in this session for exact counts.

## Interpretation
- EG-37 (unified arena) should extend `wubu_arena.c` / `wubu_mem_budget.c`, not create
  a second allocator. Check whether the arena supports compaction/relocation before
  claiming the gap is small.
- EG-35 (extensible KV) belongs at the KV sizing boundary in `wubu_mem_budget.c`;
  the UMA correction is a measurement rule, not a new allocator.
- EG-39/EG-40 (agent state, invalidated KV) land in `wubu_agentic_mem.c`,
  `wubu_agentic_kv.c` and `wubu_prefix_cache.c` release/invalidate paths.
- Verify a named gate exists before treating any of these modules as shipped.
