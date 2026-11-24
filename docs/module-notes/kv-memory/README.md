# kv-memory module notes

KV allocation, virtual/physical mapping, connectors, paging, offload, prefix reuse and placement.

## Named gates (22)

- `eg03_vram_plan`
- `eg06_plan`
- `eg07_row_reuse`
- `eg08_radix`
- `eg105_loop_place`
- `eg107_virt_kv`
- `eg108_kv_connector`
- `eg10_admit`
- `eg18_hicache`
- `eg32_capacity`
- `eg35_extkv`
- `eg37_arena`
- `eg40_invalidkv`
- `eg47_hostkv`
- `eg49_readkv`
- `eg53_rocketkv`
- `eg65_leanstream`
- `eg79_streamk`
- `eg83_pod_plan`
- `eg93_kv_rollback`
- `eg95_prefetch_feedback`
- `eg97_reclaim_margin`

## Reuse checklist

1. Read the owning header and source end to end before editing.
2. Confirm the gate links the owning source directly.
3. Extend the existing seam rather than creating a parallel module.
4. Add a REGISTER evidence paragraph with exact measurements.
5. Rebuild the full `test_eg*` suite and record the tally.
