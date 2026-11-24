# moe module notes

Expert routing, hot/cold caches, staging, load accounting and expert-side kernels.

## Named gates (8)

- `eg05_neuron`
- `eg20_expert`
- `eg36_expert_cache`
- `eg45_expert_stage`
- `eg50_ssdexpert`
- `eg52_seqmoe`
- `eg78_expert_sub`
- `eg96_early_expert`

## Reuse checklist

1. Read the owning header and source end to end before editing.
2. Confirm the gate links the owning source directly.
3. Extend the existing seam rather than creating a parallel module.
4. Add a REGISTER evidence paragraph with exact measurements.
5. Rebuild the full `test_eg*` suite and record the tally.
