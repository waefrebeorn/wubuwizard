# quant module notes

Formats, scales, bit packing, dequant correctness and format handoff.

## Named gates (10)

- `eg101_sq_agreement`
- `eg102_mx8`
- `eg28_kvquant`
- `eg30_fp8attn`
- `eg36_fp8kv`
- `eg45_varbit_vq`
- `eg48_bitnest`
- `eg66_kv8`
- `eg84_awq`
- `eg85_hadamard`

## Reuse checklist

1. Read the owning header and source end to end before editing.
2. Confirm the gate links the owning source directly.
3. Extend the existing seam rather than creating a parallel module.
4. Add a REGISTER evidence paragraph with exact measurements.
5. Rebuild the full `test_eg*` suite and record the tally.
