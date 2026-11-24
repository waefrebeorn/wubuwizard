# diagnostics module notes

Probes, identity checks, smoke gates and tests that diagnose without claiming a served model.

## Named gates (45)

- `eg01_vecdot`
- `eg02_tok_meta`
- `eg04_attention_sink`
- `eg09_overlap`
- `eg100_layer_probe`
- `eg11_grammar`
- `eg12_sampler`
- `eg13_pretok`
- `eg14_template`
- `eg15_pd`
- `eg16_sleep`
- `eg17_lora_slots`
- `eg19_recompute`
- `eg23_skipsoftmax`
- `eg26_dynact`
- `eg27_isq`
- `eg29_blob`
- `eg38_pages`
- `eg39_surgery`
- `eg42_tv`
- `eg44_gqa_index`
- `eg44_indexer`
- `eg45_lowrank`
- `eg51_spice`
- `eg54_proxy`
- `eg55_uno`
- `eg56_dflash`
- `eg59_online`
- `eg60_layerskip`
- `eg62_dspark`
- `eg66_adaptive_cap`
- `eg66_aot_pack`
- `eg66_tensorfold`
- `eg67_titan`
- `eg68_strata`
- `eg70_cert`
- `eg75_sparse_retrieve`
- `eg76_index_reuse`
- `eg77_vc_attn`
- `eg80_fly`
- `eg81_dfly`
- `eg88_adapt_len`
- `eg89_keyed_sample`
- `eg98_batch_h2d`
- `eg99_conv_blocks`

## Reuse checklist

1. Read the owning header and source end to end before editing.
2. Confirm the gate links the owning source directly.
3. Extend the existing seam rather than creating a parallel module.
4. Add a REGISTER evidence paragraph with exact measurements.
5. Rebuild the full `test_eg*` suite and record the tally.
