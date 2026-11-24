# speculative module notes

Draft generation, acceptance, tree/DAG verification, candidate policy and rollback.

## Named gates (28)

- `eg103_zero_logits`
- `eg104_markov_tree`
- `eg21_tree`
- `eg24_eagle3`
- `eg25_dag_verify`
- `eg25_recycle`
- `eg25_relaxed`
- `eg25_specfam`
- `eg25_spectr`
- `eg34_logits`
- `eg41_specsearch`
- `eg43_draftkv`
- `eg57_frspec`
- `eg58_specprefill`
- `eg61_blockverify`
- `eg63_specquant`
- `eg64_memspec`
- `eg69_replayssm`
- `eg71_ngram_pool`
- `eg72_ngram_map`
- `eg73_suffix_spec`
- `eg74_selector`
- `eg82_engram`
- `eg87_ev_depth`
- `eg90_dynamic_tree`
- `eg91_tree_accept`
- `eg92_two_level_verify`
- `eg94_accept_table`

## Reuse checklist

1. Read the owning header and source end to end before editing.
2. Confirm the gate links the owning source directly.
3. Extend the existing seam rather than creating a parallel module.
4. Add a REGISTER evidence paragraph with exact measurements.
5. Rebuild the full `test_eg*` suite and record the tally.
