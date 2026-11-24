# Inkling 2 Mini benchmark results — RTX 4050 (Vulkan)

Date: 2026-11-24. All runs use Unsloth llama.cpp build `d426a3774 (12651)`,
`Inkling-2-Mini-UD-IQ2_XXS.gguf` (26.07B, 7.89 GiB, 2.0625 bpw), explicit
`-dev Vulkan1` (RTX 4050 Laptop), `-ngl 99`, and `tools/bench_llama.sh`.
Raw JSON, manifests and CSVs are under `data/bench/`.

## Placement comparison — 5 repetitions, full profile set

Directory: `data/bench/20261124T010640Z-smoke/`.

| Profile | Metric | Baseline `-ncmoe 12` | Loop-aware `-ot` | Ratio |
|---|---|---:|---:|---:|
| chat | pp128 t/s | 127.13 | 195.31 | 1.54× |
| chat | tg32 t/s | 15.12 | 19.42 | 1.28× |
| coding | pp512 t/s | 44.22 | 349.82 | 7.91× |
| coding | tg64 t/s | 15.76 | 20.89 | 1.33× |
| agent | pp1024 t/s | 40.61 | 327.07 | 8.05× |
| agent | tg32 t/s | 15.44 | 20.65 | 1.34× |

The loop-aware placement keeps layers 6–17 on GPU and pins non-loop experts
(layers 0–5, 18–23) to CPU with
`-ot 'blk\.(0|1|2|3|4|5|18|19|20|21|22|23)\.ffn_.*_exps=CPU'`.
Large pp ratios at 512/1024 come from the baseline spilling looped expert
weights to CPU while the optimized placement keeps them resident.

## Variant sweep — coding profile, 3 repetitions

Directory: `data/bench/20261124T014927Z-variants/`.

| Variant | pp512 t/s | tg64 t/s | Note |
|---|---:|---:|---|
| loopopt | 359.41 ± 0.55 | 21.29 ± 0.65 | reference |
| loopopt_fa (`-fa 1`) | 357.86 ± 0.51 | 21.35 ± 0.59 | within noise |
| loopopt_q8kv (`-ctk/-ctv q8_0`) | 353.10 ± 0.12 | 21.18 ± 0.37 | 1.8% pp loss |
| loopopt_ub (`-b 2048 -ub 512`) | 353.99 ± 0.23 | 21.50 ± 0.15 | 1.5% pp loss, no decode win |

Decision: keep the plain loop-aware placement. The extra flags are not wins on
this host and must not be folded into the default.

## Thread sweep — coding profile, loopopt placement, 3 repetitions

| Threads | pp512 t/s | tg64 t/s |
|---:|---:|---:|
| 2 | 355.54 ± 1.30 | 17.26 ± 0.07 |
| 4 | 359.41 ± 0.55 | 21.29 ± 0.65 |
| 6 | 353.72 ± 0.50 | 22.75 ± 0.20 |
| 8 | 353.41 ± 0.93 | 22.79 ± 0.11 |

Decode improves about 7% from 4 to 6/8 threads while prefill is flat-to-slightly
worse. Online laptop evidence points the same way: llama.cpp decode often peaks
at 6–8 threads, not the default physical-core count. The next config to test in
a real server is split tuning (more threads for decode, fewer for prefill).

## Decision rule

`docs/BENCHMARK-WORKFLOW.md`: a change ships only if it wins outside combined
standard error on the same device/model/quant/flags and no correctness gate
regresses. The loop-aware placement meets that rule; the three flag variants do
not.
