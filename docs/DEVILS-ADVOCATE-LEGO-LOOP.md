# Devil's Advocate — lego gates + Inkling looped-transformer speedup

Date: 2026-11-23. Scope: wubuwizard engine work and the local Inkling 2 Mini
looped-transformer benchmark. Method: mind-palace 4-phase loop (Claim → Verify →
Risk → Mitigate), one practical pass per claim. No ceremony, no 1000-step loop.

## Claim 1 — "All work is a set of lego pieces"

Claim: every shipped technique is a named, runnable C11 gate linked to owning
sources; the whole engine is a stack of independent bricks, not a monolith.
Source: `Makefile.win` target list (114 `test_eg*` targets), `docs/ENGINE-GAP-REGISTER.md`,
`DIAGRAMS/wubuwizard-lego-pieces.html`.
Trust: high.

Verify:
- Parsed `Makefile.win`: 114 distinct `test_eg*` targets; HTML map generated from
  that list, not hand-authored.
- Register has 105 EG rows; five explicitly gated off (EG-22/31/33/46/86), one
  partial row (EG-44 trained half), all others carry named gates and evidence.
- Full suite on 2026-11-23 before EG-105: `PASS=113 FAIL=0`; EG-105 added and
  stress-run 30/30 below.
Risk:
- Confirmation bias: counting targets is not proof each target tests the named
  mechanism. Mitigation: per-gate evidence paragraphs state exact measurements;
  new gates use RED first and link owning sources directly.
- Staleness: a map built from Makefile targets goes stale when targets are added.
  Mitigation: regenerate `DIAGRAMS/wubuwizard-lego-pieces.html` from the Makefile
  at each wave and record the date in the footer.
- Scope creep: "all work" excludes the WuBuOS platform and the host agent layer.
  Mitigation: this map covers wubuwizard only; platform layers are in `.hermes/GOAL.md`.

## Claim 2 — "We actually benchmarked and sped up looped transformers"

Claim: on this host, placing only non-loop experts on CPU while keeping the
Inkling 2 Mini loop core (layers 6–17, 3 passes/token) on GPU gives 1.56× prefill
and 1.35× decode over `--n-cpu-moe 12`.
Source: `llama-bench` runs logged in `~/AppData/Local/hermes/cache/scratch/inkling_base_r3.log`
and `inkling_opt_r3.log`; register EG-105 evidence.
Trust: high for this exact setup; low for generalization.

Verify:
- Hardware: RTX 4050 Laptop GPU, 6 GB; Vulkan device `Vulkan1`; Unsloth
  llama.cpp build `d426a3774 (12651)`.
- Model: `Inkling-2-Mini-UD-IQ2_XXS.gguf`, `inkling2_loop`, 26.07B params,
  7.89 GiB, 2.0625 bpw.
- Command shape: `-p 128 -n 32 -r 3 -ngl 99 -dev Vulkan1 -t 4`; baseline
  `-ncmoe 12`; optimized `-ncmoe 0 -ot 'blk\.(0|1|2|3|4|5|18|19|20|21|22|23)\.ffn_.*_exps=CPU'`.
- Baseline: pp128 `126.19 ± 2.08`, tg32 `15.44 ± 0.35` tok/s.
- Optimized: pp128 `196.27 ± 2.07`, tg32 `20.88 ± 0.86` tok/s.
- Independent online corroboration: Thinking Machines describes the 3× middle
  loop; ComputingForGeeks and the Unsloth/loop-optimisation reports name the
  looped core and per-expert scale path as the bottleneck. Ratios here are
  locally measured, not copied.
Risk:
- Measurement error: was the GPU actually used? `Vulkan1` is explicitly the
  NVIDIA device in the log; `-ngl 99` offloads all layers; `-t 4` identical.
- Survivorship: only one prompt length/quant/backend. No claim for CUDA, other
  quants, long context, or Inkling 2 Small.
- Attribution: Inkling 2 Small is 594B and requires a ~500 GB VRAM cluster;
  the local looped target is **Mini**, not Small. The user-facing wording must
  not claim Small was benchmarked.
- Gate/measurement gap: EG-105 counts synthetic MiB of loop re-reads; it does
  **not** assert tok/s. The speedup is reported from llama-bench, not from C11
  gate timing.
Mitigate:
- Keep raw logs; re-run with a different prompt/quant before claiming a range.
- Label the C11 gate as a placement-policy gate; label tok/s as measured host
  evidence for one configuration.
- Revert threshold: if a future rerun on the same command drops below 1.2× for
  both pp and tg, mark the placement claim stale and re-investigate backend changes.

## Claim 3 — "The C11 engine absorbed the loop placement technique"

Claim: `wubu_loop_place.c/.h` encodes the tested placement rule as a gate.
Source: `tools/test_eg105_loop_place.c`, `include/wubu_loop_place.h`,
`src/wubu_loop_place.c`, Makefile targets.
Trust: medium-high.

Verify:
- RED phase observed: missing `wubu_loop_place.h` compile failure before source.
- GREEN: `test_eg105_loop_place: PASS`, exit 0; 30/30 stress runs clean.
- Gate checks loop layers 6–17 pinned GPU, non-loop experts CPU, naive first-12
  CPU placement re-reads 6 loop layers 3× (1536 vs 768 synthetic MiB), and
  refuses a loop core larger than the GPU hot budget.
Risk:
- A synthetic 64 MiB/layer byte count is a proxy. It explains the direction of
  the speedup but is not a VRAM measurement or a kernel timing.
- The gate does not parse GGUF metadata or drive the Vulkan backend; it is a
  policy primitive that future engine code can call.
Mitigate:
- Keep the measured llama-bench result attached to the register row.
- Add a later GGUF-metadata gate (phase P1/P2 in `.hermes/GOAL.md`) before
  claiming the engine itself executes Inkling 2 Mini end-to-end.

## Verdict

- Lego-map claim: ✅ verified against gate targets and full-suite run.
- Loop speedup claim: ✅ verified for Inkling 2 Mini, RTX 4050 Vulkan, IQ2_XXS,
  short prompt; ⚠️ not valid for Inkling 2 Small or untested hardware.
- Engine absorption claim: 🟡 partial — the placement policy is gated; full
  native Inkling decode remains a later phase.
