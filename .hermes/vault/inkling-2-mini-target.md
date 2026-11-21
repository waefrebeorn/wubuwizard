# wubuwizard Goal Mantra — Inkling 2 Mini (Nov 21, 2026)

> Reorients the project away from the bytropix CPU-parity goal (archived in
> `.hermes/vault/bytropix-goal-archive.md`) toward a concrete model target.
> Source of truth for the new goal: [Thinking Machines — Introducing Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/).

## THE GOAL

Make wubuwizard a local, open-weights inference engine for **looped MoE transformers**,
with **Inkling 2 Mini** as the first target:

- load the released GGUF (Unsloth / llama.cpp quantized build),
- run the **3× middle-layer loop** natively in the C11 engine,
- keep CoT legible while reasoning happens inside the loop (monitorability),
- run on local hardware (RTX 4050 6GB here; target class = consumer laptop / DGX Spark),
- verify outputs against llama.cpp / Tinker reference implementations.

Exit gate: an Inkling 2 Mini GGUF decodes on wubuwizard with correct loop depth, matched
logits against a reference, and a documented tokens/s + memory footprint.

## WHY THIS TARGET

The release is the goal because it matches what the engine is for: strong agentic coding
in a small local model. Reported numbers (vendor-reported, not independently verified):

| Property | Value |
|----------|-------|
| Architecture | Looped transformer, MoE, middle layers run 3× per token |
| Parameters | 26.4B total / 3.4B active / 5.4B weight activations per token |
| Precision | Native NVFP4 training+inference; last 4 layers MXFP8 |
| Footprint | ~20 GB weights — fits local devices |
| Coding | DeepSWE v1.1 48.2%, SWE-bench Pro v2 Hard 77.9%, Terminal Bench v2.1 71.6% |
| Search/tools | BrowseComp 79.0%, DeepSearchQA 85.3%, MCP Atlas 82.7% |
| Looping | Serial-reasoning gains continue to 10 loops (2× trained depth); no-CoT horizon 16s → 37s at 3 passes; CoT shorter but legible; no rise in sabotage tests |
| Efficiency | Intelligence/watt 0.703 on DGX Spark (vs Qwen3.6-35B-A3B 0.430) |

Key constraint: looping adds reasoning, not facts. Knowledge-intensive benchmarks do not
improve with more passes, so the model relies on search/tool use — which means the engine
must keep tool-call and long-context paths correct.

## GAP MAP — present work vs target

Already shipped in this tree (see `docs/WORKLIST.md`, 16/16 gated):

- LP-01..04 — persistent goal state, evidence-first judge, drift steering, resampling, harness bandit (host layer).
- LP-05..09 — paged KV claim, SWA reclamation, speculative decoding, seeded determinism, constrained decode.
- LP-10..16 — kernel family specialization, MX/NVFP4 KV reconstruction, quant-format KV fetch, diffusion sampler, measured-cost TTC, SIMD quant decode, state/ABI semver.

Missing for Inkling 2 Mini (new work):

1. **Loop executor** — repeat the middle block N times per token; expose loop count as a runtime parameter (research variant goes to 10).
2. **NVFP4 / MXFP8 decode** — most weights in NVFP4, final 4 layers MXFP8; current code handles IQ2/Q4 and MX/NVFP4 KV, not full NVFP4 weight path.
3. **MoE routing at 3.4B active** — top-k expert dispatch with the loop multiplier accounted for in activation count.
4. **Long context** — GraphWalks BFS and MRCR v2 improve with passes at 256K–512K; verify KV spine holds.
5. **Monitorability hooks** — preserve legible CoT across loops; make loop depth observable for oversight.
6. **Multi-agent local inference** — several persistent agents on one device; KV isolation and memory budget from LP-05/06.

## PHASE PLANE (extension)

| Phase | Deliverable | Gate |
|-------|-------------|------|
| P0 — Reorient | This doc + index pointers + archived old goal | `test -f .hermes/inkling-2-mini-goal.md` |
| P1 — Probe | Architecture probe reads an Inkling GGUF: dims, experts, loop metadata, quant formats | `make test_arch_probe` |
| P2 — Loop | Runtime loop executor over the middle block; N=1..10 | `make test_loop_depth` (logits match reference per depth) |
| P3 — Format | NVFP4 weight dequant + MXFP8 tail layers | `make test_nvfp4_weights` (vs reference tensor) |
| P4 — MoE | Top-k routing with loop-aware activation budget | `make test_moe_loop` |
| P5 — Context | 256K–512K decode with SWA reclaim + quant KV | `make test_inkling_longctx` |
| P6 — Monitor | CoT legibility + loop-depth observability | `make test_monitorability` |
| P7 — Agents | Multi-agent KV isolation on one device | `make test_multiagent_local` |

Dependency: P1 → P2 → {P3, P4} → P5 → {P6, P7}. Do not open a later phase before its gate passes.

## THE LOOP

```
read goal → pick lowest undone phase → implement one gate → run gate →
update state.md + WORKLIST → update palace → push → loop
```

Exit gate on every decision: Claim → Verify → Risk → Mitigate. A result that "looks right"
is exactly when verification is mandatory. A handoff is a claim, not evidence.

## RULES

- The engine stays ignorant of the goal it serves. Goal/state/slerm logic lives in the host module (LP-01).
- Reimplement from upstream research in C11; never vendor a third-party library.
- No stubs. A phase ships only with a real, named `make test_*` gate.
- No prestige ceremony in responses; the palace files are reference, not a response format.
- bytropix is a coding reference and a past proof run, not the target. This repo is the target.

## FILES

- `.hermes/inkling-2-mini-goal.md` — this file (authoritative goal)
- `.hermes/goal-mantra.md` — short boot pointer
- `.hermes/mind-palace/goal-mantra.md` — short boot pointer
- `.hermes/vault/bytropix-goal-archive.md` — previous goal, preserved
- `docs/WORKLIST.md` — shipped pad status
- `docs/SOTA-2026-11-ROADMAP.md` — prior lilypad set
