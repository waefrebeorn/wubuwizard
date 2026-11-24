# Benchmark Workflow — continuous local-inference optimization

Date: 2026-11-23. Scope: the reproducible workflow wubuwizard will use to keep
improving local serving on a 6 GB laptop. This is a method doc, not a result doc. [unverified]
Measured results go in `docs/DEVILS-ADVOCATE-LEGO-LOOP.md` and the gap register. [unverified]

## The workflow in one loop

1. **Pin the workload before touching the engine.** For a home coding-harness
   target, record input length, output length, turn count, tool-call cadence and
   whether the run is chat or agentic; agentic multi-turn traces change TTFT and
   TPOT enough that chat-only benchmarks mislead.[3]
2. **Isolate the run.** Execute the matrix sequentially at run level, restart the
   serving process when context size changes, warm up every run, and use fixed
   inter-run spacing; that protocol cut average coefficient of variation from
   15.2% to 2.2% in a controlled A100 study.[1]
3. **Use at least five repetitions and report variance**, not a single mean;
   single-run numbers cannot support a regression decision.[1]
4. **Select representative parameters, not all of them.** Sweep batch/concurrency
   on power-of-two points and shrink the output-token space where fidelity allows;
   FMwork reports up to 24× experiment savings and a 2.7× gain from reducing
   1024 to 128 output tokens while retaining 96.6% accuracy in one Llama 3.1 8B
   evaluation.[2]
5. **Measure both engine capacity and user experience.** Server-side/localhost
   benchmarks isolate the engine, while client-side benchmarks capture the UX path;
   both are needed, but never mix them in one comparison.[6]
6. **Archive a reproducible bundle**: exact flags, git commit, model quant/hash,
   build ID, GPU/driver, raw stdout, CSV, config and date. MLPerf-style
   submissions and Red Hat's harness both depend on version-pinned configs and
   structured trial records to make autotuning repeatable.[5]
7. **Change one variable at a time.** Compare baseline and candidate with the
   same model, quant, device, prompt lengths and repeat count; record mean,
   standard deviation and CV for prefill and decode separately. [unverified]
8. **Promote only with evidence.** A change ships when the named gate passes and
   the benchmark log shows the improvement under the fixed protocol; otherwise it
   stays a lead. [unverified]

## Metrics to keep

- Prefill: `pp128`, `pp512`, `pp1024` tokens/s and TTFT. [unverified]
- Decode: `tg32`, `tg64`, `tg128`, TPOT / inter-token latency and E2EL. [unverified]
- Serving: request throughput (req/s) at stable load, plus a saturation sweep;
  request throughput is the honest headline for agentic serving because long
  prefill requests produce few output tokens.[3]
- Resources: VRAM peak, system RAM, device/back-end string, thread count. [unverified]
- Statistics: mean, sample std-dev, CV, min/max, repetition count. [unverified]
- Correctness: gate exit code and any logits/byte-equality evidence; speed never
  overrides a correctness gate. [unverified]

## Workload profiles for this machine

| Profile | Shape | Why |
|---|---|---|
| `chat` | pp128 / tg32 | fast smoke comparison; matches the existing Inkling evidence [unverified] |
| `coding` | pp512 / tg64 | code-harness prompt plus generated patch [unverified] |
| `agent` | pp1024 / tg32 with repeated turns | long context, small output — the agentic regime[3] |
| `saturation` | concurrency 1,2,4,8 | locate the point where TTFT/TPOT stop scaling [unverified] |

Random-token fills are fine for length-shaped serving stress, but they are bad
for speculative decoding because a trained draft model cannot predict noise; use
real or code-like text when testing speculation.[6]

## Artifacts per run

```
data/bench/<UTC>-<tag>/
  manifest.json     # git rev, model, quant, build, flags, environment
  raw/<variant>.log # untouched stdout/stderr
  results.csv       # variant,prompt,gen,rep,pp_tps,tg_tps,device,notes
  summary.md        # mean/std/CV, ratios, decision, next variable
```

## Decision rule

Declare a win only when both conditions hold: [unverified]

- the new mean is better than the baseline mean outside the combined noise
  (roughly `delta > 2 * sqrt(se_base^2 + se_cand^2)`, with `se = sd/sqrt(n)`), and [unverified]
- no correctness gate regresses. [unverified]

If the difference is inside noise, increase repetitions or change the workload; [unverified]
do not call it a speedup. If the gain holds only on one prompt shape, name that [unverified]
shape and keep the change local to its profile. [unverified]

## What not to break

- Do not benchmark with a different device, quant, build or thread count between
  baseline and candidate. [unverified]
- Do not average a saturated point into an unsaturated comparison.[3]
- Do not use a synthetic byte-count proxy as a wall-clock claim; proxy gates and
  measured tok/s belong in separate sentences. [unverified]
- Do not let hardware-aside research replace the active engine loop. [unverified]

## Next concrete step

Wire this workflow into `tools/bench_llama.sh`, add a C11 statistics gate for
mean/std/CV and regression detection, then re-run the Inkling loop placement
comparison at `pp512/tg64` and the agent profile to see whether the 1.56×/1.35×
short-prompt gain survives realistic shapes. [unverified]

## Sources

[1] https://arxiv.org/html/2610.09778 — Sequential Isolation Methodology
[2] https://arxiv.org/html/2508.10251v1 — FMwork meta-metrics
[3] https://arxiv.org/html/2609.34683 — AgentPerfBench
[5] https://www.redhat.com/en/blog/efficient-and-reproducible-llm-inference-red-hat-mlperf-inference-v51-results — Red Hat MLPerf v5.1 harness
[6] https://www.baseten.co/blog/how-to-run-llm-performance-benchmarks-and-why-you-should — Baseten benchmarking
