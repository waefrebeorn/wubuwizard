# Front M — Benchmark-workflow refinements for local laptop serving and agentic coding harnesses

Date: 2026-11-23. This report extends `docs/BENCHMARK-WORKFLOW.md` for the 6 GB laptop target [unverified].
It does not repeat that document's sequential-isolation, FMwork, AgentPerfBench, MLPerf or Baseten material; it adds the tool layer, workload distributions, metrics and reproducibility rules that document lacks [unverified].
It is a method report, not a result report [unverified].

## 1. The missing layer: a serving benchmark client

`llama-bench` is a micro-benchmark of isolated prompt processing (pp) and token generation (tg); its own README states that its measurements exclude tokenization and sampling, so it cannot represent client-side streaming, arrival processes, concurrency, prefix-cache reuse or agentic turn structure.[56]

genai-bench adds token-level serving metrics (TTFT, E2E, TPOT, error rate, RPS) across traffic scenarios and concurrency levels with a live UI and Excel/plot reports.[55] NVIDIA AIPerf replaces GenAI-Perf with a multiprocess client so the measurer does not become the bottleneck, and reports TTFT, ITL, request latency and output-token throughput with percentile breakdowns plus GPU telemetry via DCGM/pynvml.[75][76]

GuideLLM adds SLO-aware sweeps over synchronous, concurrent and rate-based traffic to identify safe operating ranges.[77] inference-perf adds constant/Poisson/fixed-concurrency load, automatic saturation sweeps, trace replay and a procedural agentic workload generator.[69][70]

## 2. llama-bench flags worth pinning on a 6 GB machine

The current upstream README is explicit about the flag surface: `-m/--model`, `-p/--n-prompt`, `-n/--n-gen`, `-pg <pp,tg>`, `-d/--n-depth`, `-b/--batch-size`, `-ub/--ubatch-size`, `-t/--threads`, `-ngl/--n-gpu-layers`, `-fa/--flash-attn`, `-ctk/-ctv`, `-dev`, `-r/--repetitions`, `--delay`, `--no-warmup`, `-o/--output` and `-fitt/--fit-target` with `-fitc/--fit-ctx`.[56] Three flags matter most here. `-d <n>` prefills the KV cache with `n` tokens before the measured test, which is the closest llama-bench analogue to a cached-prefix agent step.[56]

`-fitt <MiB>` fits the model to device memory with a per-device margin and `-fitc` sets the minimum context size, which is the direct guard against a 6 GB VRAM/DDR budget.[56] `-o jsonl` or `-o sql` records every repetition plus `build_commit`, `cpu_info`, `gpu_info`, `backends`, model, quantization and all tunables, so the output doubles as the run manifest.[56]

```bash
# pp sweep (prefill), fixed output, machine-readable
./llama-bench -m models/qwen.gguf -n 0 -p 128,512,1024 -r 10 -o jsonl
# tg sweep (decode) at realistic agent output lengths
./llama-bench -m models/qwen.gguf -p 0 -n 128,214,256 -r 10 -o csv
# cached-prefix proxy: prefill 119K of history, append + generate (see section 4)
./llama-bench -m models/qwen.gguf -pg 2000,214 -d 2000 -t 4,8 -ngl 99 -r 10 -o jsonl
# fit to a 6 GB device with a 256 MiB margin and 4K minimum ctx
./llama-bench -m models/qwen.gguf -p 0 -n 64 -fitt 256 -fitc 4096 -r 5 -o sql
```

Pin `-r`, `--delay` and the warmup state explicitly; the README notes that every option except `-r`, `-o` and `-v` can be swept and that each pp/tg combination is run with all specified values.[56]

## 3. Serving-client commands to add

genai-bench installs from PyPI and its benchmark subcommand takes `--api-backend`, `--api-base`, `--api-model-name`, `--task`, `--traffic-scenario`, `--max-time-per-run` and `--max-requests-per-run`; the `excel` and `plot` subcommands turn an experiment folder into a report.[55]

```bash
pip install genai-bench
genai-bench benchmark --api-backend "your-backend" \
  --api-base "http://localhost:8080" --api-model-name "your-model" \
  --task text-to-text --traffic-scenario "I(1024,1024)" \
  --max-time-per-run 5 --max-requests-per-run 100
genai-bench excel --experiment-folder ./experiments/<tag> --excel-name results --metric-percentile mean
genai-bench plot --experiments-folder ./experiments --group-key traffic_scenario --preset 2x4_default
```

NVIDIA's AIPerf walkthrough pins synthetic lengths to exact values with zero standard deviation and `ignore_eos`, then shows a Poisson rate-mode run with an explicit seed.[76]

```bash
aiperf profile --model Qwen/Qwen3-0.6B --endpoint-type chat --streaming --url localhost:8000 \
  --synthetic-input-tokens-mean 128 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 --output-tokens-stddev 0 \
  --extra-inputs min_tokens:128 --extra-inputs ignore_eos:true
aiperf profile --model Qwen/Qwen3-0.6B --endpoint-type chat --streaming --url localhost:8000 \
  --request-rate 10 --arrival-pattern poisson \
  --synthetic-input-tokens-mean 512 --output-tokens-stddev 32 \
  --random-seed 42 --request-count 200
```

GuideLLM's documented first run is a capacity sweep bounded by wall time.[77]

```bash
guidellm run --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=sweep --constraint kind=max_duration,seconds=30 \
  --data kind=synthetic_text,prompt_tokens=256,output_tokens=128
```

inference-perf runs from a YAML config and supports a one-line constant-rate smoke run and a saturation sweep.[69][70]
It also supports SLO-constrained goodput and a procedural agentic data type.[71][73]

```bash
inference-perf --server.type vllm --server.base_url http://localhost:8000 \
  --data.type random --load.type constant \
  --load.stages '[{"rate": 10, "duration": 60}]' --api.streaming true
```
```yaml
load: {type: constant, sweep: {type: linear}}      # sweep to saturation
report:
  goodput:
    constraints: {ttft: 0.2, tpot: 0.02, ntpot: 0.25}
```

## 4. Real prompt/response length distributions for coding agents

This is where the current `pp128/tg32`, `pp512/tg64` and `pp1024/tg32` shapes are misleading [unverified].

TraceLab's cross-provider trace of ~4,300 sessions, ~350,000 LLM steps and ~430,000 tool calls from 43 developers over eight months reports a median step of about 119K prefix tokens (cached history), 875 freshly appended tokens and only 214 output tokens; an agent completes a request with ~8.8 LLM calls and ~10.8 tool calls, averaging 4.3 minutes with p90 above 6.4 minutes.[60]

Normalized decode in that trace has a median of 46.8 tokens/s for Claude and 33.9 for Codex with CV above 50%, and the global prefix-cache hit rate is 95.7%; misses cause 3.8× more prefill than genuinely new input, and tool calls over one minute are just 4% of calls but 85% of tool time.[60]

GitHub Copilot's production characterization of 5.7B LLM calls and 946T tokens finds prefix caching covers ~90% of tokens within a turn but only 52% across turn boundaries and 6% after a model switch, with five user archetypes spanning a 50× per-turn token range (23K–1.1M) and tool failures on 9% of turns amplifying compute by up to 4× via retries.[61]

A separate serving study reports empirical cache-hit ratios of 84.6–99.5% with decode dominating 91.0–98.6% of LLM time, i.e. agentic serving is a long-lived-KV-state problem rather than an isolated-long-prompt problem.[62]

SWE-chat's ~18,000 real sessions, 229,000 prompts and 2M tool calls show more than 45% of tool calls are shell commands and users push back on agent output in 50% of turns.[63]

The LMCache agentic-trace dataset makes the shape explicit: input tokens are right-skewed with a median of 21K, outputs concentrate below 200 tokens (tool calls) with a long code-generation tail, and context grows from ~7K to ~35K tokens by turn 50.[64]

Local consequence: a 119K-token KV cache cannot fit alongside a 6 GB model, so the honest local profile is a *small* cached prefix (2K–8K) plus a short append and <200 output tokens per tool step [unverified].
That makes prefix-cache hit rate, TTFT and KV residency the primary local metrics, not raw tg tok/s [unverified].
The procedural agentic generator exposes exactly these knobs — `input_tokens_per_turn`, `output_tokens_per_turn`, `turns_per_session`, `tool_loop_depth`, `parallel_tool_calls_per_step`, `tool_catalog_size_per_agent`, `fanout_probability`, `max_depth` and `shared_system_prompt_len` — and produces a byte-identical graph for a given `(config, seed)`.[73]

## 5. NTPOT and normalized latency

NTPOT amortizes the whole request lifecycle over output length: for request *i*, `NTPOT_i = L_i / N_i`, expanded as `(T_queue + I_i*T_prefill + N_i*T_decode) / N_i`, so queue contention and heavy prefill are no longer hidden by a decode-only metric.[57]

Google's GKE documentation defines the same field as `request_latency / total_output_tokens` and lists it alongside TPOT, ITL and request latency as a first-class serving indicator.[58]

The same client-side measurement-bias work shows that single-process asyncio clients inflate TTFT and TPOT via the Python GIL (modelled as an M/G/1 queue), which is why a multiprocess client is a methodological requirement and not a nicety.[57]

The IETF terminology draft defines normalized latency as end-to-end latency divided by output token count, gives `TPOT = (E2EL - TTFT) / (Output Token Count - 1)`, and warns that normalized latency must not be the sole latency metric and should be reported with the output-length distribution.[59]

inference-perf implements NTPOT as `e2e request latency / output tokens` and keeps both server-reported `usage` counts and client re-tokenizations, with `report.request_lifecycle.use_server_output_tokens: true` to normalize by the server count.[74]

Goodput is the SLO-aware complement: count only requests that complete *and* meet constraints on `ttft`, `tpot`, `itl`, `ntpot` or `request_latency`, reported as request goodput, token goodput and goodput percentage.[71] The IETF draft generalizes this as SLO attainment, goodput and agentic goodput for compound systems.[59]

## 6. Energy metrics

Software-based GPU energy measurement is portable enough for a laptop-class study: ML.ENERGY argues for measuring GPU energy in software, reporting at the whole-response granularity (not per token, which misses verbosity and task differences), and letting the user supply a latency target such as average TPOT so the tool recommends the energy-optimal configuration.[65]

Watt Counts publishes 5,000+ experiments over 50 LLMs and ten GPUs with power draw, energy per token and throughput under an MIT-licensed benchmark that needs no external power meter, and reports up to 70% energy reduction in server scenarios with negligible user-experience impact.[66]

An ACL 2025 study uses a binning strategy over input/output token distributions and batch sizes with CodeCarbon/NVML, finds that FLOP-based estimates substantially underestimate real energy, and shows up to 73% reduction from correctly applied optimizations.[67]

AI Energy Score uses CodeCarbon plus grid carbon intensity and reports GPU energy in watt-hours per 1,000 queries.[68]

The IETF draft's supplementary metrics name energy per token, tokens per joule, instantaneous power draw and peak power draw.[59]

For a 6 GB laptop, energy per token and peak power draw are also a thermal-throttling guard: integrate power over the run, report Wh/1K output tokens at a fixed TTFT constraint, and treat a tok/s gain that coincides with a power-limit clamp as a thermal artifact [unverified].

## 7. Saturation and load generation

A latency profile is a sweep from baseline arrival rate to full structural saturation, with an ideal operating zone below the knee and a post-saturation region where SLOs collapse.[57]

inference-perf implements this with `load.type: constant` or `poisson` and `load.sweep.type: linear` (sweep is unavailable for `concurrent`), first estimating the server's maximum QPS from a 1,000-concurrency burn-rate run; its report exposes `achieved_rate` versus `requested_rate` and the client `schedule_delay` distribution, with a median scheduling delay under 10 ms as the fidelity check.[70]

Maximum in-flight load is bounded by `num_workers * worker_max_concurrency`, with `worker_max_concurrency` defaulting to 100 and `num_workers` auto-set to the available CPUs.[70]

GuideLLM and AIPerf both offer reproducible rate/concurrency sweeps, but AIPerf defaults to Poisson arrivals while vLLM's `bench serve` rescales random gaps to an anchored total, so the two are not comparable at the same nominal rate.[77][76][72]

## 8. Reproducible artifacts

Cross-tool gaps are configuration differences until proven otherwise: the comparability guide requires agreement on output length (fixed plus `ignore_eos`), input length, sampling parameters, load model (closed-loop versus open-loop), warmup window and token counting, and provides a checked flag map for `vllm bench serve` v0.10.0 and `aiperf profile` v0.12.0 plus `inference-perf-convert` to emit a config from peer argv.[72]

Its verify-before-comparing checklist is to compare total tokens sent/received, compare input/output length statistics, check `token_count_mismatches` and `client_fallback_requests`, and only then compare throughput and latency.[72]

The `inference-perf-convert` emitted file records the source argv and every assumption, refuses unknown or workload-changing flags, and never writes an API key into the config.[72]

The agentic generator derives all randomness from `seed` and session index so a `(config, session_index)` pair is byte-identical across runs.[73] llama-bench's `-o jsonl`/`-o sql` output already carries build commit, CPU/GPU strings, backends and every tunable per row.[56]

A run bundle should therefore contain: the config YAML, the llama-bench SQL/JSONL row, the serving-client JSON report, the exact argv, and a one-line statement of which side counted tokens.[69][74]

## 9. Metric → why it matters for a 6 GB laptop

| Metric | Definition / source | Why it matters here |
|---|---|---|
| TTFT | first non-empty token from request send[74] | On a shared CPU/GPU laptop this is where KV-cache misses and context-switching show up first; it is the user-perceived responsiveness gate. |
| TPOT | `(E2EL - TTFT)/(N-1)`[59] | Decode speed for the short <200-token tool-call outputs that dominate coding agents.[60] |
| ITL / jitter | inter-token arrival time[74] | Streaming smoothness; spikes reveal backend scheduling or thermal throttling. |
| NTPOT | `E2EL / output_tokens`[57][58] | Single comparable number across the 2K-prefix smoke and the 8K-prefix agent profile; folds prefill and queue delays into decode-normalized latency. |
| E2E request latency | completion minus send[74] | What the harness actually waits on per tool step. |
| Prefix-cache hit rate | cached / total prompt tokens[61][62] | Production coding agents live at 52–95% hit rates; on 6 GB the cache is the scarce resource, so this drives TTFT more than raw FLOPs. |
| Fresh append tokens | new input per step (~875 median upstream)[60] | Sets the *real* prefill cost; pin it rather than the full context length. |
| Output tokens per step | ~214 median upstream, <200 concentrated[60][64] | Determines whether tg tok/s or TTFT is the bottleneck for this workload. |
| Request goodput | good requests / time under TTFT+TPOT SLOs[71] | Rewards a stable engine over a peaky one; the right promotion gate for a shared laptop. |
| Energy per token / Wh per 1K | energy divided by tokens or queries[65][66][68] | Battery and thermal budget; correlates with sustained, not burst, tok/s. |
| Peak / instantaneous power | sampled NVML/RAPL power[59][67] | Detects when a "win" is a short turbo burst before the power limit clamps. |
| VRAM peak + KV footprint | device memory high-water mark | 6 GB is a hard ceiling; `-fitt`, `-fitc` and cache type decide whether the model fits at the required ctx.[56] |
| `achieved_rate` / `requested_rate` | load-generator fidelity[70] | A benchmark that cannot generate the offered load measures the client, not the engine. |
| Client `schedule_delay` p50 | queue delay before send[70] | Must stay <10 ms or the latency curve is client-biased.[57] |
| `token_count_mismatches` | client vs server counts[74][72] | Invalidates cross-tool comparisons when nonzero. |
| Build commit / device string | emitted per row by llama-bench[56] | The only way to reproduce a 6 GB result months later. |

## 10. Five concrete additions

1. Add a serving-client tier beside `llama-bench` for engine kernels.[56] Install AIPerf and GuideLLM (or genai-bench) for the harness path.[55][76]
   Gate harness changes on TTFT/E2E/NTPOT distributions, not pp/tg alone.[77]
2. Replace the single `pg` shape with a measured agent profile: short cached prefix, ~1K fresh append, <200 output tokens, repeated steps; generate it with `synthetic_agentic` knobs so it is deterministic.[60][64][73]
3. Add NTPOT and SLO goodput to the results CSV and the promotion gate, alongside `ignore_eos`, fixed length distributions and an explicit seed.[57][71][72]
4. Add an energy column using CodeCarbon/NVML sampled power, reported as Wh/1K output tokens and energy per token at a fixed TTFT constraint.[65][66][68]
5. Add a saturation sweep with load-fidelity checks (`achieved_rate` ≈ `requested_rate`, p50 `schedule_delay` < 10 ms) and archive the full config + argv + token-count provenance with each run.[70][57][72]

Caveat: all upstream distributions cited here come from cloud or desktop-class traces with far larger context windows than a 6 GB laptop can hold; use their *ratios* (prefix:append:output, hit rates, tool cadence) and rescale the absolute lengths to what fits locally [unverified].

## Sources

[55] https://github.com/sgl-project/genai-bench/blob/main/README.md — genai-bench README (SGLang)
[56] https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md — llama.cpp llama-bench README
[57] https://arxiv.org/html/2605.24217 — NTPOT / measurement bias (arXiv 2605.24217)
[58] https://docs.cloud.google.com/kubernetes-engine/docs/concepts/machine-learning/inference — GKE inference performance metrics
[59] https://datatracker.ietf.org/doc/draft-gaikwad-llm-benchmarking-terminology — IETF LLM benchmarking terminology draft
[60] https://arxiv.org/html/2606.30560v1 — TraceLab coding-agent workload characterization
[61] https://arxiv.org/html/2608.00101 — GitHub Copilot production-scale characterization
[62] https://arxiv.org/html/2605.26297 — Agentic AI Workload Characteristics
[63] https://arxiv.org/html/2604.20779v2 — SWE-chat real coding-agent sessions
[64] https://huggingface.co/datasets/sammshen/lmcache-agentic-traces/raw/61dfc6bbbcc3b82ed3a06c907f02db908504a930/README.md — LMCache agentic traces dataset
[65] https://papers.neurips.cc/paper_files/paper/2025/file/9dc510e3d7b0b3b2a58ffed7a3ad6b0f-Paper-Datasets_and_Benchmarks_Track.pdf — ML.ENERGY Benchmark (NeurIPS 2025)
[66] https://arxiv.org/html/2604.09048 — Watt Counts energy-aware LLM benchmark
[67] https://aclanthology.org/2025.acl-long.1563.pdf — Energy Considerations of LLM Inference (ACL 2025)
[68] https://huggingface.github.io/AIEnergyScore — AI Energy Score
[69] https://github.com/kubernetes-sigs/inference-perf — inference-perf (kubernetes-sigs)
[70] https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/loadgen.md — inference-perf Load Generation Guide
[71] https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/goodput.md — inference-perf Goodput Guide
[72] https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/comparability.md — inference-perf Comparability / flag map
[73] https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/synthetic_agentic.md — inference-perf Synthetic Agentic Workload Generator
[74] https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/metrics.md — inference-perf Metrics Definitions
[75] https://github.com/ai-dynamo/aiperf — NVIDIA AIPerf
[76] https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf.md — Benchmarking LLM Inference at Scale with AIPerf
[77] https://github.com/vllm-project/guidellm — GuideLLM (vLLM project)
