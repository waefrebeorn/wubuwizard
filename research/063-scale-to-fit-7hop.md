# research/063 — SCALE-TO-FIT 7-HOP: the sizing structure for ALL hardware

> 2026-08-08. The user's directive: "seven steps online research and also
> consult local research on how to improve the stack and its sizing
> structure to fit all hardware configurations and possibilities that
> you will possibly face."
> Predecessor: THEORY/11 (the scale-to-fit planner, shipped as
> wubu_scale.c + test_scale_to_fit). This doc AUDITS the planner against
> the frontier and names the concrete axes it must gain.

## The convergence (one line)

**RAM is the weakest sizing axis.** Every paper converges on the same
truth: the load plan must be a function of BANDWIDTH (bytes/token), not
just capacity — and on constrained silicon it must also weigh ENERGY and
THERMAL headroom, plus the compute/communication split between
CPU/GPU/NPU. THEORY/11's `ram_budget` is step 0; the frontier is a
4-axis plan: capacity × bandwidth × energy × device-routing.

## The 7 hops (online) + local consultation

### Step 1 — Memory-bounded inference: Prima.cpp (arXiv:2504.08791)
Heterogeneous home clusters (mixed CPU/GPU, insufficient RAM/VRAM, slow
disks, Wi-Fi) run 30-70B models OOM-free. The two mechanisms that
matter for us:
- **pipelined-ring parallelism (PRP)**: overlap disk I/O with compute
  and communication — the mmap offload is NOT a stall, it is a pipeline.
- **Halda scheduler**: co-optimizes per-device CPU/GPU workload and
  device selection under RAM/VRAM constraints; 5-17× lower TPOT vs
  llama.cpp. KEY: it LEARNS to drop a device when that is faster.
Convergence: the plan must model DEVICES, not one RAM pool. A CM4's
storage (SD/eMMC) is a compute tier too — with prefetch.

### Step 2 — Sub-1B SLM architectures (arXiv:2501.05465 + SmolLM/TinyLlama)
Small models pack punches when the ARCHITECTURE is sized to the task:
shared heads, no excessive width, high data quality. For a CM4 boot
core, the lesson is not "fewer layers" but "right-shaped core": the
boot core's depth/width ratio should come from the checkpoint's own
shape, never a fixed default.

### Step 3 — Inference-time scaling laws for heterogeneous computing:
QEIL (arXiv:2602.06057) ← THE anchor paper
**Five architecture-agnostic theorems**: inference efficiency scales
with model size × sample budget × device parameters, and
**heterogeneous workload distribution yields SUPERLINEAR gains
invisible to homogeneous approaches**. Their three axes map directly
onto our planner:
1. inference scaling laws → our ecosystem N (the active ratio)
2. hardware-aware routing via cost models (TFLOPS, bandwidth, power,
   thermal) → the ROUTER SLOT should route by device cost, not just
   well depth
3. Intelligence Per Watt (IPW, 2-5.6×) + Price-Power-Performance →
   the planner needs an ENERGY axis (watts per active param), not just
   bytes.
Verified on 125M → 2.6B models — exactly our range.

### Step 4 — On-chip hardware-aware quantization: OHQ (arXiv:2309.01945)
Mixed-precision quantization that perceives ACTUAL hardware efficiency
instead of simulation: 15-30% latency cut vs INT8 on real silicon.
Convergence: our precision cascade (F32 core → Q2 leaves) should be
MEASURED per machine, not assumed — the plan's cascade is a starting
point, the runtime measures and tightens. (We already have the
dequant-on-demand path that makes per-machine re-quant cheap.)

### Step 5 — Adaptive inference / early exit (PALBERT arXiv:2204.03276 +
local pond: adaptive-inference-batching via policy gradients, 2607.05272)
Early-exit on input difficulty: "models may require a different number
of computing layers for different input sequences — evaluating all
layers leads to overthinking." Convergence: the plan's `fractal_depth`
should be a BUDGET, not a constant — the runtime early-exits easy
tokens. On a Pi this is the difference between "runs" and "runs at a
usable speed".

### Step 6 — MoE on constrained hardware (PROBE arXiv:2602.00509 +
Speculating-Experts 2603.19289 + local pond)
MoE inference has a fundamental tension: expert parallelism improves
memory efficiency but amplifies stragglers. PROBE: co-balancing
computation and communication via REAL-TIME PREDICTIVE PREFETCHING.
Convergence: the ecosystem router should PREFETCH the next likely
balls (the fire-count utilization we already track) during the current
token's compute — the CM4's slow storage hides behind compute.

### Step 7 — Energy-aware on-device inference: EnerInfer (arXiv:2606.23001)
**"On-device LLM inference has exploitable configuration slack:
modestly lowering NPU and memory frequencies preserves QoE while
substantially improving energy efficiency and reducing heat."** Up to
65% energy improvement without QoE violation. The most efficient
NPU/DDR setting varies per model+engine+platform — no stable ranking.
Convergence: the plan must report an ENERGY CLASS and leave frequency
as a tunable, not a constant. A CM4 throttling under load is a
THERMAL problem the planner should anticipate (the SD card especially).

### Local consultation (the ponds + prior synthesis)
- pond-a (KV/memory): 6 memory-bounded files, 98 scaling-law files,
  32 hardware-aware — the substrate confirms steps 1-4.
- pond-c (MoE): 133 edge-device files, 43 hardware-aware, PROBE +
  Speculating-Experts present.
- pond-g (systems/OS): 46 offloading files, 163 heterogeneous, 7
  raspberry-pi files (FreeRTOS RP2040 ports — the CM4 target's OS).
- research/053 (small-model playbook) + 058 (low-bit accuracy): our own
  prior synthesis already names data-quality and per-role bit ladders —
  this 7-hop extends them with the DEVICE/ENERGY axes.
- research/057 (mixed compression): the per-role ladder is the cascade
  the planner should expose per-hardware, not globally.

## The implementation mandate (what the planner must gain)

| # | Axis | Source | Concrete change |
|---|------|--------|-----------------|
| A | DEVICES | Prima.cpp/QEIL | `wubu_scale_hw_t` gains `devices[]` (CPU/GPU/NPU with TFLOPS, bandwidth, power); plan reports per-device weight split |
| B | ENERGY | EnerInfer/QEIL | plan gains `watts_estimate` + an energy class; the cascade becomes a candidate the runtime measures |
| C | BANDWIDTH | QEIL | plan reports `bytes_per_token` and the active ratio as a BANDWIDTH choice (K/N solved for BW, not just RAM) |
| D | ADAPTIVE | PALBERT | `fractal_depth` becomes a budget: plan sets a ceiling, runtime early-exits (the ecosystem fire-count already gives the signal) |
| E | PREFETCH | PROBE/PRP | the router adapter prefetches next-likely balls from the fire-count utilization (slow storage hides behind compute) |
| F | ON-CHIP MEASURE | OHQ | `wubu_scale_measure()` — a one-shot microbenchmark that tightens the cascade from the plan's starting point |

## Triple-DA

1. **Correctness**: all six changes are CPU-realizable (no Triton, no
   datacenter). The bandwidth axis is roofline-honest (bytes/token is
   the real cost on every tier from CM4 to super).
2. **Privacy**: no external services; the on-chip measure is local.
3. **Robustness**: every new axis degrades gracefully (no devices ->
   one CPU device, no power sensor -> energy class from defaults);
   the plan still computes on a bare hw struct.

## Status: `open` — A,B,C land first (the planner's core); D,E,F are
runtime companions (the router + a measure hook) queued behind them.
