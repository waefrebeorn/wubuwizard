# Concept: SSM (State Space Models)

## Summary
State Space Models for sequence modeling. Alternative to attention. Used in Qwen3.6 as Gated DeltaNet variant.

## How It Works
- Maintains a **hidden state** that evolves over the sequence
- At each step: `h_t = A * h_{t-1} + B * x_t`, `y_t = C * h_t`
- Linear time per step, constant memory per step (vs O(n) for attention)
- **Recurrence** means no KV cache needed — state is the "memory"

## Key Variants
- **Mamba** — selective SSM, data-dependent A/B/C matrices
- **Gated-DeltaNet** — Mamba variant with gating mechanism, used in Qwen3.6
- **S4** — diagonal SSM, HIPPO initialization
- **S5** — multi-input multi-output SSM
- **Mamba-2** — SSD (state space duality), dual paths

## In bytropix
- 30 SSM layers in Qwen3.6-35B (interleaved with MoE attention)
- CPU path verified vs llama.cpp (cos-sim 0.994) ✅
- GPU path diverges (cos-sim -0.66) 🔴 — state persistence bug between layers
- Chunked SSM: CS=1 exact ✅, CS>1 FP accumulation ⚠️ (see [[Chunked-SSM]])
- Fused kernels: beta/alpha, conv/silu/split, L2 norm, recurrence, gated norm, ssm_out

## Open Questions
- GPU SSM divergence root cause: recurrence state not persisted correctly between GPU kernel launches?
- Can CS>1 FP accumulation be fixed with Kahan summation?
- How does Gated DeltaNet conv state initialization differ from Mamba?

## Sources
- [[mamba-paper]]
- [[qwen36-tech-report]]
