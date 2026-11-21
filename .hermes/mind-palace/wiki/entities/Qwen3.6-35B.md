# Entity: Qwen3.6-35B

## Summary
35B parameter MoE language model with Gated DeltaNet SSM architecture. Target model for bytropix inference engine.

## Key Facts
| Property | Value |
|----------|-------|
| Parameters | 35B total, ~3B active (A3B) |
| Architecture | qwen35moe |
| Attention | GQA (Grouped Query Attention) |
| SSM | Gated DeltaNet |
| Experts | MoE with ~128 routed experts |
| Context | Up to 256K tokens |
| Quant | UD-IQ2_M (2.7 bpw, 10.7 GB) |

## Details
- Uses **Gated DeltaNet** variant of State Space Models — not traditional attention-only
- **MoE routing**: top-k expert selection across transformer layers
- **GQA**: grouped query attention with shared KV heads
- **IMRoPE**: Interleaved Multi-head Rotary Position Embeddings
- Weights distributed as GGUF format
- Vision extension via **mmproj** projection from ViT encoder to 2048-dim text space

## Relationships
- Implements: [[Gated-DeltaNet]], [[MoE]], [[GQA]], [[GGUF]]
- Extended by: [[Moondream3]] (vision encoder + mmproj)
- Reference impl: [[llama.cpp]] (qwen35moe.cpp handler)

## Sources
- [[qwen36-tech-report]]
- [[qwen36-arch-reference]] (local: `bytropix/vault/qwen36-repo/Qwen3.6_README.md`)
