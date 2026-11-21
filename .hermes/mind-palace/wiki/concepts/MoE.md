# Concept: MoE (Mixture of Experts)

## Summary
Routing architecture where each token is processed by a subset of "expert" networks instead of the full FFN.

## How It Works
- **Router**: small network that scores each expert for each token
- **Top-k selection**: pick k highest-scoring experts (typically k=1 or k=8)
- **Weighted combination**: merge expert outputs by router weights
- Result: large model capacity with small active compute per token

## In bytropix
- Qwen3.6 uses MoE with ~128 experts
- **CPU MoE**: working correctly ✅
- **GPU MoE**: cos-sim 0.9888 → garbage output 🔴 (fundamental code-path difference, not a small bug)
- GPU kernels: `gpu_moe_kernel.cu`, `wubu_moe.c`

## Variants
- **Dense** — no routing, full FFN every time (simpler, slower at scale)
- **Switch Transformer** — top-1 routing, load balancing loss
- **DeepSeek MoE** — shared experts + routed experts

## Sources
- [[qwen36-tech-report]]
- [[deepseek-v3-report]]
