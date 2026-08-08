# vault/ — Legacy Documentation & Quantization References

Archived mind palace snapshots, quantization formulas, and reference data.

## Key Files

| File | Purpose |
|------|---------|
| `context-growth-penalty.md` | Full analysis: persistent KV fix + compilation flags fix |
| `LEGACY.md` | Legacy documentation index |
| `api-server.md` | API server sandbox documentation |
| `unsloth-quantization-format.md` | Unsloth UD quantization format reference |
| `bins/` | Archived mind palace snapshots from prior sessions |

## Quantization Reference

| Format | File | Bits/Weight | Usage |
|--------|------|:-----------:|-------|
| Q4_K | `gguf_reader.h` | 5.0 | Output projection, MoE down (3L) |
| Q5_K | `gguf_reader.h` | 6.5 | SSM attn_qkv/gate, GQA attn_q/k/v/output, shared gate/up |
| Q6_K | `gguf_reader.h` | 7.5 | SSM ssm_out, shared down |
| IQ2_XXS | `gguf_reader.h` | 2.2 | MoE expert gate/up (all 40L) |
| IQ3_XXS | `gguf_reader.h` | 3.3 | MoE expert down (37L) |
| IQ4_XS | `gguf_reader.h` | 4.3 | MoE expert down (3L) |
| Q8_0 | `gguf_reader.h` | 9.0 | Quantized activation for matmul |

## Related

- `.hermes/vault/` — Research papers and active documentation
- `.hermes/mind-palace/` — Current mind palace (state, plan, prestige)
