# Module notes — reuse folders

Generated 2026-11-24 from `Makefile.win` (117 named EG gates). These notes are navigation, not a second register. Ground truth remains `docs/ENGINE-GAP-REGISTER.md`, `include/`, `src/`, and the gate sources under `tools/`.

## Folders

| Folder | Reuse seam | Gate count |
|---|---|---:|
| `kv-memory/` | KV allocation, paging, reuse, offload, placement | 22 |
| `speculative/` | drafting, verification, tree/graph policies | 28 |
| `moe/` | expert routing, caching, staging | 8 |
| `quant/` | number formats, dequant, bit packing | 10 |
| `host/` | goal, agent, telemetry, benchmark decisions | 4 |
| `diagnostics/` | probes, identity, format checks, smoke gates | 45 |

## Hive mindset

Every mechanism is one hive cell: one owner header in `include/`, one implementation in `src/`, one gate in `tools/`, one evidence paragraph in the register. A new technique joins the hive by matching an existing seam, not by creating a parallel stack. Shared facts (benchmark protocol, source catalog, gap register) live once and are linked.

## Revolver mindset

Six chambers rotate every wave: research → clone/read → RED test → GREEN implementation → measured evidence → verification/commit. Fire one chamber at a time; after committing, the cylinder returns to research with the gap register as the next target. This keeps the loop moving without inventing work.

## Model-agnostic rule

Core kernels know tensor shapes and formats, not model names. Architecture probes read GGUF metadata; family differences become compiled rows with byte-identical kernel bodies. Model-specific claims require a probe log, not a hard-coded branch.

## Anti-duplication rule

Before adding a module, grep `include/` and `src/` for the mechanism's symbols, then check the register and these notes. If the seam exists, extend it; if the gate exists, cite it. A new folder needs a named seam and at least one gate.
