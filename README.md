<div align="center">

# ⚡ wubuwizard — THE BRAIN

**The WuBu amoeba AGI** — a C11 inference/training engine, KV-cache
filesystem, universal encoder space, and research ledger. One AGI,
two repos: this Brain (thinks) + `wubuos` (the Body: kernel, GUI,
Styx/9P namespace).

**C11. Opaque structs. No third-party if we can write it ourselves.**

</div>

---

## What this repo is

wubuwizard is the **Brain** half of the WuBu project — everything that
thinks: the C11 inference engine, model loading, KV-cache compression,
training, the universal encoder space, and the research ledger.

**One sentence:** The Brain learns; the Body protects and acts.

## Architecture

- **Engine:** C11 (`-std=c11`), opaque structs, minimal includes, no god
  headers, backends register at runtime (`wubu_kernel_register()`).
- **Model:** WuBu1 (dim 512, ffn 2048, 8 heads, head_dim 64 —
  56,376,832 params), alive homogeneous amoeba: BOOT CORE / BODY
  ecosystem of hyperbolic balls / MEMORY orbits. Scale-to-fit:
  one checkpoint, any hardware (THEORY/11).
- **KV cache is a file system:** the namespace stores floats; a file
  IS a mount region; the encoder IS a mount (research/061, THEORY/05).
- **The user space is the training data:** real user files (text,
  images, video, audio, office, PDF) ingest into a compressible
  encoder space — all decoders OURS, no encoder licensing
  (research/065–067).
- **Spine principle:** decode is memory-bandwidth-bound (Roofline
  2607.02558). Every performance win attacks bytes moved.

## Branches (2026-08-09 unification)

**ONE branch: `wubu-integration`** (GitHub default). ALL work — both
agents, together — flows here. The former LFM/Distiller branch was
unified into the trunk.

Retired branches live as `archive/retired-*` tags — see
`docs/BRANCH-AUDIT-2026-08-09.md` for the no-code-lost audit.

## Build & test

```bash
make all          # full build (engine + tools)
make test_all     # the test gate — run this before claiming anything works
make test_<name>  # one subsystem (e.g. test_kvfs, test_codec, test_userfs)
```

## Directory map

| Path | What lives there |
|---|---|
| `src/wubu_*.c` | The engine modules (SSM, MoE, KV, attention, quant, encoders) |
| `include/wubu_*.h` | Public API headers — opaque types + function decls |
| `tools/*.c` | CLI tools and test drivers (`gen_text`, `wubu_cli`, `test_*`) |
| `research/` | The gap ledger (`INDEX.md`) + NNN-*.md 7-hop research docs |
| `docs/` | TOPOLOGY.md (master map), MODULES.md, BUILDING.md, ADR/ |
| `THEORY/` | Design philosophy, nesting, KV-namespace theory (01–11) |
| `ENCODERS/` | The WuBu Nesting research core (5 phases of encoder work) |
| `MATH/lean/` | Lean-verified theorems (MobiusAdd, Poincaré ball, etc.) |

## Sources of truth

- **AGENTS.md** — agent context: how to work here, the non-negotiables
- **docs/TOPOLOGY.md** — the master map
- **research/INDEX.md** — the gap ledger (open/wired discipline)
- **STATUS.md** — verified implementation state (fresh-command audited)
- **docs/BRANCH-AUDIT-2026-08-09.md** — the no-code-lost branch audit
