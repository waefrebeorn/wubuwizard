# AGENTS.md — wubuwizard (THE BRAIN)

> Agent context file. Read this before working in this repo. Kept current;
> update it when the structure changes. Humans: this is your onboarding too.

## What this repo is

wubuwizard is the **Brain** half of the WuBu project — everything that
thinks: the C11 inference engine, model loading, KV-cache compression,
training, the research ledger, and the math vault. The **Body** half is
`wubuos` (kernel, GUI, Styx/9P namespace). One AGI, two repos.

**One sentence:** The Brain learns; the Body protects and acts.

## Architecture in one paragraph

- **Engine:** C11 (no C++), `-std=c11`, opaque structs, minimal includes,
  no god headers, no third-party if we can write it ourselves.
- **Model:** WuBu-35M (12 layers, dim 448, GQA 7:1, hybrid local/global
  attention, partial RoPE, bounded SwiGLU, residual selectors, tied
  embeddings, byte-level BPE vocab 16,384).
- **Spine principle:** decode is **memory-bandwidth-bound** (Roofline
  2607.02558). Every performance win attacks bytes moved.
- **KV cache is a file system:** the KV namespace is a Styx/9P-exportable
  hierarchy (see research/061 + THEORY/05).
- **Dispatch:** backends register at runtime via `wubu_kernel_register()`;
  the engine never hardcodes which backend is live.

## Directory map

| Path | What lives there |
|---|---|
| `src/wubu_*.c` | The inference engine modules (SSM, MoE, KV, attention, quant) |
| `include/wubu_*.h` | Public API headers — opaque types + function decls |
| `tools/*.c` | CLI tools and test drivers (`gen_text`, `wubu_cli`, `api_server`, `test_*`) |
| `research/` | The gap ledger (`INDEX.md`) + NNN-*.md 7-hop research docs |
| `docs/` | TOPOLOGY.md (master map), MODULES.md, BUILDING.md, model blueprint |
| `THEORY/` | Design philosophy, nesting, the KV-namespace theory |
| `MATH/lean/` | Lean-verified theorems (MobiusAdd, Poincaré ball, etc.) |
| `tests/` (tools/test_*.c) | Test drivers, one per subsystem |

## Branches (the 2-branch reality — 2026-08-09 consolidation)

The repo was consolidated from 7 branches to 2. Do not create more:

| Branch | Purpose |
|---|---|
| `wubu-integration` | **THE trunk** (GitHub default). ALL work flows here. |
| `lfm25-adapter` | The LFM/Distiller agent's branch. Leave it alone. |

- Work on local `unified`, push to `origin/wubu-integration`
  (`git push origin unified:wubu-integration`). Fetch + rebase before
  pushing — the other agent pushes to the same branch.
- Retired branches are preserved as archive tags
  (`archive/retired-*`) — see `docs/BRANCH-AUDIT-2026-08-09.md` for
  the no-code-lost audit and recovery commands.
- The old bytropix brand is GONE. The repo is wubuwizard.

## Build & test (the two commands that matter)

```bash
make all          # full build (engine + tools)
make test_all     # the test gate — run this before claiming anything works
```

For fast iteration on one subsystem: `make test_<name>` builds+runs just
that test (e.g. `make test_kvfs`, `make test_lfm`). See `make help` for the
full target list.

## The non-negotiables (do NOT violate)

1. **No stubs.** Every called function does real work. Compile-time `#else`
   for unavailable hardware is the only exception.
2. **Opaque structs at module seams.** Public headers expose
   `typedef struct X X;` + accessors — never raw struct layouts across
   module boundaries.
3. **Minimal includes.** A header must not pull in other modules'
   implementation headers. No god headers. Split, don't bundle.
4. **No third party if we can write it** — self-contained C11 is the point.
5. **C11 strictness:** `-std=c11`, opaque pointers, single-exit error
   handling, status-as-value (never rely on `errno`).
6. **Verify before claiming.** Tests ≠ correct: read the code, run the
   target, read the FAIL lines. Never cite stale test counts.
7. **Research discipline:** the gap ledger (`research/INDEX.md`) is
   `open`/`wired` — a gap is only `wired` when it ships tested code.

## How to work here (agent workflow)

1. Read `research/INDEX.md` tail + `docs/TOPOLOGY.md` first — know what's
   open and where things live before touching code.
2. For a new feature: research (7-hop) → design (ADR if architectural) →
   implement (one module, one test) → verify (`make test_<name>`) → close
   the gap in INDEX.md → commit.
3. After refactors: re-verify with FRESH tool calls (clean rebuild, full
   `make test_all`) before reporting done. Triple Devil's-Advocate.

## Architecture decision records

Architectural decisions are recorded in `docs/adr/` (Nygard template,
append-only). Read them before changing anything structural. Write one when
you make a structural decision.
