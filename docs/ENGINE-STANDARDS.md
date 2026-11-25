# ENGINE-STANDARDS — what we build against, and what is banned

**Status: authoritative.** Where any other document in this repo names a different reference engine, that
document is stale and this one wins. Last set: 2026-11-25.

## 1. The standard

| role | engine | pinned | why it is the standard |
|---|---|---|---|
| **Model semantics** | **project-maya** | `v1.1.78` (`1530120`) | Ships `docs/INKLING2.md`, a 508-line spec for this exact model, plus `src/ik2/` kernels and a float64 reference validated to `max|dlogp| 4.7e-5`, KL `2.6e-12`, **top-1 100% of positions** against transformers over 256 tokens and all 48 layers. |
| **Product + gates** | **Strata** | `v0.2.21.1` (`2af30cc`) | The deepest Inkling 2 Mini integration by file count (247), and the only engine with a **per-family CI gate suite** for it: `b70_gate.sh`, `chat_check.py`, `longctx_chat.py`, `serve_chat.py`. |

If maya and Strata disagree on a formula, that is a real finding, not a menu — measure which one reproduces the
checkpoint's own outputs and record the result.

## 2. The workflow replacement: atomicchat

**atomic-engine** (`dev-latest`, `fa190d7b5`) replaces the entire llama.cpp-shaped workflow. It is not a
reference for arithmetic; it is the operational pattern:

- **SSD expert streaming** with model, state, compute and expert buffers under **one shared budget with an
  explicit runtime reserve**. That is the staging hierarchy this project wants, already built.
- **KL-divergence gates.** Its benchmark tree compares streaming against baseline by `kld` on logits with
  separate coherence/f16/nofa checks (`benchmarks/results/kv-stream-pr-evidence/`). This is the measurement
  standard for anything that changes the memory path — not "tok/s went up".
- **Provenance discipline.** Every result carries the file, the flags and the machine.

Adopt those three things; do not copy its code.

## 3. Banned

**llama.cpp is removed from this workspace and from every workflow.** The clone at
`research/engine-sources/llama.cpp` (351 MB) is deleted. It is not:

- a comparative spec (it cannot even load this checkpoint without rewriting its metadata — it wants *physical*
  `block_count` and physical-length per-layer arrays, while the file ships *logical* `block_count 48`),
- a quality bar,
- a throughput target,
- a semantics source.

The one thing it was still being consulted for — ggml quant **block formats** — moves to
**`ik_llama.cpp`** (`ad217a2`), which carries the same `ggml-common.h` / `ggml-quants.c` under active
maintenance. A format definition is a data-layout fact, not an engine opinion, and that is the only role
any llama.cpp lineage has here.

`FlexGen` was already deleted: it is a 2014 Java map generator, not an offloading engine.

## 4. What this changes in the existing documents

These documents reference llama.cpp and are **superseded on that point**; their measurements stand, their
provenance does not:

- `ENGINE-SPECS.md` — the pull inventory and the maya/Strata/atomic readings stand. Every llama.cpp citation in
  it is now historical.
- `ENGINE-SOURCE-INDEX.md` — the llama.cpp rows describe techniques worth having, but the source is gone.
  Re-source each from maya or Strata, or drop it. The tombstone row marks the block.
- `ENGINE-GAP-REGISTER.md` — EG-01/EG-02/EG-03 name llama.cpp files as their reference. Their **gates and
  implementations stand** (they are this repo's code and this repo's tests); only the "source" column is
  orphaned. Mark them `re-source pending`.
- `ENGINE-THROUGHPUT-BASELINE.md` — the "remaining 1.75x against llama.cpp" framing is void. The absolute
  numbers (tok/s, ms/token, GB/s in the FFN) stand as measurements of **this** engine. Restate any ratio
  against Strata or maya, or leave it as an absolute.
- `INKLING2-PORT.md` — records several llama.cpp-derived findings. They are kept as history and each is
  re-attributed where maya or transformers confirms it. The corrections log already shows one case where
  llama.cpp disagreed with maya and **llama.cpp was wrong for this model**: the loop message gate.

## 5. Standing rule

Anything that only llama.cpp could answer, by being deleted, is now answered by maya, Strata, atomic-engine, or
the model's own `transformers` implementation. If none of them answers it, it is an open question to measure —
not a reason to bring the clone back.

Agents working in this repo: **do not re-clone llama.cpp, do not cite it, do not benchmark against it.**

## 6. Building maya without fetching llama.cpp

Maya's CPU path declares its ggml as a `FetchContent` clone of llama.cpp upstream:

```cmake
FetchContent_Declare(strata_llamacpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d)
```

That fetch fails in this environment (`CMake step for strata_llamacpp failed: 1`) and, more importantly, it
would pull the banned engine back onto the box. Point it at the local `ik_llama.cpp` instead:

```
cmake -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=C:/Users/eman5/WuBu/research/engine-sources/ik_llama.cpp
```

**Measured 2026-11-25: `configure exit=0`, `NINJA_OK`.** So the standard engine builds against the ggml we
keep for quant formats, with no llama.cpp checkout anywhere. `-DMAYA_GGML_DIR=<...>` is the other documented
knob for the same purpose.

Two things that made this fail before were both mine, not maya's:

- the clone was a **sparse checkout** (1,358 absent files, `git status` clean), and
- the default fetch has no network path here.

## 7. Maya's ggml dependency is a real constraint, not a build hiccup

Three attempts to build maya on this box, three environmental failures, each distinct:

| # | failure | cause |
|---|---|---|
| 1 | `CMake step for strata_llamacpp failed: 1` | sparse checkout (1,358 files absent) + no network fetch path |
| 2 | `cmake/maya_old_names.cmake` not found | same sparse checkout |
| 3 | `native_expert.cpp:12: ggml.h` → then `ggml-cpu.h` not found | see below |

**The third is not environmental in the same way.** `third_party/ggml/` in maya holds exactly six tracked files —
`ggml-common.h`, `LICENSE`, `VERSION.txt`, and three patches. It vendors the *block layouts and codebook grids*,
not the public headers. `ggml.h` and `ggml-cpu.h` are expected from a ggml source tree, and:

- pointing at `ik_llama.cpp` satisfies `ggml.h` (`ggml/include/ggml.h`) but **not** `ggml-cpu.h`, which does not
  exist anywhere in that fork;
- `native_expert.cpp` needs it because its whole purpose is "the activation quantizers and the row dot products
  are ggml-cpu's" — i.e. maya's GGUF-form IQ experts *are* ggml-cpu's code, by design.

So the honest statement is: **maya's GGUF expert path depends on upstream ggml headers that `ik_llama.cpp` does
not carry.** `MAYA_NATIVE_EXPERTS=OFF` (CMakeLists.txt:1637) removes that path and is the only route to a maya
build with no upstream checkout — accepted only if the resulting binary can still run a GGUF IQ2 checkpoint,
which is the thing being tested.

**Resolution (2026-11-25): pulling fresh upstream is allowed.** A current llama.cpp checkout may exist as a
*build dependency* of the standard if it is fresh to date - it is not a reference, not a bar, and nothing from
it is read or copied into this repo. The distinction that matters:

- **banned**: consulting llama.cpp to decide what this model's arithmetic *is*, or benchmarking against it;
- **allowed**: maya's own build needing `ggml/include/ggml.h` and `ggml/include/ggml-cpu.h`.

A stale copy is worse than none - the deleted one was pinned at b12723 and its `inkling2` implementation
disagreed with maya on the loop gate. So the dependency is cloned fresh (`git clone --depth 1`) into
`research/upstream-ggml`, verified at clone time (`upstream-ggml/.git` present, commit date recorded), and its
path passed to maya as:

```
-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=research/upstream-ggml
-DCMAKE_CXX_FLAGS="-I<upstream>/ggml/include -D_WIN32_WINNT=0x0A00"
```

Freshness is part of the gate: record the commit hash with any oracle number taken from a maya build, exactly
as atomic-engine records provenance for its KL runs.
