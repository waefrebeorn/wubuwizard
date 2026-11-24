# Front P — project-maya (2026-11-23)

Source from the user: `https://github.com/mw00/project-maya` release v1.1.75. Two claims
that matter here: Inkling-2-Mini drafting 140 -> ~200 tok/s on one RTX 3090 at 262,144
context, and models up to 744B parameters running "from 8 GB of VRAM".

Study clone (read-only): `C:/Users/eman5/WuBu/research/engine-sources/project-maya`,
MIT, branch `main`, commit `8c618a5994e25932ff4977498d6195f8b80f91bb` (2026-11-23, v1.1.76).
Language: CUDA (5% Python driver `maya.py`/`chat.py`, the engines are C/CUDA under `src/`).

## What it actually contains

| Area | Exact location | Mechanism |
|---|---|---|
| Inkling 2 spec | `docs/INKLING2.md` | Full written specification of the model: 48 layers, every one MoE, 40 sliding (window 1024) + 8 full, GQA 48/8 heads of 128, **no RoPE** (per-head relative bias mixed to 1,024 distance), attention sink logit, sigmoid output gate, q L2 clip / k clip 22.6, weightless RMS norm on v, 4-tap short convs on k and v, attention-residual blocks of 6 layers, 384 experts top-8, width 2304, sigmoid router + selection bias, softcapped SwiGLU (gate 4, up 25), logits softcap 30, 200,058 real rows of 201,024. Experts NVFP4 (e2m1 x e4m3 per 16) and MXFP8 on layers 40-47. |
| Inkling 2 kernels | `src/ik2/` (`ik2_kernels.cu`, `ik2_lane.cpp`, `ik2_off.cu`, `ik2_draft.cu`, `ik2_iq_tables.h`, `ik2_exact.h`, `ik2_rdna.inc`) | Hand-written CUDA kernels with a CPU parity reference (`ik2_cpu.cpp`), RDNA include, draft path, and offload. |
| Placement | `src/plan/plan_main.cpp` | A standalone planner binary: computes expert/layer placement for a VRAM budget before the run. |
| Drafting policy | `src/spec/draft_policy.cpp`, `src/spec/controller.cpp`, `src/spec/suffix_drafter.cpp` (+ `*_test.cpp` for each) | `--drafting auto`: measures no-drafting / earlier-chat text / the drafter once per machine and keeps whichever wins; disables the drafter inside a long chat when it stops winning. Each piece has a unit test beside it. |
| N-gram speculation | `src/ngram/` | Prompt-lookup style n-gram drafter, separate from the model drafter. |
| Quantized KV / quant plan | `docs/QUANT-PLAN.md`, `docs/MEMORY.md` | Budget arithmetic for weights, KV and experts under a VRAM/RAM split. |

## Transferable rules

1. **Decide placement from a measurement, not a heuristic.** Maya's `--drafting auto`
   benchmarks the candidate strategies once on the user's machine and then keeps the
   winner; `plan_main.cpp` does the same for placement. wubuwizard's `wubu_adapt_len`,
   `wubu_adaptive_cap`, `wubu_ev_depth` are all static policies with no on-machine
   measurement pass.
2. **Never let a timing-dependent choice change the tokens.** Their v1.1.75 note is
   explicit that GLM-5.3-Flash "gives the same answers on every start" because choices
   that had depended on how fast something happened to run are now measured once and
   stored (`MAYA_GLM_SAME_ANSWERS=0` restores the old behaviour). Any adaptive policy
   here must have the same property or it is a correctness bug.
3. **A written model specification before kernels.** `docs/INKLING2.md` states the
   architecture, the GGUF key names and the parity reference before `src/ik2/` exists.
   That is the same discipline as `docs/ENGINE-GAP-REGISTER.md`, applied per model.
4. **Ship the routing prior.** Release notes describe shipping "each size's expert-use
   file" so a fresh install places experts the same way from the first start. Our
   `wubu_expert_cache.c` learns at runtime and starts cold.

## Relevance to wubuwizard

- The Inkling-2 GGUF layout doc is the missing piece for `LLM_ARCH_INKLING` support:
  it names the llama.cpp conversion (ggml-org/llama.cpp PR #32187, stacked on #25731),
  the exact tensor names and which tensors stay F32.
- Maya is CUDA-only for this engine and does not target 6 GB; its "8 GB VRAM" claim is
  for the 744B-class paths with heavy host offload, not for Inkling at 262K context.
- Nothing was linked or vendored; study-only.
