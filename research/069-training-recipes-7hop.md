# research/069 — Kevin-Bacon 7-hop: THE 2026 TRAINING RECIPES (Nemotron-led) (2026-08-09)

## The mandate
"Find the best sorted training recipes — we are doing a focus training for our AGI
operating system and Nemotron will be extremely helpful. Do a seven-step research of
training recipes and use another 10 web searches."

## The 7-hop chain (seed -> convergence)

| Hop | Node | What it established |
|-----|------|---------------------|
| 1 | Nemotron-3-Ultra technical report (arXiv 2606.15007) | The FULL post-training pipeline: Base -> SFT -> RLVR -> MOPD warmup -> MOPD (x2 iterations) -> MTP boosting. MOPD = Multi-teacher On-Policy Distillation: >10 domain-specialist teachers, student rollouts get dense teacher rewards, asynchronous pipelining. Recovery rates: Terminal-Bench 172.7%, GDPVal 86.4%, SWE-bench 88.1% of teacher |
| 2 | Nemotron-RL-Ultra-Training-Blends (HF) | THE OPEN RECIPE DATA: exact RLVR + MOPD blend ratios (agentic tool-use 20.4%, SWE-rebench 14.1%, instruction-following 12%, math 10.9%, coding 8.1%, ARC-AGI 4.2%...). Curriculum: samples ordered easier->harder (pass-rate ordering) |
| 3 | Nemotron 3 Nano 4B (arXiv 2512.20848) | The SMALL-model recipe: 63B tokens, 8K context, data blend = ~70% post-training data + 30% pretraining. Nemotron Nano 4B = the small-model distillation recipe (our scale class) |
| 4 | GLM-5 (arXiv 2602.15763) | The optimizer + schedule confirmation: **Muon Split** (per-head scale adaptation — the projection weights for different attention heads update at different scales), cosine decay, lr warmup 0->2e-4 -> 4e-5 (pretrain), linear 4e-5->1e-5 (mid-train). Post-train: SFT -> Reasoning RL -> Agentic RL -> General RL. GRPO with beta=2, eps_low=0.2, eps_high=0.28, group-normalized advantage |
| 5 | GLM-5.2 (Z.ai blog) | The agentic-RL scale-up: slime framework = parallel OPD merging 10+ expert models; long-horizon anti-hacking (compaction splits, variable-length traces); 99.2% AIME 2026 |
| 6 | DeepSeek-V4 (Fireworks + On-Policy Distill survey 2604.00626) | Pure multi-teacher OPD replaces the mixed RL stage: 10+ experts (math/coding/agent/instruction-following) consolidated into 1.6T params via full-vocab reverse-KL. Caches teacher last-layer hidden states, reconstructs logits on the fly. THREE reasoning modes from same weights (non-think / think-high / think-max) — reasoning effort is TRAINED BEHAVIOR |
| 7 | Interconnects frontier recipe review (2026) | The meta-convergence: "Curated prompts -> SFT -> DPO -> RLVR" is the OLD recipe; 2026 = **SFT -> ~6-10 domain-specialist teachers -> MOPD into one student**, run over 2 iterations with refreshed teachers. DeepSeek-V4 + Nemotron-3-Ultra are the two canonical executions. SmolLM3 adds the mid-training stage (175B reasoning tokens between pretrain and SFT) |

## Convergence principle
**The 2026 post-training recipe is: (1) pretrain -> (2) REASONING MID-TRAIN on distilled
traces (SmolLM3: 175B tokens) -> (3) SFT cold-start -> (4) RLVR (verifiable rewards) ->
(5) train domain-specialist teachers -> (6) MOPD consolidate into one student (x2
iterations) -> (7) MTP boosting. Muon is the optimizer of the frontier (Kimi K2, GLM-5,
Moonlight — exactly our Muon/NS5 choice). Reasoning effort is trained behavior (3 modes
from one weight set). Small models: Nemotron Nano 4B = 63B tokens, 70/30 post/pretrain
blend.**

## The AGI-OS focus training recipe (our scale, from the convergence)

The WuBu-35M/S7 focus training on the AGI-OS data (the 25GB just gathered):

| Stage | Data | Recipe knob |
|-------|------|-------------|
| 1. Pretrain (extend) | cosmopedia + finemath + openmath (existing ~6.7B tok) | Muon NS5, cosine, 20 tok/param (Chinchilla); progressive growth OK (FLM-101B: 80% perf @ 10% FLOPs — the amoeba) |
| 2. Reasoning mid-train | OpenThoughts-114k (3.4GB on disk) + OpenThoughts3-1.2M shard (10GB) + GLM-5.1 shard (8GB) | THE new tier; full CoT preserved (self-reflection is ESSENTIAL per OpenThoughts Appx H); lr mid-train linear decay (GLM-5: 4e-5->1e-5) |
| 3. SFT cold-start | GLM-5.2-Conversation (50K), GPT-5.6-Sol-Luna-Terra (15K), Fable-5 (700), DeepSeek-v4-Pro-Agent (155MB), agentic pack | LOW lr (1e-5, Llama3/Moonlight finding); role-tagged |
| 4. RLVR | Nemotron-RL blends (open!) + our verifiable oracles | GRPO beta=2 eps 0.2/0.28 (GLM-5 exact); distillation > RL at <=32B (RC03) — the RLVR is for the agentic-OS tasks with verifiable rewards |
| 5. MOPD (future, at bigger scale) | specialist teachers per domain | DeepSeek-V4-style reverse-KL; 2 iterations |

Nemotron's open RL blends + the 25GB trace tier ARE the data backbone; our Muon NS5
backprop is the optimizer the frontier converged on.

## The AGI-OS angle (the focus)
The OS-backbone role data (agentic, tool-use, syscall traces) is exactly what
Nemotron-RL-Agentic-Conversational-Tool-Use + SWE-rebench + the tau-bench lineage
provide — the 20.4%+14.1% top of the RLVR blend. Our own ancient-subsystem pack
(168 syscall Q/A) + agentic pack (1392 convos) + the new deepseek-v4-pro-agent traces
(155MB) mirror that role at 35M scale.

## The non-Euclidean optimizer split (the user's research, 2026-08-09)
The frontier recipe's Muon/NS5 orthogonalizes momentum in EUCLIDEAN space (Newton-
Schulz on the matrix's flat Gram structure). Our body is NOT flat: the amoeba's
hyperbolic cells (Poincaré GQA, Möbius layers, hyperbolic SSM blocks, hyperbolic
output proj — src/wubu_mobius*, wubu_poincare_gqa, wubu_moe_hyperbolic, wubu_hyper)
live on nested spheres where Euclidean orthogonalization is geometrically wrong
(the geodesics curve; a flat-space step leaves the manifold).

The user's research: **the non-Euclidean components take a NON-Muon optimization path**:
- RSGD (src/rsgd.c — Riemannian SGD: retraction-based step on the Poincaré ball,
  `rsgd_step(w, dw, n_vecs, dim, lr, R, clip)` — radius-aware, clipped)
- the qlearner (src/qlearner.c — Q-learning the learning RATE from the loss signal,
  `qlearner_step(ql, loss)` — the adaptive metabolism for the hyperbolic cells)
- PID + TGT optimization (the user's own researched family — proportional-integral-
  derivative control of the update trajectory; TGT = the target-state optimization)

**KEY INSIGHT (user, 2026-08-09): "these optimizers work for you in EUCLIDEAN space
too, if you use the OPTIMIZED SGD."** The RSGD/qlearner/PID/TGT family is not
hyperbolic-only — with the optimized-SGD form (the retraction becomes ordinary
gradient descent with the curvature-aware radius R -> inf), it is a general
non-Muon alternative everywhere. Muon remains the flat-matrix champ; the TGT path is
the ravine navigator for BOTH spaces.

### The TGT odometer (the gravitational-pole physics — the AGI design)
"TGT is designed as an ODOMETER to help you get INTO very large ravines and get OUT
of it — a gravitational pole physics system."

- **Odometer**: TGT tracks the optimizer's position along its trajectory (distance
  traveled, direction, curvature) — not just the current gradient. It KNOWS where the
  update has been, so it can tell "entering a ravine" (curvature rising, gradient
  aligning) from "stuck in one" (velocity ~0 in a steep basin).
- **Gravitational poles**: the loss landscape's deep basins act as mass centers. TGT
  routes the update by POLAR coordinates around them — the existing wubu_gravity_t
  (wubu_gravity_route(g, r_in, theta_in), grow/shrink cells) + wubu_orbits_t
  (nested orbit write/read) ARE this machinery: cells live at (r, theta) around the
  poles, the odometer integrates the trajectory in that polar frame.
- **INTO a ravine**: when the pole's pull dominates (r shrinking toward the basin
  center), TGT lets the update FALL — the momentum aligns with the ravine's axis
  (the gravitational gradient descent into the valley).
- **OUT of a ravine**: when the odometer detects the trap (loss flat but curvature
  high — a saddle/ravine floor), the pole physics EJECTS: the update gets a
  repulsive kick off the ravine axis (the anti-gravity escape), preventing the
  plateau/dead-ravine stall.
- **The AGI link**: this IS the amoeba metabolism — the loss landscape's ravines are
  the spheres' basins; gravity routes cells (which sphere absorbs the next token's
  mass), the odometer tells the colony whether to keep digging (absorb) or leave
  (prune/reroute). The optimizer and the routing share the polar frame.

STATUS: rsgd_step + qlearner_step are ONLY wired in the legacy trainers
(tools/train_integrated.c, tools/train_gpu.c). The current wubu_backprop.c Muon/AdamW
path gives EVERYTHING Euclidean Muon — correct today only because the S7 dense body is
flat (no hyperbolic modules in the main forward yet). The moment the hyperbolic cells
wire into the body, the optimizer must SPLIT: Muon for the Euclidean matrices, the
RSGD/PID/TGT path for the non-Euclidean params. That split is the amoeba's dual
metabolism — the flat skeleton Muons, the curved cells roll.

## Sources (persistent)
- arXiv 2606.15007 (Nemotron 3 Ultra technical report)
- https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends (the blends)
- arXiv 2512.20848 (Nemotron 3 Nano)
- arXiv 2602.15763 (GLM-5 — Muon Split, GRPO, lr schedule)
- https://z.ai/blog/glm-5.2 (slime OPD, anti-hacking RL)
- arXiv 2604.00626 (On-Policy Distillation survey — DeepSeek-V4 details)
- https://www.interconnects.ai/p/frontier-post-training-recipe-review (the meta-review)
- https://huggingface.co/blog/smollm3 (SmolLM3 mid-training recipe)
- https://huggingface.co/blog/nvidia/nemotron-3-nano-4b (Nano 4B 63B-token recipe)
