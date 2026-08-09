# research/068 — Kevin-Bacon 7-hop: THE BEST 25GB OF REASONING + INTERACTION DATA (2026-08-09)

## The mandate
"Use this mindset to do a seven-step Kevin Bacon search, 25 web searches, and find us the
best data sets. We basically need 25 GB of amazing." + the standing directive: nothing
before 2025 unless proven extreme quality; 2026 frontier traces are the priority
(Fable-5, GLM-5.2, GPT-5.6 Sol/Luna/Terra, DeepSeek-V4-Pro). The data must be VERIFIED
quality (reference answers, test cases, or a grading pass), not mass-collected.

## The 7-hop chain (seed -> convergence)

| Hop | Node | What it established |
|-----|------|---------------------|
| 1 | OpenThoughts (arXiv 2506.04178) | The SOTA open reasoning DATA RECIPE: 1000+ controlled curation experiments; QwQ-32B is a STRONGER TEACHER than DeepSeek-R1 for small models; long self-reflective traces are ESSENTIAL (stripping self-reflection: 11.6K->0.3K tok crashed scores); OpenThoughts3-1.2M = 850K math + 250K code + 100K science |
| 2 | OpenThinker3-7B (trained on OT3-1.2M) | SOTA open-data 7B: 53% AIME25, 51% LCB, 54% GPQA — beats DeepSeek-R1-Distill-7B by 15-20pts. PROOF the data recipe transfers to small models (our 35M scale) |
| 3 | GLM-5.1-Reasoning-1M-Cleaned (2026-04) | 1M cleaned reasoning traces; the cleaning pipeline documented: removed incomplete/repeated/refusal/unparseable/duplicate; meta carries teacher_model + token counts; 4 subsets incl. PHD-Science + Multilingual-STEM |
| 4 | GLM-5.2-Conversation + coding traces (2026-07) | 50K High-reasoning traces (GPT-OSS-120b prompts, GLM-5.2 answers); GLM-5.2 leads DeepSWE 46.2 / Terminal-Bench 81 — the CURRENT frontier open model's traces |
| 5 | Fable-5 (Claude) traces | The frontier agentic traces: Glint-Research (verified), Crownelius Complete-FABLE.5 library (MIT, content-verified, deduplicated, trace-atlas provenance); "the trace IS the work"; Sol rows = Codex-CLI rollouts, xhigh, acceptance-test VERIFIED |
| 6 | GPT-5.6 Sol/Luna/Terra (2026-07) | The current #1 Terminal-Bench agent (89.5% TB2.1); Sol = xhigh reasoning, acceptance-test verified Codex rollouts; Crownelius library has provenance + verification metadata |
| 7 | Nemotron-3-Ultra data disclosure | The LAB'S ACTUAL RECIPE: DeepSeek-V4-Pro math/code/science synthetic traces, GLM-5 chat reasoning (best-of-4 via GenRM), math proofs PROOF-VALIDATED (820K), agentic CLI/OpenCode traces; plus **"Synthetic Hermes Agent Reasoning Traces"** — our own agent family is a frontier-lab data source. THE validation that reasoning-trace distillation is the frontier standard |

## Convergence principle
**The 2026 frontier is REASONING-TRACE DISTILLATION at scale, verified.** Every lab
(OpenThoughts/QwQ, DeepSeek-V4-Pro, GLM-5.2, Fable-5/Claude, GPT-5.6-Sol) converges on
the same recipe: take hard problems, generate LONG self-reflective traces from the best
teacher, VERIFY (reference answers / test cases / GenRM grading), keep only the verified,
clean the refusals/duplicates/incomplete. For a small model this is the SINGLE highest-
value data tier (distillation > RL at <=32B, per RC03). User interactions (WildChat-era)
are 2023-2024 — DEPRECATED by the directive; the 2026 replacement is the verified trace
libraries (Crownelius pattern) + the new GLM/DeepSeek conversation datasets.

## The 25GB selection (final, verified sizes)

### Tier A — the reasoning core (the "extreme quality" exception for 2025)
| Source | Size | Verification |
|--------|------|--------------|
| open-thoughts/OpenThoughts-114k | 3.55 GB ✅ ON DISK | DeepSeek-R1 traces; OpenThinker3 recipe basis; 86K downloads |
| open-thoughts/OpenThoughts3-1.2M (shard) | ~10 GB | QwQ-32B teacher; SOTA open-data recipe; boxed answers |
| Jackrong/GLM-5.1-Reasoning-1M-Cleaned (shard) | ~8 GB | 2026-04; cleaned (documented pipeline); teacher metadata |

### Tier B — the 2026 frontier traces (verified, small but dense)
| Source | Size | Teacher / verification |
|--------|------|------------------------|
| Crownelius/Complete-FABLE.5-traces-2M | 0.54 GB | Claude Fable-5/Opus/Sonnet; MIT; content-verified + deduped |
| Crownelius/GPT-5.6-Sol-Luna-Terra-Traces | 0.30 GB | GPT-5.6 Sol (xhigh, acceptance-test verified) |
| ianncity/GLM-5.2-Conversation | 0.49 GB | GLM-5.2 High-reasoning, 50K traces |
| greghavens/glm-5.2-coding-and-debugging-traces | 0.04 GB | GLM-5.2 coding |
| TeichAI/DeepSeek-v4-Pro-Agent | 0.28 GB | DeepSeek-V4-Pro agent rollouts |
| Jackrong/DeepSeek-V4-Distill-8000x | 0.14 GB | DeepSeek-V4 distilled traces |
| MoreThought/Fable-5-Max-Reasoning-250x | 0.04 GB | Fable-5 filtered |
| HelioAI/Fable-5-Distill-Reasoning-462x | 0.15 GB | Fable-5 full-manifold distill |

Total: ~24 GB (Tier A ~21.5 + Tier B ~2).

### DROPPED (dated or redundant)
- WildChat-1M (2024-10), Magpie-Pro (2024-08) — pre-2025, not proven extreme quality
- OpenR1-Math-220k (2025-02) — redundant with our existing cosmopedia/openmath math tier
- AM-DeepSeek-R1-Distilled-1.4M (39.6 GB) — extreme quality (verifier-graded) but over
  budget; revisit if we go to 50GB

## The wiring (how it's used)
1. Download: `tools/wubu_download_reasoning.py` (wave 1) + `wubu_download_frontier.py`
   (wave 2) + `wubu_download_bigtickets.py` (wave 3, sharded).
2. Extract: `tools/wubu_extract_reasoning.py` — schema-aware per dataset:
   - OpenThoughts/GLM-5.1: problem + [REASONING] + [ANSWER] (full CoT preserved —
     the OpenThoughts finding: stripping self-reflection destroys performance)
   - conversations: role-tagged [USER]/[ASSISTANT] turns
3. Tokenize: `wubu_tokenc` (C11 BPE, 16384 vocab) -> .tok
4. Train: the from-scratch run mixes Tier A reasoning + Tier B frontier + the existing
   cosmopedia/openmath pretrain + SFT pack.

## Sources (persistent)
- https://arxiv.org/abs/2506.04178 (OpenThoughts — the recipe paper)
- https://github.com/open-thoughts/open-thoughts (recipe + data)
- https://huggingface.co/datasets/Jackrong/GLM-5.1-Reasoning-1M-Cleaned (schema docs)
- https://huggingface.co/datasets/Crownelius/Complete-FABLE.5-traces-2M (provenance)
- https://huggingface.co/datasets/Crownelius/GPT-5.6-Sol-Luna-Terra-Traces (verification)
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 (data disclosure)
- https://huggingface.co/blog/zai-org/glm-52-blog (GLM-5.2 benchmarks)
- https://huggingface.co/datasets/ianncity/GLM-5.2-Conversation (traces)
- https://github.com/RenBing-Sumeru/Awesome-LLM-Reasoning-Data (the atlas, MIT)
