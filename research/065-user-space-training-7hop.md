# research/065 — THE USER SPACE IS THE TRAINING DATA: 7-hop on making it the best

> 2026-08-08. The user's directive: "seven steps online research how to
> make this the best — we don't just make it, we make it the best."
> Topic: the user-files-as-namespace live-training system (wubu_userfs —
> regular user files interpreted inside the KV-FS space; every user
> trains at every time; no data gathering needed).
> Sources archived in ~/.hermes/profiles/mind-palace/cache/web/.

## The one-line answer

The frontier VALIDATES the design and names the three upgrades that make
it the best: (1) Letta proved a plain filesystem beats specialized
memory tools (74.0% vs 68.5% on LoCoMo) — our KV-FS is the right
substrate, keep the tools simple; (2) content-aware chunking doubles
retrieval quality vs naive fixed-size splitting (nDCG@5 0.459 vs 0.244);
(3) implicit user behavior (mouse/eyes/edits/opens) is a HIGHER-VALUE
training signal than explicit labels (reward 55%→64%, ~3× DPO gain), and
replay must follow the forgetting curve, not raw steps.

## STEP 1 — Letta: "Is a Filesystem All You Need?" (Aug 2025)

Letta/MemGPT's own team benchmarked memory tools on LoCoMo. The result
is a direct validation of our KV-FS doctrine:

- A simple agent with ONLY filesystem tools (grep, search_files, open,
  close) + semantic search over files **hits 74.0%** on LoCoMo with
  GPT-4o-mini and minimal prompt tuning — **above Mem0's 68.5%** for
  their top-performing graph variant.
- Why: agents are extremely good at using filesystem tools (they're in
  the training data); they generate their OWN queries and iterate until
  they find the data. "Specialized memory tools... are less effective
  than simply allowing the agent to autonomously search through data
  with iterative querying."
- The lesson for us: **the namespace + simple tools beats fancy
  memory.** Our KV-FS (path-addressable, resolve-once handles) IS the
  right substrate; the body (agent) iterates its own queries over
  /kv/user/* — no vector DB required for the agentic layer.
- Also: "Agent capabilities matter more than the tools" — the retriever
  (our physics router / ecosystem) must be trainable, not fixed.

## STEP 2 — FOREVER: forgetting-curve memory replay (arXiv:2601.03938)

"Every user trains at every time" has a forgetting problem: sequential
ingestion without catastrophic forgetting. FOREVER's findings:

- LLM forgetting **mirrors the Ebbinghaus human forgetting curve**.
- **Model time ≠ training steps**: identical steps produce varying
  parameter change. FOREVER defines model time by the MAGNITUDE OF
  OPTIMIZER UPDATES (r = μ/μ₀), aligning replay intervals with the
  model's internal evolution.
- **When to replay**: forgetting-curve intervals (Ebbinghaus-shaped,
  model-time based). **How to replay**: intensity-aware regularization
  (β scaled by the update-magnitude ratio).
- Proven on 0.6B–13B across three CL benchmarks. Consistent forgetting
  mitigation.
- For us: the replay scheduler belongs in wubu_amoeba's feed loop —
  replay /kv/user data on the forgetting curve, measured in update
  magnitude, not epochs. (Our amoeba already has diagnose→mutate;
  this gives it the WHEN and HOW of replay.)

## STEP 3 — Implicit feedback beats explicit labels (arXiv:2606.20482)

The "every user trains at every time" doctrine, proven:

- IFllm: 1,336 multi-turn questions, 59 workers, **mouse trajectories +
  eye-gaze** recorded against LLM responses.
- A reward model trained on implicit behavior **boosts the text-based
  reward model from 55% → 64% accuracy** and **nearly triples the
  relative response-quality improvement after DPO** across 8 LLMs.
- Users rarely give explicit feedback; implicit signals are "vital to
  the economic moats of Internet giants."
- For us: a user opening, editing, reading, or re-visiting a file is
  implicit preference — the training signal is already in the filesystem
  (access times, edit patterns, dwell). The userfs should RECORD usage
  (open/edit/read events) as preference annotations, and the amoeba's
  RLHF loop (RC03) consumes them instead of requiring explicit labels.

## STEP 4 — Chunking: content-aware doubles retrieval (arXiv:2603.06976)

The largest cross-domain chunking survey (36 strategies × 6 domains × 5
embedding models):

- **Content-aware chunking significantly beats naive fixed-length**
  splitting. Top: **Paragraph Group Chunking** — mean nDCG@5 ≈ 0.459,
  Precision@1 ≈ 24%, Hit@5 ≈ 59%.
- Naive fixed-size character chunking: nDCG@5 < 0.244, Precision@1 ≈
  2-3% — **~2× worse on every metric**.
- Domain differences: dynamic token sizing strongest in biology/physics/
  health; paragraph grouping strongest in legal/maths.
- More, smaller chunks → bigger index + latency; dynamic chunking
  approaches the best effectiveness/efficiency balance.
- For us: wubu_userfs ingests whole files as ONE embedding today — the
  #1 upgrade is **paragraph-group chunking**: a .md/.txt file becomes
  N paragraph-group embedding files (idea.md.emb.p1, .p2, ...). The
  INDEX ledger tracks chunks; retrieval targets the chunk, not the file.

## STEP 5 — Implicit feedback from usage (Cornell Coactive Learning)

Coactive learning (Tucker et al.): the user's natural interaction —
accepting, editing, or correcting an output — IS the feedback; the model
learns from the user's continued usage rather than explicit ratings.
Fits the "every user trains at every time" economy: no annotation cost,
no data gathering, the signal is the behavior.

## STEP 6 — Personalization from personal corpora

The practical pattern across the personal-LLM literature: user files are
already well-structured (documents, code, notes) — the win is not
re-training but (a) indexed retrieval (our namespace), (b) incremental
adaptation with replay (STEP 2), (c) fine-tune on the user's OWN files
with the user's implicit preferences as the reward (STEPS 3+5). The
three steps compound: the namespace stores, the replay schedules, the
implicit signal rewards.

## STEP 7 — Federated / on-device: the data stays home

Apple + Google on-device personalization + federated LLM surveys:
personalization succeeds when the data does NOT leave the device. The
user's files live in THEIR namespace; the model adapts locally; only
gradient/reward summaries (or nothing) ever leave. This is the
"removes the headache of gathering training data" endgame: **data
ownership and training signal are the same thing** — the user's own
files, interpreted where they live.

## The convergence → what "the best" means for wubu_userfs

| # | Finding | Implementation |
|---|---|---|
| 1 | Filesystem beats memory tools (74% vs 68.5%) | ✅ already the architecture — keep tools simple (grep/search/open semantics on /kv/user/*) |
| 2 | Forgetting-curve replay (FOREVER) | **NEW**: replay scheduler in the feed loop — model-time (update magnitude), Ebbinghaus intervals |
| 3 | Implicit feedback > explicit (55%→64%) | **NEW**: record open/edit/read as preference; RLHF consumes usage, not labels |
| 4 | Paragraph-group chunking (0.459 vs 0.244) | **NEW, #1 priority**: file → N paragraph-group chunk embeddings |
| 5 | Coactive learning | folds into 3 (edits ARE feedback) |
| 6 | Personal corpora + replay | folds into 2 |
| 7 | On-device, data stays home | ✅ architectural by design (the namespace is local) |

### The build list (when the user says go)

1. **Chunking (4)**: wubu_userfs parses text into paragraph groups
   (blank-line separated, capped ~512 ids); each chunk is its own
   embedding file `/kv/user/emb/<name>.emb.p<k>`; INDEX tracks chunks.
2. **Usage ledger (3)**: `/kv/user/meta/<name>` records open/edit/read
   events with timestamps — the implicit reward stream for RLHF.
3. **Replay scheduler (2)**: model-time (update-magnitude) Ebbinghaus
   intervals feeding wubu_amoeba's diagnose loop.
4. **The gate**: test_userfs extended — a .md file with 3 paragraphs
   yields 3 distinct chunk embeddings; editing the file re-ingests only
   the changed chunks (size-based); usage events are recorded.

## Triple-DA (the honest audit)

- **DA1 (filesystem is enough?)**: Letta's win used an LLM that ALREADY
  knows filesystem tools from training. Our body (WuBu1, 56M) has no
  such priors — the router must be trained to search. Mitigation: the
  physics router IS trainable; the ecosystem learns the retrieval
  policy. The substrate is right; the retriever needs training.
- **DA2 (chunking cost)**: more chunks = more KV blocks = more memory.
  The chunk-count is a scale-to-fit parameter (wubu_scale already
  budgets bytes/token); chunk size adapts to the budget, not the other
  way. Mitigation: dynamic chunk sizing (the survey's efficiency
  winner) + the KV-FS tiering already handles the index growth.
- **DA3 (implicit signal noise)**: opens/edits are weak, noisy signals
  (a file open ≠ a preference). Mitigation: IFllm showed even weak
  implicit signals beat nothing by a wide margin; usage events are
  WEIGHTED by recency + dwell + edit magnitude, and the density gate
  (coherence/n_tokens) decides what actually gets absorbed.

## Sources

- Letta: "Benchmarking AI Agent Memory: Is a Filesystem All You Need?"
  (Aug 12 2025, letta.com) — 74.0% vs Mem0 68.5% on LoCoMo.
- FOREVER: Forgetting Curve-Inspired Memory Replay (arXiv:2601.03938,
  Jan 2026) — model-time replay, Ebbinghaus intervals, 0.6B-13B.
- IFllm: "Your Mouse and Eyes Secretly Leak Your Preference" —
  LLM Alignment using Implicit Feedback (arXiv:2606.20482, Jun 2026) —
  reward 55%→64%, ~3× DPO gain.
- Chunking survey: "A Systematic Investigation of Document Chunking
  Strategies and Embedding Sensitivity" (arXiv:2603.06976, Mar 2026) —
  Paragraph Group Chunking nDCG@5 0.459 vs fixed <0.244.
- Coactive Learning (Tucker et al., Cornell) — edits as feedback.
- Federated LLMs (arXiv:2409.15723) + Apple/Google on-device
  personalization — data stays home.
