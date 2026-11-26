# D — Is 1M context the right goal? The premise, challenged

Status: devil's-advocate research pass, 2026-11-26. Every number carries a ledger id
(`[nnn]` → `docs/research/LEDGER.md`). Local measurements are marked **measured**;
arithmetic on top of them is marked **derived**; everything else is cited.

Companion to `INKLING2-REALITY-CHECK.md` (throughput) and `INKLING2-KV-CACHE.md` (cache layout).
This document does not re-derive those. It asks a different question: **should 1M be the target at all?**

---

## 0. Verdict first

**The premise is wrong, and wrong in a way that has been measurably wrong since 2024.**

Not because 1M is impossible to *allocate* — it is 12.65 GB of KV here, which is fine — but because
the quantity being optimised, "tokens the model will accept", is not the quantity that produces utility.
The quantity that produces utility is "tokens the model can still *use*", and that number is:

- **1K–16K** for most models on non-literal association tasks (NoLiMa, 85%-of-base criterion) [123][124];
- **16–26%** of a 1M window for the best 2026 frontier models on 8-needle disambiguation [160];
- **~55–80%** of a 1M window for the best models on *literal* retrieval, which is the easy case [122][146].

Three independent lines of evidence say the same thing from different directions:

1. **Capability**: claimed length and effective length are different numbers, and the gap is 10x–1000x,
   not 20% [121][123][136]. The most quoted single fact in this literature is that Llama 4 Scout
   advertises 10M and clears the 85% bar at **1K** [124].
2. **Utility**: adding context past a task-dependent knee makes answers *worse*, not just slower —
   including when retrieval is perfect and the irrelevant tokens are masked out [135].
3. **Cost**: for a dense 12B, attention FLOPs exceed the entire rest of the model at **n ≈ 24.4k tokens**
   (**derived**), and for this hybrid 48-layer model at **n ≈ 176k** (**derived**). Past that point every
   extra token costs more than the last and buys less.

So the honest target is **not 1M**. It is: *make 16–32K genuinely fast and genuinely usable, and put the
effort that 1M would have consumed into exact retrieval plus compaction.* 1M is defensible as a **ceiling
that never crashes**; it is indefensible as a **requirement that must work**.

The rest of this document is the evidence, including the parts that cut against that verdict.

---

## 1. What the numbers actually are, 2024 → 2026

### 1.1 The claimed-vs-effective record

| model | claimed | effective | criterion | source |
|---|---|---|---|---|
| LWM 7B | 1M | **<4K** | RULER, 85.6% floor | [121] |
| ChatGLM 6B | 128K | **4K** | RULER | [121] |
| GPT-4 (1106) | 128K | **64K** | RULER | [121] |
| Llama 4 Scout | **10M** | **1K** | NoLiMa, 85% of base | [124] |
| Gemini 2.0 Flash | 1M | **4K** | NoLiMa | [124] |
| Gemini 2.5 Flash | 1M | **2K** | NoLiMa | [124] |
| GPT-4o | 128K | **8K** | NoLiMa | [123] |
| GPT-4.1 | 1M | **16K** | NoLiMa | [124] |
| Gemma 3 12B | 128K | **1K** | NoLiMa | [123] |
| Claude 3.5 Sonnet | 200K | **4K** | NoLiMa | [123] |
| Qwen2.5-14B-Instruct-1M | 1M | **>128K** | RULER (synthetic) | [122] |
| Qwen3-14B | 128K | **>128K** | RULER (synthetic) | [122] |
| InternLM2.5-7B | 1M | **4K** | RULER | [122] |

Read the last four rows against the eight above them. **The same 1M claim maps to effective lengths
spanning 1K to >128K depending on the benchmark.** That is the central methodological fact of this
literature and it should make anyone suspicious of a single headline number — in either direction.

### 1.2 The 5x methodology shock

The cleanest demonstration that "1M works" is a property of the test, not the model:

> Gemini 2.5 Pro, 1M tokens, MRCR:
> **82.9%** with 1 needle (old methodology) → **16.4%** with 8 needles (current methodology). [129][130]

Same model, same length, same family of task. A 5x swing from *how many similar things you ask it to
disambiguate*. Any claim of the form "model X handles 1M" is a claim about a specific needle count, and
vendors choose the needle count. Google's own model card says so explicitly: they moved to 8-needle
"as we have decided to focus on a harder version" [130].

### 1.3 2026: the frontier is better, and still not at 1M

| model | 128K | 1M | ratio | source |
|---|---|---|---|---|
| Claude Opus 4.6 | — | **76.0%** | — | [160] |
| Claude Sonnet 4.6 | — | 65.8% | — | [160] |
| GPT-5.4 | — | 36.6% | — | [160] |
| Gemini 3 Flash | — | 32.6% | — | [160] |
| Gemini 3 Pro | **77.0%** | **26.3%** | **2.9x fall** | [160] |
| Gemini 2.5 Pro | 58.0% | 16.4% | 3.5x fall | [129] |

Two things to take from this. First, a genuine and large improvement: 76% at 1M in 2026 vs 16.4% in
mid-2025 is a 4.6x gain in two releases. Second, **even the best model in the world loses a quarter of
its 1M-window capability**, and the mid-tier 2026 models sit at 25–37% — i.e. a 1M-token prompt to
GPT-5.4 gets you a *worse* answer than a 32K prompt to the same model would.

The trend is real. The conclusion "therefore build for 1M" does not follow, because the same releases
also improved 32K. The *ratio* is what matters for a build decision, and the ratio has not moved much.

### 1.4 RULER is saturating; that is not good news

RULER's live leaderboard now shows Qwen3-14B at 85.1% at 128K and Qwen2.5-14B-1M at 92.2% at 128K [122].
On synthetic retrieval, a 14B model from 2025 clears the bar a 2024 405B model (Llama 3.1, 66.6% at 128K)
did not [121][122]. Taken alone this says "long context is solved at 128K for mid-size models".

NoLiMa says the same model class falls to 7–40% of baseline at 32K when the needle and question share no
literal tokens [123][124]. Both are true. **Synthetic literal retrieval has been solved; semantic
association under load has not.** A benchmark suite that only measures the first will keep reporting
progress that users cannot feel.

---

## 2. The utility curve — what actually degrades and by how much

### 2.1 It degrades on every model, on every task, from the first token

Chroma, 18 models, 194,480 calls, task complexity held fixed and *only* input length varied [126]:

- performance declines with length for **all 18 models** on **all** tasks, including copying a list of
  repeated words;
- the effect is not a cliff at the window edge but a continuous slope;
- lower needle-question similarity → faster decay;
- **a single distractor** degrades accuracy; four compound it; and models diverge in failure mode
  (Claude abstains, GPT hallucinates);
- counterintuitively, a *shuffled* incoherent haystack beat a coherent one on all 18 models.

The conversational-memory experiment is the one to keep, because it is the shape of every "just put the
whole history in" design:

| condition | tokens | result |
|---|---|---|
| focused (only the relevant turns) | **≈300** | all models near-ceiling |
| full LongMemEval history | **≈113k** | every model materially worse [126] |

113k is ~11% of a 1M window. The models were nowhere near their limit and still could not reliably find
the relevant turn. 306 prompts, three question families (knowledge update, temporal reasoning,
multi-session) [126][157].

### 2.2 Length alone hurts, with retrieval held perfect

This is the result that kills the "just retrieve better" defence. Du et al. (EMNLP 2025 Findings)
gave 5 models the *complete* relevant evidence — retrieval accuracy 100% by construction — and varied
only how much padding surrounded it [135]:

- degradation **13.9% – 85%** across math, QA and code, all *within* claimed window;
- the drop persists when irrelevant tokens are replaced with **whitespace**;
- it persists when irrelevant tokens are **masked out entirely** and the model is forced to attend only
  to relevant ones;
- it persists when all evidence is placed **immediately before the question**;
- mitigation that works: make the model **recite the retrieved evidence** before answering, i.e. convert
  a long-context task into a short-context one. Up to +4% on RULER for GPT-4o on an already strong baseline.

If length hurts even when attention is masked to the right tokens, then no amount of better retrieval,
better KV compression, or better position encoding fully fixes it. **The fix is to not have the tokens there.**

### 2.3 Where the knee is, per task family

| task family | knee | evidence |
|---|---|---|
| literal single-needle retrieval | ~128K–1M on frontier, ~32–64K on 7–14B | RULER [122]; LongCodeQA GPT-4.1 80.0% at 1M [146] |
| non-literal association (NoLiMa) | **1K–16K**, 2K typical | [123][124] |
| multi-needle disambiguation (MRCR-8) | **128K**, and only for 2026 frontier | [129][160] |
| multi-hop / aggregation | 128K → 1M cliff, 61.7%→19.0% on graph BFS | [160], Graphwalks via [160] |
| whole-context reasoning (LongBench v2 "long" bucket) | not the worst bucket — 71.0 vs 56.1 medium for Gemini 2.5 Pro | [128] |
| repository-scale *comprehension* (MCQ) | 512K–1M for strong models, **512K** for 14B | [146] |
| repository-scale *repair* | **broken at every length**: 0–2% accuracy | [146] |
| conversational memory | ~300 tokens of focus beats 113K | [126] |

That table is the whole argument in one object. **The knee is task-dependent by two orders of magnitude,
and the tasks a hobbyist actually runs sit at the low end.**

Two readings deserve emphasis because they cut against the doom:

- **LongBench v2's "long" bucket (128K–2M words) is not the worst bucket** for Gemini 2.5 Pro: 71.0 vs
  56.1 on the medium bucket [128]. Tasks that genuinely need the whole document are often *easier* at
  length than tasks that need to find one thing among many. This is a real counter-example to a naive
  "longer is always worse" story.
- **LongCodeQA improves with length for strong models**: GPT-4.1 72.6% at 32K → 80.0% at 1M [146]. More
  repository context genuinely helps a model that can use it. The catch is the same table: Qwen2.5-14B
  goes 70.2% at 512K → **40.0% at 1M**. The capability is model-size-gated, and 12–14B is on the wrong
  side of the gate.

### 2.4 Lost in the middle, still the baseline

Liu et al., TACL 2024, remains the reference point and has not been overturned [134]:

- GPT-3.5-Turbo, 20 documents: **75.8%** (answer first) / 57.2 / **53.8** (10th) / 55.4 / **63.2** (last);
- closed-book (no documents at all) **56.1%**, oracle **88.3%**;
- so the middle of the context is **worse than no context**, a 22.0-point swing purely from position;
- 50 retrieved documents instead of 20 buys **+1.5%**.

Everything since is a refinement of this: 2025–2026 models have flattened the U somewhat, but Chroma
found position effects persist and NoLiMa found the decay is *similarity-gated* rather than
position-gated [125][126].

---

## 3. Cost: what the last 900k tokens actually cost

### 3.1 FLOPs — the crossover is closer than anyone assumes

Attention FLOPs per token scale as `4·n·d·L_full`; weight FLOPs are constant at `2·N_active`.
Setting them equal gives the context length at which **attention costs more than the entire model**:

| model class | layers | d | crossover | at 128K | at 1M |
|---|---|---|---|---|---|
| dense 12B | 48 | 5120 | **24,414** | 5.37x weights | 42x |
| Inkling-2-Mini (8 of 72 visits full) | 8 full | 2304 | **176,324** | 0.74x | 5.95x |

**Derived** from [150]'s `F_a(n)=2n²dh_q` halved for causality, against 13.0 GFLOP/token for this model
(measured MACs in `INKLING2-REALITY-CHECK.md` §2).

Two consequences:

1. For a **dense** 12B, every token past ~24k is more expensive in attention than in the FFN and every
   weight in the model combined. "Context is cheap because the weights are small" is false in exactly
   the regime people want context.
2. **This architecture is unusually good at long context** — the 8/72 hybrid pushes the crossover to
   176k, 7.2x further out than a dense 12B. If any 12B-class model has a defensible 1M, it's this shape.
   That is the strongest argument *for* the premise in this whole document, and it still loses to §3.3.

Length is not free even before inference: at a fixed pretraining token budget, models trained with longer
windows score *worse* on average downstream, and scheduling short→long recovers up to 3.7% while training
22% faster [156]. The industry's own training recipes treat window size as a cost centre, not a free axis.

### 3.2 Wall clock — measured, across the literature

| setup | length | cost | source |
|---|---|---|---|
| Llama-3-8B, no KV cache, A100-class | 128K | 32.8 s **per token** | [151] |
| " | 256K | 111 s/token | [151] |
| " | 512K | 465 s/token | [151] |
| " | 1M | **1,765 s/token**, >96% attention | [151] |
| ChatGLM3-6B, A100 | 1M | attention alone **1,555 s**, >90% of TTFT | [153] |
| Qwen3-4B, H20 | 256K | attention 57.76 s = 89.5% of TTFT | [152] |
| " | 1M | attention **>1 h** = 98.6% of latency | [152] |
| Llama3-405B, **128 H100** | 1M | 77 s total prefill | [154] |
| Llama-4 class, **1 H100 host** | 1M | <60 s (CP + fast attention) | [155] |

Note the ratio between rows 4 and 7: **1,765 s/token on one 8B model vs 77 s for the whole 1M prefill of
a 405B model.** Same token count. The difference is 128 GPUs, context parallelism, and a KV-cache-free
kernel. The "1M is here" story is a story about the datacentre.

Also note the architectural divergence: Epoch measured TTFT for GPT-5.6 as **quadratic** (0.071 s at
0.1M → 0.216 s at 1M → 1.66 s at 10M) while Claude Sonnet 5 was **flat** (0.129 s at all three) [149].
Hybrid/linear attention does not shave the constant, it removes the term. This model has that property
partially — 8 of 72 visits carry the quadratic — which is why §3.3 is 222 h rather than 2,000 h.

### 3.3 What 1M costs *here* — the number that decides it

Measured: **0.11 s/token prefill at f32**, and **12.06 KB/token KV at 6-bit** [local].

| n | KV footprint | attn/weights FLOP ratio | prefill time | fits in 16 GB RAM with 7.9 GB weights? | decode ceiling @192 GB/s |
|---|---|---|---|---|---|
| 8,192 | 94 MiB | 0.05x | 0.25 h | yes | 24.1 tok/s |
| 16,384 | 188 MiB | 0.10x | 0.50 h | yes | 23.8 |
| 32,768 | 377 MiB | 0.19x | **1.19 h** | yes (8.3 GB) | 23.1 |
| 65,536 | 754 MiB | 0.38x | 2.4 h | yes (8.7 GB) | 22.2 |
| 131,072 | 1.51 GB | 0.74x | 4.9 h | yes (9.5 GB) | 20.3 |
| 262,144 | 3.02 GB | 1.49x | **19.9 h** | yes (11.1 GB) | 17.4 |
| 1,048,576 | **12.65 GB** | **5.95x** | **222 h** | **no (20.5 GB)** | 9.3 |

Prefill column = measured 0.11 s/token × n, scaled by (1 + attention/weight ratio) — **derived, not
measured**; the anchor is the f32 measurement at short context and the quadratic term is added
analytically. Treat 222 h as ±30%, not ±3%.

Decode ceiling is `(7.9 GB + n·12.06 KB) / 192 GB/s` — **derived** from spec-sheet bandwidth, and it is a
*ceiling*: it assumes 100% bandwidth utilisation, which nothing achieves. The real number is lower.

Three independent walls at 1M:

1. **Time**: 222 h of prefill for one request. 9.3 days. Not a slow feature; not a feature.
2. **Memory**: a 20.5 GB working set on a 16 GB machine. The KV cache alone (12.65 GB) exceeds the
   model (7.9 GB) and the VRAM (6 GB) by 2x each.
3. **Bandwidth**: 9.3 tok/s ceiling, and the KV *read* per decode step (12.06 GB) is 1.5x the weights.
   Decode becomes KV-bound, so the SSD-expert-streaming trick that makes the model fit stops helping —
   it was solving the wrong bottleneck.

At 32K all three walls are gone: 377 MiB, 1.19 h prefill for the *whole* 32K, 23 tok/s ceiling. **The
difference between 32K and 1M here is 187x in prefill time and 32x in memory, for a capability gain that
§2 says is between +4 points and −30 points.**

### 3.4 Dollars, for the cloud comparison

1M-token input call, published rates (2026): MiniMax M3 ≈ $1.21, Gemini 2.5 Pro ≈ **$2.58**, Claude Opus
4.7 ≈ $5.13, GPT-5.5 ≈ **$10.23** [159]. Gemini doubles above 200K *on the whole request*, not just the
excess [131][159]; OpenAI re-prices the whole session above 272K [159]; Anthropic removed its 1M
surcharge on 2026-03-13 [159].

The price architecture is itself evidence: **three of four vendors charge more per token for the tokens
that work worse.** The surcharge is not gouging, it is the honest price of quadratic attention — and it
is why "just send everything" is a design that loses to retrieval on cost by 26x [158] and 50x [141].

---

## 4. Who actually needs 1M, and does anything cheaper match it

### 4.1 The four real claims

| claim | is 1M load-bearing? | verdict |
|---|---|---|
| **Long audio** — full-day meeting/call/library | **yes** | 1M tokens ≈ **8.4 h** of audio at Gemini's rate [133]; Gemini 1.5 did 22 h / 2M [132]. No retrieval shortcut exists for raw audio without an ASR pass. |
| **Repo-scale comprehension** | **partly** | LongCodeQA rewards length for strong models (GPT-4.1 80.0% at 1M) [146]; but targeted acquisition beats stuffing (FastCode 78.0% at 1M at **$0.073** vs $2.600 direct) [147], and the entire production answer is grep+LSP, not a big window [142][143][145]. |
| **Multi-document one-shot analysis** (legal, medical, financial) | **partly** | LC 56.3% vs RAG 49.0% on dense structured corpora [138]; LC beats RAG by 7.6–13.1% when fully resourced [139]. Real but modest, and it is a *single-document-class* task, not a 1M-token task: LongBench v2 median is 54k words [127]. |
| **Agent memory / conversation history** | **no** | focused 300 tokens beat 113K on every model [126]. Window size is not the mechanism; retrieval and compaction are [142][157]. |

### 4.2 The repo-scale story, in detail, because it is the one people cite

This is worth doing carefully because "1M for codebases" is the most-copied justification and it is
**half true in a way that matters**.

What the benchmark says [146]:

| model | 32K | 128K | 512K | 1M |
|---|---|---|---|---|
| GPT-4.1 (LongCodeQA) | 72.6 | 78.3 | 78.7 | **80.0** |
| Llama 4 Scout | 66.4 | 70.7 | 78.7 | 76.0 |
| Qwen2.5-14B | 61.9 | 68.5 | 70.2 | **40.0** |
| Gemini 2.5 Pro | 75.2 | 71.7 | 68.1 | 69.8 |
| Claude 3.5 Sonnet (LongSWE-Bench repair) | 29 | 15 | — | **3** |

- **Comprehension (MCQ over a repo) does improve with length — for models with headroom.** +7.4 points
  for GPT-4.1 from 32K to 1M. This is the real finding and it is not nothing.
- **Repair does not work at any length.** 0–2% for nearly every model at every size [146]. A 1M window
  does not buy you "fix this bug in my monorepo"; it buys you "answer a question about it".
- **The 14B falls off a cliff at 1M** (70.2 → 40.0), which is precisely the size class in question here.

What production actually does [142][143][144][145]:

- Claude Code: file traversal + **grep** + LSP symbol lookup, just-in-time, with `CLAUDE.md` as the only
  up-front context. Anthropic's own docs: RAG-index approaches "can fail" at scale because the pipeline
  lags the repo [143].
- Anthropic's 2026 best-practices page has *retired* the advice "the context window is your main
  constraint" — because 1M windows and auto-compaction make running out rare [144]. But the same page
  says the failure mode moved: "you can still *pay* for context you don't need", because every request
  re-sends the whole conversation.
- Snap runs **grep across thousands of repos** for agents, sub-second p95, no embeddings [145].
- FastCode gets **78.0% at 1M for the price of a 32K call** by acquiring evidence selectively [147].
  Its ablation is the punchline: naive file-level BM25 RAG *destroys* performance (65.5 → 25.55) [147] —
  so "RAG" is not a free win either; the win is *targeted acquisition*, which is a different thing.

So the accurate statement is: **1M context substitutes for a good retrieval layer, at 30x the cost and
lower accuracy, for comprehension tasks only.** When you have the retrieval layer, the window is a
convenience.

### 4.3 Long audio is the one genuine 1M use case — and it is not this model

8.4 hours per 1M tokens [133]; Gemini 1.5's headline demo was 22 hours [132]. There is no way to answer
"what did we decide in the Q3 planning marathon" across 9 hours of raw audio without either (a) a huge
window or (b) an ASR pass that throws away prosody, overlap, and speaker identity.

But: this is a **multimodal encoder** problem. A text-only 12B cannot consume audio at all, and 8.4 h of
*transcript* is ~76k words ≈ **100k tokens** [derived: 150 wpm × 8.4 h, 1.3 tok/word], not 1M. Audio's
1M requirement comes from 32 acoustic frames/second [133], not from information density — the audio path
inflates token count ~10x relative to its text content. **The 1M audio use case does not transfer to text.**

### 4.4 What the counter-position actually claims, and its limits

The pro-retrieval case is strong but not universal, and the honest version is a routing decision:

| finding | number | source |
|---|---|---|
| LC vs RAG, Wikipedia-dense QA | LC **56.3%** vs RAG 49.0% | [138] |
| LC vs RAG at 32K, averaged over 11 models | LC **+2.4%** | [137] |
| LC vs RAG at 128K, averaged | RAG **+3.68%** | [137] |
| LC vs RAG at 128K, **Mistral-Nemo-12B** | RAG **+38.12%** | [137] |
| LC vs RAG, Llama-3.2-3B | RAG **+6.48%** | [137] |
| RAG win-rate on NIAH-style tasks | **82.58%**, 50x fewer tokens | [141] |
| DOS RAG at 30K budget | **93.1%** vs vanilla RAG 87.8% | [140] |
| long-context vs semantic RAG, small models | 73.1% vs 65.4% (**+7.7 pts at 26x cost**) | [158] |

The pattern is unambiguous and it is the answer to the key question:

> **Below ~32K, long context wins. Above ~128K, retrieval wins. The crossover is model-size-dependent,
> and it moves *down* as the model gets smaller.** A 12B is on the retrieval side of the crossover at
> 128K by 38 points [137].

The pro-LC results ([138][139][158]) come from strong models on dense single-document corpora. The
pro-RAG results come from weak models on everything else. **A 12B on 16 GB is the weak-model case.**

---

## 5. The answer to the key question

### 5.1 Utility per unit of compute, by budget

| context | prefill (this box) | KV | what you gain vs 8K | what it costs | verdict |
|---|---|---|---|---|---|
| 8K | 0.25 h | 94 MiB | baseline | 1x | floor — too small for a repo file or a long paper |
| **16K** | 0.50 h | 188 MiB | most of the non-literal-task curve already lost by here [123] | 2x | minimum viable |
| **32K** | **1.19 h** | **377 MiB** | +2.4% over RAG at this length [137]; full paper/chapter; attention still 0.19x weights | 4.8x | **sweet spot** |
| 64K | 2.4 h | 754 MiB | small gains on comprehension for strong models [146]; nothing for a 12B | 9.6x | upper useful bound |
| 128K | 4.9 h | 1.51 GB | RAG now beats LC by 3.7% avg, 38% on a 12B [137] | 20x | **negative marginal utility for this class** |
| 262K | 19.9 h | 3.02 GB | strictly worse than 128K on every non-literal benchmark | 80x | waste |
| 1M | 222 h | 12.65 GB | recall 16–26% on the hard half of the task space [129][160] | 890x | **unreachable, and would not help if it were** |

**Verdict: 32K is the utility maximum; 64K is the practical ceiling; 128K is where the curve turns
negative for a 12B-class model on this hardware.** The "most utility per unit of compute" point is not a
single number but a band, and the band is 16K–48K. Above it, you are paying 2x per doubling for a
capability that benchmarks say is falling.

### 5.2 The three reasons 1M is the wrong goal, stated as falsifiable claims

1. **The capability is not there at 1M, at any model size, on the tasks that need it.** Best 2026 result
   at 1M on 8-needle MRCR is 76% [160]; for a 12–14B it is 40% on code comprehension [146] and 7.3% on
   NoLiMa at 32K [123]. Falsifier: a 12B-class open model scoring >70% on NoLiMa-Hard at 1M, or >70%
   on MRCR-8 at 1M. Nothing in the 2025–2026 record does.
2. **The marginal token past ~48K is worth less than it costs, on this hardware and any hardware.**
   Falsifier: a task family where a 12B gains >10 points going 128K→1M. LongCodeQA is the closest and
   it goes *down* for 14B (70.2 → 40.0) [146].
3. **The engineering effort is better spent elsewhere.** The two things that actually moved the needle
   in 2025–2026 production systems were (a) context *management* — compaction, sub-agents, just-in-time
   retrieval [142] — and (b) targeted acquisition [147]. The harness study quantifies the substitution
   exactly: the benefit of good context management over naive stuffing is **35.7 points at a 32K
   window and 2.7 points at 128K** [148]. A small window plus management ≈ a big window plus nothing.
   Falsifier: a system where 1M-without-management beats 32K-with-management on a real task. It exists
   in one place — single-pass dense-corpus QA [138] — and that task's median length is 54k words [127],
   not 1M tokens.

### 5.3 What the data supports building instead

In priority order, each with the number that justifies it:

1. **Make 16–32K fast.** At 32K this model needs 1.19 h of prefill (derived). A 32K request that takes
   an hour is as unusable as a 1M request that takes 9 days. The 9,600x throughput gap in
   `INKLING2-REALITY-CHECK.md` is the actual blocker; context length is decoration on top of it.
2. **Exact retrieval over a local corpus** (grep/ripgrep + symbol index, not embeddings). Precedent:
   Claude Code, Snap, FastCode [143][145][147]. Cost: 2–3% of the tokens; benefit: the whole corpus,
   not 1M tokens of it.
3. **KV eviction / quantization at 32–64K** so the cache is a rounding error rather than 3 GB. The
   sliding-window layers already do this for 64 of 72 visits; the remaining 8 are where all the cost is.
4. **Compaction.** The one mechanism that turns an unbounded task into a fixed window and is measured to
   work [142][144].
5. **1M as a ceiling, not a target** — i.e. the model must not *crash* at 1M, and need not be *good* at it.
   That is cheap to guarantee (it's an allocation question, 12.65 GB) and expensive to optimise.

### 5.4 Where this analysis could be wrong

Stated plainly, because the premise deserves real opposition:

- **The 222 h figure is an extrapolation.** The anchor is 0.11 s/token at short context. If the f32
  prefill path is currently 100x off peak (as the reality-check suggests for decode), the *ratio* between
  32K and 1M holds (187x) but the absolute number could be far smaller once kernels are vectorised.
  The conclusion survives because the ratio is what kills 1M, but the specific hour count should not be
  quoted as measured.
- **Capability is improving fast.** 16.4% → 76% at 1M in 14 months [129][160] is a 4.6x gain. If that
  rate continued for two more releases, 1M would be genuinely usable on frontier models. It would still
  not be usable on a 12B — the size gap is the binding constraint, not the generation.
- **LongBench v2's long bucket is not the worst bucket** [128], and LongCodeQA improves with length for
  strong models [146]. There exist real tasks where a big window pays. They are not the tasks a
  hobbyist runs, and they are not 1M-token tasks in the median case (54k words) [127].
- **This model's hybrid attention is unusually long-context-friendly** (crossover at 176k vs 24k for a
  dense 12B, §3.1). The strongest pro-1M argument in this document is that the architecture was *built*
  for it. That argues for not throwing the capability away — not for making it the goal.

---

## 6. Bottom line

1M context is a **marketing number and an allocation decision**, not a capability. The 2025–2026 record
is consistent across five independent benchmark families: claimed length exceeds effective length by
10x–1000x [121][123][136], every model degrades continuously from the first extra token [126], and the
degradation survives perfect retrieval [135].

For a 12B-class quantised model on 16 GB, the utility curve peaks at **~32K**, is still mildly positive
at 64K, and **turns negative at 128K** — where retrieval beats long context by 38 points on exactly this
model size [137]. 1M would cost 222 h of prefill and 20.5 GB of working set for a capability that
frontier models score 16–26% on and 12–14B models score 40% on, in the one task family (code
comprehension) where length helps at all [146][160].

The premise "1M context is unreachable on this box" is **true and irrelevant**. It is unreachable, and
reaching it would not produce a better system. The reachable, useful target is 32K with retrieval and
compaction — and the real gap to close is the 9,600x throughput deficit, which no amount of context
would fix.
