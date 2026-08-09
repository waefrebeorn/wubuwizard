# research/064 — THE GEMMA UNIVERSAL ENCODER SPACE (and the encoder-free turn)

> 2026-08-08. The user's directive: "research the Gemma 12B cause that
> had a universal encoder space that they were using."
> Sources: Gemma 3 Technical Report (arXiv:2503.19786, Mar 2025) + Gemma 4
> Technical Report (arXiv:2607.02770, Jul 2026). Full texts archived in
> ~/.hermes/profiles/mind-palace/cache/web/.

## The one-line answer

Gemma 3 12B's "universal encoder space" = **every modality becomes a
sequence of soft tokens in ONE embedding space that the decoder reads**.
Images are encoded by a frozen SigLIP-400M into 256 vectors; text uses
the tokenizer's embeddings; both flow into the SAME decoder-only
transformer. But the frontier already moved past it: **Gemma 4 12B
(Jul 2026) is encoder-FREE** — it throws the separate encoder away
entirely and projects raw image patches and audio chunks straight into
the LLM embedding space with a single matmul. That is the design our
"make our own encoder" should copy.

## Gemma 3 12B — the universal encoder space (2503.19786)

| Element | Design |
|---|---|
| Encoder | **SigLIP-400M** (a ViT trained with CLIP-style contrastive loss), tailored for Gemma |
| Input | square images resized to **896×896**, fixed resolution |
| Output | a sequence of **soft tokens** the LLM reads directly — no discretization |
| Condensation | vision embeddings condensed to a **fixed 256 vectors** per image (the inference-cost cut) |
| Flexible resolution | **Pan & Scan (P&S)**: inference-time adaptive windowing — segment the image into non-overlapping crops, resize each to 896×896, feed all; only when needed, capped crops |
| Sharing | the encoder is **shared across 4B/12B/27B and FROZEN** during training (only the LLM backbone trains) |
| KV slashing | 5:1 local/global attention interleave, local span 1024 → 128K context without KV explosion |

The "universal" part: one embedding space, every input encoded into it —
exactly the KV-FS doctrine's "all inputs are encoded". The cost: a 400M
frozen encoder is 7× the size of our entire model, and it is an import.

## Gemma 4 12B — the encoder-free turn (2607.02770)

The June 2026 report **kills the separate encoder**:

- **Vision**: 48×48×3 RGB patches → **a single large matmul (35M params)**
  replaces the 550M vision encoder → add **2D coordinate-based positional
  embeddings** → a final LayerNorm. That is the whole "vision encoder".
- **Audio**: the 305M USM conformer encoder is **discarded entirely**.
  Raw audio → 40ms chunks @16kHz → 640-dim vectors per chunk → projected
  **directly into the LLM embedding space**. No positional encoding
  needed (audio is temporal).
- **Model table**: the 12B row shows NO audio encoder, NO vision encoder —
  just a 1,000M embedder + the 10,890M transformer + a 400M drafter.
- **Rationale**: "alleviating the need for separate encoders and reducing
  memory fragmentation" — fewer params, less latency, one fine-tune pass
  over the whole model.

The pattern: **Gemma 3 = import a big frozen encoder. Gemma 4 = the LLM
IS the encoder.** The backbone learns to read raw patches itself; the
projection is a learnable linear layer.

## The convergence for WuBu (the user's "make our own" is validated)

| Approach | Params | Ours? | Verdict |
|---|---|---|---|
| Gemma 3 SigLIP import | 400M encoder | ❌ import | rejected — we make our own |
| Full ViT from scratch (wubu_imgenc CC01) | ~1-2M | ✅ ours | good, but it's the Gemma-3-style path |
| **Gemma 4 encoder-free (single matmul projection)** | **35M-class projection** | ✅ ours | **THE frontier — the LLM learns to read raw patches** |

What we already shipped this session (`wubu_encoder_impl`) is the
Gemma-3-style seam: our own ViT (imgenc) + our own mel (audio) into a
shared 128-dim. The Gemma 4 finding says the SIMPLER design wins: a
single learned projection from raw patches/chunks straight into the
model's embedding space — no separate encoder network at all. That is
smaller, ours entirely, and the backbone (which we ARE training) learns
the perception itself.

### The concrete next step (when the user says go)

`wubu_encoder_impl` gains a third path, `our-patches`: raw 48×48×3
patches → one linear layer (the 35M-class matmul, ours) → 2D coord
positional embedding → LayerNorm → the shared space. The test then
proves image-patches, audio-chunks, and text ALL land in the same
embedding space through LEARNED projections only — no ViT, no mel
filterbank, no import. The imgenc/audio front-ends stay as the
Gemma-3-style alternatives behind the same slot (Revolver: the slot
precedes the module; the payload rotates).

## SHIPPED (2026-08-08): the encoder IS a mount (AN16 G3)

The user's directive: "if we can also make the encoder part of the
universal file system space because then we are already lifting heavy
work." Done — `wubu_enc_fs.c` mounts the encoder INTO the KV namespace:

```
write raw input  -> /kv/enc/<id>   (the mount boundary encodes)
read embedding   -> /kv/emb/<id>   (pure KVFS read — heavy work lifted)
```

The encode transform is OUR encoder (imgenc/audio/text via the slot);
the embedding lands as a FILE in the KV tensor region, where tiering,
eviction, paging, persistence, and mirroring are all ALREADY lifted by
wubu_kvfs (ADR-003). The encoder re-implements NO storage — it is just
another mount. Gate: test_enc_fs PASS (mount, encode-through-fs,
read-back, bytes-in-tensor, namespace integrity), ASan clean. AN16 G3
marked wired in research/INDEX.md.

## Sources

- Gemma 3 Technical Report — arXiv:2503.19786 (Mar 2025): SigLIP-400M,
  896×896, 256 soft tokens, Pan & Scan, 5:1 interleave, frozen shared encoder.
- Gemma 4 Technical Report — arXiv:2607.02770 (Jul 2026): encoder-free
  12B — 48×48×3 patches → single 35M matmul + 2D pos embeds + LayerNorm;
  audio 40ms/640-dim chunks projected directly; no encoders in the 12B
  model table.
