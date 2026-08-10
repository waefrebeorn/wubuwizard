# Decoder-Synthesis Model — architecture + training plan (2026-08-09)

The user's directive: *"We can also do this to have media creation through
a decoder synthesis model because I have created an encoder decoder system,
and I have given you the methodologies to create those, but we do not have
those trained yet — you would have to train them."*

This doc is the honest scope: the encoder side is SHIPPED (our ViT, the
VHF canvas, the universal codec); the DECODER-SYNTHESIS model that
GENERATES media from the latent space is not trained yet. Training is a
multi-day GPU effort on the existing pipeline (wubu-agi-training-pipeline);
this doc fixes the architecture + the plan so the training run is
mechanical, not a research question.

## What exists (the encoder side, all shipped + gated)

| Piece | Where | What it does |
|---|---|---|
| the SLOT | `include/wubu_encoder.h` + `src/wubu_encoder.c` | one vtable (encode/decode, modality enum, dim probed at load) |
| our-image | `src/wubu_encoder_impl.c` (wubu_imgenc) | ViT patch embedding from scratch — the CLS readout is the 128-dim embedding |
| our-audio | same | mel-spectrogram + radix-2 FFT |
| our-text | same | deterministic token-id projection |
| encoder-as-mount | `src/wubu_enc_fs.c` | write raw at `/kv/enc/<id>`, the embedding is a FILE at `/kv/emb/<id>` |
| user-space data | `src/wubu_userfs.c` | real user files bridge into the namespace — every user action trains |
| the VHF canvas | `src/wubu_canvas.c` | the latent FIELD: a grid of quaternion+amplitude cells, bilinear decode, positional encoding (sin/cos at 2^k·π) |
| the universal codec | `src/wubu_codec.c` | audio ↔ image reversibly, ours on our FFT (5 perceptual bands → image rows; the peak in the metadata pixel) |

## The decoder-synthesis model (what to train)

**Goal**: a neural DECODER `D(latent) -> media` that generates new media
(image/audio/video) from a latent field or an embedding — the generative
side of the encoder-decoder the user built.

**The training objective (self-supervised, no labels)**: the encoder `E`
already maps media → the latent space (the canvas cells / the 128-dim
embedding). Train `D` to INVERT `E`:

```
loss = || media - D(E(media)) ||        (the reconstruction path)
     + lambda * smoothness on D        (the latent field must be
                                         continuous — the canvas
                                         invariants from the codec traps)
```

The reconstruction target is REAL: our corpus (images, audio, the
userfs stream). `E` is frozen; `D` learns the inverse.

**The architecture** (the P3 generative lineage: VQ-VAE vocabulary +
transformer Conductor, per `ENCODERS/` in the repo):
- `D` = a small transposed-ViT (image) / a transposed mel-GRU (audio) /
  a canvas-field sampler (video — the GAAD phi-spiral foveated patching
  is the input segmentation, already designed)
- the latent CONDITIONING = the canvas field coordinates + the positional
  encoding (sin/cos at 2^k·π — the canvas's native geometry)

**The plug-in**: the trained weights land in the model zoo
(`/home/wubu/models/`), the decoder registers into the SLOT as
`"our-decoder"` (a new encoder-slot entry with the decode direction
primary), and the canvas/codec round-trip becomes a GENERATION path:
the AGI writes a latent, the decoder synthesizes the media.

## The training plan (mechanical once the architecture is fixed)

1. **The corpus**: the existing SFT token stream (12M tok) + the userfs
   stream + the codec round-trip fixtures (the canvas traps doc lists
   the exact decode invariants to test against).
2. **The run**: the wubu-agi-training-pipeline harness (Muon optimizer,
   the backprop path — the same loop that trains WuBu-35M), a small
   decoder (10-100M params — the body repo's GPU is a 6GB RTX 4050).
3. **The gate**: reconstruction error on the held-out corpus + the
   canvas continuity invariant (adjacent latent cells decode to
   adjacent media — the coherence the canvas encodes by construction).
4. **The deliverable**: `wubu_decoder_<name>.safetensors` + the slot
   registration + a `media-synthesis` test that writes a latent,
   decodes, and asserts the media is coherent (the pixel/audio
   statistics match the training distribution).

## The body-side surface (what the OS shows)

Once trained, the WuBuOS GUI gets a **Media Studio** app (the era-apps
grid + the start menu): write a latent (or let the AGI write one), the
decoder synthesizes an image/audio/video — the OS as a CREATION space,
not just a consumption space. The `dosgui` canvas app already exists as
the latent editor shell.

## Status

- [x] the encoders (our-image/audio/text)
- [x] the canvas + the universal codec (the encoder-decoder system)
- [ ] **the decoder-synthesis training run** (the plan above — a
      multi-day GPU job on the Brain side)
- [ ] the Media Studio app (trivial once the decoder exists)
