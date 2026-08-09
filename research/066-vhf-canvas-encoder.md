# research/066 — THE VHF CANVAS: the coordinate-addressable latent field

> 2026-08-08. The user's directive: "my vhf encoder is i think previously
> made research that helps this encoder space work."
> Source: AUDIO/wubusynth/vhf_tool.py + vhf_demos.py + vhf_audio.py +
> ENCODERS/README.md (the WuBu Nesting research core — 5 phases).
> The user solved "video and audio in one morning" (Mar 9, 2026 commit).

## The VHF encoder (what it is)

**VHF = the latent space as a coordinate-addressable FIELD, shaped like
the VGA signal itself:**

- **The canvas**: 525 lines × 656 columns. 480 visible lines carry
  VIDEO; 45 VBI (vertical-blanking) lines + 16 HBI columns carry AUDIO.
  Every modality occupies its own coordinates in ONE canvas.
- **The encoder** (`HamiltonEncoder`): a conv pyramid (4×4 stride-2,
  gelu) collapses the input to a latent grid (e.g. 96×96); a 1×1 conv
  emits **5 values per cell: a unit quaternion (4) + an amplitude (1)** —
  the orientation + presence of the local field.
- **The decoder** (`VHFDecoder`): samples the latent grid at (y,x)
  coordinates (bilinear `map_coordinates`), concatenates a
  **positional encoding** (sin/cos at 2^k·π freqs, k=0..9) + the global
  context vector + the local field sample, then 4 gelu-Dense layers →
  tanh → RGB. It is an **implicit neural field**: the network IS the
  decode function; any coordinate can be queried.
- **The philosophy** (ENCODERS/README, Phase 2): "represent information
  in the *parameters of the laws that generate it*" — a QAE compresses
  an image to 3 Hamiltonian coefficients. Extreme compression, not bit
  packing.

## Why it "helps this encoder space work" (the convergence)

Everything we built this session composes with the VHF insight:

| VHF canvas concept | The C11 stack (today) |
|---|---|
| ONE canvas, all modalities at coordinates | **KVFS = the universal namespace** — every datum is a file with a path; /kv/user holds text+image+video+office |
| Encoder emits quaternion+amplitude per cell | **our ViT** (imgenc) emits CLS tokens; the encoder slot (wubu_encoder) is the modality seam |
| Decoder samples the field at any (y,x) | **the router** (wubu_router): top-K by potential over the ecosystem balls — sampling the field |
| Positional encoding (sin/cos freqs) | **hyperbolic geometry** (wubu_mobius, wubu_hyper) — our balls live in a curved coordinate space |
| Audio in HBI, video in visible lines | **one namespace, all file types** (test_ingest_all: PNG/JPEG/AVI/docx/PDF/wav) |
| Extreme compression via law-parameters | **the Escha precision ladder** (AM03): 28MB for WuBu1 — quality density |

The VHF canvas is the *visual* proof of the KV-FS doctrine: **the
encoding is a place you can point at.** The namespace is the canvas;
paths are coordinates; the encoder writes the field; the router reads
it. The user's prior art and the C11 architecture are the same shape.

## What the C11 stack should carry forward (the build list)

1. **`wubu_canvas`** — the coordinate-addressable latent field in C11:
   a latent grid (cells × 5: quaternion4 + amplitude) written by the
   encoder slot, decoded by coordinate sampling with positional
   encoding. The namespace IS the canvas: /kv/canvas/(y,x) addresses a
   field value.
2. **Quaternion-field cells** — replace/augment the flat embedding
   files with orientation+amplitude cells (the Hamilton encoder's
   output shape), so the field is addressable AND interpolable.
3. **Coordinate decode path** — the router already samples; a
   `wubu_canvas_decode(coords)` entry makes the decode function itself
   a first-class module (the VHFDecoder's MLP, but ours).
4. **Audio-in-HBI parity** — the canvas carries audio and video in one
   address space, exactly like /kv/user already does (wav + avi in the
   same namespace).

## Sources

- `AUDIO/wubusynth/vhf_tool.py` — VHF end-to-end (HamiltonEncoder,
  VHFDecoder, positional encoding, Q-learning LR controller).
- `AUDIO/wubusynth/vhf_demos.py` / `vhf_audio.py` — canvas constants
  (VBI_LINES=45, VISIBLE_H=480, AUDIO_HBI_WIDTH=16, VISIBLE_W=640).
- `ENCODERS/README.md` — the 5-phase nesting arc (symmetric geometric
  AE → topological QAE → generative → HashMind → Hamilton geodesics).
- `AUDIO/README.md` — "I solved video and audio in one morning."
