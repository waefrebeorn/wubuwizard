# 🎨 ~ WUBU ~ VIBE SYSTEM — The Aesthetic Engine
> *How to make things that generate VIBES, not just descriptions.* Pair with
> [[CAST]] (who) and [[SCRIPT_TEMPLATE]] (what). Sole brand: ~ WuBu ~.

## THE TOOLS (learned, not guessed)
- **FLUX-3** — hero model. "Spielberg CGI" quality. $0.06/s (1080p). Image-to-
  video via FAL. Native audio free. One architecture does picture + synced audio.
- **LTX-2.3** — cheapest at 720p ($0.06/s), audio free. Bulk/workhorse.
- **Seedance 2.0** — 1-3 reference images, high consistency (8.5/10).
- **Gateway text-to-video** works WITHOUT FAL_KEY. **I2V needs hosted image
  URLs** (file:// fails). → we host rotational reference images on an HTTP server.

## THE ROTATIONAL I2V WORKFLOW (user's method)
We can only hold **3 live image links** at once. So we host **3 rotational
reference images** of the Old UNC and cycle them per generation:
1. Start an HTTP server (Python `http.server` or a lightweight host) serving
   `UNC_ref_1.png`, `UNC_ref_2.png`, `UNC_ref_3.png` (3 clean WuBu sheets).
2. Each generation pulls the 3 URLs as **reference images** into FLUX-3 I2V.
3. After a batch, **rotate** the 3 images (swap in sheet_03/Sheet_05/etc.) so the
   model sees varied poses but the SAME identity → consistency without drift.
4. Never more than 3 links live; rotate to cover all 6+ pose-sets over time.

> Why this works (2026 research): "Stop asking the model to invent AND animate
> simultaneously. Lock identity FIRST (reference image), then animate." Our
> rotational 3-link system IS that discipline, within the link limit.

## THE FLUX-3 PROMPT GRAMMAR (from flick.art 2026)
```
[Shot type / camera movement] of [subject] in [setting].
[Subject action over time].
[Environmental motion over time].
Audio: [dialogue, ambience, effects, silence, music direction].
Style: [cinematic style, lens, lighting, texture].
Constraints: [identity lock, no text, no logos, no extra characters, realistic motion].
```
Example (UNC, ORIGINALS):
```
Medium shot, slow push-in, of a heavyset Black man with a large rounded afro,
thick mustache and goatee, BLACK hoodie with "WuBu" in YELLOW/GOLD letters,
holding a teal-orb staff, standing in a narrow fluorescent storage unit.
He reaches for a door handle; his hand passes through it; the handle becomes a
face; he nods, satisfied, and walks off.
The hallway walls breathe — expanding and contracting like lungs.
Audio: bass-driven dark-trap Wizardcore, a low sub hum, the hallway's breath as
wind-through-tube; no dialogue.
Style: 2D cartoon, thick black outlines, flat colors, cel-shaded, VHS REC-cam —
chroma bleed, lifted blacks, scanlines, vignette, tape noise, REC/date burn-in.
Constraints: keep the man's face and hoodie identical to reference images, no
extra characters, no text in frame except "~ WuBu ~" corner watermark, realistic
weight, no Ken Burns.
```

## THE GENERICS VOCABULARY (trademark-safe descriptors — chain-bootleg/13)
Use these instead of protected names. Build the *vibe* from:
- **Color:** amber, gold, teal, fluorescent white, steel gray, muted, vibrant
- **Lighting:** found-footage available light, flickering single-source, high-key
  game-show, warm sitcom, spotlight talk-show
- **Framing:** handheld shaky, Dutch angle, medium close-up, wide low-angle
- **Camera:** handheld operator, slow push, snap zoom, dolly in, static
- **Texture:** VHS degradation, scan lines, chroma bleed, film grain, cinderblock,
  linoleum, fluorescent, carpet
- **Style era:** 1980s neon+VHS, 1990s warm analog, contemporary LED-minimal

## PER-CHANNEL VIBE TABLE
| Channel | Palette | Lighting | Texture | Vibe keyword |
|---|---|---|---|---|
| NEWS | steel blue + amber CRT | bright even, blue-white | scanlines, tracking | "broadcast found footage" |
| ORIGINALS | teal + fluorescent white | available light, overexposed | tape noise, chroma bleed | "found room, generated crowd" |
| SATIRE | gold + prismatic | game-show high-key | slight bleed, confetti | "absurd ceremony" |
| LABS | cyan + graphite | tech blue, screen glow | clean grain | "transparent technique" |
| MEDIA SCHOOL | red+black + amber | interrogation single-source | heavy VHS decay | "the fiction you accepted" |
| PUBLIC ACCESS | wood + warm + neon | living-room lamp | soft VHS, laugh-track sheen | "domestic liminal" |
| PUPPET THEATER | neon + clipart primary | storybook key | cel-shaded, no decay | "surreal puppet" |
| NETWORK | blue+white + static | 24h news blue | rolling tracking lines | "continuous transmission" |
| ARCHIVE | raw + desaturated | none (diegetic) | max degradation | "the room was found" |

## VHS / ANALOG-HORROR KEYWORDS (prompt lexicon — chain-meme/19, comedy-deep/23)
scanlines · chroma bleed · tape noise · tracking errors · CRT curvature ·
rolling distortion bands · lifted blacks · vignette · record-date burn-in ·
prismatic color shift · soft interlaced edges · low apparent resolution ·
RGB tone drift · image decay · timestamp overlay.
> The degradation signals "this is real, this was found" → makes the absurd
> funnier (authenticity as comedy tool).

## CONSISTENCY RULES (from 2026 research)
1. **Reference image first, animate second** — never text-only identity.
2. **Repeat key features in every prompt:** "Black man, large rounded afro,
   mustache+goatee, black WuBu hoodie, teal-orb staff."
3. **One character per frame** unless scripted — multi-character still blurs
   (2026 unsolved). Peter/the orb only when the bit needs them.
4. **Same angle, same lighting** when continuing a scene.
5. **3 rotational links max** — rotate, don't exceed.

## Links
- Cast: [[CAST]] · Voice: [[VOICES]] · Script: [[SCRIPT_TEMPLATE]]
- Research root: `chain-bootleg/13` · `chain-meme/19` · `chain-comedy-deep/23`
- Tool proof: `PIPELINE.md`

~ WuBu ~
