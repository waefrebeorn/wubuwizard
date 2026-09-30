# VISUAL SPECS v1 — the five recurring non-UNC characters

Written 2026-09-26 from **actual reference frames on the human's machine**, not
from memory. Where a ref is thin I say so, because a wrong spec costs a reroll.

**House style for every character below:** 2D cartoon, thick black outlines, flat
colors, cel-shaded, 1970s public-access broadcast, VHS chroma bleed, scanlines.
`~ WuBu ~` watermark bottom-right. All 16:9 landscape, subjects CENTRED.

---

## 1. LAURA — the counter-voice
**Ref:** `WuBuMovie/raw_footage/wubuclips/wubu_laura_s1..s5.mp4` (real, on disk)

- Young woman, fair/pale complexion, oval-to-heart face, small straight nose
- Dark brown/black hair, straight, shoulder-length bob, parted slightly off-centre,
  falls over the ears
- Dark eyebrows, dark eyes, neutral almost blank expression, lips closed in a thin line
- Slender build
- **Wardrobe:** dark forest-green or black hoodie, hood down, drawstring visible,
  small white chest graphic (a tilde-like squiggle + wordmark)
- **Signature:** she is the only person in the canon who does NOT do deadpan.
  She is the counter-voice. She reacts. That contrast is the whole engine — if
  you make her deadpan you have deleted the show.

**LIVE DEFECT IN THE REF — do not reproduce.** I frame-extracted her s1 clip and
the chest wordmark **changes between frames**: "WuB" → "WuF" → "Wu". The model is
regenerating the lettering. So: if you use chest text on Laura, **spell it in the
Text block and 6-tile it.** Safest is a plain unlettered hoodie.

---

## 2. THE COLONEL — News voice
**Ref:** thin. No dedicated ref sheet exists. Described from `codec-call-001.md`.

- Stocky middle-aged Black man, short grey/white beard
- Dark navy military dress uniform, silver buttons, small name plate, worn with
  total comfort — it is a job, not a costume
- Upright posture, hands folded on the desk
- **Expression: serious, professional, composed.** This is the one place deadpan
  is explicitly WRONG. He has no irony in his face. The irony is in the words,
  and it works precisely because his face is straight. Making him smirk kills it.
- **Set:** television news desk, glowing orange edge; two studio monitors behind;
  stacks of paper; `ON AIR` indicator. No real network logos, ever.

---

## 3. PETER — the loud one
**Ref:** `WuBuMovie/raw_footage/rapman/peter_sheet.png` (real, on disk)

- Balding, ring of short brown hair around the sides
- Round thin-rimmed glasses
- Thick brown handlebar mustache
- Mustard-yellow/gold polo shirt with collar; dark green shorts above the knee
- White crew socks, white low-top sneakers
- Stocky/overweight, large belly
- **Function:** Peter is enthusiasm with no evidence. He is the one who is
  excited about the thing that is not real. He is the straight man who is wrong,
  which is worse and funnier than a straight man who is right.

---

## 4. THE WIZARD — primary magical-order figure
**Ref:** `WuBuMovie/raw_footage/rapman/sheets/` (exists; I have not verified
frame-by-frame, so treat the sheet as authoritative and do not improvise him)

- Described in canon as the primary magical-order figure. The sheet is the truth.
- If you need him generated, **load the sheet as the reference and give it a named
  job in the prompt** ("Image 1 is THE WIZARD reference") rather than describing
  him from text. Inventing a wizard from a one-line description will produce a
  different man every episode, and a wizard who changes is not a wizard.

---

## 5. THE TEAL ORB — the recurring non-human
**Ref:** generated and stable (o1-mega-vending 6-tile verified: same sphere, same
cyan, same internal swirl, no morph, no resize)

- Glowing teal/cyan sphere, small enough to hold in one hand
- Pale white swirling internal pattern, like contained energy
- **Speaks by pulsing once. Never opens a mouth. Never has a face.**
- The Old UNC narrates its silence. That is the entire relationship.
- **Memetic anchor — the "gas mask" equivalent.** It must be the most
  recognizable silhouette in the frame. If it drifts, the network loses its logo.
- Appears as: Satire contestant, Public Access prop, Labs logo. Same design
  everywhere, to the pixel.

---

## 6. CLOUDWU — flying cloud-wizards
**Ref:** none. Listed in UNIVERSE personas. Neon clouds, sunglasses, flying.
- Treat as a bumper-only element until a ref exists. Do not put them in an
  episode yet — a recurring character with no sheet is how continuity dies.

---

## RULES THAT APPLY TO ALL OF THEM
1. **2-3 references max per scene.** More and the model treats them as different
   people. That has bitten repeatedly.
2. **Give every reference a named job in the prompt text.** Never assume the
   model knows which board means what.
3. **Restate the identity lock in every prompt.** Permanent features only.
4. **Pin by screen position** — "centred", "on the left" survives cuts better
   than clothing descriptions.
5. **6-tile every load-bearing detail**, especially any text on clothing. Laura's
   wordmark proved the failure is real and it happens on the *show's own canon*.
6. **Landscape, centred, never cropped.** See the reframe arithmetic in
   TRANSFER_PROTOCOL — off-centre props get deleted by a 9:16 crop.

~ WuBu ~
