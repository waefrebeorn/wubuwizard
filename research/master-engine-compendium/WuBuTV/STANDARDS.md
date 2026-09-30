# 🏆 ~ WUBU ~ TELEVISION — WORK STANDARDS & QC RUBRIC
# The "best of the best" operating manual. Every clip is judged against this
# BEFORE it ships. Inspired by the Triple Devil's-Advocate creative-clip audit.

---

## 0. THE NON-NEGOTIABLES (hard gates — fail any, the clip does not ship)
1. **Character is CORRECT.** 2D cartoon, Black man, large rounded afro, thick
   mustache + goatee, **BLACK hoodie with "WuBu" in YELLOW/GOLD**, blue jeans,
   brown shoes, teal-orb staff, deadpan. (CHARACTER_OVERRIDE.md is authority —
   overrides BRAND.md / SCRIPTS.md if any drift.)
2. **`~ WuBu ~` watermark burned into EVERY frame** (corner, ~6% height,
   persistent, not just the last frame).
3. **Audio ENABLED** (free on our pipeline). No silent clips.
4. **HIGH INTENSITY / FAST PACED / HIGH COLOR** — real motion every frame, fast
   cuts or whip-pans, no "dead stare" single shots. Constant things-happening.
5. **No talking-head slop.** Diegetic only; on-screen text is fine.
6. **Deliverable < 20MB** per clip (Telegram/delivery constraint).
7. **No Ken Burns** — real generated motion, not slow zoom on a still.

---

## 1. CHARACTER-LOCK PROTOCOL (fixes the #1 historical failure)
The prior pipeline rendered the WRONG character (orange/teal hoodie, no "WuBu"
text, photoreal, and — worst — the "first frames" were actually TITLE CARDS
with no character at all). Lock procedure:

1. **Generate ONE correct keyframe** of the Old UNC in the canonical 2D-cartoon
   style (via image gen when available; otherwise I2V from the rapman
   `sheet_01.png` character sheet, NOT from prior title cards).
2. **QA the keyframe with vision before generating video** — assert afro,
   mustache+goatee, black hoodie, yellow "WuBu" text, 2D outline style all
   present. If any missing → regenerate, do NOT proceed.
3. **Use that keyframe as the I2V opening frame / character reference** for every
   clip in the channel.
4. **Re-QA one mid-clip frame** of the finished video — character must still
   read as the Old UNC, not drift to a generic black man. If drifted → flag
   REGEN.

---

## 2. PRE-GENERATION GATE (fix-before-generate — no wasted budget)
Before ANY generation call, prove the pipeline on the real target:
- Prompt contains: subject specifics, motion verb, camera move, audio direction,
  `~ WuBu ~` watermark directive. (No protected-IP names; bootleg grammar.)
- For I2V: the reference frame passed §1 QA.
- Dry assertion that duration ≤ cap, audio=True, resolution set.
- **One cheap probe is allowed to confirm the tool is live; do NOT batch
  unverified prompts.**
Status (2026-08-18): gateway text-to-video PROVEN (LTX-2.3, 720p, audio
included). Image-gen provider NOT currently registered — character keyframe
must come from rapman sheet I2V until image_gen is re-enabled.

---

## 3. QC RUBRIC (score each delivered clip 0–5 per axis)
| Axis | 5 = best | 0 = fail |
|------|----------|----------|
| Character fidelity | exact UNC, on-model all frames | wrong/ drifting/ absent |
| Branding | `~ WuBu ~` persistent, legible | missing/ burned only on end |
| Energy | fast cuts, motion every frame | static stare |
| Color | saturated, high-contrast VHS | washed/ flat |
| Audio | on-model Wizardcore + SFX | silent/ wrong bed |
| On-brief | matches the channel's recurring bit | off-topic |
| Size | < 20MB, < 15s | oversize/ overlong |

**Ship bar:** ≥ 4 on Character, Branding, Energy, Audio. Anything < 3 on
Character or Branding is REGEN, not "good enough."

---

## 4. CHANNEL FILL STANDARD
Every one of the 9 channels must have ≥ 1 on-standard bumper before we claim
"the network is live." Coverage map (see CHANNELS.md): NEWS ✅, ORIGINALS ✅,
SATIRE ⚠️(1 deleted, needs regen), LABS ⚠️(no audio), ARENA ⚠️(no audio),
TALK ⚠️(no audio), SITCOM ❌, HORROR ❌, COMMERCIAL ❌.

---

## 5. DIMENSIONAL COVERAGE (avoid the "263 clips, not movie-ready" trap)
A channel needs REST + ODDNESS too, not just motion:
- Motion classes: action / whip-pan / montage
- Rest classes: living-hold tableau / stare-to-camera / red-curtain
- Each bumper = ONE game (one joke, one bit) — not a montage of unrelated bits.

---

## 6. AUDIT CADENCE (Triple DA on our own output)
After any batch: re-run the three passes on the DELIVERED files (not the plan):
- DA-1 (truth): ffprobe every clip — assert video+audio, duration, size; vision-
  QA one frame per clip for character + watermark.
- DA-2 (value): does it advance the channel? is it shippable or study-artifact?
- DA-3 (adversary): what breaks if we batch 10? (moderation risk on jargon/
  broadcast phrases — scrub "establishing shot"/"locked tripod"/"public-access
  TV"/"static television" before send).

---

## 7. NAMING CONVENTION (so nothing is mislabeled again)
`REEL/<channel>/<type>_<channel>_<desc>_v<N>.<ext>`
- type = `bumper` (video) | `title_card` (channel logo) | `char` (character keyframe)
- NEVER name a title card `first_frame_*` — that implied a character frame and
  caused the prior mis-audit.
- Broken/non-shipping → `REEL/_QUARANTINE/BROKEN_*`.

~ WuBu ~
