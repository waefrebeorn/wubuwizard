# 📚 ~ WUBU ~ TELEVISION — ASSET LIBRARY (real inventory)
# Generated 2026-08-18 by scanning the actual moved files (ffprobe).
# Status legend: GOOD = usable, BROKEN = wrong character / no branding,
# RAW = unrendered source, REGEN = needs regeneration.

---

## REEL INVENTORY (21 clips, 53.0 MB total)

### 📺 NEWS  (6 clips, 16.9 MB)
| File | Dur | A/V | Notes |
|------|-----|-----|-------|
| wubutv_news_v3.mp4 | 7.0s | V+A | breathing-hallway bumper, I2V |
| wubutv_news_breathinghallway_v2.mp4 | 7.0s | V+A | alt take |
| wubutv_news_i2v_v4.mp4 | 10.1s | V+A | image-to-video |
| video_news_seedance_v1.mp4 | 15.0s | V | Seedance, no audio track |
| video_news_ltx_v1.mp4 | 5.0s | V | LTX, no audio |
| video_news_flux3_v1.mp4 | 5.0s | V | FLUX-3, no audio, PREMIUM |

### 🎬 ORIGINALS (incl MEGA flagship)  (6 clips, 20.9 MB)
| File | Dur | A/V | Notes |
|------|-----|-----|-------|
| wubutv_originals_v3.mp4 | 7.0s | V+A | door bumper |
| wubutv_originals_door_v2.mp4 | 7.0s | V+A | alt take |
| wubutv_originals_i2v_v4.mp4 | 10.1s | V+A | I2V |
| video_originals_seedance_v1.mp4 | 15.0s | V | Seedance, no audio |
| video_originals_ltx_v1.mp4 | 5.0s | V | LTX, no audio |
| video_mega_ltx_v1.mp4 | 5.0s | V | MEGA flagship cut, no audio |

### 🎭 SATIRE (incl LEGACY)  (3 clips, 6.6 MB)
| File | Dur | A/V | Notes |
|------|-----|-----|-------|
| wubutv_satire_gameshow_v2.mp4 | 7.0s | V+A | game-show bumper |
| wubutv_satire_i2v_v4.mp4 | 10.1s | V+A | I2V |
| video_satire_ltx_v1.mp4 | 5.0s | V | LTX, no audio |
> ⚠️ `video_legacy_ltx_v1.mp4` (LEGACY→SATIRE) was DELETED in cleanup — must REGEN.

### 🔬 LABS (incl ARCHIVE)  (2 clips, 2.8 MB)
| File | Dur | A/V | Notes |
|------|-----|-----|-------|
| video_labs_ltx_v1.mp4 | 5.0s | V | LTX, no audio |
| video_archive_ltx_v1.mp4 | 5.0s | V | ARCHIVE→LABS, no audio |

### 🏟️ ARENA (SPECTACLE)  (1 clip, 1.4 MB)
| File | Dur | A/V | Notes |
|------|-----|-----|-------|
| video_spectacle_ltx_v1.mp4 | 5.0s | V | no audio |

### 🎙️ TALK (TRANSMISSION)  (1 clip, 1.8 MB)
| File | Dur | A/V | Notes |
|------|-----|-----|-------|
| video_transmission_ltx_v1.mp4 | 5.0s | V | no audio |

### 🚫 _QUARANTINE  (2 clips)
| File | Why |
|------|-----|
| video_bumper_talking_head.mp4 | BANNED slop (talking head) |
| test_ltx.mp4 | pipeline test, broken character |

---

## CHARACTER STATUS — ALL PRIOR RENDERS BROKEN (DA-verified 2026-08-18)
Vision-QA on the old `first_frame_*.png` (now `title_card_*.png`) proved they are
**CHANNEL TITLE CARDS** — logo graphics with "WUBU <CHANNEL>" text and a teal
mascot/orb/hallway. **NONE contain the Old UNC character.** Separately, the
rendered video clips used the WRONG character (orange/teal raglan hoodie, no
"WuBu" text, photoreal, not 2D cartoon). So:
- The "character frame" layer was entirely mislabeled — there are ZERO correct
  character keyframes on disk.
- Character keyframe must be regenerated (I2V from rapman `sheet_01.png`, or
  image-gen when re-enabled) and QA'd before any new video.
- The `title_card_*.png` files are VALID as channel logos — keep them, just
  don't confuse them with character references.

## TITLE CARDS (valid — one per channel, in REEL/<ch>/)
news, originals, mega, satire, legacy, labs, archive, transmission, spectacle
(9 cards). These are branding assets, not character locks.

---

## SOURCE MEDIA (NOT moved — lives in WuBuMovie/, separate git repo)
These are the RAW anchors. Listed as paths (not symlinks — per directive).
- Character sheets (7+): `/home/wubu/WuBuMovie/raw_footage/rapman/sheets/`
- Location sets (100): `/home/wubu/WuBuMovie/raw_footage/rapman/`
- Title cards (10): `/home/wubu/WuBuMovie/raw_footage/rapman/cards/`
- Raw clips (40): `/home/wubu/WuBuMovie/raw_footage/wubuclips/`
- Album tracks (9): `/home/wubu/WuBuMovie/audio/music/`
- Movie cuts: `/home/wubu/WuBuMovie/AGI_MADE_THIS_MOVIE/`

---

## ACTION REQUIRED
1. REGEN SATIRE/LEGACY clip (deleted in cleanup).
2. REGEN correct character first-frames (all channels) — black hoodie, yellow
   "WuBu", 2D cartoon, `~ WuBu ~` watermark.
3. Re-render top bumpers with AUDIO ON (most `video_*` clips lack audio).
4. Fill empty channels: COMMERCIAL, SITCOM, HORROR (scripts exist in SCRIPTS.md).

~ WuBu ~
