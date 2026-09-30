# 📺 ~ WUBU ~ TELEVISION — THE NETWORK
# Canonical home for WuBuTV. Established 2026-08-18.
# "We are formless. We are the discipline." — The Colonel (re: context, not content)

---

## WHAT THIS IS
WuBuTV is the **television layer** of the WuBu universe — a 9-channel network
broadcasting from the liminal spaces. This directory is the single source of
truth for channel structure, branding, and the asset LIBRARY. It does NOT
duplicate source media; it SYMLINKS to the real footage in `WuBuMovie/`.

## THE 9 CHANNELS (CANONICAL TAXONOMY — v1)
Unified from the produced-clip set (8 distinct + MEGA flagship). Every channel
maps to a recurring bit from the movie's world rules (chain-television/21).

| # | Channel | Handle | Format | Recurring Bit |
|---|---------|--------|--------|---------------|
| 1 | NEWS | @WuBuNews | Colonel's Codec Call | "The Order Reports" / breathing hallway briefing |
| 2 | ORIGINALS | @WuBuOriginals | Found-footage series | "The MEGA" / impossible doors, vending-machine lore |
| 3 | SATIRE | @WuBuSatire | Seymour Ponderable | "The Order's Game Show" / teal-orb contestant |
| 4 | LABS | @WuBuLabs | Technique breakdown | "Found Room, Generated Crowd" / HALLWAY-AID infomercial |
| 5 | ARENA | @WuBuArena | Sports broadcast | "Height Opposition" (height competition) |
| 6 | SITCOM | @WuBuSitcom | Multi-cam sitcom | "The Short Family" (height inversion) |
| 7 | HORROR | @WuBuHorror | Analog horror | "Liminal Dread" / the hallway breathes |
| 8 | TALK | @WuBuTalk | Talk show | "The Old UNC Interviews" / teal-orb guest |
| 9 | COMMERCIAL | @WuBuCommercial | Ad break | "WuBu Products" / storage units with galaxies inside |

> NOTE: The earlier flat `PRODUCTION/` first-frames used a DIFFERENT set of
> names (mega/archive/legacy/transmission/spectacle). Those are now folded into
> the 9-channel grid above: MEGA→ORIGINALS flagship, ARCHIVE→NEWS, LEGACY→SATIRE,
> TRANSMISSION→TALK, SPECTACLE→ARENA. See `LIBRARY.md` for the mapping.

## THE BRAND
- **Name:** ~ WuBu ~ (tildes + spaces)
- **Watermark:** `~ WuBu ~` burned into EVERY frame
- **Style:** Found-footage VHS REC-cam — chroma bleed, lifted blacks, scanlines,
  vignette, tape noise, REC/date burn-in
- **Audio:** Wizardcore — bass-driven, hypnotic, dark trap, dreamlike (9 album tracks)
- **No narrator:** diegetic only; on-screen text (TikTok text-clip aesthetic)

## THE CHARACTER (LOCKED — authoritative spec)
**The Old UNC** (host / "The Voice"):
- Black man, **LARGE ROUNDED AFRO**
- **THICK MUSTACHE + GOATEE** (soul patch)
- **BLACK hoodie with "WuBu" in YELLOW/GOLD letters**  ← canonical
- Blue jeans, brown shoes, stocky/heavyset build
- Glowing **teal-orb staff**
- Deadpan philosopher, gravelly flat affect
- **2D CARTOON STYLE**: thick black outlines, flat colors, cel-shaded
- Sheet ref: `WuBuMovie/raw_footage/rapman/sheets/sheet_01.png`

> Earlier assets (first_frame_*.png, portrait_old_unc.png) rendered the WRONG
> character (orange/teal raglan hoodie, no "WuBu" text, photoreal). Those are
> quarantined in `LIBRARY.md` as BROKEN and must be regenerated before use.

## DIRECTORY MAP (real files, moved — no symlinks)
```
WuBuTV/
├── README.md          (this file — network bible)
├── STANDARDS.md       (work standards + QC rubric + pre-gen gate)
├── BRAND.md           (watermark spec, audio style, VHS LUT recipe, current file map)
├── LIBRARY.md         (asset inventory + status GOOD/BROKEN/REGEN)
├── CHANNELS.md        (per-channel specs: format, voice, cadence, first-post)
├── SCRIPTS.md         (9 bumper scripts, corrected character, intensity rules)
├── PIPELINE.md        (verified generation path: gateway + model + cost)
├── LOOP_LOG.md        (session history — AMENDED reorg notes)
├── BOOT.md            (next-session startup)
├── WRITING_ROOM/      (the creative engine)
│   ├── README.md           (workflow + planning layer pointer)
│   ├── CHARACTER_BIBLE.md  (canonical cast + world rules + memetic design)
│   ├── CRAFT_INDEX.md      (185 research files → "what each gives the room")
│   ├── CHANNEL_BRIEFS.md   (9 briefs: rule + engine + bit + prompt seed)
│   ├── SCRIPT_TEMPLATE.md  (fillable beat sheet + canonical prompt)
│   ├── JOKE_TEST.md        (the 5-question gate — pass or don't generate)
│   └── PLANNING/           (meta-layer: plan any task with our 3 info skills)
│       ├── PLANNING_SYSTEM.md   (methodology: Kevin Bacon + 25-lilypad + Triple DA)
│       ├── PLAN_TEMPLATE.md     (fillable plan for any writing task)
│       └── EXAMPLE_PLAN_SATIRE.md (worked plan, real citations)
├── REEL/              (organized output — one dir per channel + _QUARANTINE)
├── ASSETS/audio/      (voice lines + SFX, moved from WuBuMovie)
└── docs/             (reference notes: ffmpeg, roadblocks, assessment)
├── ASSETS/audio/      (voice lines + SFX, moved from old PRODUCTION/)
└── docs/             (reference notes: ffmpeg, distribution, craft, etc.)
```

> Source media (character sheets, raw clips, album) stays in WuBuMovie/ — a
> separate git repo. Paths are listed in LIBRARY.md. We do NOT duplicate or
> move those gigabytes; only WuBuTV's own production output lives here.

## GENERATION STATUS (as of 2026-08-18, DA-verified)
- ✅ Gateway text-to-video PROVEN WORKING (LTX-2.3, 720p, audio included free) — executed 2026-08-18
- ✅ 21 clips on disk (REEL/), sorted into 9 channels + _QUARANTINE
- ⚠️ Character consistency BROKEN in all prior renders (wrong hoodie; and the
  old "first_frame_*.png" were actually TITLE CARDS, not character frames — see LIBRARY.md)
- ⚠️ 11 of 21 clips have NO audio track (legacy silence bug) — must re-render w/ audio
- ⚠️ Image-gen provider NOT currently registered → character keyframe must come
  from rapman `sheet_01.png` I2V until re-enabled
- ❌ WIKI.md / INDEX.md are STALE (claim video "blocked", "0 videos", "11 bumper
  scripts") — superseded by this dir. See STANDARDS.md for the live truth.

~ WuBu ~
