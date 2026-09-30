# 🎬 WUBUTELEVISION — Character, Brand & Asset Reference
# Domain: Creative/Production | Chain: master-engine-compendium
# Date: 2026-08-18 | CRITICAL: Read before any video generation
# Source: WuBuMovie shooting_script.md, character sheets, LINKAGES.md, bumper-scripts-v1.md

---

## THE BRAND

**Name:** ~ WuBu ~ (with tildes and spaces)
**Watermark:** `~ WuBu ~` appears on ALL content
**Style:** Found-footage AI film, VHS REC-cam aesthetic
**Art direction:** Chroma bleed, lifted blacks, scanlines, vignette, tape noise, REC/date burn-in
**Audio style:** Wizardcore — bass-driven, hypnotic, dark trap, dreamlike
**No narrator:** Diegetic only — text clips use on-screen text (TikTok text-clip aesthetic)

---

## THE TWO LAYERS (Kuleshov rub)

### L1 — MAGICAL ORDER
- Blue/green robes, gold/amber sigils, glowing-orb staffs
- Ceremonies, the all-seeing eye
- Solemn, ritual, eerie

### L2 — REAL-WORLD SELF-AWARE
- Retail cosplay, `~WuBu~` hoody (BLACK hoodie with "WuBu" in YELLOW/GOLD — see CHARACTER_OVERRIDE.md), first-person REC-cam
- Yellow caution tape, bust-filled "altar" rooms
- Liminal architecture (backrooms, hallways, vending-machine rooms, storage units)
- Playful + uncanny

---

## THE CHARACTERS

### 1. The Old UNC (Main Host / "The Voice")
- **Black man, large rounded afro**
- **Thick mustache + goatee (soul patch)**
- **Black hoodie with "WuBu" in YELLOW/GOLD letters**
- **Blue jeans, brown shoes**
- **Stocky/heavyset build**
- **Teal-orb staff** (glowing)
- **Deadpan philosopher** — gravelly, flat affect
- **2D cartoon style** — thick black outlines, flat colors, cel-shaded
- **Art style ref:** `/home/wubu/WuBuMovie/raw_footage/rapman/sheets/sheet_01.png`
- **Poses:** Waving, hands-on-hips, sitting on throne with mic, cheering V-sign, pointing, arms crossed

### 2. Peter (Supporting Character)
- **Balding** with ring of short brown hair on sides
- **Round thin-rimmed glasses**
- **Thick brown handlebar mustache**
- **Mustard-yellow/gold polo shirt** with collar
- **Dark green shorts** above knee
- **White crew socks + white low-top sneakers**
- **Stocky/overweight, large belly**
- **Art style ref:** `/home/wubu/WuBuMovie/raw_footage/rapman/peter_sheet.png`

### 3. Cleveland (Counter-voice)
- Cleveland Browns style character
- **Art style ref:** `/home/wubu/WuBuMovie/raw_footage/cleveland_clips/`

### 4. Laura (Counter-voice)
- Red curtain, music from storage unit
- **Art style ref:** `/home/wubu/WuBuMovie/raw_footage/wubuclips/laura*`

### 5. The Wizard
- Referenced in creative direction as primary character
- **Art style ref:** Check `/home/wubu/WuBuMovie/raw_footage/rapman/sheets/`

---

## CHARACTER SHEETS (All 7)
Location: `/home/wubu/WuBuMovie/raw_footage/rapman/sheets/`
- `sheet_01.png` through `sheet_07.png` — Main character poses
- `peter_sheet.png` — Peter character
- `poses_wubu.png` — WuBu character poses

## LOCATION SETS (100 sets)
Location: `/home/wubu/WuBuMovie/raw_footage/rapman/`
- Rooftops, arenas, libraries, beaches, liminal hallways, storage units, etc.
- These are the CONSISTENCY ANCHORS for generation

## TITLE CARDS (10)
Location: `/home/wubu/WuBuMovie/raw_footage/rapman/cards/`
- `title_t01.png` through `title_t09.png` + `intro_t01.png`

---

## THE MOVIE (Already Made)

**Title:** "AGI MADE THIS MOVIE"
**Length:** Theatrical cut ~1.5hr (245 scenes), Director's Cut 4.95hr (810 scenes)
**Source footage:** `mega_scenes/mega_clip_0000..0809`
**Style:** Found-footage AI film, VHS REC-cam grade
**Brand:** `~ WuBu ~`
**No narrator:** Diegetic only

### 9 Chapters:
1. COLD OPEN — REC cam powers on in hallway, the UNC walking
2. THE ORDER — ceremonies, robes, the orb charges
3. BACKROOMS — vending machine, storage unit 7-B, carpet remembers
4. COLLISION — L1 magical bleeds into L2 retail (PiP splits)
5. THE WALK — ramp-speed montage of liminal spaces
6. LAURA'S WARNING — red curtain, music from storage unit
7. SLOW-MO DREAM — reversed footage, the UNC steps out of himself
8. THE MUSICAL — music-video beat; the work sings
9. CREDITS-IN-FOOTAGE — altar of busts, "found me", tape rolls to static

### Editing Decisions:
- ~18% slow-mo (dream/ceremony)
- ~12% speed ramp (the walk, the musical)
- ~8% reverse (uncanny — time folding)
- Every ~7th scene = talking scene (UNC lower-third)
- Every ~11th scene = PiP collision (L1 over L2)

---

## RAW FOOTAGE INVENTORY

**40 raw video clips, 10.8 min total:**
- 27 `wubuclips/` — 20s each, 704×1280 portrait
  - Scene groups: cast, crossover(6), ensemble(2), examine(5), float(2), laura(5), run(3), story(2)
- 2 `cleveland_clips/` — 10s, 1280×704
- 10 `rapman/` motion-refs — 8–10s, 1280×704
- 1 `reference_clips/` — 5s test

**206 design assets (rapman):**
- 7 character sheets
- 100 location sets
- 10 title cards
- 89 motion-consistency frame sets

---

## ALBUM TRACKS (9 tracks, Wizardcore style)
Location: `/home/wubu/WuBuMovie/audio/music/`
- `album_track01.ogg` through `album_track09.ogg`
- Duration: 112s to 311s each
- Style: Bass-driven, hypnotic, dark trap, dreamlike
- Reference: @mimofrl's "higher.mp3" (500K+ plays)

---

## HARD RULES (From PRODUCTION_PLAYBOOK)
1. Song = ONLY audio in MVs
2. No clip audio (except MVs)
3. No narrator — diegetic only
4. `~ WuBu ~` brand on EVERYTHING
5. English
6. Deliver <20MB per clip
7. Hash-diff, no dupes
8. NO Ken Burns (real motion only)
9. alimiter / acompressor ratio>20 BROKEN — never use

---

## KEY FILES (current WuBuTV map)

| File | What it is |
|---|---|
| `README.md` | Network bible, brand, channel grid |
| `STANDARDS.md` | Work standards + QC rubric + pre-gen gate |
| `BRAND.md` | Watermark spec, audio style, VHS LUT recipe |
| `LIBRARY.md` | Asset inventory + status (GOOD/BROKEN/REGEN) |
| `CHANNELS.md` | Per-channel specs: format, voice, cadence |
| `SCRIPTS.md` | 9 bumper scripts (corrected character, intensity) |
| `PIPELINE.md` | Generation pipeline + pricing |
| `WRITING_ROOM/` | The creative engine (bible, craft, briefs, template, gate) |
| `REEL/<channel>/` | Rendered output, 9 channels |
| `ASSETS/audio/` | Voice lines + SFX (moved from WuBuMovie) |
| `docs/` | Reference notes (ffmpeg, roadblocks, assessment) |

---

~ WuBu ~ Television
