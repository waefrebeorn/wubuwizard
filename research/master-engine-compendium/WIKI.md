# 📚 ~ WUBU ~ MASTER WIKI — Complete Knowledge Base
# Domain: Meta/Organization | Chain: master-engine-compendium
# Date: 2026-08-18 | Files: ~175 | Size: ~750KB
# "Know what we have. Know what we don't."

---

> ## ⚠️ AMENDED 2026-08-18 — THIS WIKI IS STALE BELOW THE FOLD
> The Production pipeline section claims video generation is **"BLOCKED"** and
> **"Videos generated: 0"** and that **image_generate is confirmed working**.
> All three are FALSE as of 2026-08-18 (Triple-DA verified):
> - Gateway text-to-video is **PROVEN WORKING** (LTX-2.3, 720p, audio free).
> - **21 clips** exist in `WuBuTV/REEL/`.
> - `image_gen` provider is **NOT currently registered** (the WIKI's "tested
>   working" claim is stale prose).
> - `PRODUCTION/` was dissolved; its assets moved into **`WuBuTV/`** (the
>   canonical TV home). The live truth lives in `WuBuTV/README.md`,
>   `WuBuTV/STANDARDS.md`, `WuBuTV/LIBRARY.md`, `WuBuTV/CHANNELS.md`.
> Treat this WIKI as historical research context only. Do not use its pipeline
> status as current fact.

## 📊 THE NUMBERS

| Metric | Value |
|---|---|
| Total research files | ~180 |
| Total knowledge | ~760KB+ |
| Chains | 15 + PRODUCTION/ |
| Video generation spaces tested | 65+ |
| Working (free, no quota) | 5 (4 audio + 1 image) |
| Working (ZeroGPU quota limited) | 26 (exhausted) |
| Broken/down | 12+ |
| Images generated | 10 (9 channels + 1 portrait) |
| Audio files generated | 18 (9 voices + 9 SFX) |
| Videos generated | 0 (non-slop) |

---

## 🎬 PRODUCTION PIPELINE (Designed, Not Yet Executed)

### Step 1: Image Generation ✅ TOOL EXISTS
- **Tool**: `image_generate` (FAL.ai, FLUX 2 Klein 9B)
- **Cost**: Uses FAL credits (Nous subscription)
- **Status**: Confirmed working — generated `first_frame_old_unc.png`
- **Output**: `PRODUCTION/first_frame_old_unc.png` (1.18MB)

### Step 2: Video Generation ❌ BLOCKED
- **Problem**: ALL HuggingFace video spaces use ZeroGPU
- **Free quota**: 90 seconds per request — exhausted
- **Without HF token**: Limited to ~3-5 generations per day across ALL spaces
- **Spaces confirmed broken**: MusicGen, AnimateDiff-Lightning, LTX-2.5 workflow
- **Spaces confirmed quota-exhausted**: Stable Audio 3, CogVideoX-2B/5B, MiniMax-H3

### Step 3: Audio Generation ✅ PARTIALLY WORKING
- **TangoFlux** ✅ — Text→SFX, no ZeroGPU, generated ambient SFX
- **Bark** ✅ — Text→Speech, no ZeroGPU, generated Old UNC voice
- **MusicGen** ❌ — Broken ("always down", "incredibly busy")
- **Stable Audio 3** — ZeroGPU quota exhausted
- **ACE-Step** — Not yet tested (likely ZeroGPU)

### Step 4: Post-Processing ✅ TOOL EXISTS
- **Tool**: ffmpeg (installed)
- **Pipeline**: Color grading, VHS effects, text overlay, audio mixing
- **Status**: Recipes documented, not yet tested on real output

### Step 5: Distribution ❌ NO ACCOUNTS
- No TikTok, IG, YouTube, or Telegram channel set up
- Strategy documented but not executed

---

## 🔧 CONFIRMED WORKING TOOLS

### Free (No Quota) — 4 Audio Tools + 1 Image Tool
| Tool | What | Space | Status |
|---|---|---|---|
| KittenTTS | Text→Speech | KittenML/KittenTTS-Demo | ✅ Tested |
| Qwen3-TTS | Voice Design | Qwen/Qwen3-TTS | ✅ Tested (best quality) |
| Bark | Text→Speech | suno/bark | ✅ Tested |
| TangoFlux | Text→SFX | declare-lab/TangoFlux | ✅ Tested |
| FAL image_generate | Text→Image | FAL (Nous Portal) | ✅ Tested |

### Quota-Limited (ZeroGPU)
| Tool | What | Status |
|---|---|---|
| CogVideoX-2B/55B | Text/Image→Video | Quota exhausted |
| MiniMax-H3 | Image+Audio→Video | Needs 302s GPU (too long) |
| Stable Audio 3 | Text→Music | Quota exhausted |
| All Wan2.2 spaces | Image→Video | frame_multiplier bug |

### Broken
| Tool | What | Issue |
|---|---|---|
| MusicGen | Text→Music | "Always down" |
| AnimateDiff-Lightning | Text→Video | TypeError |
| LTX-2.5 workflow | Image→Video | Silent error |
| MusicGen Streaming | Text→Music | seed≤10, still fails |
| Stable Audio Open | Text→Audio | RuntimeError |

---

## 🧠 KNOWLEDGE CHAINS

### chain-writing-room/ (25 lilypads)
Comedy writing theory: joke structure, South Park philosophy, misdirection, joke stacking, absurdist comedy, etc.

### chain-comedy-deep/ (25 research nodes)
Deep comedy research: 7-step Kevin Bacon chains, character comedy, dark humor mechanics

### chain-bootleg/ (25 lilypads)
Visual grammar: trademark-safe prompt engineering for FLUX video

### chain-television/ (13 files)
TV research: Roddenberry's "world with rules", genre overlap, 25-lilypad TV universe

### chain-creators/ (2+ files)
AI creator ecosystem: Smoking Wizard, Kane Parsons blueprint

### chain-production/ (1 file)
AI video landscape 2026: tools, roadblocks

### PRODUCTION/ (20+ files)
- 9 channel bumper scripts
- ffmpeg post-processing pipeline
- Audio post-production & thumbnail design
- Distribution strategy (TikTok algorithm, hooks)
- Growth strategy (Buffer data: 2-5 posts/week)
- Community building & pitching
- Complete API reference (all 54 spaces)
- What Actually Works (tier list)
- Self-assessment of weaknesses
- Gap map
- Loop log

---

## 🚧 THE BLOCKER

**We cannot generate video because:**
1. ALL HuggingFace video spaces use ZeroGPU
2. Free ZeroGPU quota is exhausted (needs HF token to reset)
3. No HF token is configured on this machine
4. The spaces that DON'T use ZeroGPU (TangoFlux, Bark) only do audio

**What we CAN do:**
1. Generate images (FAL — uses subscription credits)
2. Generate audio SFX (TangoFlux — free)
3. Generate voice (Bark — free)
4. Post-process with ffmpeg (free)
5. Write scripts, plan strategy (free)

**What we NEED to proceed:**
1. **HF token** — to reset ZeroGPU quota OR
2. **A non-ZeroGPU video generation method** — local GPU? different platform? OR
3. **A paid video generation subscription** — Kling, HeyMarmot, etc.

---

## 📋 FILES GENERATED

### 9 Channel First Frames (FAL FLUX 2 Klein)
| File | Size | Channel |
|---|---|---|
| `first_frame_news.png` | 1.03MB | NEWS |
| `first_frame_originals.png` | 981KB | ORIGINALS |
| `first_frame_satire.png` | 1.10MB | SATIRE |
| `first_frame_labs.png` | 881KB | LABS |
| `first_frame_mega.png` | 973KB | MEGA |
| `first_frame_archive.png` | 1.08MB | ARCHIVE |
| `first_frame_legacy.png` | 1.10MB | LEGACY |
| `first_frame_transmission.png` | 1.21MB | TRANSMISSION |
| `first_frame_spectacle.png` | 1.00MB | SPECTACLE |

### 9 Channel Voice Lines (Qwen3-TTS + KittenTTS)
| File | Size | Tool |
|---|---|---|
| `audio_bumpers/news_voice.wav` | 379KB | Qwen3-TTS |
| `audio_bumpers/originals_voice.wav` | 260KB | Qwen3-TTS |
| `audio_bumpers/satire_voice.wav` | 375KB | Qwen3-TTS |
| `audio_bumpers/labs_voice.wav` | 306KB | Qwen3-TTS |
| `audio_bumpers/mega_voice.wav` | 262KB | KittenTTS |
| `audio_bumpers/archive_voice.wav` | 366KB | KittenTTS |
| `audio_bumpers/legacy_voice.wav` | 359KB | KittenTTS |
| `audio_bumpers/transmission_voice.wav` | 404KB | KittenTTS |
| `audio_bumpers/spectacle_voice.wav` | 430KB | KittenTTS |

### 9 Channel SFX (TangoFlux)
| File | Size |
|---|---|
| `audio_bumpers/*_sfx.wav` | 882KB each |

### Other Files
| File | Type | Size | Source |
|---|---|---|---|
| `first_frame_old_unc.png` | Image | 1.18MB | FAL FLUX 2 Klein |
| `portrait_old_unc.png` | Image | 1.21MB | FAL FLUX 2 Klein |
| `audio_ambient_sfx.wav` | Audio | 1.76MB | TangoFlux |
| `audio_old_unc_voice.wav` | Audio | 263KB | Bark |
| `audio_kittentts.wav` | Audio | 362KB | KittenTTS |
| `audio_qwen3tts.wav` | Audio | 360KB | Qwen3-TTS |

---

## 🔄 LOOP STATUS

- Iteration 1: KNOW ✅ (research built)
- Iteration 2: KNOW ✅ (gaps closed, toolkit found)
- Iteration 3: KNOW ✅ (growth strategy, content matrix)
- Iteration 4: KNOW ✅ (craft improvement, LUTs, comedy timing)
- Iteration 5: KNOW ✅ (complete API reference, what works)
- Iteration 6: KNOW ✅ (post-production workflow, what makes non-slop, banned talking heads)
- Iteration 7: KNOW ✅ (Kuleshov effect, 4 dimensions of editing, comedy timing techniques, non-slop pipeline)
- Iteration 8: APPLY ✅ (9 channel first frames + 9 voice lines + 9 SFX = 27 assets generated)
- MEASURE: ⏸️ (need video to measure)
- LEARN: ⏸️ (need completed video to learn)

## 🎬 KEY LEARNING: The Kuleshov Effect
**Meaning comes from the JUXTAPOSITION of shots, not the shots themselves.**
- Same face + soup = hunger
- Same face + coffin = grief
- Same face + woman = desire

**This is what we were bad at**: generating single shots and not thinking about how they COMBINE.

**The fix**: Generate shots SEPARATELY, edit them TOGETHER with intention.

## 💰 KEY LEARNING: Nous Portal FAL Video = THE PATH
**Seedance 2.0 image-to-video is the winning pipeline:**
- Cost: $0.0147 flat (4-15 seconds, same price!)
- Input: Our FLUX first frame + prompt
- Output: High-quality video with optional native audio
- Reference: Up to 9 images + 3 videos + 3 audio clips
- Character consistency: YES (reference image maintains identity)

**Cost per bumper: $0.0263** (FLUX first frame $0.0116 + Seedance $0.0147)
**Cost for 100 bumpers: $2.63**

## 🚫 BANNED (Slop)
- Talking heads (LeapTalk, etc.)
- Static single-shot clips with no timing
- "Just add VHS filter" found footage
- Random AI video with no setup-payoff structure

---

## 🎯 HONEST ASSESSMENT

**We are at ~3%.**

**What WORKS:**
- FAL video generation through Nous Portal gateway (text-to-video)
- Image-to-video via Serveo tunnel + local HTTP server
- Audio generation (free on most models)
- 12 videos generated (3 with audio)

**What we LACK:**
- Character consistency (I used generic descriptions, NOT our actual characters)
- Proper branding (no `~ WuBu ~` watermark on generated content)
- Fast-paced content (user wants HIGH INTENSITY, not slow staring)
- Using our existing assets (40 raw clips, 100 location sets, 7 character sheets)

**CRITICAL LEARNINGS 2026-08-18:**
1. Our character is SPECIFIC: Black man, afro, mustache+goatee, "WuBu" hoodie (yellow letters), 2D cartoon style
2. User wants FAST-PACED, HIGH-INTENSITY, HIGH-COLOR content — NOT slow liminal staring
3. `~ WuBu ~` branding MUST be on every video
4. We have 9 album tracks, 40 raw clips, 100 location sets, 7 character sheets — USE THEM
5. Always generate with AUDIO (it's free)
6. Image-to-video needs public URL (Serveo tunnel works)

**Budget spent: ~$3.50 of $5.00**
**Remaining: ~$1.50**

~ WuBu ~
