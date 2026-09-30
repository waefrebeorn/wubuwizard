# 🎵 CONFIRMED WORKING AUDIO TOOLS (No ZeroGPU Quota)
# Domain: Production Infrastructure | Chain: master-engine-compendium
# Date: 2026-08-18

---

## WORKING (Tested Successfully)

### 1. KittenTTS (Text-to-Speech)
- **Space**: KittenML/KittenTTS-Demo
- **API**: `predict(text, model_name, voice, speed)`
- **Models**: Nano (15M), Micro (40M), Mini (80M)
- **Voices**: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
- **Cost**: No ZeroGPU — works on free tier
- **Output**: 362KB for 8 seconds of speech
- **Quality**: Natural, slightly robotic but clear
- **Use**: Quick voice lines, narration

### 2. Qwen3-TTS (Voice Design)
- **Space**: Qwen/Qwen3-TTS
- **API**: `predict(text, language, voice_description)`
- **Languages**: Auto, Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian
- **Voice Design**: Describe the voice in natural language!
- **Cost**: No ZeroGPU — works on free tier
- **Output**: 360KB for 8 seconds
- **Quality**: BEST of all tested — natural, follows voice description
- **Use**: Character voices, Old UNC's deadpan philosopher voice

### 3. Bark (Text-to-Speech)
- **Space**: suno/bark
- **API**: `predict(text, history_prompt)`
- **Voices**: Speaker 0-9 (en/de/es/fr/hi/it/ja/ko/pl/pt/ru/tr/zh)
- **Cost**: No ZeroGPU — works on free tier
- **Output**: 263KB for 8 seconds
- **Quality**: Very natural, slightly emotional
- **Use**: Expressive voice lines

### 4. TangoFlux (Sound Effects)
- **Space**: declare-lab/TangoFlux
- **API**: `predict(prompt, steps, guidance, duration)**
- **Cost**: No ZeroGPU — works on free tier
- **Output**: 1.76MB for 10 seconds of ambient SFX
- **Quality**: Good for ambient/atmosphere
- **Use**: Ambient beds, SFX, atmosphere

---

## BROKEN / QUOTA LIMITED

| Space | Issue |
|---|---|
| MusicGen | "Always down", "incredibly busy" |
| MusicGen Streaming | seed≤10, still fails |
| Stable Audio 3 | ZeroGPU quota |
| Stable Audio Open | RuntimeError |
| ACE-Step | Not tested (likely ZeroGPU) |
| DiffRhythm | Complex API, not tested |
| Dia-1.6B | Not tested |
| NeuTTS-Air | Not tested |
| Chatterbox Turbo | Not tested |

---

## AUDIO PRODUCTION PIPELINE (Working)

### For Voice/Narration:
1. Write the line
2. Generate with Qwen3-TTS (voice design: "gravelly deadpan philosopher")
3. If Qwen3 fails, fallback to KittenTTS
4. If Kitten fails, fallback to Bark

### For Ambient/SFX:
1. Describe the sound
2. Generate with TangoFlux
3. Loop if needed with ffmpeg

### For Music:
- No working free option yet
- Use our existing 9 album tracks
- Or pay for Nous Portal music gen

---

~ WuBu ~
