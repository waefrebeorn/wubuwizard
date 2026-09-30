# 🔬 WHAT ACTUALLY WORKS — Tested & Verified
# Domain: Production Infrastructure | Chain: master-engine-compendium
# Date: 2026-08-18 | Method: Read each space's docs + discussions + tested

---

## RELIABILITY TIER LIST

### Tier 1: CONFIRMED WORKING (Tested Successfully)
| Space | What It Does | API Call | Notes |
|---|---|---|---|
| **declare-lab/TangoFlux** | Text→SFX | `predict(prompt, steps, guidance, duration)` | ✅ Generated 10s SFX in 6.2s |
| **suno/bark** | Text→Speech | `predict(text, history_prompt)` | ✅ Generated voice in 10.9s |

### Tier 2: WORKING BUT QUOTA LIMITED
| Space | What It Does | Issue | Workaround |
|---|---|---|---|
| **stabilityai/stable-audio-3** | Text→Music | ZeroGPU quota (0s left) | Try at midnight UTC, or use non-ZeroGPU spaces |
| **multimodalart/minimax-h3** | Image→Video | ZeroGPU quota | Same |
| **multimodalart/minimax-h3-audio-to-video** | Image+Audio→Video | ZeroGPU quota | Same |

### Tier 3: BROKEN / OVERLOADED (Discussions Confirm)
| Space | What It Does | Issue | Source |
|---|---|---|---|
| **facebook/MusicGen** | Text→Music | "incredibly busy", "NOT working", "always down" | 10+ discussion threads |
| **sanchit-gandhi/musicgen-streaming** | Text→Music | seed max is 10, still fails silently | Tested + error |
| **MiniMaxAI/MiniMax-Music3** | Text→Music | 11 endpoints, complex, errors | Not tested yet |

### Tier 4: NEED TO TEST (Not Yet Verified)
| Space | What It Does | Why It Might Work |
|---|---|---|
| **ACE-Step/ACE-Step** | Text→Music | 12 endpoints, music foundation model |
| **ASLP-lab/DiffRhythm** | Text→Song | "Blazingly Fast" |
| **ameerazam08/stableaudio-open-1.0** | Text→Audio | Stable Audio Open, no ZeroGPU mentioned |
| **Qwen/Qwen3-TTS** | Text→Speech | 3 endpoints, voice design |
| **nari-labs/Dia-1.6B** | Text→Dialogue | 1 endpoint, simple |
| **neuphonic/neutts-air** | Voice Cloning TTS | 1 endpoint |
| **KittenML/KittenTTS-Demo** | Text→Speech | 1 endpoint, simple |
| **ResembleAI/chatterbox-turbo-demo** | Text→Speech | 1 endpoint |
| **zai-org/CogVideoX-2B-Space** | Text→Video | 2 endpoints |
| **zai-org/CogVideoX-5B-Space** | Text/Image→Video | 3 endpoints |
| **ByteDance/AnimateDiff-Lightning** | Text→Video | 2 endpoints, fast |

---

## WHY SPACES FAIL (Lessons Learned)

1. **ZeroGPU Quota** — HF free tier gives 90s of GPU time per request. Big models (Stable Audio 3, MiniMax-H3) exhaust it instantly.
   - Fix: Try at different times of day, or use smaller models

2. **Wrong API Name** — Many spaces have different endpoint names than expected
   - Fix: Always call `client.view_api()` first to get exact names

3. **Parameter Range Errors** — e.g., MusicGen streaming seed must be ≤10
   - Fix: Read the actual parameter ranges from view_api()

4. **Spaces Overloaded** — Popular spaces (MusicGen) have thousands of users
   - Fix: Use less popular alternatives

5. **melodies Required but Optional** — MusicGen's melodies param is required but can be None
   - Fix: Pass `handle_file(None)` or check if param truly accepts None

---

## MINIMAX-H3 PROMPT STRUCTURE (From Official Guide)

The MiniMax-H3 requires a VERY specific prompt structure:

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Cinematic, a deadpan man with an afro stands in a liminal hallway with fluorescent lights. The camera holds a static shot as he stares directly at the viewer. He wears a black hoodie with yellow WuBu lettering and holds a teal orb staff.

overall_soundscape: Fluorescent lights hum steadily. Distant air conditioning drone. Footsteps echo faintly.

non_diegetic_music: Dark bass trap beat, 80 BPM, eerie atmosphere
```

This is the CORRECT way to prompt MiniMax-H3, not just a simple sentence.

---

## NEXT STEPS

1. Test Tier 4 spaces systematically using correct API names
2. Use MiniMax-H3's official prompt structure when quota allows
3. For music: try ACE-Step, DiffRhythm, Stable Audio Open (non-ZeroGPU)
4. For voice: we have Bark working, try Qwen3-TTS and KittenTTS
5. For video: try CogVideoX and AnimateDiff (simpler, less quota)

~ WuBu ~
