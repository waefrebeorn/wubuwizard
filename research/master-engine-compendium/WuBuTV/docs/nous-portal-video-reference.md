# 🎬 NOUS PORTAL FAL VIDEO GENERATION — Complete Reference
# Domain: Production Infrastructure | Chain: master-engine-compendium
# Date: 2026-08-18
# Sources: fal.ai API docs, Nous Portal pricing page, github.com/fal-ai/seedance-2.0-api

---

## THE KEY INSIGHT: "Reference" = Image/Video/Audio Conditioning

The most powerful FAL video models accept **references**:
- **Image references** (@Image1, @Image2... up to 9) — character consistency, style transfer, scene composition
- **Video references** (@Video1... up to 3) — motion transfer, style transfer
- **Audio references** (@Audio1... up to 3) — soundtrack, lip-sync

This means: generate a first frame with FLUX → feed it as reference → the video KEEPS our character/look.

---

## AVAILABLE MODELS (Through Nous Portal FAL Gateway)

### 1. Seedance 2.0 Reference to Video (MOST POWERFUL)
- **Endpoint**: `bytedance/seedance-2.0/reference-to-video`
- **Cost**: $0.0147/billable unit (Nous Portal)
- **Capabilities**: Up to 9 images + 3 videos + 3 audio references
- **Duration**: 4-15 seconds (or auto)
- **Resolution**: 720p, 1080p
- **Aspect ratios**: 16:9, 9:16, 1:1, 4:3, 3:4, 21:9, auto
- **Native audio**: YES (generate_audio parameter)
- **Features**: Character consistency, style transfer, motion transfer, lip-sync, multi-scene composition
- **Prompt syntax**: `@Image1`, `@Video1`, `@Audio1` to reference inputs
- **Scene cuts**: "Cut scene to..." in prompt for multi-scene videos

### 2. Seedance 2.0 Image to Video (SIMPLER)
- **Endpoint**: `bytedance/seedance-2.0/image-to-video`
- **Cost**: $0.0147/billable unit
- **Capabilities**: Single image + prompt
- **Duration**: 4-15 seconds
- **Features**: Start-to-end frame transition (image → video, or image → image)

### 3. Seedance 2.0 Fast Reference to Video (CHEAPER)
- **Endpoint**: `bytedance/seedance-2.0/fast/reference-to-video`
- **Cost**: $0.0074/billable unit (mini), $0.0147 (standard)
- **Capabilities**: Same as Reference to Video but faster/cheaper

### 4. Seedance 2.5 Image to Video (NEWER)
- **Endpoint**: `bytedance/seedance-2.5/image-to-video`
- **Cost**: $0.0225/billable unit
- **Capabilities**: Latest model, better quality

### 5. LTX-2.3 Text to Video (CHEAPEST)
- **Endpoint**: `fal-ai/ltx-2.3/text-to-video`
- **Cost**: $0.0017/billable unit (Nous Portal)
- **Per-second pricing**: $0.06/s at 1080p, $0.12/s at 1440p, $0.24/s at 2160p
- **Duration**: 6-20 seconds (fast model)
- **Resolution**: 1080p, 1440p, 2160p
- **FPS**: 24, 25, 48, 50
- **Aspect ratios**: 16:9, 9:16
- **Native audio**: YES
- **Features**: Portrait 9:16 support, last-frame interpolation, LoRA fine-tuning
- **Note**: Text-to-video only (no image reference), but very cheap

### 6. LTX-2.3 Image to Video
- **Endpoint**: `fal-ai/ltx-2.3/image-to-video`
- **Cost**: $0.0017/billable unit
- **Capabilities**: Image + prompt → video

### 7. FLUX-3 Text to Video (PREMIUM)
- **Endpoint**: `fal-ai/flux-3/text-to-video` (via blackforestlabs)
- **Cost**: $0.0893/billable unit
- **Capabilities**: FLUX-based, highest quality

### 8. MiniMax H3 Text to Video
- **Endpoint**: `fal-ai/minimax-h3/text-to-video`
- **Cost**: $0.273/billable unit
- **Capabilities**: Text + image + audio → video with soundtrack

---

## COST COMPARISON (Per Second of Video)

| Model | 720p | 1080p | 1440p | 4K |
|---|---|---|---|---|
| LTX-2.3 | — | $0.06/s | $0.12/s | $0.24/s |
| Seedance 2.0 mini | $0.0074 flat | — | — | — |
| Seedance 2.0 | $0.0147 flat | — | — | — |
| Seedance 2.5 | $0.0225 flat | — | — | — |
| FLUX-3 | $0.0893 flat | — | — | — |
| MiniMax H3 | $0.273 flat | — | — | — |

**For 5-second bumper at 1080p:**
- LTX-2.3: $0.30
- Seedance 2.0: $0.0147 (FLAT regardless of duration!)
- Seedance 2.5: $0.0225
- FLUX-3: $0.0893

**Seedance 2.0 is the clear winner** — flat pricing means a 15-second video costs the same as a 4-second one.

---

## THE WINNING PIPELINE FOR ~ WUBU ~

### Budget Path (Image → Video):
1. Generate first frame with FLUX 2 Klein ($0.0116)
2. Feed to Seedance 2.0 image-to-video ($0.0147)
3. **Total: $0.0263 per video**

### Quality Path (Reference → Video):
1. Generate first frame with FLUX 2 Klein ($0.0116)
2. Feed to Seedance 2.0 reference-to-video with @Image1 ($0.0147)
3. Add audio reference (@Audio1) from our generated voice/SFX
4. **Total: $0.0263 per video** (same cost, better quality!)

### Premium Path:
1. Generate first frame with FLUX 2 Pro ($0.0315)
2. Feed to Seedance 2.5 image-to-video ($0.0225)
3. **Total: $0.054 per video**

### For 100 bumpers:
- Budget: $2.63
- Quality: $2.63
- Premium: $5.40

---

## HOW TO CALL (Python via fal_client)

```python
import fal_client

# Seedance 2.0 Image to Video
result = fal_client.subscribe(
    "bytedance/seedance-2.0/image-to-video",
    arguments={
        "prompt": "The teal orb pulses with energy. Subtle camera push in. The liminal hallway stretches infinitely behind it. Fluorescent lights flicker.",
        "image_url": "https://v3b.fal.media/files/.../first_frame.png",
        "resolution": "1080p",
        "duration": "8",
        "aspect_ratio": "9:16",
        "generate_audio": False,
    },
)
video_url = result["video"]["url"]

# Seedance 2.0 Reference to Video (with audio!)
result = fal_client.subscribe(
    "bytedance/seedance-2.0/reference-to-video",
    arguments={
        "prompt": "@Image1 shows the WuBu NEWS orb. Slow dolly in. The orb pulses once. Cut to black.",
        "image_urls": ["https://v3b.fal.media/files/.../first_frame.png"],
        "audio_urls": ["https://.../voice_line.wav"],
        "resolution": "1080p",
        "duration": "5",
        "aspect_ratio": "9:16",
        "generate_audio": True,
    },
)
```

---

## SEEDANCE 2.0 REFERENCE CAPABILITIES

### Character Consistency
Use an image reference to maintain character identity across scenes:
```
prompt: "@Image1 walks confidently down a neon-lit city street at night, camera tracking alongside."
image_urls: ["character.jpg"]
```

### Style Transfer
Apply the visual style of a reference image:
```
prompt: "A bustling Tokyo street scene at night in the style of @Image1, neon reflections on wet pavement."
image_urls: ["art-style-reference.jpg"]
```

### Motion Transfer
Transfer motion from a reference video onto a character:
```
prompt: "@Image1 performs the dance moves from @Video1 on a rooftop at sunset."
image_urls: ["dancer.jpg"]
video_urls: ["dance-reference.mp4"]
```

### Multi-Scene Composition
Use scene cuts in the prompt to control pacing:
```
prompt: "@Image1 finds a football in the ocean and picks it up excitedly. Cut scene to @Image1 calling friends over. Cut scene to a group playing football underwater."
```

---

## LTX-2.3 SPECIFICS

### Text to Video
```python
result = fal_client.subscribe(
    "fal-ai/ltx-2.3/text-to-video",
    arguments={
        "prompt": "Through-the-veil shot of a bride's face...",
        "duration": "10",
        "resolution": "1080p",
        "aspect_ratio": "16:9",
        "fps": "24",
        "generate_audio": True,
    },
)
```

### Key Features
- Portrait 9:16 support (for vertical/mobile)
- Last-frame interpolation (for seamless transitions)
- LoRA fine-tuning support (for custom styles)
- Native audio generation

---

## MAXIMIZING VALUE (The "Stingy" Strategy)

1. **Always use image references** — Seedance 2.0 flat pricing means you pay the same for 4s or 15s. Make longer videos!
2. **Generate audio WITH video** — Seedance 2.0's generate_audio is FREE (same cost regardless)
3. **Use the cheapest first frame** — FLUX 2 Klein ($0.0116) is fine, no need for Pro
4. **Batch generate** — submit multiple jobs in parallel
5. **Use scene cuts** — one Seedance call can produce multiple scenes (saves money vs. multiple calls)

---

~ WuBu ~
