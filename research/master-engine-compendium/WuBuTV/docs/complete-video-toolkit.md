# 🎬 COMPLETE VIDEO GENERATION TOOLKIT — All Free Methods
# Domain: Production Infrastructure | Chain: master-engine-compendium
# Date: 2026-08-18 | Tested: 31 spaces, 28 working

---

## TESTED HUGGINGFACE SPACES (28/31 WORKING)

### Tier 1: Audio-to-Video (Our Primary Pipeline)
| Space | Status | Input | Output | Notes |
|---|---|---|---|---|
| **multimodalart/minimax-h3-audio-to-video** | ✅ | Image + Audio + Prompt | Video + Audio | OUR MAIN TOOL. 9:16 vertical. 2-14s. |
| **multimodalart/minimax-h3** | ✅ | Image + Prompt | Video + Audio | Text/image to video. 9:16 available. |
| **multimodalart/minimax-h3-reference** | ✅ | Image + Audio + Video refs | Video + Audio | Multi-reference generation |
| **MiniMaxAI/MiniMax-H3-Turbo-Lora** | ✅ | Image + Prompt + LoRA | Video | Turbo speed + LoRA support |
| **mrfakename/minimax-h3-ultra-fast** | ✅ | Image + Prompt | Video | Ultra-fast NVFP4 |
| **akhaliq/MiniMax-H3-Turbo-Lora** | ✅ | Image + Prompt | Video | Turbo LoRA variant |

### Tier 2: Image-to-Video (Wan 2.2 Family)
| Space | Status | Input | Output | Notes |
|---|---|---|---|---|
| **kulkas2pintu/wan555** | ✅ | Image + Prompt | Video | Wan2.2 14B fast |
| **r3gm/wan2-2-fp8da-aoti-preview** | ✅ | Image + Prompt | Video | Wan2.2 FP8 |
| **amisima/Wan_2.2_I2V_14B_Custom_Lora** | ✅ | Image + Prompt + LoRA | Video | Custom LoRA support |
| **hugging-apps/wan2-2-animate-2-14b** | ✅ | Character + Driving video | Video | Character animation |
| **wank3r/Wan_2.2_I2V_14B-Clean** | ✅ | Image + Prompt | Video | Clean I2V |
| **Saravutw/Omni-videos-custom** | ✅ | Image + Prompt | Video | Custom fast motion |
| **Saravutw/WAN2.2_I2V_LIGHTNING_4-8step_custom** | ✅ | Image + Prompt | Video | 4-8 step lightning |
| **dream2589632147/Dream-wan2-2-fp8da-aoti-preview-2** | ✅ | Image + First/Last frame | Video | Dream motion |
| **obsxrver/wan22-i2v-omni-lora** | ✅ | Image + Prompt | Video | Blink effect |
| **Parthiban543/wanmanlove** | ✅ | Image + Prompt | Video | WAN2.2 I2V |
| **cinderholm/wan-i2v** | ✅ | Image + Prompt | Video | Simple I2V |
| **Rchoks/wan555** | ✅ | Image + Prompt | Video | Fast preview |
| **zerogpu-aoti/wan2-2-fp8da-aoti-faster** | ✅ | Image + Prompt | Video | ZeroGPU cost |
| **wank3r/wan2-2-i2v-v3-WOW** | ✅ | Image + Prompt | Video | v3 WOW |

### Tier 3: Text-to-Video
| Space | Status | Input | Output | Notes |
|---|---|---|---|---|
| **zai-org/CogVideoX-2B-Space** | ✅ | Prompt | Video | Text-to-video 2B |
| **zai-org/CogVideoX-5B-Space** | ✅ | Prompt + Image | Video + GIF | Text/image to video 5B |
| **maxin-cn/Cinemo** | ✅ | Image + Prompt | Video | Multimodal I2V |
| **ByteDance/AnimateDiff-Lightning** | ✅ | Prompt + Style | Video | 1/2/4/8 step options |
| **BestWishYsh/MagicTime** | ✅ | Prompt + Model | Video | Time-lapse style |
| **BestWishYsh/ConsisID-preview-Space** | ✅ | Prompt + Image | Video | Identity-preserving |
| **akhaliq/LTX-2.5-workflow** | ✅ | Prompt + Image | Video | LTX 2.5 workflow |
| **Fighterdan/LTX-2.3-10Eros_I2V** | ✅ | Image + Prompt | Video | LTX 2.3 I2V |
| **bleckhert/uncensored-image-to-video-alchemist** | ✅ | Image + Prompt | Video | Uncensored I2V |

### Not Working
| Space | Status | Reason |
|---|---|---|
| KingNish/Instant-Video | ⏸️ PAUSED | Space paused by owner |
| Lightricks/LTX-2.5 | ❌ FAIL | Can't fetch config |

---

## THE COMPLETE GENERATION CODE

### MiniMax-H3 Audio-to-Video (Primary)
```python
from gradio_client import Client, handle_file

client = Client("multimodalart/minimax-h3-audio-to-video")
result = client.predict(
    prompt="A deadpan wizard stares at a breathing hallway",
    image_path=handle_file("first_frame.png"),
    audio_path=handle_file("funny_audio.mp3"),
    canvas="768x1344 · 9:16 full",
    duration=10,
    steps=7,
    seed=42,
    upsample=False,
    api_name="/generate"
)
video_path = result[0]
```

### Wan 2.2 Image-to-Video (Fast)
```python
client = Client("kulkas2pintu/wan555")
result = client.predict(
    input_image=handle_file("image.png"),
    prompt="make this image come alive, cinematic motion",
    steps=6,
    guidance_scale=1,
    seed=42,
    randomize_seed=True,
    api_name="/generate_video"
)
```

### CogVideoX Text-to-Video
```python
client = Client("zai-org/CogVideoX-2B-Space")
result = client.predict(
    prompt="A deadpan wizard stares at a breathing hallway",
    num_inference_steps=50,
    guidance_scale=6.0,
    api_name="/generate"
)
```

### AnimateDiff-Lightning (Fastest)
```python
client = Client("ByteDance/AnimateDiff-Lightning")
result = client.predict(
    prompt="wizard staring",
    base="ToonYou",
    motion="guoyww/animatediff-motion-lora-zoom-in",
    step="4",
    api_name="/generate_image"
)
```

---

## MULTI-BROWSER STRATEGY

Since each space has rate limits, use multiple browser sessions:
- **Session 1**: MiniMax-H3 audio-to-video (primary)
- **Session 2**: Wan 2.2 image-to-video (secondary)
- **Session 3**: CogVideoX text-to-video (backup)
- **Session 4**: AnimateDiff (fast drafts)

Stagger requests across sessions to maximize throughput.

---

## 7-STEP KEVIN BACON: HuggingFace Video Spaces

1. **HuggingFace Spaces** → Free Gradio-hosted AI demos
2. **MiniMax-H3** → Audio-to-video, joint generation, 9:16 vertical
3. **Wan 2.2** → Image-to-video, 14B params, consumer GPU friendly
4. **CogVideoX** → Text-to-video, Z-AI, 2B/5B options
5. **LTX 2.5** → Lightricks, text+image to video
6. **AnimateDiff** → ByteDance, fastest (1-8 steps)
7. **Application** → 28 working spaces = 28x generation capacity

---

## THE COMPLETE PRODUCTION PIPELINE

```
1. WRITE JOKE → Writing-room toolkit (25 lilypads)
2. SOURCE AUDIO → Free libraries / memes / TV / games
3. GENERATE FIRST FRAME → FAL image_generate
4. GENERATE VIDEO → MiniMax-H3 (audio-to-video)
   ├── Fallback 1: Wan 2.2 (image-to-video)
   ├── Fallback 2: CogVideoX (text-to-video)
   └── Fallback 3: AnimateDiff (fast draft)
5. POST-PROCESS → ffmpeg pipeline
   ├── Color grading (LUTs + curves)
   ├── VHS degradation (scan lines, noise)
   ├── Audio mixing (ducking, normalization)
   └── Text overlay (~ WuBu ~ watermark)
6. DISTRIBUTE → Content calendar (2-5x/week)
   ├── TikTok (9:16 vertical)
   ├── IG Reels (same content)
   ├── YT Shorts (under 60s)
   └── Telegram (direct to community)
7. MEASURE → Metrics framework
   ├── Completion rate (70%+)
   ├── CTR (5%+)
   ├── Share rate (3%+)
   └── Follower conversion (2%+)
```

---

## Session Context

This is the breakthrough document. We went from "FLUX 3 is dead, we're blocked" to "28 working video generation spaces, all free, all programmatic." The pipeline is real. The only remaining step is execution.

~ WuBu ~
