# 🎵 COMPLETE AUDIO + VIDEO GENERATION TOOLKIT (54 Working Spaces)
# Domain: Production Infrastructure | Chain: master-engine-compendium
# Date: 2026-08-18 | Tested: 65 spaces, 54 working, 9 failed, 2 paused

---

## 🎬 VIDEO GENERATION (28 Working — Confirmed Earlier)

### Audio-to-Video (Primary Pipeline)
| Space | Endpoints | Notes |
|---|---|---|
| multimodalart/minimax-h3-audio-to-video | 3 | MAIN TOOL: image+audio→video |
| multimodalart/minimax-h3 | 2 | image→video+audio |
| multimodalart/minimax-h3-reference | 3 | Multi-reference |
| MiniMaxAI/MiniMax-H3-Turbo-Lora | 1 | Turbo + LoRA |
| mrfakename/minimax-h3-ultra-fast | 2 | Ultra-fast NVFP4 |
| akhaliq/MiniMax-H3-Turbo-Lora | 1 | Turbo variant |

### Image-to-Video (Wan 2.2 Family)
| Space | Endpoints | Notes |
|---|---|---|
| kulkas2pintu/wan555 | 2 | Wan2.2 14B fast |
| r3gm/wan2-2-fp8da-aoti-preview | 2 | Wan2.2 FP8 |
| amisima/Wan_2.2_I2V_14B_Custom_Lora | 2 | Custom LoRA |
| hugging-apps/wan2-2-animate-2-14b | 2 | Character animation |
| wank3r/Wan_2.2_I2V_14B-Clean | 2 | Clean I2V |
| Saravutw/Omni-videos-custom | 2 | Custom fast motion |
| Saravutw/WAN2.2_I2V_LIGHTNING_4-8step_custom | 2 | 4-8 step lightning |
| dream2589632147/Dream-wan2-2-fp8da-aoti-preview-2 | 2 | Dream motion |
| obsxrver/wan22-i2v-omni-lora | 2 | Blink effect |
| Parthiban543/wanmanlove | 2 | WAN2.2 I2V |
| cinderholm/wan-i2v | 8 | Simple I2V |
| Rchoks/wan555 | 2 | Fast preview |
| zerogpu-aoti/wan2-2-fp8da-aoti-faster | 2 | ZeroGPU |
| wank3r/wan2-2-i2v-v3-WOW | 2 | v3 WOW |

### Text-to-Video
| Space | Endpoints | Notes |
|---|---|---|
| zai-org/CogVideoX-2B-Space | 2 | Text→video 2B |
| zai-org/CogVideoX-5B-Space | 3 | Text/image→video 5B |
| maxin-cn/Cinemo | 4 | Multimodal I2V |
| ByteDance/AnimateDiff-Lightning | 2 | 1-8 steps |
| BestWishYsh/MagicTime | 2 | Time-lapse |
| BestWishYsh/ConsisID-preview-Space | 1 | Identity-preserving |
| akhaliq/LTX-2.5-workflow | 1 | LTX 2.5 |
| Fighterdan/LTX-2.3-10Eros_I2V | 1 | LTX 2.3 I2V |
| bleckhert/uncensored-image-to-video-alchemist | 1 | Uncensored I2V |

### NEW Video Spaces (This Search)
| Space | Endpoints | Notes |
|---|---|---|
| wxDai/joyai-video-edit | 0 | Real-time webcam edit |
| hugging-apps/minimax-h3-spatial-physics-lora | 4 | Spatial physics LoRA |
| amisima/minimax-h3-reference-4-step-lora | 44 | 4-step turbo LoRA |
| TencentARC/scope-camera-video-generation | 9 | Camera-controlled |
| Akuyakufree/Omni-videos | 15 | Text/image→video + extend |
| brnawy2/Wan_Animate_2_motion_transfer_V2V | 1 | V2V motion transfer |
| mpasila/minimax-h3-reference-4-step-lora | 19 | 4-step turbo LoRA v2 |
| hugging-apps/leaptalk-talking-head | 3 | Audio-driven talking head |

---

## 🎵 AUDIO GENERATION (18 Working — New)

### Music Generation
| Space | Endpoints | Model | Notes |
|---|---|---|---|
| **facebook/MusicGen** | 3 | MusicGen | Text+melody→music. THE standard. |
| **facebook/MelodyFlow** | 5 | MelodyFlow | Text+melody→music, 30s |
| **sanchit-gandhi/musicgen-streaming** | 11 | MusicGen | Streaming music gen |
| **MiniMaxAI/MiniMax-Music3** | 12 | MiniMax Music 3 | Text+lyrics→music. Studio quality. |
| **ACE-Step/ACE-Step** | 12 | ACE-Step 3.5B | Music foundation model |
| **ASLP-lab/DiffRhythm** | 10 | DiffRhythm | Fast song generation |
| **declare-lab/TangoFlux** | 5 | TangoFlux | Text→audio (SFX) |
| **ameerazam08/stableaudio-open-1.0** | 6 | Stable Audio Open | Text→audio |
| **artificialguybr/Stable-Audio-Open-Zero** | 9 | Stable Audio | Custom audio from text |
| **smola/higgs_audio_v2** | 4 | Higgs Audio v2 | High-quality audio |
| **stabilityai/stable-audio-3** | 9 | Stable Audio 3 | SA3: music + SFX |

### Text-to-Speech (Voice Generation)
| Space | Endpoints | Model | Notes |
|---|---|---|---|
| **suno/bark** | 2 | Bark | Realistic speech + sounds |
| **Qwen/Qwen3-TTS** | 3 | Qwen3-TTS | Voice design + cloning |
| **ResembleAI/chatterbox-turbo-demo** | 10 | Chatterbox Turbo | Fast TTS |
| **neuphonic/neutts-air** | 6 | NeuTTS-Air | Voice cloning TTS |
| **KittenML/KittenTTS-Demo** | 2 | KittenTTS | Natural-sounding TTS |
| **nari-labs/Dia-1.6B** | 2 | Dia 1.6B | Dialogue generation |

### Multi-Modal Audio
| Space | Endpoints | Model | Notes |
|---|---|---|---|
| **FunAudioLLM/Fun-Audio-Chat-8B** | N/A (401) | Fun-Audio-Chat | Any-to-any audio |

---

## ❌ FAILED SPACES (9)
| Space | Reason |
|---|---|
| haoheliu/audioldm-text-to-audio-generation | RUNTIME_ERROR |
| haoheliu/audioldm2-text2audio-text2music | RUNTIME_ERROR |
| artificialguybr/musicgen-songstarter-demo | CONFIG_ERROR |
| coqui/xtts | RUNTIME_ERROR |
| declare-lab/mustango | BUILD_ERROR |
| AIGC-Audio/AudioGPT | BUILD_ERROR |
| AIGC-Audio/AudioLCM | BUILD_ERROR |
| AIGC-Audio/Make-An-Audio-3 | BUILD_ERROR |
| FunAudioLLM/Fun-Audio-Chat-8B | 401 auth required |

### Paused (2)
| Space | Reason |
|---|---|
| reach-vb/musicgen-prompt-upsampling | PAUSED |
| FunAudioLLM/ThinkSound | PAUSED |

---

## 🔧 CODE SNIPPETS

### MusicGen (Text→Music)
```python
from gradio_client import Client, handle_file
client = Client("facebook/MusicGen")
result = client.predict(
    texts="dark bass wizardcore trap beat with eerie atmosphere",
    melodies=handle_file(None),
    api_name="/predict_batched"
)
```

### MiniMax Music 3 (Text+Lyrics→Music)
```python
client = Client("MiniMaxAI/MiniMax-Music3")
result = client.predict(
    prompt="dark ambient trap beat, 120 BPM, bass-heavy, eerie",
    lyrics=None,
    api_name="/generate"
)
```

### Bark (Text→Speech)
```python
client = Client("suno/bark")
result = client.predict(
    text="The hallway breathes. I have seen it.",
    api_name="/predict"
)
```

### Stable Audio 3
```python
client = Client("stabilityai/stable-audio-3")
result = client.predict(
    prompt="dark ambient drone, fluorescent hum, liminal space",
    api_name="/generate"
)
```

### LeapTalk (Audio→Talking Head Video)
```python
client = Client("hugging-apps/leaptalk-talking-head")
result = client.predict(
    image=handle_file("portrait.png"),
    audio=handle_file("speech.wav"),
    api_name="/generate"
)
```

### SCoPE (Camera-Controlled Video from Image)
```python
client = Client("TencentARC/scope-camera-video-generation")
result = client.predict(
    image=handle_file("first_frame.png"),
    camera_path="pan_left",
    api_name="/generate"
)
```

---

## 📊 THE COMPLETE TOOLKIT

| Category | Working Spaces | Use Case |
|---|---|---|
| Audio-to-Video | 6 | Image + Audio → Video (PRIMARY) |
| Image-to-Video | 14 | Image → Video |
| Text-to-Video | 9 | Prompt → Video |
| Music Generation | 10 | Text → Music |
| Text-to-Speech | 6 | Text → Voice |
| Talking Head | 1 | Audio + Face → Video |
| Camera Control | 1 | Image + Camera → Video |
| V2V Editing | 1 | Video → Video |
| **TOTAL** | **~54** | |

---

## 🔄 UPDATED PRODUCTION PIPELINE

```
1. WRITE JOKE → Writing-room toolkit
2. GENERATE MUSIC → MusicGen / MiniMax Music 3 / Stable Audio 3
3. GENERATE VOICE → Bark / Qwen3-TTS (Old UNC's lines)
4. CREATE FIRST FRAME → FAL image_generate
5. GENERATE VIDEO → MiniMax-H3 (audio-to-video)
   ├── Fallback: Wan 2.2 (image-to-video)
   ├── Fallback: CogVideoX (text-to-video)
   └── Fallback: AnimateDiff (fast draft)
6. POST-PROCESS → ffmpeg pipeline
7. DISTRIBUTE → Content calendar
8. MEASURE → Metrics framework
```

---

~ WuBu ~
