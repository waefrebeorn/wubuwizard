# 🔧 COMPLETE API REFERENCE — All Working Generation Spaces
# Domain: Production Infrastructure | Chain: master-engine-compendium
# Date: 2026-08-18 | Spaces: 20 core + 34 extended = 54 total

---

## HOW TO USE THIS DOCUMENT

Every space below has:
1. Exact Python `gradio_client` call
2. Parameter names, types, defaults
3. What it returns
4. Recommended use case for ~ WuBu ~

---

## 🎬 VIDEO: Audio-to-Video

### multimodalart/minimax-h3-audio-to-video (PRIMARY)
```
Client: multimodalart/minimax-h3-audio-to-video

ENDPOINT: /generate
  prompt: str (required) — "A deadpan wizard stares at a breathing hallway"
  image_path: filepath (required) — First frame image
  audio_path: filepath (required) — Audio file to sync to
  canvas: str (default: "960x544 · 16:9 fast")
    Options: "768x1344 · 9:16 full" (VERTICAL!), "640x1152 · 9:16", "544x960 · 9:16 fast"
  duration: float (default: 6) — 2 to 14 seconds
  steps: float (default: 7) — More = better quality, slower
  seed: int (default: 42)
  upsample: bool (default: False)
RETURNS: (video_filepath, markdown, textbox)

USE: Primary bumper generation. Image + Audio → Vertical video.
```

### multimodalart/minimax-h3 (Image-to-Video+Audio)
```
Client: multimodalart/minimax-h3

ENDPOINT: /generate
  prompt: str (optional)
  image_path: filepath (optional) — First frame
  last_image_path: filepath (optional) — Last frame
  canvas: str (default: "960x544 · 16:9 fast")
  duration: float (default: 5)
  steps: float (default: 28)
  seed: int (default: 42)
  upsample: bool (default: False)
RETURNS: (video_with_soundtrack, markdown, textbox)

USE: Image → Video with auto-generated audio.
```

---

## 🎬 VIDEO: Image-to-Video

### cinderholm/wan-i2v
```
Client: cinderholm/wan-i2v

ENDPOINT: /generate_video
  input_image: filepath (required)
  last_image: filepath (required)
  prompt: str (optional, default: "make this image come alive, cinematic motion")
  steps: float (default: 6)
  negative_prompt: str (optional)
  duration_seconds: float (default: 3.5)
  guidance_scale: float (default: 1)
  guidance_scale_2: float (default: 1)
  seed: float (default: 42)
  randomize_seed: bool (default: True)
  quality: float (default: 6)
  scheduler: str (default: "UniPCMultistep")
  flow_shift: float (default: 3.0)
  frame_multiplier: str (default: "16")
  safe_mode: bool (default: True)
RETURNS: (generated_video_filepath, download_filepath, seed)

USE: High-quality image-to-video with first+last frame control.
```

### zai-org/CogVideoX-2B-Space (Text-to-Video)
```
Client: zai-org/CogVideoX-2B-Space

ENDPOINT: /generate
  prompt: str (required)
  num_inference_steps: float (default: 50)
  guidance_scale: float (default: 6.0)
RETURNS: (video_dict, download_video, download_gif)

USE: Pure text-to-video. Good for simple scenes.
```

### ByteDance/AnimateDiff-Lightning (Fastest)
```
Client: ByteDance/AnimateDiff-Lightning

ENDPOINT: /generate_image
  prompt: str (required)
  base: str (default: "ToonYou") — Options: "ToonYou", "epiCRealism"
  motion: str (default: "") — Options: "guoyww/animatediff-motion-lora-zoom-in", "zoom-out", "tilt-up", "tilt-down", "pan-left", "pan-right", "rolling-anticlockwise", "rolling-clockwise"
  step: str (default: "4") — Options: "1", "2", "4", "8"
RETURNS: (video_dict)

USE: Fastest generation. 1-8 steps. Good for rapid prototyping.
```

### TencentARC/scope-camera-video-generation (Camera Control)
```
Client: TencentARC/scope-camera-video-generation

ENDPOINT: /generate
  image: filepath (optional)
  prompt: str (required)
  trajectory: str (default: "dolly_in")
    Options: dolly_in, dolly_out, truck_left, truck_right, orbit_left, crane_up_fwd, snake_fwd, grand_tour, push_sweep, wide_orbit, spiral_climb, greek_spiral_rise, crane_arc, flyover_left, s_curve_reveal, pullback_rise
  steps: float (default: 4)
  motion_scale: float (default: 1.0)
  fov_degrees: float (default: 80.7)
  seed: float (default: 42)
  randomize_seed: bool (default: True)
RETURNS: (generated_video, seed, markdown)

USE: Camera-controlled video from still image. Cinematic movements.
```

### hugging-apps/leaptalk-talking-head
```
Client: hugging-apps/leaptalk-talking-head

ENDPOINT: /generate
  portrait_image: filepath (required)
  speech_audio: filepath (required)
  max_seconds: float (default: 9)
  num_inference_steps: float (default: 1)
  guidance_scale: float (default: 1.0)
  seed: int (default: 42)
  auto_crop_face: bool (default: True)
RETURNS: (result_video, markdown)

USE: Audio-driven talking head. Old UNC speaks → talking head video.
```

---

## 🎵 AUDIO: Music Generation

### facebook/MusicGen
```
Client: facebook/MusicGen

ENDPOINT: /predict_batched
  texts: str (required) — "dark ambient trap beat, bass heavy, eerie"
  melodies: filepath (required) — Can be None but must pass handle_file(None)
RETURNS: (generated_audio_filepath)

USE: Text → Music. THE standard music generator.
```

### sanchit-gandhi/musicgen-streaming
```
Client: sanchit-gandhi/musicgen-streaming

ENDPOINT: /generate_audio
  text_prompt: str (default: "80s pop track with synth and instrumentals")
  audio_length_in_s: float (default: 15)
  play_steps_in_s: float (default: 1.5)
  seed: float (default: 5)
RETURNS: (generated_music_filepath)

USE: Streaming music generation. Simple API.
```

### MiniMaxAI/MiniMax-Music3
```
Client: MiniMaxAI/MiniMax-Music3

ENDPOINT: (11 endpoints — complex)
  prompt: str (required) — Music description
  lyrics: str (optional) — Song lyrics
  duration: float (default: 95) — Seconds
  infer_step: float (default: 60)
  guidance_scale: float (default: 15.0)
  seed: float (default: 0)
  randomize_seed: bool (default: True)
RETURNS: (generated_audio, parameters_json)

USE: Studio-quality music with lyrics. Best music quality.
```

### stabilityai/stable-audio-3
```
Client: stabilityai/stable-audio-3

ENDPOINT: /generate_audio
  prompt: str (required)
  sampler_type_dropdown: str (default: "dpmpp-3m-sde")
  seconds_total: float (default: 20) — 0 to 47 seconds
  steps: float (default: 80) — 10 to 150
  cfg_scale: float (default: 7) — 1 to 15
  sigma_min_slider: float (default: 0.3)
  sigma_max_slider: float (default: 500)
RETURNS: (generated_audio_filepath)

USE: High-quality audio. Music + SFX.
```

### ACE-Step/ACE-Step
```
Client: ACE-Step/ACE-Step

ENDPOINT: /__call__
  audio_duration: float (default: -1) — -1 = auto, up to 240s
  prompt: str (default: "funk, pop, soul...")
  lyrics: str (optional) — Song lyrics
  infer_step: float (default: 60) — 1 to 200
  guidance_scale: float (default: 15.0) — 0 to 30
  scheduler_type: str (default: "euler") — "euler", "heun"
  cfg_type: str (default: "apg") — "cfg", "apg", "cfg_star"
  omega_scale: float (default: 10.0)
  manual_seeds: str (optional)
  guidance_interval: float (default: 0.5)
  guidance_interval_decay: float (default: 0.0)
  min_guidance_scale: float (default: 3.0)
  use_erg_tag: bool (default: True)
  use_erg_lyric: bool (default: False)
  use_erg_diffusion: bool (default: True)
  oss_steps: str (optional)
  guidance_scale_text: float (default: 0.0)
  guidance_scale_lyric: float (default: 0.0)
RETURNS: (generated_audio, parameters_json)

USE: Full music generation with lyrics. Most control.
```

### declare-lab/TangoFlux
```
Client: declare-lab/TangoFlux

ENDPOINT: /predict
  prompt: str (required)
  steps: float (default: 25) — 10 to 100
  guidance: float (default: 4.5) — 1 to 10
  duration: float (default: 10) — 1 to 30 seconds
RETURNS: (generated_audio_filepath)

USE: Sound SFX generation. Simple API.
```

### ameerazam08/stableaudio-open-1.0
```
Client: ameerazam08/stableaudio-open-1.0

ENDPOINT: /generate_audio
  prompt: str (required)
  sampler_type_dropdown: str (default: "dpmpp-3m-sde")
  seconds_total: float (default: 20)
  steps: float (default: 80)
  cfg_scale: float (default: 7)
  sigma_min_slider: float (default: 0.3)
  sigma_max_slider: float (default: 500)
RETURNS: (generated_audio_filepath)

USE: Ambient soundscapes. Good for liminal atmosphere.
```

---

## 🎵 AUDIO: Text-to-Speech

### suno/bark
```
Client: suno/bark

ENDPOINT: /gen_tts
  text: str (default: "Hello, my name is Suno...")
  history_prompt: str (default: "Speaker 1 (en)")
    Options: Unconditional, Announcer, Speaker 0-9 (en/de/es/fr/hi/it/ja/ko/pl/pt/ru/tr/zh)
RETURNS: (generated_audio_filepath)

USE: Realistic speech. Old UNC voice lines.
```

### Qwen/Qwen3-TTS
```
Client: Qwen/Qwen3-TTS

ENDPOINT: /generate_voice_design
  text: str (optional)
  language: str (default: "Auto") — Auto, Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian
  voice_description: str (optional) — "Speak in a gravelly deadpan tone..."
RETURNS: (generated_audio_filepath, status_text)

ENDPOINT: /generate_custom_voice
  text: str (optional)
  language: str (default: "English")
  speaker: str (default: "Ryan") — Aiden, Dylan, Eric, Ono_anna, Ryan, Serena, Sohee, Uncle_fu, Vivian
  instruct: str (required) — Voice instructions
  model_size: str (default: "1.7B") — "0.6B", "1.7B"
RETURNS: (generated_audio_filepath, status_text)

ENDPOINT: /generate_voice_clone
  ref_audio: filepath (required) — Reference voice sample
  ref_text: str (required) — Transcript of reference
  target_text: str (required) — What to say
  language: str (default: "Auto")
  use_xvector_only: bool (default: False)
  model_size: str (default: "1.7B")
RETURNS: (generated_audio_filepath, status_text)

USE: Voice design, custom voices, voice cloning. Most flexible TTS.
```

### nari-labs/Dia-1.6B
```
Client: nari-labs/Dia-1.6B

ENDPOINT: /generate_audio
  text_input: str (default: dialogue example)
  audio_prompt_input: filepath (required)
  transcription_input: str (required)
  max_new_tokens: float (default: 3072)
  cfg_scale: float (default: 3.0)
  temperature: float (default: 1.8)
  top_p: float (default: 0.95)
  cfg_filter_top_k: float (default: 45)
  speed_factor: float (default: 1.0)
RETURNS: (generated_audio_filepath)

USE: Dialogue generation. Multiple speakers.
```

### neuphonic/neutts-air
```
Client: neuphonic/neutts-air

ENDPOINT: /predict
  ref_text: str (optional)
  ref_audio_path: filepath (optional)
  gen_text: str (optional)
RETURNS: (generated_speech_filepath)

USE: Voice cloning TTS. Clone Old UNC voice.
```

### KittenML/KittenTTS-Demo
```
Client: KittenML/KittenTTS-Demo

ENDPOINT: /synthesize
  text: str (required)
  model_name: str (default: "Micro (40M - Balanced)") — Nano, Micro, Mini
  voice: str (default: "Jasper") — Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
  speed: float (default: 1.0)
RETURNS: (output_filepath)

USE: Fast, natural TTS. Simple API.
```

### ResembleAI/chatterbox-turbo-demo
```
Client: ResembleAI/chatterbox-turbo-demo

ENDPOINT: /generate
  text: str (optional)
  audio_prompt_path: filepath (optional)
  temperature: float (default: 0.8)
  seed_num: float (default: 0)
  min_p: float (default: 0.0)
  top_p: float (default: 0.95)
  top_k: float (default: 1000)
  repetition_penalty: float (default: 1.2)
  norm_loudness: bool (default: True)
RETURNS: (output_audio_filepath)

USE: Fast TTS with voice cloning.
```

---

## 📋 QUICK REFERENCE: What to Use When

| Goal | Best Space | Backup |
|---|---|---|
| Image + Audio → Video | multimodalart/minimax-h3-audio-to-video | multimodalart/minimax-h3 |
| Image → Video | cinderholm/wan-i2v | kulkas2pintu/wan555 |
| Text → Video | zai-org/CogVideoX-2B-Space | ByteDance/AnimateDiff-Lightning |
| Fast draft | ByteDance/AnimateDiff-Lightning | zai-org/CogVideoX-2B-Space |
| Camera move from image | TencentARC/scope-camera-video-generation | — |
| Talking head from audio | hugging-apps/leaptalk-talking-head | — |
| Text → Music | facebook/MusicGen | MiniMaxAI/MiniMax-Music3 |
| Studio music with lyrics | MiniMaxAI/MiniMax-Music3 | ACE-Step/ACE-Step |
| Sound SFX | declare-lab/TangoFlux | stabilityai/stable-audio-3 |
| Ambient atmosphere | ameerazam08/stableaudio-open-1.0 | stabilityai/stable-audio-3 |
| Text → Speech | suno/bark | KittenML/KittenTTS-Demo |
| Voice design | Qwen/Qwen3-TTS | suno/bark |
| Voice cloning | neuphonic/neutts-air | Qwen/Qwen3-TTS |
| Dialogue generation | nari-labs/Dia-1.6B | — |

---

~ WuBu ~
