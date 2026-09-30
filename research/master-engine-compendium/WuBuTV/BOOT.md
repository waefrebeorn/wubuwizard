# 🚀 WUBUTV — NEXT SESSION STARTUP GUIDE
# Read this FIRST when starting a new session
# Date: 2026-08-18 | Session end state

---

## FIRST ACTIONS (in order)

1. **Read `wubutv-character-brand-reference.md`** — Our characters, brand, assets
2. **Read `bumper-scripts-v1.md`** — Complete scripts for all 9 channels
3. **Read `technical-reference-video-generation.md`** — Working code patterns
4. **Read `loop-log.md`** — What we did and what we got wrong

---

## CRITICAL CREATIVE DIRECTION (User's corrections)

### ❌ What the user REJECTS:
- Slow-paced staring content ("nothing happens")
- Generic character descriptions
- Silent video
- No branding
- Wasting existing assets

### ✅ What the user WANTS:
- **HIGH INTENSITY, FAST PACED, HIGH COLORS**
- **THINGS HAPPENING** in every frame
- Our SPECIFIC 2D cartoon character (afro, mustache+goatee, "WuBu" hoody)
- `~ WuBu ~` branding on EVERYTHING
- Audio ALWAYS (it's free)
- Use our existing assets (40 clips, 100 locations, 7 sheets, 9 tracks)

---

## WHAT WE HAVE

### Generated Assets (this session):
- 10 first-frame images (9 channels + 1 portrait)
- 9 voice lines (Qwen3-TTS + KittenTTS)
- 9 SFX tracks (TangoFlux)
- 22 videos (mix of LTX-2.3, Seedance 2.0, FLUX-3)
- 3 image-to-video bumpers WITH audio (NEWS, ORIGINALS, SATIRE)

### Existing Assets (from WuBuMovie):
- 40 raw video clips at `/home/wubu/WuBuMovie/raw_footage/wubuclips/`
- 7 character sheets at `/home/wubu/WuBuMovie/raw_footage/rapman/sheets/`
- 100 location sets at `/home/wubu/WuBuMovie/raw_footage/rapman/`
- 9 album tracks at `/home/wubu/WuBuMovie/audio/music/`
- Movie shooting script at `/home/wubu/WuBuMovie/AGI_MADE_THIS_MOVIE/shooting_script.md`

### Working Infrastructure:
- FAL video generation through Nous Portal gateway
- Image-to-video via Serveo tunnel + local HTTP server
- Audio generation: Qwen3-TTS (best), KittenTTS, Bark, TangoFlux
- HF token configured for free-tier spaces

### Budget:
- Spent: ~$3.50 of $5.00
- Remaining: ~$1.50

---

## WORKING GENERATION CODE

```python
import sys, time, os, urllib.request
sys.path.insert(0, '/home/wubu/.hermes/hermes-agent')

from plugins.video_gen.fal import register
from agent import video_gen_registry

class MockCtx:
    def register_video_gen_provider(self, provider):
        video_gen_registry._providers[provider.name] = provider
register(MockCtx())

provider = video_gen_registry.list_providers()[0]

result = provider.generate(
    prompt="HIGH INTENSITY prompt with our character...",
    model="flux-3",  # or "seedance-2.0", "ltx-2.3"
    duration=5,
    aspect_ratio="16:9",
    resolution="720p",
    audio=True,  # ALWAYS!
)

if result.get('video'):
    urllib.request.urlretrieve(result['video'], "output.mp4")
```

### For Image-to-Video (needs public URL):
1. Start local HTTP server on port 8080
2. Start Serveo tunnel: `ssh -R 80:localhost:8080 serveo.net`
3. Use the public URL as `image_url` parameter

---

## 9 CHANNELS TO PRODUCE

1. **NEWS** — "The Order Reports" (news desk, breathing hallway)
2. **ORIGINALS** — "The MEGA" (found footage, impossible doors)
3. **SATIRE** — "The Order's Game Show" (game show with teal orb contestant)
4. **LABS** — "Found Room, Generated Crowd" (infomercial, HALLWAY-AID)
5. **ARENA** — "Height Opposition" (sports, height competition)
6. **SITCOM** — "The Short Family" (multi-camera, height inversion)
7. **HORROR** — "Liminal Dread" (found footage, breathing walls)
8. **TALK** — "The Old UNC Interviews" (talk show, teal orb guest)
9. **COMMERCIAL** — "WuBu Products" (storage units with galaxies inside)

Full scripts: `bumper-scripts-v1.md`

---

## MODEL PRICING (Per-Second)

| Model | 480p | 720p | 1080p | Notes |
|---|---|---|---|---|
| FLUX-3 | — | $0.06/s | $0.06/s | "Spielberg CGI" quality |
| LTX-2.3 | — | $0.06/s | — | Cheapest at 720p |
| Seedance 2.0 | $0.0721/s | $0.1547/s | — | Supports image refs |
| Seedance 2.0 Mini | $0.0721/s | $0.1547/s | — | Faster/cheaper |
| Grok Imagine 1.5 | $0.08/s | $0.14/s | $0.25/s | Native audio |
| Pixverse v6 | Cheap | — | — | SLOP — avoid |
| Veo 3.1 | — | — | $0.42/s | Too expensive |

### Rate Limits:
- Max 3 concurrent generations
- Wait 60s between generations

---

## KEY FILES

| File | Path |
|---|---|
| Character/Brand Reference | `WuBuTV/BRAND.md` |
| Bumper Scripts | `WuBuTV/SCRIPTS.md` |
| Technical Reference | `WuBuTV/PIPELINE.md` |
| Loop Log | `WuBuTV/LOOP_LOG.md` |
| Wiki | `WIKI.md` |
| Character Sheets | `/home/wubu/WuBuMovie/raw_footage/rapman/sheets/` |
| Raw Clips | `/home/wubu/WuBuMovie/raw_footage/wubuclips/` |
| Album Tracks | `/home/wubu/WuBuMovie/audio/music/` |
| Movie Script | `/home/wubu/WuBuMovie/AGI_MADE_THIS_MOVIE/shooting_script.md` |

---

~ WuBu ~ Television
