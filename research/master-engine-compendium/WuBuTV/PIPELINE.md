# 🔧 WUBUTV VIDEO GENERATION — Complete Technical Reference
# Domain: Production Infrastructure | Chain: master-engine-compendium
# Date: 2026-08-18 | Status: LEARNED (not guessed)

---

## LESSONS LEARNED (The Hard Way)

### 1. FAL Image Upload Requires FAL_KEY (NOT gateway token)
- `fal_client.upload_file("path/to/image.png")` → needs `FAL_KEY` env var
- The Nous Portal gateway token does NOT work for CDN uploads
- Without FAL_KEY, image-to-video with local files is impossible
- **Workaround**: Use text-to-video with detailed prompts (no image reference)
- **Better workaround**: Get a free FAL_KEY from fal.ai/dashboard/keys

### 2. Nous Portal Gateway Works for Inference
- The managed gateway (`fal-queue-gateway.nousresearch.com`) works for video generation
- Routes through `_submit_fal_video_request()` → `_ManagedFalSyncClient.submit()`
- Text-to-video works perfectly through the gateway
- Image-to-video FAILS because FAL can't download `file://` URLs

### 3. PRICING IS PER-SECOND — THE HARD WAY (CORRECTED FROM REAL LEDGER 2026-08-18)
> Source of truth: the actual billing CSV (`usage-day-summary-eman56447-2026-08-18.csv`, 1015 rows).
> The Portal bill shows "Tool use" as a SEPARATE line from model inference.
> **All FAL video/image calls bill under `fal.queue.*` = the "Tool use" line.**
>
> **REAL Aug 18 breakdown (total debit $16.90; "Tool use" subset = $16.72):**
> - **$7.50 — 6× `fal.queue.blackforestlabs.flux-3.text-to-video`** ← THE SPIKE
> - $4.78 — 2× `seedance-2.0.mini.text-to-video`
> - $2.97 — 8× `seedance-2.0.image-to-video`
> - $1.33 — 11× `ltx-2.3-22b.text-to-video`
> - $0.13 — 20× `flux-2.klein.9b` (image)
> - $0.12 — 164× firecrawl.search · $0.05 browser-use · $0.02 firecrawl.scrape
>
> **Root cause (corrected):** NOT the failed I2V call — it was **6 FLUX-3
> text-to-video jobs in one day**. FLUX-3 is the premium per-second model; 6 of
> them = budget gone. Week total Aug 12-18 = **$37.12**; worst day was actually
> **Aug 14 ($12.05)** from a 98-req deepseek run + search, then Aug 18 ($16.90)
> from the video spree. Full breakdown is in `REEL/_GEN_LOG.md`.
>
> **Lesson:** FLUX-3 T2V is the most expensive thing we do. Treat every FLUX-3
> submit as ~$1.25/job. 6 in a day = $7.50. Cap it.

| Model | 480p | 720p | 1080p | Notes |
|---|---|---|---|---|
| Grok Imagine 1.5 | $0.08/s | $0.14/s | $0.25/s | T2V only, no refs |
| Seedance 2.0 Mini | $0.0721/s | $0.1547/s | — | refs + scene cuts |
| LTX-2.3 | — | $0.06/s | $0.06/s | native audio, cheapest workhorse |
| FLUX-3 | — | ~$0.03/s est | $0.06/s | "Spielberg CGI" — **PREMIUM (~$1.25/job), the budget killer** |
| Seedance 2.0 | ~$0.0147/s est | — | — | better than mini |
| Seedance 2.5 | ~$0.0225/s est | — | — | up to 30s |
| Veo 3.1 | — | — | $0.42/s | too expensive |
| Kling v3 4K | — | — | $0.441/s | too expensive |
| Pixverse v6 | $0.0053/s (360p) | — | — | **SLOP — banned** |

**Cost-guardrail rules (MUST):**
1. Before ANY video gen, compute `duration × per-second rate`. FLUX-3 ≈ **$1.25/job** — budget for it.
2. **Cap FLUX-3 at ≤2 jobs/day** unless explicitly approved. Default engine = LTX-2.3 (720p, ≤10s, audio ON ≈ $0.60/job).
3. Log every generation to `REEL/_GEN_LOG.md` (model, tool_action, duration, est. cost).
4. "Tool use" on the Portal ≠ model inference cost. Watch BOTH. The real ledger CSV is the source of truth.

### 4. Audio Generation is FREE on Most Models
- LTX-2.3: `generate_audio=True` (free)
- Seedance 2.0: `generate_audio=True` (free)
- FLUX-3: `generate_audio=True` (free)
- Pixverse v6: `generate_audio=True` (free)
- Grok Imagine 1.5: Native audio always on (no parameter)
- MiniMax H3: Native audio always on (no parameter)

### 5. Model Quality Tiers (User's Classification)
- **FLUX-3** = "Spielberg CGI" — highest quality, worth the cost
- **Seedance 2.0** = High quality, supports image references
- **Seedance 2.0 Mini** = Good quality, cheaper
- **LTX-2.3** = Good quality, cheapest at 720p
- **Pixverse v6** = SLOP (user rejected)
- **Talking heads** = SLOP (user banned)

### 6. The Plugin System
- `hermes plugins enable video_gen/fal` enables the FAL backend
- Plugin must be loaded in Python process to use `video_gen_registry`
- Default model is `pixverse-v6` (cheap but slop)
- Override with `model=` parameter in generate() call

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

# Generate with AUDIO ENABLED
result = provider.generate(
    prompt="Your detailed prompt here...",
    model="ltx-2.3",           # or "seedance-2.0", "flux-3", etc.
    duration=5,                 # seconds
    aspect_ratio="16:9",        # or "9:16" for vertical
    resolution="720p",          # "480p" for cheapest
    audio=True,                 # ENABLE AUDIO (free!)
)

if result.get('video'):
    urllib.request.urlretrieve(result['video'], "output.mp4")
```

---

## OPTIMAL STRATEGY FOR WUBUTV (post-ledger correction)

### Real budget reality (Aug 12-18 actual): $37.12 total
- Aug 14: $12.05 (deepseek 98-req run + search) — worst day
- Aug 18: $16.90 (video spree: 6× FLUX-3 T2V = $7.50 + Seedance/LTX)
- The FAL `fal.queue.*` video calls are the "Tool use" line and the real budget risk.

### Best Value (per job, 720p, ≤10s, audio ON):
1. **LTX-2.3** ≈ $0.60/job — DEFAULT. Cheapest, native audio, good quality.
2. **Seedance 2.0 Mini** ≈ $0.72-1.55/job — better quality, refs + scene cuts.
3. **FLUX-3** ≈ $1.25/job — PREMIUM. Hero shots ONLY. **Cap ≤2/day.**

### Recommendation:
- DEFAULT everything to **LTX-2.3** (cheapest, audio free).
- FLUX-3 only for hero/bumper-poster shots, max 2/day, pre-approved.
- ALWAYS enable audio (free!).
- Use scene cuts in prompts to pack multiple beats per generation.
- Log every gen to `REEL/_GEN_LOG.md`.

### What NOT to do:
- Don't fire 6 FLUX-3 jobs in a day (that's the $7.50 spike).
- Don't use Pixverse (slop) or talking heads (banned).
- Don't generate without audio.
- Don't use file:// URLs for I2V (use hosted HTTP + Serveo).
- Don't assume flat pricing (it's per-second; "Tool use" ≠ model inference).

---

~ WuBu ~ Television
