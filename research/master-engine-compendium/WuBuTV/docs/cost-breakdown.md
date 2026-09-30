# 💰 COST BREAKDOWN: Video Generation Options
# Domain: Financial Planning | Chain: master-engine-compendium
# Date: 2026-08-18

---

## OPTION 1: HuggingFace ZeroGPU (What We've Been Trying)

### Free Tier
| Item | Cost | Reality |
|---|---|---|
| ZeroGPU quota | FREE | ~5 min GPU/day total across ALL spaces |
| Account required | FREE | HF account |
| Token (optional) | FREE | HF token auth |
| **Actual cost** | **$0** | **But quota runs out in minutes** |

### PRO Tier ($9/mo)
| Item | Cost | Reality |
|---|---|---|
| Subscription | $9/mo | 40 min GPU/day |
| Extended quota | $1 per 10 min | Overages if you exceed |
| **40 min/day** | **$9/mo** | **Enough for ~8-20 video generations** |

### Cost Per Video Generation (PRO)
| Model | GPU Time | Cost |
|---|---|---|
| LeapTalk (talking head) | ~3s | ~$0.01 |
| CogVideoX-2B | ~360s | ~$0.60 |
| Wan2.2 Lightning | ~60s | ~$0.10 |
| MiniMax-H3 | ~302s | ~$0.50 |
| AnimateDiff | ~30s | ~$0.05 |

**Verdict: $9/mo PRO + ~$0.05-0.60 per video = CHEAPEST option**

---

## OPTION 2: Nous Portal (FAL Video Generation)

### Plans
| Plan | Price | Credits | Rollover |
|---|---|---|---|
| Free | $0 | $0 | — |
| Plus | $20/mo | $22 | $10 |
| Super | $100/mo | $110 | $50 |
| Ultra | $200/mo | $220 | $100 |

### Video Generation Costs (per "billable unit")
| Model | Type | Cost/Unit | What's a "Unit"? |
|---|---|---|---|
| **LTX-2.3-22b** | text-to-video | **$0.0017** | Per generation |
| **LTX-2.3-22b** | image-to-video | **$0.0017** | Per generation |
| **PixVerse v6** | text-to-video | **$0.0053** | Per generation |
| **PixVerse v6** | image-to-video | **$0.0053** | Per generation |
| **Seedance 2.0 mini** | text-to-video | **$0.0074** | Per generation |
| **Seedance 2.0 mini** | image-to-video | **$0.0074** | Per generation |
| **Seedance 2.0** | text-to-video | **$0.0147** | Per generation |
| **Seedance 2.0** | image-to-video | **$0.0147** | Per generation |
| **Seedance 2.5** | text-to-video | **$0.0225** | Per generation |
| **Seedance 2.5** | image-to-video | **$0.0225** | Per generation |
| **FLUX-3** | text-to-video | **$0.0893** | Per generation |
| **FLUX-3** | image-to-video | **$0.0893** | Per generation |
| **MiniMax H3** | text-to-video | **$0.273** | Per generation |
| **MiniMax H3** | image-to-video | **$0.273** | Per generation |
| **Veo 3.1** | text-to-video | **$0.42** | Per generation |
| **Veo 3.1** | image-to-video | **$0.42** | Per generation |
| **Kling v3 4K** | text-to-video | **$0.441** | Per generation |
| **Kling v3 4K** | image-to-video | **$0.441** | Per generation |

### Image Generation Costs (for first frames)
| Model | Cost/Unit | Notes |
|---|---|---|
| Ideogram v4 instant | $0.0001 | Cheapest |
| Qwen Image 3 | $0.0002 | Very cheap |
| Krea v2 medium | $0.0011 | Cheap |
| FLUX 2 Klein 9B | $0.0116 | What we use now |
| FLUX 2 Pro | $0.0315 | Higher quality |
| Seedance 2.5 img | $0.0368 | High quality |

### Cost Per Complete Bumper (Nous Portal)
| Pipeline | Image | Video | Total/Video | For 100 videos |
|---|---|---|---|---|
| **Budget: LTX-2.3** | $0.0011 (Krea) | $0.0017 | **$0.0028** | **$0.28** |
| **Mid: Seedance 2.0** | $0.0116 (FLUX Klein) | $0.0147 | **$0.0263** | **$2.63** |
| **High: FLUX-3** | $0.0315 (FLUX Pro) | $0.0893 | **$0.1208** | **$12.08** |
| **Premium: Veo 3.1** | $0.0315 (FLUX Pro) | $0.42 | **$0.4515** | **$45.15** |

**Verdict: $20/mo Plus + ~$0.003-0.45 per video = BEST QUALITY per dollar**

---

## OPTION 3: HF PRO + ZeroGPU (Hybrid)

| Item | Cost |
|---|---|
| HF PRO subscription | $9/mo |
| 40 min GPU/day | included |
| Extended quota | $1 per 10 min |
| LeapTalk (3s/gen) | ~$0.01/gen |
| CogVideoX (360s/gen) | ~$0.60/gen |

**Cost per bumper (using LeapTalk for talking head):**
- Image: $0.0116 (FLUX Klein via FAL/Nous)
- Video: ~$0.01 (LeapTalk, 3s GPU)
- Total: **~$0.02 per video**
- 100 videos: **~$2.00**

**Cheapest for talking-head style bumpers.**

---

## COMPARISON: Cost to Produce 100 Bumpers

| Method | Subscription | Per-Video Cost | Total for 100 | Quality |
|---|---|---|---|---|
| **HF PRO + LeapTalk** | $9/mo | $0.02 | **$11** | Medium (talking head) |
| **Nous Portal + LTX-2.3** | $20/mo | $0.003 | **$20.30** | Medium |
| **Nous Portal + Seedance 2** | $20/mo | $0.026 | **$22.60** | High |
| **Nous Portal + FLUX-3** | $20/mo | $0.12 | **$32** | Very High |
| **Nous Portal + Veo 3.1** | $20/mo | $0.45 | **$65** | Highest |

---

## RECOMMENDATION

### For Right Now (Proving the Pipeline):
**HF PRO ($9/mo) + LeapTalk** = $11 for 100 videos
- We already confirmed LeapTalk works (2.76s GPU time)
- We have FAL image generation (already paid for via Nous subscription)
- Total out of pocket: $9 for HF PRO

### For Production Quality:
**Nous Portal Plus ($20/mo) + Seedance 2.0** = ~$23 for 100 videos
- Seedance 2.0 is a top-tier video model
- Image-to-video means we generate a first frame with FLUX, then animate
- No GPU quota management — just pay per generation
- Higher quality than HF free-tier models

### For Premium Quality:
**Nous Portal Plus ($20/mo) + FLUX-3 or Veo 3.1** = $32-65 for 100 videos
- Best quality available
- Still cheaper than any competitor

### The Winner:
**Nous Portal Plus ($20/mo)** — because:
1. We're ALREADY paying for it (Nous subscription)
2. FAL image gen is ALREADY included
3. No GPU quota juggling
4. Higher quality models than HF free tier
5. Simpler workflow (no gradio_client API debugging)
6. Image-to-video is the right pipeline for bumpers

**But for immediate testing: HF PRO ($9/mo)** to get the pipeline working NOW, then switch to Nous Portal for production.

---

~ WuBu ~
