# ffmpeg Post-Production Pipeline — WuBu Media
# Domain: Video Post-Production | Chain: master-engine-compendium
# Source: tessl.io reels-producer-skill ffmpeg-recipes.md + original research
# Date: 2026-08-18

---

## COLOR GRADING RECIPES

### "WuBu Gritty" (Teal shadows / warm orange mids — our signature look)
```bash
-vf "eq=contrast=1.08:saturation=1.12,
colorbalance=rs=-0.06:gs=-0.02:bs=0.08:rm=0.05:gm=0.01:bm=-0.05:rh=0.04:bh=-0.04,
curves=all='0/0.02 0.5/0.5 1/0.98',
vignette=PI/5:mode=backward"
```

### "WuBu Analog" (Muted, faded film — for found footage)
```bash
-vf "eq=contrast=1.04:saturation=0.82:gamma=1.02,
colorbalance=rs=0.02:gs=-0.01:bs=-0.03:rm=0.05:gm=0.00:bm=-0.05:rh=0.03:gh=0.00:bh=-0.04,
curves=all='0/0.03 0.5/0.5 1/0.97',
noise=alls=5:allf=t+u"
```

### "WuBu Punch" (Hard contrast + grain — for game show/sitcom)
```bash
-vf "eq=contrast=1.20:saturation=0.94:gamma=0.99,
colorbalance=rs=0.02:bs=-0.03:rm=0.06:bm=-0.05:rh=0.03:bh=-0.04,
curves=all='0/0.0 0.25/0.19 0.75/0.82 1/1.0',
noise=alls=10:allf=t+u"
```

---

## VHS / ANALOG HORROR EFFECTS

### Full VHS Degradation Chain
```bash
-vf "
# Slightly desaturate
eq=saturation=0.85:contrast=1.05,
# Add scan lines
drawgrid=w=2:h=2:t=1:c=black@0.3,
# Chroma shift (RGB channel separation)
chromashift=cbh=2:cbv=0:crh=-2:crv=0,
# Tracking noise
noise=alls=8:allf=t+u,
# Slight blur (VHS softness)
boxblur=0.5:1,
# Vignette
vignette=PI/4:mode=backward,
# Color grade to teal/warm
curves=all='0/0.03 0.5/0.5 1/0.97',
colorbalance=bs=0.05:rm=0.03"
```

### Emergency Broadcast System (EBS) Look
```bash
-vf "
# High contrast black and white with slight color
eq=saturation=0.3:contrast=1.3,
# Scan lines
drawgrid=w=1:h=1:t=1:c=black@0.5,
# Glitch bars
geq='lum=lum(X,Y)*if(between(mod(Y,40),0,3),0.7,1)',
# Noise
noise=alls=15:allf=t+u"
```

---

## PLATFORM EXPORT SPECS

| Platform | Resolution | Max Length | Video Codec | Audio | Notes |
|---|---|---|---|---|---|
| **TikTok** | 1080x1920 | 10 min | H.264 high, ≤16 Mbps | AAC 192k, -14 LUFS | CRF 19 enough |
| **IG Reels** | 1080x1920 | 15 min | H.264 high | AAC 192k, -14 LUFS | CRF 18 (IG recompresses hardest) |
| **YT Shorts** | 1080x1920 | 3 min | H.264 high, BT.709 tags | AAC 192k, -14 LUFS | Wrong color tags = washed out |
| **Telegram** | 480x854 (vertical) | <20MB | H.264 | AAC 128k | Our delivery format |

### Standard Export Command
```bash
ffmpeg -i input.mp4 \
  -vf "<grade_recipe>" \
  -c:v libx264 -crf 19 -preset slow \
  -pix_fmt yuv420p \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -c:a aac -b:a 192k -ar 48000 \
  -af "loudnorm=I=-14" \
  -movflags +faststart \
  -r 30 \
  output.mp4
```

### Vertical Crop (from horizontal source)
```bash
# Center crop to 9:16
-vf "crop=ih*9/16:ih:(W-ih*9/16)/2:0"

# Animated punch-in (Ken Burns)
-vf "zoompan=z='min(zoom+0.0008,1.15)':d=1:s=1080x1920:fps=30"

# Letterbox with blurred background
-vf "split[a][b];[a]scale=1080:1920,boxblur=20[bg];[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2"
```

---

## LOWER THIRD / TEXT OVERLAY

### Burn-in Subtitles (via .srt file)
```bash
ffmpeg -i input.mp4 -vf "subtitles=subs.srt:force_style='FontName=Arial,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=0,Alignment=2,MarginV=50'" output.mp4
```

### Simple drawtext watermark
```bash
ffmpeg -i input.mp4 -vf "drawtext=text='~ WuBu ~':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:x=(w-text_w)/2:y=h-th-20" output.mp4
```

---

## AUDIO NORMALIZATION

### Platform-standard loudness (-14 LUFS)
```bash
-af "loudnorm=I=-14"
```

### Two-pass loudness normalization (for precise matching)
```bash
# Pass 1: Measure
ffmpeg -i input.mp4 -af "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json" -f null -
# Pass 2: Apply measured values
ffmpeg -i input.mp4 -af "loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=-23.5:measured_TP=-3.2:measured_LRA=8.1:measured_thresh=-33.7:offset=0.5:linear=true" output.mp4
```

---

## TROUBLESHOOTING

| Problem | Cause | Fix |
|---|---|---|
| A/V desync after concat | Mismatched audio params | Re-encode to 48kHz stereo |
| Washed-out on YouTube | Missing color tags | Add `-colorspace bt709 -color_primaries bt709 -color_trc bt709` |
| Black frames at cuts | Keyframe-snapped stream copy | Re-encode instead of `-c copy` |
| Slow renders on Apple Silicon | Software encoding | Use `h264_videotoolbox -b:v 12M` |
| Audio too loud/quiet | No normalization | Add `loudnorm=I=-14` |

---

## Session Context

This document is the output of the recursive self-improvement loop, iteration 1. We identified that we're BAD at video post-production. This document gives us the actual technical recipes to fix that. Combined with the comedy-writing toolkit (chain-writing-room/), the bootleg grammar (chain-bootleg/), and the distribution knowledge (TikTok algorithm research), we now have a complete production pipeline:

1. **Write** → writing-room toolkit + comedy-deep philosophy
2. **Generate** → bootleg grammar prompts + AI video tools
3. **Post-process** → ffmpeg recipes (THIS DOCUMENT)
4. **Distribute** → platform specs + TikTok algorithm strategy
5. **Measure** → analytics + completion rate tracking
6. **Improve** → recursive loop back to step 1

~ WuBu ~
