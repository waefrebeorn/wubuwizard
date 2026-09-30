# CRAFT IMPROVEMENT — What We're BAD At (And How to Get Better)
# Domain: Self-Improvement | Chain: master-engine-compendium
# Date: 2026-08-18 | Loop Iteration 4

---

## The Honest Truth

We have 190+ research files but we've never published anything. The research has become a procrastination mechanism. Let me stop that and address what we're ACTUALLY bad at:

### What We're BAD At (Ranked by Impact)

1. **We've never made anything people can see** — 0 videos published
2. **We don't know if our jokes are funny** — no testing, no audience
3. **We don't know if our visuals look good** — no quality bar
4. **We don't know how to edit for timing** — the difference between funny and not
5. **We don't know how to color grade** — everything looks "AI"
6. **We don't know how to mix audio** — bad audio = instant skip
7. **We don't know how to grow an audience** — no accounts, no followers

### What We CAN Do Right Now (Without Accounts)

1. **Write jokes and test them against the writing-room toolkit**
2. **Generate first-frame images and judge if they look good**
3. **Study color grading and build the ffmpeg pipeline**
4. **Study audio mixing and build the layer stack**
5. **Write 100 jokes and filter them through the joke test**

---

## 1. COLOR GRADING (Making Things Not Look Like AI)

### The Problem
AI-generated video looks "plastic" — too clean, too saturated, too perfect. Real film has grain, color shifts, imperfections.

### The Solution: LUTs + ffmpeg

**LUT** = Lookup Table = a preset color transform.
- 1D LUT = gamma curve only (simple)
- 3D LUT = maps hue, saturation, brightness independently (powerful)
- File formats: .lut (1D), .cube (3D)

### Free LUT Sources
| Source | URL | Type |
|---|---|---|
| Color Grading Central | colorgradingcentral.com/free-luts/ | 10 free cinematic LUTs |
| StudioBinder | studiobinder.com/blog/what-is-lut/ | Ridley Scott inspired pack |
| Motion Array | motionarray.com/learn/video-effects/free-luts-color-grading-packs/ | 30 free LUT packs |
| The Resolve Store | theresolve.store | 30 free CinematicX LUTs |

### ffmpeg Color Grading Without LUTs (Built-in Filters)

#### The "WuBu Gritty" Look
```bash
ffmpeg -i input.mp4 -vf "
eq=brightness=-0.05:contrast=1.1:saturation=0.85,
curves=r='0/0.05 0.5/0.45 1/0.95':g='0/0.05 0.5/0.5 1/0.95':b='0/0.1 0.5/0.55 1/0.9',
noise=alls=20:allf=t+u,
format=yuv420p
" -c:a copy output.mp4
```

#### The "Analog Horror VHS" Look
```bash
ffmpeg -i input.mp4 -vf "
eq=brightness=-0.1:contrast=1.2:saturation=0.6,
curves=r='0/0 0.5/0.4 1/1':g='0/0 0.5/0.5 1/1':b='0/0.1 0.5/0.6 1/0.9',
noise=alls=30:allf=t+u,
format=yuv420p,
drawbox=x=0:y=0:w=iw:h=2:color=black@0.5:t=fill,
drawbox=x=0:y=ih-2:w=iw:h=2:color=black@0.5:t=fill
" -c:a copy output.mp4
```

#### The "WuBu VHS Grade" Look
```bash
ffmpeg -i input.mp4 -vf "
eq=brightness=-0.02:contrast=1.05,
curves=r='0/0 0.3/0.2 0.7/0.8 1/1':g='0/0 0.3/0.3 0.7/0.7 1/1':b='0/0.05 0.3/0.4 0.7/0.85 1/1',
noise=alls=10:allf=t+u
" -c:a copy output.mp4
```

### The Key Insight
The difference between "AI slop" and "cinematic" is:
1. **Lower saturation** (0.7-0.9 instead of 1.0+)
2. **Slightly crushed blacks** (lift the floor, don't go pure 0)
3. **Film grain** (noise filter, 10-30 intensity)
4. **Color shift** (teal in shadows, warm in highlights — or vice versa)
5. **Vignette** (dark edges draw the eye to center)

### Applying .cube LUTs in ffmpeg
```bash
# If you download a .cube LUT:
ffmpeg -i input.mp4 -vf "lut3d=file=film_look.cube" -c:a copy output.mp4
```

---

## 2. AUDIO MIXING (Making Things Not Sound Like AI)

### The Layer Stack (Priority Order)
1. **Dialogue/Narration** — loudest (-14 LUFS)
2. **Sound Effects** — medium (-18 LUFS)
3. **Ambient/Room Tone** — quiet (-24 LUFS)
4. **Music Bed** — quietest (-28 LUFS, ducked when dialogue plays)

### The ffmpeg Audio Chain
```bash
ffmpeg -i video.mp4 -i music.mp3 -i sfx.mp3 -filter_complex "
[0:a]volume=1.0,loudnorm=I=-14[voice];
[1:a]volume=0.15,loudnorm=I=-28[music];
[2:a]volume=0.5,loudnorm=I=-18[sfx];
[voice][music]sidechaincompress=threshold=0.015:ratio=8:attack=50:release=200[ducked];
[ducked][sfx]amix=inputs=2:duration=first[aout]
" -map 0:v -map "[aout]" -c:v copy output.mp4
```

### The "Found Footage" Audio Treatment
```bash
ffmpeg -i input.mp4 -af "
highpass=f=80,
lowpass=f=8000,
afftdn=nf=-25,
volume=0.9,
aecho=0.8:0.9:500|1000:0.3|0.25
" output.mp4
```

---

## 3. COMEDY TIMING (The Difference Between Funny and Not)

### The Rules
1. **The pause before the punchline** — 0.5-1 second of silence is funnier than immediate delivery
2. **The rule of three** — setup, setup, subvert (not setup, setup, setup)
3. **Cut on the laugh** — don't let the joke breathe too long or too short
4. **Visual punchline first** — if the visual is funny, cut away from the setup
5. **Audio punchline first** — if the sound is funny, let it play out

### The 10-Second Timing Formula
| Time | What Happens | Duration |
|---|---|---|
| 0:00-0:02 | HOOK (grab attention) | 2 sec |
| 0:02-0:05 | SETUP (establish the premise) | 3 sec |
| 0:05-0:06 | PAUSE (let the audience anticipate) | 1 sec |
| 0:06-0:08 | PUNCHLINE (the surprise) | 2 sec |
| 0:08-0:10 | TAG (optional second laugh) | 2 sec |

### Editing for Comedy
- **Cut BEFORE the laugh** — don't show the reaction, cut to the next thing
- **Cut AFTER the surprise** — show the reaction for 0.5 sec, then cut
- **Never hold a shot longer than 3 seconds in a 10-second bumper**
- **The cut IS the joke** — sometimes the edit itself is the punchline

---

## 4. THE JOKE TEST (Are We Actually Funny?)

### From Lilypad 23 (Minimal Scene Joke Test)
Every joke must pass:
1. ✅ **Punchline exists** — there IS a surprise
2. ✅ **Misdirection exists** — the setup points one way, punchline goes another
3. ✅ **Not obvious** — if you see it coming, it's not funny
4. ✅ **Actually funny** — not just clever, actually makes someone laugh
5. ✅ **ONE thing happening** — not 3 jokes competing for attention

### Testing Without an Audience
1. **The "would I share this?" test** — if I wouldn't send it to a friend, it's not ready
2. **The "explain the joke" test** — if you have to explain it, it doesn't work
3. **The "second viewing" test** — if it's not funny the second time, it's a one-note joke
4. **The "swap the audio" test** — if the joke works with different audio, it's visual (good)
5. **The "swap the visual" test** — if the joke works with different visuals, it's audio (good)

---

## 5. THE RECURSIVE LOOP (Updated)

### What We Learned This Iteration
1. **LUTs are the shortcut to "not looking like AI"** — download free ones, apply in ffmpeg
2. **Audio layering has a strict priority** — dialogue > SFX > ambient > music
3. **Comedy timing is about the pause** — 0.5-1 second before the punchline
4. **The cut IS the joke** — editing timing matters as much as the content
5. **We've been using research as avoidance** — at some point you have to make something

### The Real Loop
```
MAKE something (even if it's bad)
  → SHOW it to someone (even if it's just me)
    → IDENTIFY what's bad (be honest)
      → RESEARCH how to fix it
        → MAKE it again (better)
```

Not: research → research → research → research → never make anything.

### Our Actual Next Step
1. Download free LUTs
2. Write 10 jokes (test them against the checklist)
3. Generate 10 first-frame images
4. Source 10 audio clips
5. Generate 10 videos (MiniMax-H3)
6. Post-process all 10 (ffmpeg pipeline)
7. Judge: are these funny? Do they look good? Would I share these?

If yes → publish. If no → figure out why and fix it.

---

## Session Context

This document is different from the others. It's not research — it's a mirror. The recursive loop only works if we complete cycles. We've been stuck in KNOW for 3 iterations. Time to move to APPLY.

~ WuBu ~
