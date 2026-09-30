# Audio Post-Production & Thumbnail Design — WuBu Media
# Domain: Post-Production | Chain: master-engine-compendium
# Sources: ffmpeg-cookbook.com, vidIQ, alici.ai, miraflow.ai
# Date: 2026-08-18

---

## PART 1: AUDIO POST-PRODUCTION

### The Rule: Audio Is Half the Picture
Bad video with good audio = watchable.
Good video with bad audio = instant skip.

### ffmpeg Audio Recipes

#### Audio Ducking (Lower music when narration plays)
```bash
ffmpeg -i video.mp4 -i narration.mp3 -filter_complex "
[0:a]volume=1.0[music];
[1:a]volume=1.0[voice];
[music][voice]sidechaincompress=threshold=0.015:ratio=8:attack=50:release=200[ducked];
[voice][ducked]amix=inputs=2:duration=first[aout]
" -map 0:v -map "[aout]" -c:v copy output.mp4
```

#### Platform-Standard Loudness (-14 LUFS)
```bash
-af "loudnorm=I=-14"
```

#### Two-Pass Loudness (Precise)
```bash
# Pass 1: Measure
ffmpeg -i input.mp4 -af "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json" -f null -
# Pass 2: Apply measured values
ffmpeg -i input.mp4 -af "loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=XX:measured_TP=XX:measured_LRA=XX:measured_thresh=XX:offset=XX:linear=true" output.mp4
```

#### Remove Background Noise
```bash
-af "afftdn=nf=-25"
```

#### Add Ambient Sound / Room Tone
```bash
# Generate 10 seconds of room tone from a silent section
ffmpeg -i input.mp4 -ss 00:00:05 -t 10 -af "volume=0.1" room_tone.mp3
# Mix it under the whole video
ffmpeg -i input.mp4 -i room_tone.mp3 -filter_complex "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0" output.mp4
```

#### Music Bed (Loop background music under video)
```bash
ffmpeg -i video.mp4 -stream_loop -1 -i music.mp3 -filter_complex "
[0:a]volume=1.0[voice];
[1:a]volume=0.15[music];
[voice][music]amix=inputs=2:duration=first[aout]
" -map 0:v -map "[aout]" -shortest output.mp4
```

### Audio Layer Stack (Priority Order)
1. **Dialogue/Narration** (loudest, -14 LUFS)
2. **Sound Effects** (medium, -18 LUFS)
3. **Ambient/Room Tone** (quiet, -24 LUFS)
4. **Music Bed** (quietest, -28 LUFS, ducked when dialogue plays)

### Sound Design Principles for WuBu

#### Found Footage / Analog Horror
- **Ambient hum**: Fluorescent lights (60Hz hum), HVAC, distant machinery
- **Tape noise**: Hiss, warble, tracking errors
- **Foley**: Footsteps, door handles, breathing
- **Silence**: The ABSENCE of sound is scarier than sound

#### Comedy Bumpers
- **Timing**: The pause before the punchline is everything
- **SFX**: Subtle whooshes, pops, or musical stings on the punchline
- **Music**: Bed at -28 LUFS, never competing with dialogue
- **Deadpan**: The Old UNC's lines should be dry — no reverb, no effects

#### Game Show / Sitcom
- **Laugh track**: Pre-recorded, timed to punchlines
- **Buzzer**: On wrong answers
- **Applause**: On correct answers
- **Music stings**: On transitions

---

## PART 2: THUMBNAIL DESIGN

### The Data (vidIQ study of 500 breakout videos, June 2026)
- **69%** featured a human face (80% of the biggest overperformers)
- **89%** used either a face OR high-contrast color
- **56%** used high-contrast color
- **Median text**: 5 words or fewer

### The Formula
1. **One focal point** (face, object, or key visual)
2. **High contrast** (bright subject on dark background, or complementary colors)
3. **3-5 words of text** (bold, sans-serif, outlined)
4. **Emotional expression** (surprise, curiosity, excitement)
5. **Consistent style** (recognizable across all videos)

### Thumbnail Rules

#### Text
- **Under 4 words** = 30% higher CTR than text-heavy
- **Bold sans-serif fonts** (Arial, Impact, Bebas)
- **Minimum 30pt** equivalent
- **Thick outline or drop shadow** for readability
- **Position where it won't cover faces**

#### Color
- **High contrast** = 30% higher CTR
- **Complementary colors** (red/green, yellow/purple, teal/orange)
- **Avoid YouTube's palette** (red, white, black — blends into interface)
- **Use teals, oranges, yellows, purples** — they pop against YouTube's chrome

#### Faces
- **Fill 25-35%** of the frame
- **Genuine expression** beats forced/exaggerated
- **Curiosity, focus, or real smile** perform best
- **Well-lit, clearly visible**

#### Composition
- **One subject, one message, one second to understand**
- **Max 3 distinct visual elements** (more = 23% lower CTR)
- **Arrows/circles** can boost CTR by 25% (use 1-2 max)
- **Design for mobile** (most views happen on small screens)

### WuBu Thumbnail Templates

#### Template 1: "The Old UNC Stares"
- Frame: The Old UNC's face fills 30% of the frame, deadpan expression
- Background: Liminal hallway or storage unit (dark, moody)
- Text: "~ WuBu ~" or a 3-word hook (top or bottom)
- Colors: Teal orb glow against dark background (high contrast)

#### Template 2: "The Absurd Event"
- Frame: The absurd element (breathing hallway, chicken, galaxy in storage unit)
- Background: The Old UNC in the distance, unfazed
- Text: 3-word hook ("THE HALLWAY BREATHES")
- Colors: Warm absurdity against cool background

#### Template 3: "The Height Opposition"
- Frame: Tall Old UNC next to short characters (visual contrast)
- Background: Game show set or sitcom living room
- Text: "HEIGHT OPPOSITION" (2 words)
- Colors: Bright game show colors vs dark Old UNC

### Thumbnail Production with ffmpeg
```bash
# Create a thumbnail from a video frame
ffmpeg -i video.mp4 -ss 00:00:03 -frames:v 1 thumbnail.png

# Add text overlay
ffmpeg -i thumbnail.png -vf "
drawtext=text='THE HALLWAY':fontcolor=white:fontsize=48:font=Arial-Bold:
x=(w-text_w)/2:y=20:bordercolor=black:borderw=3,
drawtext=text='BREATHES':fontcolor=0x00FFFF:fontsize=48:font=Arial-Bold:
x=(w-text_w)/2:y=80:bordercolor=black:borderw=3
" thumbnail_final.png
```

---

## Session Context

This document closes two critical gaps identified in the self-assessment:
1. **Sound design / audio mixing** (was 0 research, now has recipes)
2. **Thumbnail design** (was 0 research, now has data-backed rules)

These are Tier 1 gaps — they will make or break our content. Bad audio = instant skip. Bad thumbnail = no click. Both are now solvable.

~ WuBu ~
