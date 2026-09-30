# SATIRE S-14 / S-15 — CORRECTED PROMPTS

Rewritten 2026-09-29 after WuBu reported two hard generation blockages. **Both
were failures in how I wrote the seeds, not in the pipeline.** WuBu reported
rather than shipping something wrong, which is the correct call.

## The bug class, named

I have now made this mistake **three times**, in three different media:

| where | what I wrote | what happened |
|---|---|---|
| audio prompts | "no applause, no music, silence" | clip had nothing to render |
| ident prompts | "no animation of ANY element" | frozen frame, rejected |
| **S-14 (this)** | **"exactly ONE dark tube"** | **every tube rendered lit** |

**A negative in a load-bearing position is a request for absence.** A generator
that reads "one tube is dark" produces all-lit tubes, because that is the only
positive image available. Never put a *what must not happen* where a *what must
be visible* belongs.

The fix is always the same move: **restate the negative as a positive object.**

---

# S-14 — THE MALFUNCTION (rewritten)

The gag: one tube in the row is fitted with a **dead bulb** — blackened, cold,
visibly darker than its neighbours — and he waits under it. The failure is that
the dead tube is a thing that *exists*, not a thing that is absent.

```
CHANNEL: satire   |   FORMAT: game show   |   ENGINE: normal -> absurd
JOKE TYPE: observation (he waits out a dead bulb like a scheduled item)
RECURRING BIT: the empty podium   |   CHARACTER: Old UNC   |   DURATION: 30s
```

**BEAT SHEET**
```
[0:00-0:10] SETUP:   Wide. A row of overhead tubes above the stage. Six are
                   bright. One, above him, is fitted with a blackened dead
                   bulb - visibly darker, grey filament, no glow. He looks up.
[0:10-0:20] DEEPEN:  He stands under the dead tube and waits. Patient, motionless,
                   no impatience, no sigh. The spheres do not react. The marquee
                   reads INF:0.
[0:20-0:30] PUNCHLINE: He steps sideways out from under it, squares himself, faces
                   camera again, as though the maintenance had been scheduled.
                   The dead tube is still dead at the final frame.
```

**GENERATION PROMPT**
```
Hand-painted 2D cartoon, thick black outlines, flat comic palette, 1970s
public-access broadcast, VHS chroma bleed, scanlines.

SUBJECTS: a heavyset Black man with a large rounded afro, thick mustache and
goatee, wearing a BLACK hoodie with WuBu in YELLOW/GOLD letters. A teal-and-
orange neon game show set. Overhead: a row of SEVEN fluorescent tube fixtures.
SIX are lit and glowing bright white. ONE, directly above him, is fitted with a
BLACKENED DEAD BULB - the glass is dark grey rather than clear, the filament is
a visible dull grey coil, and this fixture casts NO glow and sits visibly darker
than its six neighbours. The dead fixture must be readable as a physical object
that exists, not as a hole. A white podium, centre frame. Tiered seating holding
identical teal spheres in black domino eye-masks. An overhead dot-matrix marquee.

TIMELINE:
[0s-10s] Wide. He stops and looks up at the dead-bulb fixture. His head tilts
        slightly. The spheres do not react. The marquee reads INF:0.
[10s-20s] He stands in the unlit patch beneath the dead fixture and waits.
         Patient, motionless, no impatience, no sigh, no checking.
[20s-30s] He steps sideways out of the unlit patch, squares himself, and faces
         camera again. The dead fixture is still dark in the final frame.

CAMERA: locked off wide, no push in, no pan, no cuts, no reframing.

AUDIO: studio room tone with the electrical hum of the six working tubes, 0s to
30s. At 21s one soft footstep. Nothing else. No music, no audience, no applause,
no SFX not listed here.

LOAD-BEARING DETAIL: exactly ONE fixture is a blackened dead bulb, it is the SAME
fixture in every frame, and it is dark at the FIRST frame and still dark at the
LAST frame. The six lit fixtures must not flicker. The marquee must read INF:0
throughout. 6-tile the fixture row and the marquee.

SILENT-FRAME GAMBLE: a man waiting patiently under a dead light bulb and then
moving on as though the repair had been scheduled. Reads with the sound off.

TEXT: marquee reads "INF:0", legible and unchanged. Corner bug ~ WuBu ~ over
WuBuTV, two lines.

CANON: identity lock - a heavyset Black man with a large rounded afro, thick
mustache and goatee, wearing a BLACK hoodie with WuBu in YELLOW/GOLD letters.
Wardrobe does not change. Set: neon game show, one blackened dead-bulb fixture
among six lit ones, white podium centre, teal spheres with domino eye-masks.
Load-bearing: the dead fixture is a visible object, unchanged in every frame.

NEGATIVE: no dissolves, no subtitles, no watermark other than the two-line corner
bug, no invented text, no letter-shaped noise, no flickering lit tubes, no cloth
mask, no face on a stand, no licensed cartoon characters, no franchise faces, no
animal mascots, no real people, no real logos, no smile, no laugh, no applause,
no audience reaction, no reaction shot, no frustration, no sigh, no looking at a
watch, no music, no stinger.
```

---

# S-15 — THE DUPLICATE (rewritten, mechanism changed)

**Why the original was refused.** Asking a generator for two *identical men in one
frame* reads as a duplication artefact and is genuinely ambiguous. The pipeline
integrity-refused two attempts, which is the policy working correctly — better a
refusal than a second distinct man.

**The fix: keep the gag, change the mechanism.** The joke was never really the
double — it is that **the show is set up for two contestants and only one is
there.** That is the empty-podium joke the channel runs on, and it renders
trivially because there is only one person in frame and no identity risk.

Option (a) is the replacement. The podium is dressed for two: two chairs, two
microphones, two name placards — one occupied, one not.

```
CHANNEL: satire   |   FORMAT: game show   |   ENGINE: normal -> absurd
JOKE TYPE: observation (a two-contestant set with one contestant)
RECURRING BIT: the empty podium   |   CHARACTER: Old UNC   |   DURATION: 30s
```

**BEAT SHEET**
```
[0:00-0:10] SETUP:   Wide. The stage is dressed for TWO: two chairs, two
                   microphones on stands, two identical name placards. He sits
                   in the left chair. The right chair is empty, with its
                   microphone standing in front of it and nothing behind it.
[0:10-0:20] DEEPEN:  He looks at the empty chair once. He does not wave at it,
                   does not address it, does not shrug. He looks back at camera.
                   The spheres do not react to the vacancy at any point.
[0:20-0:30] PUNCHLINE: He reads from the empty chair's placard - the one that is
                   not his - holds it up flat to camera, and puts it back. The
                   set stays dressed for two. Nobody is coming.
TAG:           The placard he picks up belongs to the empty chair. That is the
               whole joke and it must not be underlined.
```

**GENERATION PROMPT**
```
Hand-painted 2D cartoon, thick black outlines, flat comic palette, 1970s
public-access broadcast, VHS chroma bleed, scanlines.

SUBJECTS: ONE man only - a heavyset Black man with a large rounded afro, thick
mustache and goatee, wearing a BLACK hoodie with WuBu in YELLOW/GOLD letters. He
sits in the LEFT of two identical chairs on a teal-and-orange neon game show
stage. The RIGHT chair is EMPTY. Each chair has its own microphone on a stand in
front of it, and its own identical rectangular name placard propped on the desk
edge. Tiered seating holding identical teal spheres in black domino eye-masks.
An overhead dot-matrix marquee.

TIMELINE:
[0s-10s] Wide. He is seated in the left chair. The right chair is empty, its
        microphone standing in front of it, its placard propped on the desk. The
        two chairs are identical and symmetrical.
[10s-20s] He turns his head once toward the empty chair. He does not wave at it,
         does not speak to it, does not shrug, does not mouth anything. He looks
         back at camera. The spheres do not react at any point.
[20s-30s] He reaches over, lifts the EMPTY chair's placard, holds it flat toward
         camera at chest height, then sets it back in exactly its original
         position. The set stays dressed for two. Cut to end.

CAMERA: locked off wide, symmetrical, no push in, no pan, no cuts, no reframing.
Both chairs and the marquee fully in frame with equal margin.

AUDIO: studio room tone throughout. At 12s a chair creak. At 22s a placard lifted
and set back, one light movement. Nothing else. No music, no audience, no
applause, no SFX not listed here.

LOAD-BEARING DETAIL: exactly TWO chairs, two microphones, two placards, and
exactly ONE person. The right chair must be empty in every single frame. The two
placards must be identical in size and shape. 6-tile the chair row.

SILENT-FRAME GAMBLE: a two-contestant set with one contestant, and he picks up
the empty contestant's name card. Reads with the sound off.

TEXT: marquee reads "INF:0", legible and unchanged. Corner bug ~ WuBu ~ over
WuBuTV, two lines.

CANON: identity lock, restate for the one figure - a heavyset Black man with a
large rounded afro, thick mustache and goatee, wearing a BLACK hoodie with WuBu
in YELLOW/GOLD letters. Wardrobe does not change. Set: neon game show, two
identical chairs, two microphones, two identical placards, teal spheres with
domino masks. Load-bearing: two chairs, one person.

NEGATIVE: no dissolves, no subtitles, no watermark other than the two-line corner
bug, no invented text, no letter-shaped noise, no cloth mask, no face on a stand,
NO SECOND PERSON, no second distinct figure, no duplicate of any kind, no
licensed cartoon characters, no franchise faces, no animal mascots, no real
people, no real logos, no smile, no laugh, no applause, no audience reaction, no
reaction shot, no waving, no shrugging, no confusion, no music, no stinger.
```

~ WuBu ~

---

## WHY OPTION (a) IS BETTER THAN THE ORIGINAL

The original gag was "two identical hosts and neither acknowledges the other."
That is a *metaphysical* joke. The replacement is "a set dressed for two with one
contestant" — which is **the same joke the channel has been running since S-1**,
told structurally rather than mysteriously. It also renders safely because there
is only ever one person in frame, so the duplication artefact that triggered the
integrity refusal cannot occur.

The lesson generalises: **when a prompt gets integrity-refused for ambiguity,
the fix is usually to make the gag *smaller and more literal*, not to retry the
same ask with different words.** A refusal is information about the request, not
an obstacle to it.
