# 📐 PLAN — FILLING THE WRITING ROOM'S CONCEPTUAL GAPS
# Author: Hermes (Producer) | Date: 2026-08-18
# Method: Seven-Step Kevin Bacon (learned the room) → Triple DA (gaps) → Plan
# This is the "learn what the room is, find the gaps, then plan" deliverable
# the user asked for. It is a PLAN, not yet execution.

---

## PART 0 — WHAT THE WRITING ROOM ACTUALLY IS (what I learned)

The writing room has TWO parent brains that were never connected:

**Brain A — the TV-BUMPER research** (`chain-television/` + `chain-writing-room/`)
- A complete comedy curriculum: 7 joke types, 4 scene engines, humor blueprint
  (premise→lens→norm→twist), joke-test gate, 9-channel bumper application.
- Built for **10-second video bumpers** of the Old UNC.

**Brain B — the UNIVERSE definition** (`MANIFESTO.md` + `ROADMAP.md` +
`channel-identity-*.md` + `codec-call-001.md`)
- The Ted Turner thesis: "create context, not control content."
- 7 Originality Principles (the room was found; the crowd was generated; the
  Active Decoy is hollow; it's all information).
- A **different 9-channel grid** than the bumper research: NEWS (Colonel Codec
  Call), ORIGINALS (Bambi the Builder), SATIRE (Seymour Ponderables), LABS
  (Found Room), MEDIA SCHOOL, PUBLIC ACCESS, PUPPET THEATER, NETWORK, ARCHIVE.
- A **bootleg-identities table**: we don't copy, we bootleg (same energy /
  different name / better execution). Each channel = a bootleg of a real
  competitor (WORM TV, The Onion, Dr. Dreams, Robot Chicken…).
- **Distinct voices**: Colonel (codec: urgent→info→personal→non-sequitur→
  [STATIC]), Seymour (academic absurdist), Old UNC (deadpan), Laura, CLOUDWU.

**The disconnect:** The writing room I built encodes Brain A perfectly but
ignores Brain B. So a writer using the room would produce bumpers that don't
know about the Colonel codec, the bootleg strategy, the originality principles,
or the real channel identities already claimed. That's the core gap.

---

## PART 1 — THE GAPS (Triple DA findings)

### DA-1 (Truth) — what's actually missing on disk
| # | Gap | Evidence | Severity |
|---|-----|----------|----------|
| G1 | **No MANIFESTO link** — originality thesis + 7 principles not in the room | writing room has 0 refs to MANIFESTO/ROADMAP | HIGH |
| G2 | **No bootleg-identity layer** — "we bootleg X better" strategy absent | no doc mapping each channel→competitor→how | HIGH |
| G3 | **NEWS brief is WRONG about its own format** — says "news desk," but the real NEWS identity is the *Colonel Codec Call* | channel-identity-wubunews.md vs CHANNEL_BRIEFS.md NEWS | HIGH |
| G4 | **No voice calibration per persona** — Old UNC/Colonel/Seymour/Laura/CLOUDWU each need operational voice spec | only Old UNC has one (partial) | HIGH |
| G5 | **No premise bank** — humor blueprint Step 1 (emotional truth) has no accumulated WuBu premises | 0 premise files; technique only | MED |
| G6 | **No lens library** — "what-if" WuBu lenses (height opposition, silence-as-language, Order-ritual, dimension-storage) uncataloged | 0 lens files | MED |
| G7 | **No callback / continuity web** — which bit recurs where, serialized story | only "recurring bit" one-liners per channel | MED |
| G8 | **Channel-grid mismatch** — room's 9 (incl. ARENA/SITCOM/HORROR/TALK/COMMERCIAL) ≠ MANIFESTO's 9 (incl. MEDIA SCHOOL/PUBLIC ACCESS/PUPPET THEATER/NETWORK/ARCHIVE) | two different grids coexist | HIGH |
| G9 | **No "genre-overlap how-to"** — we assert 5-genre overlap but never operationalize colliding them in one bumper | 0 how-to files | MED |
| G10 | **Horror-comedy / VHS / analog-horror not synthesized for WuBu** — Lynch + Local 58 + Tales from Crypt exist but no WuBu-specific horror-comedy craft | scattered, unintegrated | MED |

### DA-2 (Value) — which gaps actually block "best writing room"
G1, G2, G3, G4, G8 are BLOCKERS: without them the room produces off-brand,
wrong-format, disconnected work. G5–G10 are DEPTH gaps (make it better, not
possible). Plan prioritizes blockers first.

### DA-3 (Adversary) — what breaks if we ignore this
- If we generate from the current room, we ship bumpers that contradict the
  claimed channel identities (e.g. a "news desk" bumper for a channel whose
  real identity is a Colonel codec call). Brand fracture.
- The bootleg strategy (our entire competitive edge per MANIFESTO) is invisible
  to the writer → we'd accidentally copy instead of bootleg → legal/identity risk.

---

## PART 2 — THE PLAN (what to build, in order)

### PHASE A — RECONCILE & CONNECT (blockers, do first)
**A1. `WRITING_ROOM/UNIVERSE.md`** — the bridge doc. Summarizes MANIFESTO's
  thesis + 7 principles + the bootleg-identities table + the real 9-channel
  grid (from channel-identities). One page the writer reads first.
**A2. Fix `CHANNEL_BRIEFS.md`** — rewrite to match the MANIFESTO grid + real
  identities. NEWS → Colonel Codec Call format. Each brief cites its
  `channel-identity-*.md` + its bootleg-of target.
**A3. `WRITING_ROOM/VOICES.md`** — operational voice spec for each persona:
  Colonel (codec structure + cadence), Seymour (academic absurdist + clipart +
  quote), Old UNC (deadpan + silence rules), Laura, CLOUDWU. With DO/DON'T.

### PHASE B — BOOTLEG STRATEGY LAYER
**B1. `WRITING_ROOM/BOOTLEG.md`** — the bootleg playbook: how to take a
  competitor's energy, change the name, execute better. Per-channel
  bootleg-of mapping (WORM→NEWS, Onion→SATIRE, Dr.Dreams→ORIGINALS,
  Robot Chicken→PUPPET THEATER, etc.) + the "Active Decoy" principle applied
  to every bumper.

### PHASE C — FUEL THE BLUEPRINT (depth)
**C1. `WRITING_ROOM/PREMISE_BANK.md`** — 30-50 WuBu emotional-truth premises
  (storage units are creepy; game shows are rigged; the audience fills gaps;
  silence is a language; the Order is real but useless…). Sourced from MANIFESTO
  + lilypads + channel-identities.
**C2. `WRITING_ROOM/LENS_LIBRARY.md`** — catalog of WuBu "what-if" lenses
  (height opposition, dimension-storage, silence-as-ritual, found-room,
  bootleg-of, Colonel-cut, memetic-anchor). Each with example bumper.
**C3. `WRITING_ROOM/CONTINUITY_WEB.md`** — the recurring-bit map: which bit
  lives where, what callbacks exist, the serialized through-line.

### PHASE D — CRAFT SYNTHESIS (depth)
**D1. `WRITING_ROOM/GENRE_OVERLAP.md`** — operationalize the 5-genre collision
  (found footage + anthology + game show + horror + comedy) into a per-bumper
  recipe.
**D2. `WRITING_ROOM/HORROR_COMEDY.md`** — WuBu-specific horror-comedy craft
  (Lynch + Local 58 + Tales from Crypt → Old UNC unbothered in dread).

### PHASE E — RESEARCH WE STILL NEED (the "more research" mandate)
Per the user's "we need to do more research for what we need," these are OPEN
research threads to run (Seven-Step Kevin Bacon) before/while building:
- **E1.** Full read of all 8 Seymour Ponderables drafts → extract the academic
  absurdist *voice patterns* into VOICES.md (currently only codec-call-001 is read).
- **E2.** Read the 4 other `channel-identity-*.md` (labs/originals already seen;
  confirm media-school/public-access/puppet-theater/net archive specs).
- **E3.** Pull the competitor bootleg targets' actual formats (WORM TV posting
  style, Onion structure) from `chain-*` research → feed BOOTLEG.md.
- **E4.** Survey `chain-lynch/`, `chain-meme/19-analog-horror` for horror-comedy
  technique → feed HORROR_COMEDY.md.
- **E5.** Audit the 9 WuBuTV channels vs MANIFESTO 9 → decide: merge the two
  grids or pick MANIFESTO as canonical (recommend: MANIFESTO canonical; TV
  bumpers become the "bumper idents" inside MANIFESTO's channels).

---

## PART 3 — EXECUTION ORDER (recommended)
```
1. E5 (grid decision)         ← resolves G8 first, everything depends on it
2. A1 UNIVERSE.md             ← G1, connects Brain A+B
3. A3 VOICES.md               ← G4, per-persona calibration
4. A2 fix CHANNEL_BRIEFS      ← G3, correct the formats
5. B1 BOOTLEG.md              ← G2, the edge
6. C1 PREMISE_BANK            ← G5, fuel
7. C2 LENS_LIBRARY            ← G6, fuel
8. C3 CONTINUITY_WEB          ← G7, serialization
9. D1 GENRE_OVERLAP           ← G9, the USP
10. D2 HORROR_COMEDY          ← G10, craft
+ run E1-E4 research alongside as input to the above.
```

## PART 4 — SUCCESS CRITERIA
- Every channel brief cites a real `channel-identity-*.md` + bootleg-of target.
- A writer can open UNIVERSE.md → VOICES.md → their channel brief → template →
  gate, with NO contradiction between the room and the MANIFESTO.
- Premise bank has ≥30 premises; lens library has ≥7 lenses; each usable in
  SCRIPT_TEMPLATE.
- Triple-DA the whole room again after build: 0 contradictions, 0 missing voices.

~ WuBu ~
