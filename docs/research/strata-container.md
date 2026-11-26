# Strata `.strata` container — what it is, and what is worth stealing

Source: `github.com/Niko1221/Strata` (MIT), cloned to `C:\Users\eman5\WuBu\repos\strata` (205 MB, engine 0.1.69).
Evidence below is from that tree plus the first 4 KB of `Inkling-2-Mini-SQ-10.strata`, read locally while downloading.

## Observed bytes (SQ-10 pack, partial download)

    offset 0   53 54 52 41 54 41 00 01     "STRATA\0\1"      magic, 1-byte format minor in the last byte
    offset 24  1c 2a 00 00 00 00 00 00     10780             plausible section count / manifest length
    offset 32  32 bytes of high-entropy data                  SHA-256
    offset 80  00 10 00 00 00 00 00 00     4096              page / frame size
    offset 96  32 more bytes                                  SHA-256

Matches `terminal/serve/test_inkling_e2e.py:50` (`MAGIC = b"STRATA\x00\x01"`) and
`engine/src/hy4flash/run/hy4_cpu_experts.cpp:1668` (compares `"STRATA\0\1"`, reads in 4096-byte frames).

## Header layout (`engine/include/strata/artifact/strata_reader.hpp`)

    struct StrataHeader {          // :48
        uint32_t flags;            // low byte = format minor  (:68)
        StrataProfile profile;     // Tiered=0, Unified=1, EngramShard=2, VisionShard=3  (:18-21)
        uint16_t num_sections;     // must equal 6  (:14, reader.cpp:56)
        StrataSectionEntry sections[6];
        uint8_t engram_shard_sha256[32];
    };

    struct StrataSectionEntry {    // :24, #pragma pack, sizeof == 64
        uint64_t file_offset;
        uint64_t byte_length;
        uint8_t  sha256[32];       // per-section integrity
        uint32_t section_id;
        uint32_t flags;            // section[3].flags & 1 => load-order table present
        uint64_t reserved;
    };

    struct StrataExpertEntry {     // :34, sizeof == 32
        uint16_t expert_id;
        uint8_t  gu_code_id;       // WHICH CODEBOOK this expert's gate/up uses
        uint8_t  dn_code_id;       // ... and its down
        uint16_t cluster_id;       // experts are clustered
        uint16_t flags;
        uint64_t body_offset;
        uint32_t body_bytes;
        uint32_t dn_offset;
        uint32_t aux_offsets[2];
    };

Section map as far as verified: **section 3 = expert directory**, **section 4 = expert bodies**
(`strata_reader.cpp:113-136`, `hpp:76-77`). Six sections total; 2, 0, 1, 5 not yet read.

Notable: with `flags & 1` the directory stride becomes **34 bytes** = 32 B entry + **2 B `vram_load_order`**
(`reader.cpp:124`), i.e. the pack ships a *priority table* telling the loader which experts to put in VRAM first.

## What is worth stealing, in order of value to this project

1. **Per-expert codebook ids.** `gu_code_id` / `dn_code_id` mean experts do not share one global quantisation: each points
   at a codebook chosen for it. That is how 3.17 bits/parameter (10.47 GB for 26.4B) reaches 92 % same-next-token. Our engine
   uses one grid for all experts (IQ2_XXS) - per-expert codebooks are the obvious quality lever at equal size.
2. **Shipped VRAM load order.** A priority list computed at pack time, not at load time. For a 6 GB card with 26.4B
   parameters that is the whole trick: decide residency offline, stream the rest. We currently decide at runtime
   (`IN2_RES_GB`, 97.7 % hit rate) - theirs is precomputed.
3. **`cluster_id`.** Experts grouped into clusters with adjacent bodies, so a routing burst hits one file region instead of
   scattering. Directly applicable to our SSD streaming (`excess = 0.00597·N` per-token read cost).
4. **Per-section SHA-256.** Cheap integrity per section, not just per file.
5. **The loop remap.** `num_passes = get_num({"num_loops","loop.count"}, 3)` (`ink_gguf_loader.h:390`) and
   `n_phys = n_visits_meta - (num_passes-1)*core_len` (:399-405) with a virtual visit resolved as
   `core_start + (rel % core_len)` (`ink_cpu_backend.cpp:807`). 72 visits -> 48 physical blocks, which matches our own
   GGUF `block_count=48` and the 48 MoE calls we measured.
6. **Epoch-stamped cache slots.** `slot_pos_[pos % kRing] = ((int64_t)epoch_ << 32) | pos`, `++epoch_` on rewind
   (`dflash2.cpp:875`, `:854`, `:973`). This is the rewind primitive our `pos % cap` ring lacks.

## What is NOT taken

The `.strata` file is a GGUF-derived but incompatible container; nothing here loads it. The value is design and the
quality/size ratios, not the bytes. Their SQ-10 pack is 92.35 % same-next-token, which would fail this project's
bit-identical oracle by construction.

## Verified against the downloaded pack (10,471,526,400 bytes)

Header read with the offsets their own reader uses (`strata_reader.cpp:48-66`: magic at 0, flags u32 at 8, profile u16 at 12,
num_sections u16 at 14, table at 0x10, engram hash at 0x190). Six entries, each 64 B:

    sec 0: off=10,471,514,112  len=      10,780  flags=0   manifest JSON (matches the 0x2a1c seen at byte 24)
    sec 1: off=          4,096 len=1,871,628,224 flags=0   weights part A
    sec 2: off=              0 len=           0 flags=0   empty
    sec 3: off=10,471,403,520  len=     104,448  flags=1   EXPERT DIRECTORY (34 B stride -> load order present)
    sec 4: off= 1,871,634,432  len=8,599,769,088 flags=0   expert bodies
    sec 5: off=10,471,510,016  len=         138  flags=0

**All five non-empty sections hash to their recorded SHA-256.** The pack is intact; the download is not merely the right size.

Section 3 decodes to **3,072 entries at stride 34**, i.e. **24 MoE layers x 128 experts** - exactly the topology our own GGUF
reports (48 blocks, experts on layers 0-23, 128 experts, 8 used). Independent confirmation from a second engine's file format.

Per-expert quantisation is real and not uniform:

    distinct gu_code_id  = 220      (top: id 0 used by 1,901 experts, then 32:149, 3:145, 43:114, 1:90)
    distinct dn_code_id  = 150      (top: id 0 used by 2,015 experts, then 2:151, 79:146, 112:119)
    distinct cluster_id  = 472      (top: 0 for 1,372, 32:148, 20224:146, 28672:121)

So most experts share a default codebook, but ~1,100 gate/up and ~1,050 down codebooks are specialised per expert or cluster.
That is the mechanism behind 3.17 bits/parameter reaching 92 % same-next-token, and it is the concrete thing to copy: our engine
gives all 128 experts one IQ2_XXS grid.

The trailing 2-byte `vram_load_order` field parses to values like `[18, 771, 0, 0, 32768, 291, 0, 0, 32768, 48]` - alternating
zeros and 0x8000 suggest a bit-packed priority rather than a plain index, and it is **not** yet decoded. Recorded as unverified
rather than guessed.
