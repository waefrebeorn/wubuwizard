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

Section 3 decodes to **3,072 entries, and the layout is NOT interleaved**: `reader.cpp:123-131` copies all
`n * 32 = 98,304` bytes of `StrataExpertEntry` first, then `n * 2 = 6,144` bytes of `uint16_t vram_load_order` - 104,448
total, which matches the section length exactly. Parsing it as 34-byte interleaved records (my first attempt) produces
plausible-looking but wrong statistics, and those numbers were committed before this was checked.

Correct directory contents:

    entries          3,072 = 24 MoE layers x 128 experts, expert_id 0..127 repeated per layer, flags all 0
    distinct gu_code_id  =   3      (id 3: 2,472 experts, id 2: 585, id 15: 15)
    distinct dn_code_id  =   3      (id 2: 2,399, id 3: 658, id 15: 15)
    cluster_id           =   0      for every entry - clustering is not used in this pack
    vram_load_order      =   a true permutation of 0..3071 (first 12: 2176, 2049, 1920, 2177, 2048, 2181, 2178,
                             1921, 2179, 1408, 1024, 1152)

So per-expert codebooks are NOT the mechanism here: three gate/up and three down codebooks cover all 3,072 experts, which
means the specialisation is per *tensor group*, not per expert. **Retracted: "220 distinct gu codebooks, ~1,100 specialised"**
- that came from the misaligned read and is wrong.

What IS real and valuable: `vram_load_order` is a complete priority permutation over all 3,072 expert bodies, shipped in the
pack. Residency is decided offline, once, and the engine just walks the list - whereas our engine decides at runtime with
`IN2_RES_GB`. The first entries cluster in high expert indices (2176+, i.e. layers 17+), consistent with ordering by use.
rather than guessed.

## Section 0 is a ZSTD manifest - and it contains the drafter

`strata_reader.cpp:71-107` decompresses section 0 with ZSTD into `manifest_json_`. Decompressed: 92,499 bytes of JSON. This is
the single most informative artifact in the pack.

    format        strata-v1.1
    build         SQ-10
    arch          inkling2_loop
    allocation    global_hessian_knapsack_v2
    trunk_tensors 646        dspark_tensors 54
    num_physical_layers 24   num_experts_per_layer 128
    quant         modelopt / NVFP4

### The codebooks are per-expert BIT ALLOCATION, not per-tensor-group

    code_counts: S3_K+S3_K : 658, S3_K+S2_K : 1,814, S2_K+S2_K : 585, SF4+SF4 : 15   (sum 3,072 = every expert)

Each expert gets a gate/up codebook and a down codebook selected from {2-bit, 3-bit, 4-bit-float}, chosen by
`global_hessian_knapsack_v2`. That is the answer to the earlier measurement (`distinct gu_code_id = 3`) - three *schemes*, not
three shared codebooks: **mixed precision per expert, allocated by a global Hessian knapsack**. My earlier reading that this was
"per tensor group" was wrong in substance; the three ids are three bit-widths and the assignment is per expert.

### DSpark: the drafter, fully specified

`mtp_config` plus 54 tensors, **216.2 MB total**:

    3 layers, each: q/k/v/o proj (plus *_ctx cross-attention variants), mlp gate/up/down (intermediate 9216),
                    input/post layernorm, dflare_norm, q_norm/k_norm
    markov_head:    markov_w1 [201024, 256] Q8_ROW (52 MB), rnn_proj [768, 5120], hidden_out [2304, 256]
    mask_embed [2304], fuse_weight [4,4], norm [2304]
    formats: 27 x S5_K, 24 x F32, 3 x Q8_ROW

    block_size 8            -> predicts 8 tokens per step
    markov_rank 256         -> the Markov head's rank over the 201,024-token vocabulary
    target_layer_ids [5, 23, 41, 48]   -> consumes hidden states from those four target-model layers
    local_layer_ids [0, 2]  -> sliding_window_size 1024 on those
    rope_theta 1e6, 24 heads / 8 KV heads, head_dim 128, hidden 2304

So a real drafter exists for this model, is 216 MB, is a 3-layer cross-attention + rank-256 Markov head that reads four
intermediate layers of the target and emits 8 tokens per step. That is why upstream reports 1.7-3.1x at unchanged output: their
acceptance comes from this module, not from n-gram matching.

**To use it here** we would need: (a) extract 216 MB by the manifest's own offsets, (b) dequantise or port `S5_K` and `Q8_ROW`,
(c) implement the 3-layer cross-attention against our layers 5/23/41/48, (d) implement the Markov head. That is a project with a
clear specification, not a session. It is now the only path to beating upstream's speed at identical output.

### One open discrepancy, flagged rather than guessed

`target_layer_ids` ends at **48**, which implies indices 0..48 exist (49 blocks), while our GGUF reports `block_count = 48` and
the trunk here lists 646 tensors against our 734. Either the target ids are 1-based, or the two distributions differ in layer
count. Not resolved; it matters because the drafter keys off those exact indices.

## Extraction proven for F32; Q8_ROW is NOT decoded and my first read of it was wrong

**Offsets are relative to section 1's start (4,096).** Proven three ways with tensors whose values can be sanity-checked:

    dspark.layers.0.input_layernorm.weight  off=1,644,895,424  base sec1 -> [1.093, 1.899, -0.542, 0.976, 1.063]  sane
    dspark.layers.0.self_attn.q_norm.weight off=1,693,870,272  base sec1 -> [0.500, 1.078, 0.614, 0.463, 0.527]  sane
    dspark.norm.weight                      off=1,871,619,008  base sec1 -> [-2.593, 3.094, 2.751, -3.438, -3.328] sane

The third is decisive positionally: 1,871,619,008 + 4,096 = 1,871,623,104, and section 1 ends at 1,871,632,320 - the drafter sits
in section 1's **tail**, immediately before the expert bodies. The other bases produce float garbage from the same offsets.

**Q8_ROW: not decoded.** `markov_head.hidden_out.weight` is `[2304, 256]` with 599,040 bytes, and 599,040 = 2304 x 260, so 4 bytes of
overhead per 256 int8 values is the obvious reading. The first three rows decode cleanly under both "scale first" and "scale last"
(|w| <= 0.21, stdev ~0.1), which looked like confirmation - and a 200-row check destroyed it:

    rows saturating at |q| = 127: 165/200 under BOTH layouts
    recovered scale range: -6.9e37 .. 9.1e35   (absurd)

Identical saturation counts under two different offsets, plus absurd recovered scales on most rows, means the rows are **not** a
fixed 260-byte stride and the overhead is not one fp32 per row. My three-row check was three lucky rows - the same failure this
project keeps repeating: a plausible sample from a small window treated as proof.

The right next move is to read `engine/include/strata/quant/sk_codecs.hpp` and `engine/src/quant/` for the actual Q8_ROW and S5_K
decoders rather than reverse-engineering them from bytes. Recorded as unresolved, with the structural fact (599,040 = 2304 x 260)
that any future attempt should start from.

## Q8_ROW cracked and verified; S_K is a super-block codec we must port, not guess

The answer is in their code, not in the bytes: `ink_cpu_backend.cpp:50` and `hip_codecs:2836` agree -

    Q8_ROW layout: [rows] float32 scales, followed by [rows, cols] int8_t weights

i.e. **a separate scale plane, not an interleaved per-row header**. For `markov_head.hidden_out` `[2304, 256]`:
`2304*4 + 2304*256 = 599,040` bytes, an exact match. Decoded and checked:

    scales: min 1.12e-3, max 2.57e-3, median 1.48e-3, all finite and positive
    per-row stdev across rows 0, 1, 2, 1152, 2303: 0.06254, 0.06259, 0.06238, 0.06261, 0.06261   (uniform, as a real matrix)
    sampled 34,816 weights: mean|x| 0.0502, stdev 0.0626, max|x| 0.266

So my earlier `599,040 = 2304 x 260` reading was a numerical coincidence: 260 x 2304 and 4 x 2304 + 256 x 2304 are the same
number for the wrong reason. The lesson repeats - the layout was never in the byte count.

`S5_K` is not decodable by inspection at all: it goes through `dequant_sk_row(src, SkCode, out, n)`, the `strata::quant::sk`
super-block codec (`sk_codecs.hpp`: `QK_S = 256`, four 64-weight sub-groups). Porting it means porting that codec, not
reverse-engineering rows.

### And their own source confirms our rollback design

Two comments in `CpuSlot` settle questions we answered by measurement and inference:

    std::vector<SlotShortConvState> sconv_snaps;  // [0..8] exact post-step ShortConv snapshots for DSpark partial-accept rollback!
    std::vector<std::vector<uint16_t>> dspark_k, dspark_v;   // [3][max_ctx * kv_dim] BF16

They snapshot the **short-conv state per step** for partial-accept rollback - precisely the second half of the rollback problem we
identified independently today (the 4-tap conv history that invalidating K/V rows does not undo), and they keep snapshots for
steps 0..8, which matches `block_size = 8`. Our `in2_kv_conv_save/restore` over both caches is the same idea with one snapshot
instead of a ring of eight.
