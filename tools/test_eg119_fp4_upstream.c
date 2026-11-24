/* test_eg119_fp4_upstream.c -- EG-119: fp4 / q2_0 dequant must match ggml's layout.
 *
 * Three independent checks, none of them self-referential:
 *  1. Golden blocks: hand-built NVFP4 / MXFP4 / Q2_0 blocks with codes 0..15 and
 *     known scale bytes, compared against values computed from ggml's tables.
 *  2. File structure: every type-40/42 tensor in the Inkling GGUF must advance the
 *     data offset by exactly the raw size this module reports.
 *  3. Lattice: dequantized expert weights must land on the E2M1 code set times a
 *     per-sub-block scale - which the previous 4-entry table cannot produce.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"
#include "wubu_dequant_fp4.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

/* ggml's E2M1 code set (magnitudes; sign is in the code). */
static const int codes[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

/* Q2_0 stores a signed scale (ggml does not enforce d > 0), so the check runs
 * against |scale| and both signs of every code are already in the table. */
static int on_lattice(float v, float scale)
{
    if (scale == 0.0f) return v == 0.0f;
    for (int i = 0; i < 16; ++i) {
        const double want = (double)codes[i] * (double)scale;
        if (fabs((double)v - want) <= 1e-5 * (1.0 + fabs(want))) return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    /* ---- 1. golden blocks ------------------------------------------------- */
    {
        /* NVFP4: 4 sub-blocks, scale byte UE4M3 0x38 = (1+0)*2^(7-7)*0.5 = 0.5.
         * Codes 0..15 written as low/high nibbles of 8 bytes per sub-block. */
        unsigned char blk[36];
        memset(blk, 0, sizeof(blk));
        for (int s = 0; s < 4; ++s) blk[s] = 0x38;
        for (int j = 0; j < 8; ++j) {
            const int lo = j;          /* codes 0..7  -> elements j        */
            const int hi = j + 8;      /* codes 8..15 -> elements j+8      */
            blk[4 + j] = (unsigned char)((lo & 0xF) | ((hi & 0xF) << 4));
        }
        float out[64];
        dequantize_row_nvfp4(blk, out, 64);
        for (int j = 0; j < 8; ++j) {
            CHECK(fabsf(out[j] - 0.5f * (float)codes[j]) < 1e-6f);
            CHECK(fabsf(out[j + 8] - 0.5f * (float)codes[j + 8]) < 1e-6f);
        }
        /* Low nibble first, high nibble in the upper half - not interleaved. */
        CHECK(out[0] == 0.0f && out[1] == 0.5f && out[15] == -6.0f);
    }
    {
        /* MXFP4: E8M0 byte 128 -> scale 2^0 = 1; codes low->j, high->j+16. */
        unsigned char blk[17];
        blk[0] = 128;
        for (int j = 0; j < 16; ++j) blk[1 + j] = (unsigned char)((j & 0xF) | (((j + 0) & 0xF) << 4));
        float out[32];
        dequantize_row_mxfp4(blk, out, 32);
        for (int j = 0; j < 16; ++j) {
            CHECK(fabsf(out[j] - (float)codes[j]) < 1e-6f);
            CHECK(fabsf(out[j + 16] - (float)codes[j]) < 1e-6f);
        }
    }
    {
        /* Q2_0: d = 2.0 (fp16 0x4000), codes 0,1,2,3 -> -2, 0, 2, 4. */
        unsigned char blk[18];
        blk[0] = 0x00; blk[1] = 0x40;
        for (int j = 0; j < 16; ++j) blk[2 + j] = 0xE4;   /* 0b11100100 */
        float out[64];
        dequantize_row_q2_0(blk, out, 64);
        CHECK(fabsf(out[0] + 2.0f) < 1e-6f);
        CHECK(fabsf(out[1] - 0.0f) < 1e-6f);
        CHECK(fabsf(out[2] - 2.0f) < 1e-6f);
        CHECK(fabsf(out[3] - 4.0f) < 1e-6f);
    }

    /* ---- 2 + 3. the real file -------------------------------------------- */
    const char *path = argc > 1 ? argv[1] : getenv("INKLING_GGUF");
    if (!path) path = "C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { printf("test_eg119: SKIP file check (no %s)\n", path); }
    else {
        if (!ctx->data_blob && !gguf_buffer_data(ctx)) {
            fprintf(stderr, "EG-119 could not map the data blob\n");
            ++failures;
        }
        int checked_types = 0, bad_stride = 0, lattice_bad = 0, lattice_tested = 0;
        for (int64_t i = 0; i + 1 < ctx->n_tensors; ++i) {
            const gguf_tensor_info *t = &ctx->tensors[i];
            if (t->ggml_type != 40 && t->ggml_type != 42) continue;
            ++checked_types;
            int64_t n = 1;
            for (int d = 0; d < t->n_dims; ++d) n *= t->dims[d];
            const long want = wubu_fp4_raw_size(t->ggml_type, (long)n);
            /* Stride to the next tensor's offset must equal the raw size. */
            const int64_t here = (int64_t)t->data_offset;
            int64_t next = -1;
            for (int64_t k = 0; k < ctx->n_tensors; ++k) {
                const int64_t off = (int64_t)ctx->tensors[k].data_offset;
                if (off > here && (next < 0 || off < next)) next = off;
            }
            if (next > here && (next - here) != (int64_t)want) {
                if (bad_stride < 4)
                    printf("EG-119 stride %s: %lld bytes between offsets, module says %ld\n",
                           t->name, (long long)(next - here), want);
                ++bad_stride;
            }
            /* Lattice test on the per-expert scale tensors we can afford to decode. */
            /* Expert stacks are hundreds of MB; decode a prefix slice instead. The
             * formats are blocked, so the first 4096 elements need only the first
             * 64 (type 40) or 64 (type 42) blocks. */
            if (lattice_tested < 200) {
                const int64_t slice = n < 4096 ? n : 4096;
                float *buf = (float *)malloc((size_t)slice * sizeof(float));
                if (buf) {
                    gguf_dequantize((const uint8_t *)ctx->data_blob + t->data_offset,
                                    t->ggml_type, slice, buf);
                    ++lattice_tested;
                    /* Recover the per-sub-block scale from the block header bytes. */
                    const uint8_t *raw = (const uint8_t *)ctx->data_blob + t->data_offset;
                    for (int64_t e = 0; e < slice; ++e) {
                        float scale;
                        if (t->ggml_type == 40) {
                            const int64_t blk = e / 64, sub = (e % 64) / 16;
                            const uint8_t sb = raw[blk * 36 + sub];
                            const int ex = (sb >> 3) & 0xF;
                            const int mn = sb & 7;
                            scale = 0.5f * (ex == 0 ? (float)mn / 512.0f
                                                    : (1.0f + (float)mn / 8.0f) * (float)(1u << (ex - 7)));
                        } else {
                            const int64_t blk = e / 64;
                            uint16_t h; memcpy(&h, raw + blk * 18, 2);
                            scale = gguf_f16_to_f32(h);
                        }
                        if (!on_lattice(buf[e], scale)) {
                            if (lattice_bad < 4)
                                printf("EG-119 off-lattice %s[%lld]=%g scale=%g\n",
                                       t->name, (long long)e, (double)buf[e], (double)scale);
                            ++lattice_bad;
                            break;
                        }
                    }
                    free(buf);
                }
            }
        }
        printf("EG-119 file: types checked=%d bad_stride=%d lattice_tested=%d off_lattice=%d\n",
               checked_types, bad_stride, lattice_tested, lattice_bad);
        CHECK(checked_types >= 20);
        CHECK(bad_stride == 0);
        CHECK(lattice_tested > 0);
        CHECK(lattice_bad == 0);
        gguf_close(ctx);
    }

    if (failures) { fprintf(stderr, "test_eg119_fp4_upstream: FAIL (%d)\n", failures); return 1; }
    printf("test_eg119_fp4_upstream: PASS\n");
    return 0;
}
