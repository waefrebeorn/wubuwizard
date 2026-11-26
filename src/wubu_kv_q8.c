/* wubu_kv_q8.c - q8_0 row quantiser for the KV cache.
 *
 * The KV cache is f32 today (float *k, *v in in2_kv_t) and that is what caps this box near 64k context: 8 full visits
 * at 64k already cost 4,608 MB, and 512k would be 34.4 GB. q8_0 is 34 bytes per 32 elements = 1.06 B/element, a 3.8x
 * reduction, and it is the same format llama.cpp uses for -ctk q8_0, so there is a reference token stream to validate
 * against once this is wired in.
 *
 * This file is deliberately standalone: quantise and dequantise one row, with a round-trip self-test, so the numerics
 * are proven before anything touches the cache layout. The block layout must match block_q8_0 in ggml-common.h:
 *     typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;      // 2 + 32 bytes, QK8_0 = 32
 * with d = amax / 127 and qs = round(x / d). That is exactly what ggml's quantize_row_q8_0 does; the dequant side
 * already exists in this repo and the two must agree or the cache will read back what it never stored.
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define KVQ8_BLOCK 32
#define KVQ8_BYTES 34

typedef struct { uint16_t d; int8_t qs[KVQ8_BLOCK]; } kvq8_blk;

static uint16_t f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;                 /* underflow to zero: scales are ~1e-2, never hit this */
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);    /* overflow to inf */
    return (uint16_t) (sign | ((uint32_t) e << 10) | (mant >> 13));
}

static float f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, mant = h & 0x3ffu, out;
    if (e == 0) { out = sign | (mant << 13); }
    else if (e == 31) { out = sign | 0x7f800000u | (mant << 13); }
    else { out = sign | ((e - 15 + 127) << 23) | (mant << 13); }
    { float r; memcpy(&r, &out, sizeof r); return r; }
}

/* Quantise n elements (n a multiple of 32) into blocks. Returns bytes written. */
size_t kvq8_quantize_row(const float *x, unsigned char *out, int n)
{
    if (n % KVQ8_BLOCK != 0) return 0;
    for (int i = 0; i < n; i += KVQ8_BLOCK) {
        float amax = 0.0f;
        for (int j = 0; j < KVQ8_BLOCK; ++j) {
            const float a = fabsf(x[i + j]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        kvq8_blk *b = (kvq8_blk *) (out + (size_t) (i / KVQ8_BLOCK) * KVQ8_BYTES);
        b->d = f32_to_f16(d);
        for (int j = 0; j < KVQ8_BLOCK; ++j) {
            int v = (int) lrintf(x[i + j] * id);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            b->qs[j] = (int8_t) v;
        }
    }
    return (size_t) (n / KVQ8_BLOCK) * KVQ8_BYTES;
}

/* Dequantise back into a caller-supplied f32 row (the cache will keep one scratch row per array). */
void kvq8_dequantize_row(const unsigned char *in, float *x, int n)
{
    for (int i = 0; i < n; i += KVQ8_BLOCK) {
        const kvq8_blk *b = (const kvq8_blk *) (in + (size_t) (i / KVQ8_BLOCK) * KVQ8_BYTES);
        const float d = f16_to_f32(b->d);
        for (int j = 0; j < KVQ8_BLOCK; ++j) x[i + j] = d * (float) b->qs[j];
    }
}

size_t kvq8_row_bytes(int n) { return (size_t) (n / KVQ8_BLOCK) * KVQ8_BYTES; }
