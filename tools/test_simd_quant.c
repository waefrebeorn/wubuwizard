/* test_simd_quant.c -- LP-15 gate: per-shape native SIMD quant decode.
 * Checks per-shape dispatch, forced AVX2/SSE/scalar kernels and exact NaN/Inf
 * bit identity on every path. Prints: test_simd_quant: PASS. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_simd_quant.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static float fabs_(float x) { return x < 0.0f ? -x : x; }

static uint32_t bits_of(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static float float_of(uint32_t u)
{
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

enum { N = 16 };
static const uint32_t specials[N / 2] = {
    0x7F800000u, /* +Inf  */
    0xFF800000u, /* -Inf  */
    0x7FC00001u, /* qNaN  */
    0xFFC00002u, /* -qNaN */
    0x7F800001u, /* sNaN  */
    0xFF800003u, /* -sNaN */
    0x7FFFFFFFu, /* qNaN max payload */
    0xFFC00000u  /* -qNaN */
};

/* Verify finite accuracy and NaN/Inf bit identity after a decode pass. */
static void check_output(const float *in, const float *out, const char *tag)
{
    int bad_bits = 0;
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        if (i & 1) {
            if (bits_of(out[i]) != bits_of(in[i])) bad_bits++;
        } else {
            float e = fabs_(out[i] - in[i]);
            if (e > max_err) max_err = e;
        }
    }
    printf("LP-15 %s: max finite err=%.4f, bit mismatches=%d\n", tag, max_err, bad_bits);
    CHECK(bad_bits == 0);
    CHECK(max_err <= 0.25f); /* scale 0.5 -> half-step 0.25 */
}

int main(void)
{
    float in[N], out[N];
    uint64_t tok[N];
    wubu_simd_quant_t *s;
    int avx2;

    CHECK(wubu_simd_quant_create(0.0f) == NULL);
    CHECK(wubu_simd_quant_create(-1.0f) == NULL);
    s = wubu_simd_quant_create(0.5f);
    CHECK(s != NULL);
    if (!s) return 1;

    for (int i = 0; i < N; i++) {
        if (i & 1) in[i] = float_of(specials[i >> 1]);
        else in[i] = (float)(((i * 7) % 13) - 6) * 0.25f;
    }
    CHECK(wubu_simd_quant_encode(s, in, tok, N) == 0);

    /* Per-shape dispatch. */
    avx2 = wubu_simd_quant_has_avx2();
    if (avx2) CHECK(wubu_simd_quant_dispatch(24) == WUBU_SIMD_Q_AVX2_8);
    CHECK(wubu_simd_quant_dispatch(10) == WUBU_SIMD_Q_SSE_2);
    CHECK(wubu_simd_quant_dispatch(13) == WUBU_SIMD_Q_SCALAR);

    /* Every native path must produce identical results and exact special bits. */
    CHECK(wubu_simd_quant_decode_path(s, WUBU_SIMD_Q_SCALAR, tok, out, N) == 0);
    CHECK(wubu_simd_quant_last_path(s) == WUBU_SIMD_Q_SCALAR);
    check_output(in, out, "scalar");

    CHECK(wubu_simd_quant_decode_path(s, WUBU_SIMD_Q_SSE_2, tok, out, N) == 0);
    CHECK(wubu_simd_quant_last_path(s) == WUBU_SIMD_Q_SSE_2);
    check_output(in, out, "sse2");

    if (avx2) {
        CHECK(wubu_simd_quant_decode_path(s, WUBU_SIMD_Q_AVX2_8, tok, out, N) == 0);
        CHECK(wubu_simd_quant_last_path(s) == WUBU_SIMD_Q_AVX2_8);
        check_output(in, out, "avx2");
    }

    /* Auto dispatch for a shape of 16 and an odd tail of 13. */
    CHECK(wubu_simd_quant_decode(s, tok, out, N) == 0);
    CHECK(wubu_simd_quant_last_path(s) ==
          (avx2 ? WUBU_SIMD_Q_AVX2_8 : WUBU_SIMD_Q_SSE_2));
    check_output(in, out, "auto-16");

    CHECK(wubu_simd_quant_decode(s, tok, out, 13) == 0);
    CHECK(wubu_simd_quant_last_path(s) == WUBU_SIMD_Q_SCALAR);
    for (int i = 0; i < 13; i++) {
        if (i & 1) CHECK(bits_of(out[i]) == bits_of(in[i]));
    }

    wubu_simd_quant_free(s);

    if (failures == 0) printf("test_simd_quant: PASS\n");
    return failures ? 1 : 0;
}
