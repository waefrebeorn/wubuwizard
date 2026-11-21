/* test_kv_qfmt.c -- LP-11 gate: KV reconstruction in compute format.
 * Checks: anchors decoded from fp16, finite output, bounded max error, and
 * attention dot computed from f32-decoded values with NO fp16 intermediate.
 * Prints: test_kv_qfmt: PASS  (only when every check holds). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_kv_qfmt.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

/* -ffast-math makes isfinite() unusable; test the IEEE bit pattern instead. */
static int finite_f32(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (u & 0x7F800000u) != 0x7F800000u;
}

static float fabs_(float x) { return x < 0.0f ? -x : x; }

int main(void)
{
    enum { N = 32 };
    float src[N], rec[N], q[N];
    uint8_t packed[32];
    uint8_t me = 0;
    float max_err = 0.0f, dot = 0.0f, ref_dot = 0.0f, ref_rec = 0.0f;
    double acc_ref = 0.0, acc_rec = 0.0;
    float sum_abs_q = 0.0f;
    wubu_kv_qfmt_t *ctx;

    ctx = wubu_kv_qfmt_create();
    CHECK(ctx != NULL);
    if (!ctx) return 1;

    /* Deterministic FP16 anchor set decoded exactly to f32. */
    CHECK(fabs_(wubu_kv_qfmt_anchor(ctx, 0) - (-6.0f)) < 1e-6f);
    CHECK(fabs_(wubu_kv_qfmt_anchor(ctx, 3) - (-0.5f)) < 1e-6f);
    CHECK(fabs_(wubu_kv_qfmt_anchor(ctx, 4) - 0.5f) < 1e-6f);
    CHECK(fabs_(wubu_kv_qfmt_anchor(ctx, 7) - 6.0f) < 1e-6f);
    CHECK(wubu_kv_qfmt_anchor(ctx, -1) == 0.0f);
    CHECK(wubu_kv_qfmt_anchor(ctx, WUBU_KV_QFMT_NANCHORS) == 0.0f);
    CHECK(wubu_kv_qfmt_packed_bytes(N) == 32u);

    /* Deterministic K-cache-like vector spread across the anchor range. */
    for (int i = 0; i < N; i++) {
        src[i] = -3.6f + 7.2f * (float)((i * 37 + 11) % 101) / 100.0f;
        q[i]   = 0.10f + 0.013f * (float)((i * 17 + 5) % 41);
        sum_abs_q += fabs_(q[i]);
    }

    CHECK(wubu_kv_qfmt_encode(ctx, src, N, packed, &me) == 0);
    CHECK(me > 0 && me < 255);

    /* Reconstruct directly into f32 and check finiteness + max error. */
    CHECK(wubu_kv_qfmt_decode(ctx, packed, me, rec, N) == 0);
    for (int i = 0; i < N; i++) {
        float e;
        CHECK(finite_f32(rec[i]));
        e = fabs_(rec[i] - src[i]);
        if (e > max_err) max_err = e;
        acc_ref += (double)src[i] * (double)q[i];
        acc_rec += (double)rec[i] * (double)q[i];
    }
    ref_dot = (float)acc_ref;
    ref_rec = (float)acc_rec;

    printf("LP-11 max reconstruction error = %.6f (bound 0.20)\n", max_err);
    CHECK(max_err <= 0.20f); /* nearest anchor <= 0.75, scale=0.125, E2M1 err <= 1 scale */

    /* Attention dot: decoded f32 values accumulated in f64 -> returned f32. */
    CHECK(wubu_kv_qfmt_dot(ctx, packed, me, q, N, &dot) == 0);
    CHECK(finite_f32(dot));

    /* No fp16 intermediate: result matches the f64/f32 reference to ~1e-5
     * relative. One fp16 round of the attention value drifts ~5e-4 relative
     * and would fail this check. */
    printf("LP-11 dot = %.8f, f64 reference = %.8f, f32-decode reference = %.8f\n",
           dot, ref_dot, ref_rec);
    CHECK(fabs_(dot - ref_rec) <= 1e-5f * (1.0f + fabs_(ref_rec)));

    /* Finite-output bound on the dot, derived from element error. */
    CHECK(fabs_(dot - ref_dot) <= (max_err + 1e-4f) * sum_abs_q);

    /* Corrupt anchor nibble is refused outright, never half-decoded. */
    packed[3] = (uint8_t)(0xF0u | (packed[3] & 0x0Fu));
    CHECK(wubu_kv_qfmt_decode(ctx, packed, me, rec, N) == -1);
    CHECK(wubu_kv_qfmt_decode(ctx, packed, 0, rec, N) == -1);

    wubu_kv_qfmt_free(ctx);

    if (failures == 0) printf("test_kv_qfmt: PASS\n");
    return failures ? 1 : 0;
}
