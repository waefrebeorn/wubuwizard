/* test_kv_fetch.c -- LP-12 gate: tolerance-parameterised KV fetch.
 * Checks: no host fetch when estimate <= eps, forced fetch when >= delta,
 * uncertain band flagged, pivotal tokens kept exactly on the fast side, and
 * fetched output matching the full-page reference.
 * Prints: test_kv_fetch: PASS  (only when every check holds). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_kv_fetch.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static float fabs_(float x) { return x < 0.0f ? -x : x; }

enum { N = 6, D = 4, NP = 2 };
static int pivots[NP] = { 1, 4 };
static float tk[N * D], tv[N * D], q[D];
static int host_calls_seen;

static float sk_k[N * D], sk_v[N * D];
static float pk[NP * D], pv[NP * D];

/* Host callback: supplies the full page and counts every call. */
static int host_page(void *ud, float *k, float *v, int n, int d)
{
    (void)ud;
    if (n != N || d != D) return -1;
    host_calls_seen++;
    memcpy(k, tk, sizeof(tk));
    memcpy(v, tv, sizeof(tv));
    return 0;
}

/* Independent full-page reference attention. */
static void ref_attention(float *out)
{
    double mx = -1e300, z = 0.0;
    float sc[N];
    float inv = 1.0f / sqrtf((float)D);
    for (int i = 0; i < N; i++) {
        double s = 0.0;
        for (int j = 0; j < D; j++) s += (double)tk[i * D + j] * (double)q[j];
        sc[i] = (float)(s * (double)inv);
        if (sc[i] > mx) mx = sc[i];
    }
    for (int i = 0; i < N; i++) { sc[i] = expf(sc[i] - (float)mx); z += sc[i]; }
    memset(out, 0, sizeof(float) * D);
    for (int i = 0; i < N; i++) {
        float p = (float)((double)sc[i] / z);
        for (int j = 0; j < D; j++) out[j] += p * tv[i * D + j];
    }
}

static float max_diff(const float *a, const float *b)
{
    float m = 0.0f;
    for (int j = 0; j < D; j++) {
        float e = fabs_(a[j] - b[j]);
        if (e > m) m = e;
    }
    return m;
}

/* Build sketch with a given per-dimension offset on pivotal tokens. */
static void build_local(float pivot_off)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            float noise = 0.001f * (float)(((i * 3 + j * 5) % 7) - 3);
            sk_k[i * D + j] = tk[i * D + j] + noise;
            sk_v[i * D + j] = tv[i * D + j] + noise;
        }
    }
    for (int p = 0; p < NP; p++) {
        for (int j = 0; j < D; j++) {
            sk_k[pivots[p] * D + j] = tk[pivots[p] * D + j] + pivot_off;
            sk_v[pivots[p] * D + j] = tv[pivots[p] * D + j] + pivot_off;
            pk[p * D + j] = tk[pivots[p] * D + j];
            pv[p * D + j] = tv[pivots[p] * D + j];
        }
    }
}

int main(void)
{
    float ref[D], out[D];
    wubu_kv_fetch_t *f;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            tk[i * D + j] = 0.25f * (float)(((i * 7 + j * 3) % 9) - 4);
            tv[i * D + j] = 0.30f * (float)(((i * 5 + j * 2) % 7) - 3);
        }
    }
    for (int j = 0; j < D; j++) q[j] = 0.2f + 0.1f * (float)j;
    ref_attention(ref);

    f = wubu_kv_fetch_create(N, D, 0.01f, 0.50f);
    CHECK(f != NULL);
    if (!f) return 1;

    /* 1. estimate below eps: NO host fetch, output close to full page. */
    build_local(0.001f);
    CHECK(wubu_kv_fetch_set_local(f, sk_k, sk_v, pivots, pk, pv, NP) == 0);
    host_calls_seen = 0;
    CHECK(wubu_kv_fetch_estimate(f, q) < 0.01f);
    CHECK(wubu_kv_fetch_attention(f, q, host_page, NULL, out) == 0);
    CHECK(wubu_kv_fetch_decision(f) == WUBU_KV_FETCH_NO);
    CHECK(wubu_kv_fetch_host_calls(f) == 0);
    CHECK(host_calls_seen == 0);
    printf("LP-12 no-fetch est=%.5f maxdiff=%.6f\n",
           wubu_kv_fetch_last_estimate(f), max_diff(out, ref));
    CHECK(max_diff(out, ref) <= 0.01f);

    /* 2. estimate above delta: forced host fetch, exact full-page output. */
    build_local(2.0f);
    CHECK(wubu_kv_fetch_set_local(f, sk_k, sk_v, pivots, pk, pv, NP) == 0);
    host_calls_seen = 0;
    CHECK(wubu_kv_fetch_estimate(f, q) > 0.50f);
    CHECK(wubu_kv_fetch_attention(f, q, host_page, NULL, out) == 0);
    CHECK(wubu_kv_fetch_decision(f) == WUBU_KV_FETCH_FORCED);
    CHECK(wubu_kv_fetch_host_calls(f) == 1);
    CHECK(host_calls_seen == 1);
    printf("LP-12 forced est=%.5f maxdiff=%.8f\n",
           wubu_kv_fetch_last_estimate(f), max_diff(out, ref));
    CHECK(max_diff(out, ref) <= 1e-4f);

    /* 3. estimate inside (eps, delta): uncertain, still no fetch. */
    build_local(0.10f);
    CHECK(wubu_kv_fetch_set_local(f, sk_k, sk_v, pivots, pk, pv, NP) == 0);
    host_calls_seen = 0;
    CHECK(wubu_kv_fetch_attention(f, q, host_page, NULL, out) == 0);
    CHECK(wubu_kv_fetch_decision(f) == WUBU_KV_FETCH_UNCERTAIN);
    CHECK(wubu_kv_fetch_host_calls(f) == 0);
    CHECK(host_calls_seen == 0);

    /* 4. a failed host fetch is not half-applied and does not count. */
    build_local(2.0f);
    CHECK(wubu_kv_fetch_set_local(f, sk_k, sk_v, pivots, pk, pv, NP) == 0);
    host_calls_seen = 0;
    CHECK(wubu_kv_fetch_attention(f, q, NULL, NULL, out) == -1);
    CHECK(wubu_kv_fetch_host_calls(f) == 0);

    /* 5. bad inputs refused. */
    CHECK(wubu_kv_fetch_set_local(f, sk_k, sk_v, pivots, pk, pv, N + 1) == -1);
    pivots[1] = N; /* out of range */
    CHECK(wubu_kv_fetch_set_local(f, sk_k, sk_v, pivots, pk, pv, NP) == -1);

    wubu_kv_fetch_free(f);

    if (failures == 0) printf("test_kv_fetch: PASS\n");
    return failures ? 1 : 0;
}
