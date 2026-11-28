/* test_f16_group.c - WuBuMM-style correctness test for the grouped f16 GEMM export.
 *
 * Discipline (why this exists): per-SHAPE PASS/FAIL, a double-precision reference, cosine + max|abs diff|, and a
 * POISON check that proves every output element was written (an unwritten element means a bounds/tiling bug that a
 * plain cos test can hide behind). Boundary-heavy on purpose: M from 1 to 65 (odd, prime, tile-1, tile, tile+1),
 * K in {32,64,2304}, N in {16,32,3072}, and a group whose n_cols differ (N, N, N/2) so per-op offsets are exercised.
 *
 * usage: test_f16_group.exe <path to dll>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

typedef int (*grp_t)(const long long *, const void *const *, const unsigned long long *, const int *, int,
                     const float *, int, int, float *);
typedef int (*one_t)(long long, const void *, unsigned long long, const float *, int, int, float *, int);

/* exact-enough float <-> half so the reference uses the SAME values the GPU does */
static unsigned short f2h(float f) {
    union { float f; unsigned u; } v; v.f = f;
    unsigned u = v.u, sign = (u >> 16) & 0x8000u;
    int e = (int)((u >> 23) & 0xffu) - 127 + 15;
    unsigned m = u & 0x7fffffu;
    if (f == 0.0f) return (unsigned short) sign;
    if (e <= 0) { if (e < -10) return (unsigned short) sign; m |= 0x800000u; return (unsigned short)(sign | (m >> (13 - e))); }
    if (e >= 31) return (unsigned short)(sign | 0x7c00u);
    return (unsigned short)(sign | ((unsigned) e << 10) | (m >> 13));
}
static float h2f(unsigned short h) {
    unsigned sign = (unsigned)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, u;
    union { unsigned u; float f; } v;
    if (e == 0) { if (m == 0) u = sign; else { e = 1; while (!(m & 0x400u)) { m <<= 1; --e; } m &= 0x3ffu; u = sign | ((unsigned)(e + 127 - 15) << 23) | (m << 13); } }
    else if (e == 31) u = sign | 0x7f800000u | (m << 13);
    else u = sign | ((unsigned)(e + 127 - 15) << 23) | (m << 13);
    v.u = u; return v.f;
}
static unsigned rng_state = 12345u;
static unsigned rng(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state >> 8; }
static float rndf(void) { return ((float)(rng() % 2001) - 1000.0f) / 1000.0f; }   /* [-1,1] */

#define POISON 12345.0f

static int run_case(grp_t grp, one_t one, int M, int K, int N, int n_ops, const char *tag) {
    /* Weights: n_ops slabs of [n_cols][K] f16, plus exact-f32 twins for the reference.
     * IMPORTANT (this bit me): the resident cache is keyed by (ident, wbytes) and assumes ident <-> immutable weight
     * bytes, exactly like the engine's file_offset keys. So derive ident from (n_cols,K) AND seed the RNG from the same
     * ident: the same ident always yields the same weights, which makes cache hits legitimate and the comparison valid. */
    int n_cols[4]; unsigned short *wh[4]; float *wf[4]; long long idents[4];
    for (int i = 0; i < n_ops; ++i) {
        n_cols[i] = (i == n_ops - 1 && n_ops > 1) ? (N / 2 > 0 ? N / 2 : 1) : N;
        idents[i] = 1000000 + (long long) n_cols[i] * 7919 + (long long) K;
        rng_state = (unsigned) (idents[i] * 2654435761u + 12345u);
        wh[i] = (unsigned short *) malloc((size_t) n_cols[i] * K * 2);
        wf[i] = (float *) malloc((size_t) n_cols[i] * K * 4);
        for (long long t = 0; t < (long long) n_cols[i] * K; ++t) { float f = rndf(); wh[i][t] = f2h(f); wf[i][t] = h2f(wh[i][t]); }
    }
    float *act = (float *) malloc((size_t) M * K * 4);
    for (long long t = 0; t < (long long) M * K; ++t) act[t] = rndf();

    long long nout = 0, off[4];
    for (int i = 0; i < n_ops; ++i) { off[i] = nout; nout += (long long) n_cols[i] * M; }

    /* --- grouped: ONE H2D, n GEMMs, ONE D2H --- */
    float *g = (float *) malloc((size_t) nout * 4);
    for (long long t = 0; t < nout; ++t) g[t] = POISON;
    int rc = grp(idents, (const void *const *) wh,
                 (const unsigned long long[]){ (unsigned long long) n_cols[0] * K * 2, (unsigned long long) n_cols[1] * K * 2,
                                               (unsigned long long) n_cols[n_ops - 1] * K * 2, 0 },
                 n_cols, n_ops, act, M, K, g);

    /* --- single calls, one round trip each, for an A/B of the grouped path --- */
    float *s = (float *) malloc((size_t) nout * 4);
    for (long long t = 0; t < nout; ++t) s[t] = POISON;
    int rc1 = 0;
    for (int i = 0; i < n_ops; ++i)
        rc1 |= one(idents[i], wh[i], (unsigned long long) n_cols[i] * K * 2, act, M, K, s + off[i], n_cols[i]);

    /* --- reference in double --- */
    double *ref = (double *) malloc((size_t) nout * 8);
    for (int i = 0; i < n_ops; ++i)
        for (int m = 0; m < M; ++m)
            for (int o = 0; o < n_cols[i]; ++o) {
                const float *wr = wf[i] + (size_t) o * K, *ar = act + (size_t) m * K;
                double acc = 0.0;
                for (int k = 0; k < K; ++k) acc += (double) wr[k] * (double) ar[k];
                ref[off[i] + (long long) m * n_cols[i] + o] = acc;
            }

    /* --- score --- */
    double maxd_g = 0, maxd_s = 0, mag = 0, dg = 0, ds = 0, ng = 0, ns = 0;
    long long un_g = 0, un_s = 0;
    for (long long t = 0, idx = 0; t < nout; ++t) {
        if (g[t] == POISON) un_g++;
        if (s[t] == POISON) un_s++;
    }
    for (int i = 0; i < n_ops; ++i)
        for (int m = 0; m < M; ++m)
            for (int o = 0; o < n_cols[i]; ++o) {
                long long e = off[i] + (long long) m * n_cols[i] + o;
                double r = ref[e];
                double a = fabs((double) g[e] - r), b = fabs((double) s[e] - r);
                if (a > maxd_g) maxd_g = a;
                if (b > maxd_s) maxd_s = b;
                if (fabs(r) > mag) mag = fabs(r);
                dg += (double) g[e] * r; ds += (double) s[e] * r; ng += (double) g[e] * g[e]; ns += (double) s[e] * s[e];
            }
    double cos_g = (ng > 0 && dg != 0) ? dg / sqrt(ng * (dg / (dg / 1.0)) * 0 + 1e-30) : 0;   /* recompute properly below */
    {   double rr = 0; for (long long t = 0; t < nout; ++t) rr += ref[t] * ref[t];
        cos_g = (ng > 0 && rr > 0) ? dg / sqrt(ng * rr) : 0; }
    double cos_s;
    {   double rr = 0; for (long long t = 0; t < nout; ++t) rr += ref[t] * ref[t];
        cos_s = (ns > 0 && rr > 0) ? ds / sqrt(ns * rr) : 0; }

    int pass = (rc == 0) && (rc1 == 0) && (un_g == 0) && (un_s == 0) && (cos_g > 0.999999) && (cos_s > 0.999999)
               && (maxd_g < 1e-3 * (mag > 1 ? mag : 1));
    if (!pass) {
        /* attribute the failure to a specific op in the group, so the bug is localised not guessed */
        for (int i = 0; i < n_ops; ++i) {
            double d = 0, ng2 = 0, rr2 = 0, md = 0;
            for (int m = 0; m < M; ++m)
                for (int o = 0; o < n_cols[i]; ++o) {
                    long long e = off[i] + (long long) m * n_cols[i] + o;
                    double r = ref[e]; double a = fabs((double) g[e] - r);
                    if (a > md) md = a;
                    d += (double) g[e] * r; ng2 += (double) g[e] * g[e]; rr2 += r * r;
                }
            printf("      op%d n_cols=%-5d grp cos=%.9f max|d|=%.3e   (first got=%.4f exp=%.4f)\n",
                   i, n_cols[i], (ng2 > 0 && rr2 > 0) ? d / sqrt(ng2 * rr2) : 0, md, g[off[i]], ref[off[i]]);
            {   long long bad = 0, first = -1;
                for (int m = 0; m < M; ++m)
                    for (int o = 0; o < n_cols[i]; ++o) {
                        long long e = off[i] + (long long) m * n_cols[i] + o;
                        if (fabs((double) g[e] - ref[e]) > 0.1) { bad++; if (first < 0) first = (long long) m * n_cols[i] + o; }
                    }
                printf("        -> %lld/%d elements off by >0.1, first bad at local idx %lld (row %lld col %lld)\n",
                       bad, n_cols[i] * M, first, first / n_cols[i], first % n_cols[i]); }
        }
    }
    printf("  %-28s M=%-3d K=%-4d N=%-4d ops=%d  grp rc=%d cos=%.9f max|d|=%.3e | one rc=%d cos=%.9f max|d|=%.3e | unwritten g=%lld s=%lld  %s\n",
           tag, M, K, N, n_ops, rc, cos_g, maxd_g, rc1, cos_s, maxd_s, un_g, un_s, pass ? "PASS" : "** FAIL **");
    for (int i = 0; i < n_ops; ++i) { free(wh[i]); free(wf[i]); }
    free(act); free(g); free(s); free(ref);
    return pass;
}

int main(int argc, char **argv) {
    const char *dll = (argc > 1) ? argv[1] : "wubu_cuda.dll";
    HMODULE h = LoadLibraryA(dll);
    if (!h) { printf("load fail %lu\n", (unsigned long) GetLastError()); return 3; }
    grp_t grp = (grp_t)(void *) GetProcAddress(h, "wubu_cuda_f16_group");
    one_t one = (one_t)(void *) GetProcAddress(h, "wubu_cuda_resident_f16_gemm");
    if (!grp || !one) { printf("missing export (grp=%p one=%p)\n", (void *) grp, (void *) one); return 4; }

    int total = 0, ok = 0;
    printf("== bounds sweep: K=64 N=32, M from 1..65 (odd / prime / tile-1 / tile / tile+1) ==\n");
    {   int Ms[] = { 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65 };
        for (unsigned i = 0; i < sizeof(Ms) / sizeof(Ms[0]); ++i) { total++; ok += run_case(grp, one, Ms[i], 64, 32, 3, "sweep"); } }
    printf("== engine shapes ==\n");
    total++; ok += run_case(grp, one, 1, 2304, 3072, 3, "decode M=1");
    /* discriminators for the op2(n_cols=N/2) failure: is it position, the halved width, or accumulation? */
    total++; ok += run_case(grp, one, 1, 2304, 3072, 1, "decode 1 op (no halve)");
    total++; ok += run_case(grp, one, 1, 2304, 3072, 2, "decode 2 ops (2nd halved)");
    total++; ok += run_case(grp, one, 1, 2304, 1536, 1, "decode single 1536 wide");
    total++; ok += run_case(grp, one, 16, 2304, 3072, 3, "prefill M=16");
    total++; ok += run_case(grp, one, 7, 2304, 16, 3, "odd M,K,N");
    total++; ok += run_case(grp, one, 3, 32, 3072, 1, "single op");
    printf("\nVERDICT: %d/%d cases %s\n", ok, total, ok == total ? "PASS" : "FAIL");
    return ok == total ? 0 : 1;
}
