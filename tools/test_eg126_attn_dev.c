/* test_eg126_attn_dev.c -- EG-126: device attention over a populated KV cache at the
 * 12B model's real dimensions (16 q heads, 8 kv heads, hd 256, 20 positions).
 *
 * EG-124 gates the whole composite but at HIDDEN=256/HD=64 with two kv heads, and it
 * passes there while the composite is wrong on the real model. The attention kernel is
 * the stage that has never been compared at real width, so this gate isolates it: plain
 * float reference, no quantisation anywhere. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <windows.h>
typedef int (*fn_t)(const float *, const float *, const float *, float *,
                    int, int, int, int, int, int, int);

static unsigned long long rng = 0x9E3779B97F4A7C15ULL;
static float rnd(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((double)(rng >> 40) / 8388608.0 - 1.0);
}

int main(int argc, char **argv)
{
    HMODULE h = LoadLibraryA(argc > 1 ? argv[1] : "wubu_cuda.dll");
    if (!h) { printf("EG-126: SKIP (no dll)\n"); return 0; }
    fn_t attn = (fn_t)(void *)GetProcAddress(h, "wubu_cuda_attn_host");
    if (!attn) { printf("EG-126: SKIP (no export)\n"); FreeLibrary(h); return 0; }

    /* Two shapes: the sliding layers (hd 256, 8 kv heads) and the full-attention layers
     * (hd 512, 1 kv head) - the model uses both, and only the first was ever compared. */
    const int both = (argc > 2 && argv[2][0] == 'f');
    const int NHEADS = 16, NKV = both ? 1 : 8, HD = both ? 512 : 256;
    const int KVD = NKV * HD, CAP = 32;
    const int KVSTART = 0, KVLEN = 20, QD = NHEADS * HD;

    float *q = (float *)malloc(sizeof(float) * QD);
    float *kc = (float *)calloc((size_t)CAP * KVD, sizeof(float));
    float *vc = (float *)calloc((size_t)CAP * KVD, sizeof(float));
    float *out = (float *)malloc(sizeof(float) * QD);
    float *ref = (float *)malloc(sizeof(float) * QD);
    for (int i = 0; i < QD; ++i) q[i] = rnd();
    for (size_t i = 0; i < (size_t)(KVSTART + KVLEN) * KVD; ++i) { kc[i] = rnd(); vc[i] = rnd(); }

    const int rc = attn(q, kc, vc, out, NHEADS, NKV, HD, KVSTART, KVLEN, KVD, CAP);
    if (rc != 0) { printf("EG-126 FAIL rc=%d\n", rc); return 1; }

    for (int hq = 0; hq < NHEADS; ++hq) {
        const int kv_h = hq / (NHEADS / NKV);
        double sc[64], mx = -1e300, sum = 0.0;
        for (int kp = 0; kp < KVLEN; ++kp) {
            const float *kvp = kc + (size_t)(KVSTART + kp) * KVD + (size_t)kv_h * HD;
            double s = 0.0;
            for (int d = 0; d < HD; ++d) s += (double)q[hq * HD + d] * (double)kvp[d];
            sc[kp] = s;
            if (s > mx) mx = s;
        }
        for (int kp = 0; kp < KVLEN; ++kp) { sc[kp] = exp(sc[kp] - mx); sum += sc[kp]; }
        for (int d = 0; d < HD; ++d) {
            double a = 0.0;
            for (int kp = 0; kp < KVLEN; ++kp)
                a += sc[kp] / sum * (double)vc[(size_t)(KVSTART + kp) * KVD + (size_t)kv_h * HD + d];
            ref[hq * HD + d] = (float)a;
        }
    }
    double worst = 0.0;
    int worst_i = -1;
    double scale = 0.0;
    for (int i = 0; i < QD; ++i) if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    for (int i = 0; i < QD; ++i) {
        const double d = fabs((double)out[i] - (double)ref[i]);
        if (d > worst) { worst = d; worst_i = i; }
    }
    const double rel = scale > 0 ? worst / scale : worst;
    printf("EG-126 attn dev vs float reference: worst=%.3e rel=%.3e at %d (scale %.3e)\n",
           worst, rel, worst_i, scale);
    printf("EG-126 out[0..3]: %.6f %.6f %.6f %.6f\n", out[0], out[1], out[2], out[3]);
    printf("EG-126 ref[0..3]: %.6f %.6f %.6f %.6f\n", ref[0], ref[1], ref[2], ref[3]);
    const int ok = rel < 1e-4;
    printf("EG-126 %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
