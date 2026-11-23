/* test_eg77_vc_attn.c -- EG-77 gate: V residual around a per-tile mean. */
#include <stdio.h>
#include <math.h>

#include "wubu_vc_attn.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float v[256], back[256], logits[256], got, want;
    unsigned char codes[256];
    float means[2], scales[2];
    double maxerr = 0.0;
    for (int i = 0; i < 256; ++i) {
        float mean = i < 128 ? 4.0f : -2.0f;
        v[i] = mean + 0.05f * (float)((i * 7 + 3) % 11);
        logits[i] = 0.01f * (float)((i * 5) % 23);
    }
    CHECK(wubu_vc_quant(v, 256, codes, means, scales) == 2);
    CHECK(fabs(means[0] - 4.0f) < 0.3f && fabs(means[1] + 2.0f) < 0.3f);
    CHECK(wubu_vc_dequant(codes, means, scales, 256, back) == 0);
    for (int i = 0; i < 256; ++i) {
        double e = fabs(back[i] - v[i]);
        if (e > maxerr) maxerr = e;
    }
    CHECK(maxerr < 0.03);
    CHECK(wubu_vc_attention(logits, back, 256, &got) == 0);
    CHECK(wubu_vc_attention(logits, v, 256, &want) == 0);
    CHECK(fabs(got - want) < 0.03);
    printf("EG-77 tiles=2 dequant maxerr=%.5f attention %.5f vs %.5f\n",
           maxerr, got, want);
    CHECK(wubu_vc_quant(v, 255, codes, means, scales) == -1);
    CHECK(wubu_vc_dequant(NULL, means, scales, 256, back) == -1);

    if (failures) { fprintf(stderr, "test_eg77_vc_attn: FAIL (%d)\n", failures); return 1; }
    printf("test_eg77_vc_attn: PASS\n");
    return 0;
}
