/* test_eg84_awq.c -- EG-84 gate: AWQ uint4 group scales + zero points. */
#include <stdio.h>
#include <math.h>

#include "wubu_awq.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float w[16], back[16];
    unsigned char codes[16];
    float scales[2];
    int zeros[2], groups;
    double maxerr = 0.0;
    for (int i = 0; i < 16; ++i) w[i] = 0.1f * (float)((i * 5 + 3) % 17) - 0.8f;

    groups = wubu_awq_u4_pack(w, 16, 8, codes, scales, zeros);
    CHECK(groups == 2);
    CHECK(wubu_awq_u4_unpack(codes, scales, zeros, 16, 8, back) == 0);
    for (int i = 0; i < 16; ++i) {
        double e = fabs(back[i] - w[i]);
        if (e > maxerr) maxerr = e;
    }
    CHECK(maxerr < 0.08);
    printf("EG-84 groups=%d scales=[%.4f %.4f] zeros=[%d %d] maxerr=%.5f\n",
           groups, scales[0], scales[1], zeros[0], zeros[1], maxerr);

    /* Constant groups are lossless and groups are independent. */
    {
        float cw[16], cb[16], w2[16];
        unsigned char c2[16];
        float s2[2]; int z2[2];
        for (int i = 0; i < 16; ++i) cw[i] = 0.5f;
        CHECK(wubu_awq_u4_pack(cw, 16, 8, codes, scales, zeros) == 2);
        CHECK(scales[0] > 0.0f && zeros[0] == 0);
        CHECK(wubu_awq_u4_unpack(codes, scales, zeros, 16, 8, cb) == 0);
        for (int i = 0; i < 16; ++i) CHECK(cb[i] == 0.5f);
        CHECK(wubu_awq_u4_pack(w, 16, 8, codes, scales, zeros) == 2);
        for (int i = 0; i < 16; ++i) w2[i] = w[i];
        w2[12] = 0.75f;
        CHECK(wubu_awq_u4_pack(w2, 16, 8, c2, s2, z2) == 2);
        for (int i = 0; i < 8; ++i) {
            CHECK(c2[i] == codes[i]);
            CHECK(c2[i] <= 15);
        }
        printf("EG-84 constant group exact; group 0 codes unchanged\n");
        CHECK(wubu_awq_u4_pack(NULL, 16, 8, c2, s2, z2) == -1);
        CHECK(wubu_awq_u4_pack(w, 16, 0, c2, s2, z2) == -1);
        CHECK(wubu_awq_u4_pack(w, 16, 17, c2, s2, z2) == -1);
        CHECK(wubu_awq_u4_unpack(NULL, s2, z2, 16, 8, cb) == -1);
    }

    if (failures) { fprintf(stderr, "test_eg84_awq: FAIL (%d)\n", failures); return 1; }
    printf("test_eg84_awq: PASS\n");
    return 0;
}
