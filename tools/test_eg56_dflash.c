/* test_eg56_dflash.c -- EG-56 gate: fused hidden injection acceptance gain. */
#include <stdio.h>
#include <math.h>

#include "wubu_dflash.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define NT 8
#define D 4

int main(void) {
    float ht[NT*D], hd[NT*D], u[NT], pf[NT], pb[NT];
    int af, ab;

    for (int t = 0; t < NT; t++) {
        u[t] = 0.5f;
        for (int d = 0; d < D; d++) {
            ht[t*D+d] = cosf(0.4f * (float)t + 0.7f * (float)d);
            hd[t*D+d] = 0.08f * sinf(1.3f * (float)(t * 7 + d));
        }
    }
    af = wubu_dflash_block(ht, hd, NT, D, 2, 0.5f, u, pf);
    ab = wubu_dflash_block(ht, hd, NT, D, 2, 0.0f, u, pb);
    printf("EG-56 fused_p0=%.4f plain_p0=%.4f accepted fused=%d/%d plain=%d/%d\n",
           pf[0], pb[0], af, NT, ab, NT);
    CHECK(af >= 6);
    CHECK(ab <= 2);
    CHECK(af > ab * 2);

    CHECK(wubu_dflash_block(NULL, hd, NT, D, 2, 0.5f, u, pf) == -1);
    CHECK(wubu_dflash_block(ht, hd, NT, D, 0, 0.5f, u, pf) == -1);
    CHECK(wubu_dflash_block(ht, hd, NT, D, 2, -1.0f, u, pf) == -1);

    if (failures) {
        fprintf(stderr, "test_eg56_dflash: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg56_dflash: PASS\n");
    return 0;
}
