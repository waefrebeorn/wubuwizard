/* test_eg102_mx8.c -- EG-102 gate: MX8 block microscaling. */
#include <stdio.h>
#include <math.h>

#include "wubu_mx8.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float x[64], out[64], noisy[64];
    wubu_mx8_block_t blocks[2];
    int nb = 0, i;
    float err = 0.0f;

    for (i = 0; i < 64; i++) {
        int code = (i % 16) - 8;
        x[i] = (float)code * 0.125f; /* exact multiples of 2^-3 */
    }
    CHECK(wubu_mx8_encode(x, 64, blocks, &nb) == 0);
    CHECK(nb == 2);
    CHECK(wubu_mx8_decode(blocks, nb, 64, out) == 0);
    for (i = 0; i < 64; i++) {
        float e = fabsf(x[i] - out[i]);
        if (e > err) err = e;
    }
    printf("EG-102 exact block err=%.6f exp0=%d exp1=%d\n",
           err, blocks[0].exponent, blocks[1].exponent);
    CHECK(err == 0.0f);

    /* Non-multiple values are lossy but within half a scale step. */
    for (i = 0; i < 64; i++) noisy[i] = x[i] + 0.03f;
    CHECK(wubu_mx8_encode(noisy, 64, blocks, &nb) == 0);
    CHECK(wubu_mx8_decode(blocks, nb, 64, out) == 0);
    err = 0.0f;
    for (i = 0; i < 64; i++) {
        float e = fabsf(noisy[i] - out[i]);
        if (e > err) err = e;
    }
    printf("EG-102 noisy maxerr=%.6f\n", err);
    CHECK(err <= 0.0625f + 1e-6f);

    /* Tail padding: 33 values -> 2 blocks, decode stops at n. */
    CHECK(wubu_mx8_encode(x, 33, blocks, &nb) == 0);
    CHECK(nb == 2);
    CHECK(wubu_mx8_decode(blocks, nb, 33, out) == 0);

    CHECK(wubu_mx8_encode(NULL, 64, blocks, &nb) == -1);
    CHECK(wubu_mx8_encode(x, 0, blocks, &nb) == -1);
    CHECK(wubu_mx8_decode(NULL, 2, 64, out) == -1);
    CHECK(wubu_mx8_decode(blocks, 0, 64, out) == -1);

    if (failures) {
        fprintf(stderr, "test_eg102_mx8: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg102_mx8: PASS\n");
    return 0;
}
