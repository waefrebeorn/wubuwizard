/* test_eg45_varbit_vq.c -- EG-45b gate: variable-bit packed VQ codes. */
#include <stdio.h>
#include <math.h>

#include "wubu_varbit_vq.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float v[3] = {0.9f, -0.5f, 0.2f}, back[3];
    int bits[3] = {4, 2, 8};
    unsigned char packed[4] = {0, 0, 0, 0};
    size_t bytes = wubu_varbit_vq_encoded_bytes(bits, 3);
    double maxerr = 0.0;
    CHECK(bytes == 2);
    CHECK(wubu_varbit_vq_encode(v, bits, 3, packed) == 0);
    CHECK(wubu_varbit_vq_decode(packed, bits, 3, back) == 0);
    for (int i = 0; i < 3; ++i) {
        double e = fabs(back[i] - v[i]);
        if (e > maxerr) maxerr = e;
    }
    CHECK(maxerr < 0.20);
    printf("EG-45b bits=[4 2 8] packed=%zu bytes maxerr=%.5f\n", bytes, maxerr);

    CHECK(wubu_varbit_vq_encoded_bytes(NULL, 3) == 0);
    CHECK(wubu_varbit_vq_encode(NULL, bits, 3, packed) == -1);
    {
        int bad[1] = {9};
        CHECK(wubu_varbit_vq_encoded_bytes(bad, 1) == 0);
        CHECK(wubu_varbit_vq_encode(v, bad, 1, packed) == -1);
    }
    CHECK(wubu_varbit_vq_decode(packed, bits, 0, back) == -1);
    {
        float wide[2] = {0.5f, -0.5f}, back2[2];
        int wb[2] = {8, 8};
        unsigned char p2[2] = {0, 0};
        float oob[1] = {1.5f};
        int one[1] = {1};
        CHECK(wubu_varbit_vq_encoded_bytes(wb, 2) == 2);
        CHECK(wubu_varbit_vq_encode(wide, wb, 2, p2) == 0);
        CHECK(wubu_varbit_vq_decode(p2, wb, 2, back2) == 0);
        CHECK(fabs(back2[0] - 0.5f) < 0.01 && fabs(back2[1] + 0.5f) < 0.01);
        CHECK(wubu_varbit_vq_encode(oob, one, 1, p2) == -1);
        CHECK(wubu_varbit_vq_encoded_bytes(bits, 0) == 0);
        printf("EG-45b 8-bit pair error <0.01; out-of-range value rejected\n");
    }

    if (failures) { fprintf(stderr, "test_eg45_varbit_vq: FAIL (%d)\n", failures); return 1; }
    printf("test_eg45_varbit_vq: PASS\n");
    return 0;
}
