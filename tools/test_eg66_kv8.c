/* test_eg66_kv8.c -- EG-66d gate: kv8 INT8 K/V codec. */
#include <stdio.h>
#include <math.h>

#include "wubu_kv8.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float row[8] = {0.25f, -1.5f, 2.0f, 0.0f, -0.125f, 1.0f, -2.0f, 0.5f};
    signed char codes[8];
    float back[8], scale;
    double maxerr = 0.0;
    CHECK(wubu_kv8_pack(row, 8, codes, &scale) == 0);
    CHECK(scale > 0.0f);
    CHECK(wubu_kv8_unpack(codes, scale, 8, back) == 0);
    for (int i = 0; i < 8; ++i) {
        double e = fabs(back[i] - row[i]);
        if (e > maxerr) maxerr = e;
    }
    CHECK(maxerr < 0.05);
    CHECK(wubu_kv8_storage_bytes(8) == 12);  /* 8 codes + 4-byte f32 scale */
    printf("EG-66d kv8 scale=%.6f maxerr=%.6f bytes=12 vs 32 f32\n", scale, maxerr);

    CHECK(wubu_kv8_pack(NULL, 8, codes, &scale) == -1);
    CHECK(wubu_kv8_pack(row, 0, codes, &scale) == -1);
    CHECK(wubu_kv8_unpack(NULL, scale, 8, back) == -1);
    CHECK(wubu_kv8_storage_bytes(0) == 0);

    if (failures) { fprintf(stderr, "test_eg66_kv8: FAIL (%d)\n", failures); return 1; }
    printf("test_eg66_kv8: PASS\n");
    return 0;
}
