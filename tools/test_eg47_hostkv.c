/* test_eg47_hostkv.c -- EG-47 gate: 1152-byte INT8 host KV record. */
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "wubu_hostkv.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define HEADS 8
#define DIM 128
#define NELEM (HEADS * DIM)

int main(void) {
    float row[NELEM], out[NELEM];
    unsigned char rec[WUBU_HOSTKV_RECORD_BYTES];
    unsigned char moved[WUBU_HOSTKV_RECORD_BYTES];
    int used, i;
    double sumerr = 0.0, sumabs = 0.0, maxerr = 0.0;

    for (int h = 0; h < HEADS; h++)
        for (int d = 0; d < DIM; d++) {
            float amp = 0.05f * (float)(h + 1);
            row[h * DIM + d] = amp * sinf(0.13f * (float)d + 0.7f * (float)h);
        }

    used = wubu_hostkv_pack(row, HEADS, DIM, rec, sizeof(rec));
    printf("EG-47 used=%d record=%u raw_f32=%zu save=%.4f\n",
           used, WUBU_HOSTKV_RECORD_BYTES, (size_t)NELEM * 4u,
           1.0 - (double)used / ((double)NELEM * 4.0));
    CHECK(used == 1104);
    CHECK(used <= (int)WUBU_HOSTKV_RECORD_BYTES);
    CHECK((size_t)used == WUBU_HOSTKV_HEADER_BYTES + HEADS * 2u + NELEM);

    CHECK(wubu_hostkv_unpack(rec, (size_t)used, out, HEADS, DIM) == 0);
    for (i = 0; i < NELEM; i++) {
        float e = fabsf(out[i] - row[i]);
        if (e > maxerr) maxerr = e;
        sumerr += e;
        sumabs += fabsf(row[i]);
    }
    printf("EG-47 maxerr=%.5f mean_relerr=%.5f\n",
           maxerr, sumerr / sumabs);
    CHECK(maxerr < 0.03f);
    CHECK(sumerr / sumabs < 0.02);

    CHECK(wubu_hostkv_mover_copy(rec, moved, sizeof(rec)) == 0);
    CHECK(memcmp(rec, moved, sizeof(rec)) == 0);
    printf("EG-47 mover opaque copy 1152 bytes identical\n");

    CHECK(wubu_hostkv_pack(row, 9, DIM, rec, sizeof(rec)) == -1);
    CHECK(wubu_hostkv_pack(row, HEADS, DIM, rec, 1024) == -1);
    CHECK(wubu_hostkv_unpack(rec, (size_t)used, NULL, HEADS, DIM) == -1);
    rec[0] ^= 0xff;
    CHECK(wubu_hostkv_unpack(rec, (size_t)used, out, HEADS, DIM) == -1);

    if (failures) {
        fprintf(stderr, "test_eg47_hostkv: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg47_hostkv: PASS\n");
    return 0;
}
