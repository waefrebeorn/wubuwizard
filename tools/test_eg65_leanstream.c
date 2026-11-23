/* test_eg65_leanstream.c -- EG-65 gate: memory cut with output parity. */
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "wubu_leanstream.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define N 64
#define D 8

int main(void) {
    float input[N*D], full[N*D], stream[N*D];
    size_t peak_full, peak_stream;
    double maxerr = 0.0;

    for (int i = 0; i < N; i++)
        for (int d = 0; d < D; d++)
            input[i*D+d] = 0.3f * cosf(0.11f * (float)i + 0.5f * (float)d);

    CHECK(wubu_leanstream_full(input, N, D, full, &peak_full) == 0);
    CHECK(wubu_leanstream_stream(input, N, D, 8, stream, &peak_stream) == 0);
    for (int i = 0; i < N*D; i++) {
        double e = fabs(full[i] - stream[i]);
        if (e > maxerr) maxerr = e;
    }
    printf("EG-65 peak full=%zu stream=%zu reduction=%.3fx maxerr=%.8f\n",
           peak_full, peak_stream, (double)peak_full / peak_stream, maxerr);
    CHECK(peak_full == (size_t)N * D * sizeof(float));
    CHECK(peak_stream < peak_full / 4);
    CHECK(maxerr < 1e-6f);
    CHECK(memcmp(full, stream, sizeof(full)) == 0);

    CHECK(wubu_leanstream_stream(input, N, D, 0, stream, &peak_stream) == -1);
    CHECK(wubu_leanstream_full(NULL, N, D, full, &peak_full) == -1);

    if (failures) {
        fprintf(stderr, "test_eg65_leanstream: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg65_leanstream: PASS\n");
    return 0;
}
