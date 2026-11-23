/* test_eg98_batch_h2d.c -- EG-98 gate: batched expert H2D uploads. */
#include <stdio.h>
#include <string.h>

#include "wubu_batch_h2d.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define N_BLOBS 17
#define BLOB_BYTES 16

int main(void) {
    unsigned char src[N_BLOBS][BLOB_BYTES], dev0[N_BLOBS * BLOB_BYTES];
    unsigned char dev1[N_BLOBS * BLOB_BYTES];
    wubu_batch_h2d_t *a = wubu_batch_h2d_create(0, N_BLOBS);
    wubu_batch_h2d_t *b = wubu_batch_h2d_create(1, N_BLOBS);
    CHECK(a && b);

    for (int i = 0; i < N_BLOBS; ++i) {
        for (int k = 0; k < BLOB_BYTES; ++k) src[i][k] = (unsigned char)(i * 16 + k);
        CHECK(wubu_batch_h2d_add(a, src[i], (size_t)i * BLOB_BYTES, BLOB_BYTES, 1) == 0);
        CHECK(wubu_batch_h2d_add(b, src[i], (size_t)i * BLOB_BYTES, BLOB_BYTES, 1) == 0);
    }
    memset(dev0, 0, sizeof(dev0));
    memset(dev1, 0, sizeof(dev1));
    CHECK(wubu_batch_h2d_issue(a, dev0, sizeof(dev0)) == N_BLOBS);
    CHECK(wubu_batch_h2d_issue(b, dev1, sizeof(dev1)) == 1);
    CHECK(memcmp(dev0, dev1, sizeof(dev0)) == 0);
    printf("EG-98 blobs=%d mode0_submissions=%d mode1_submissions=%d bytes_equal=1\n",
           N_BLOBS, wubu_batch_h2d_groups(a), wubu_batch_h2d_groups(b));

    /* Groups are capped at 128; a submission failure commits nothing. */
    {
        unsigned char many_src[129][4], many_dev[129 * 4], zero[129 * 4] = {0};
        wubu_batch_h2d_t *c = wubu_batch_h2d_create(1, 129);
        wubu_batch_h2d_t *f = wubu_batch_h2d_create(2, 2);
        CHECK(c && f);
        for (int i = 0; i < 129; ++i) {
            many_src[i][0] = (unsigned char)i;
            many_src[i][1] = (unsigned char)(i >> 8);
            many_src[i][2] = 0;
            many_src[i][3] = 0;
            CHECK(wubu_batch_h2d_add(c, many_src[i], (size_t)i * 4, 4, 1) == 0);
        }
        CHECK(wubu_batch_h2d_issue(c, many_dev, sizeof(many_dev)) == 2);
        CHECK(memcmp(many_dev, zero, sizeof(zero)) != 0);
        printf("EG-98 blobs=129 mode1_submissions=2 (cap 128)\n");
        CHECK(wubu_batch_h2d_add(f, src[0], 0, BLOB_BYTES, 1) == 0);
        CHECK(wubu_batch_h2d_add(f, src[1], BLOB_BYTES, BLOB_BYTES, 1) == 0);
        CHECK(wubu_batch_h2d_set_fail_at(f, 0) == 0);
        CHECK(wubu_batch_h2d_issue(f, dev1, sizeof(dev1)) == -1);
        CHECK(memcmp(dev1, dev0, sizeof(dev0)) == 0); /* no stale/partial weights */
        CHECK(wubu_batch_h2d_set_fail_at(f, -1) == 0);
        CHECK(wubu_batch_h2d_issue(f, dev1, sizeof(dev1)) == 1);
        wubu_batch_h2d_free(c);
        wubu_batch_h2d_free(f);
    }

    /* Overlapping destinations and unpinned sources are rejected. */
    {
        wubu_batch_h2d_t *q = wubu_batch_h2d_create(1, 2);
        unsigned char probe[32] = {0}, zero_probe[32] = {0};
        wubu_batch_h2d_t *u = wubu_batch_h2d_create(1, 1);
        CHECK(q && u);
        CHECK(wubu_batch_h2d_add(q, src[0], 0, 16, 1) == 0);
        CHECK(wubu_batch_h2d_add(q, src[1], 8, 16, 1) == 0);
        CHECK(wubu_batch_h2d_issue(q, probe, sizeof(probe)) == -1);
        CHECK(memcmp(probe, zero_probe, sizeof(probe)) == 0);
        CHECK(wubu_batch_h2d_add(u, src[0], 0, 16, 0) == 0); /* not pinned */
        CHECK(wubu_batch_h2d_issue(u, probe, sizeof(probe)) == -1);
        CHECK(wubu_batch_h2d_create(3, 2) == NULL);
        wubu_batch_h2d_free(q);
        wubu_batch_h2d_free(u);
    }

    wubu_batch_h2d_free(a);
    wubu_batch_h2d_free(b);
    if (failures) { fprintf(stderr, "test_eg98_batch_h2d: FAIL (%d)\n", failures); return 1; }
    printf("test_eg98_batch_h2d: PASS\n");
    return 0;
}
