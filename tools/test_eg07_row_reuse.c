/* test_eg07_row_reuse.c -- EG-07 gate: sliding-window weight-row reuse and
 * coalesced reads. Checks data fidelity, read/byte counts against a naive
 * full-reread baseline, zero-I/O same-window access and jump refills.
 * Prints: test_eg07_row_reuse: PASS. */
#include <stdio.h>
#include <string.h>

#include "wubu_ssd_row_reuse.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

enum { ROWS = 64, ROW_BYTES = 4096, WINDOW = 8, COALESCE = 32768 };

static void fill(uint8_t *data) {
    for (int i = 0; i < ROWS; i++)
        memset(data + (size_t)i * ROW_BYTES, (unsigned char)(i & 0xFF), ROW_BYTES);
}

static int window_ok(const uint8_t *win, const uint8_t *data, int start) {
    for (int r = 0; r < WINDOW; r++) {
        const uint8_t *expect = data + (size_t)(start + r) * ROW_BYTES;
        const uint8_t *got = win + (size_t)r * ROW_BYTES;
        if (memcmp(expect, got, ROW_BYTES) != 0) return 0;
    }
    return 1;
}

int main(void) {
    uint8_t *data = (uint8_t *)malloc((size_t)ROWS * ROW_BYTES);
    CHECK(data != NULL);
    if (!data) return 1;
    fill(data);

    wubu_ssd_row_reader_t *r = wubu_ssd_row_reader_create(
        data, (size_t)ROWS * ROW_BYTES, ROW_BYTES, WINDOW, COALESCE);
    CHECK(r != NULL);
    if (!r) return 1;

    const uint8_t *win = wubu_ssd_row_window(r, 0);
    CHECK(win != NULL);
    CHECK(window_ok(win, data, 0));
    CHECK(wubu_ssd_row_reader_reads(r) == 1);      /* 8 rows in one 32 KiB read */
    CHECK(wubu_ssd_row_reader_bytes(r) == COALESCE);
    printf("EG-07 first window: reads=%zu bytes=%llu\n",
           wubu_ssd_row_reader_reads(r),
           (unsigned long long)wubu_ssd_row_reader_bytes(r));

    /* Same window is free. */
    CHECK(wubu_ssd_row_window(r, 0) == win);
    CHECK(wubu_ssd_row_reader_reads(r) == 1);

    /* Slide seven times: one entering row per step, data stays correct. */
    for (int s = 1; s < WINDOW; s++) {
        win = wubu_ssd_row_window(r, s);
        CHECK(win != NULL);
        CHECK(window_ok(win, data, s));
    }
    size_t reads = wubu_ssd_row_reader_reads(r);
    uint64_t bytes = wubu_ssd_row_reader_bytes(r);
    size_t naive_reads = (size_t)WINDOW * WINDOW;                 /* 64 */
    uint64_t naive_bytes = (uint64_t)WINDOW * WINDOW * ROW_BYTES; /* 256 KiB */
    printf("EG-07 slid window: reads=%zu bytes=%llu deltas=%d "
           "(naive %zu reads / %llu bytes)\n", reads,
           (unsigned long long)bytes, wubu_ssd_row_reader_delta_updates(r),
           naive_reads, (unsigned long long)naive_bytes);
    CHECK(reads == 8);
    CHECK(bytes == 61440);
    CHECK(wubu_ssd_row_reader_delta_updates(r) == 7);
    CHECK(reads <= naive_reads / 8);
    CHECK(bytes <= naive_bytes / 4);

    /* A jump cannot reuse the window: full coalesced refill. */
    win = wubu_ssd_row_window(r, 20);
    CHECK(win != NULL);
    CHECK(window_ok(win, data, 20));
    CHECK(wubu_ssd_row_reader_reads(r) == 9);

    /* Bounds and construction fail closed. */
    CHECK(wubu_ssd_row_window(r, -1) == NULL);
    CHECK(wubu_ssd_row_window(r, ROWS - WINDOW + 1) == NULL);
    CHECK(wubu_ssd_row_reader_create(NULL, 1024, 64, 2, 128) == NULL);
    CHECK(wubu_ssd_row_reader_create(data, 1024, 0, 2, 128) == NULL);
    CHECK(wubu_ssd_row_reader_create(data, 1024, 64, 0, 128) == NULL);

    /* Coalescing: a 16-row window is two 32 KiB reads on full refill. */
    wubu_ssd_row_reader_t *r2 = wubu_ssd_row_reader_create(
        data, (size_t)ROWS * ROW_BYTES, ROW_BYTES, 16, COALESCE);
    CHECK(r2 != NULL);
    if (r2) {
        CHECK(wubu_ssd_row_window(r2, 0) != NULL);
        CHECK(wubu_ssd_row_reader_reads(r2) == 2);
        printf("EG-07 16-row refill: reads=%zu\n",
               wubu_ssd_row_reader_reads(r2));
    }

    wubu_ssd_row_reader_free(r);
    wubu_ssd_row_reader_free(r2);
    free(data);
    if (failures) {
        fprintf(stderr, "test_eg07_row_reuse: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg07_row_reuse: PASS\n");
    return 0;
}
