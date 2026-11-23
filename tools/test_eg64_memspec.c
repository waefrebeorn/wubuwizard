/* test_eg64_memspec.c -- EG-64 gate: memory-aware backfill vs FIFO stall. */
#include <stdio.h>

#include "wubu_memspec.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_memstep_t steps[6] = {
        { 0, 40, 2 }, { 1, 70, 2 }, { 2, 40, 10 },
        { 3, 40, 2 }, { 4, 70, 2 }, { 5, 40, 10 }
    };
    size_t peak_aware = 0, peak_fifo = 0;
    int aware, fifo;

    aware = wubu_memspec_schedule(steps, 6, 100, 0, &peak_aware);
    fifo = wubu_memspec_schedule(steps, 6, 100, 1, &peak_fifo);
    printf("EG-64 cap=100 ticks aware=%d fifo=%d peak_aware=%zu peak_fifo=%zu\n",
           aware, fifo, peak_aware, peak_fifo);
    CHECK(aware > 0 && fifo > 0);
    CHECK(peak_aware <= 100 && peak_fifo <= 100);
    CHECK(aware < fifo * 4 / 5);

    CHECK(wubu_memspec_schedule(steps, 6, 0, 0, &peak_aware) == -1);
    CHECK(wubu_memspec_schedule(steps, 6, 100, 2, &peak_aware) == -1);
    {
        wubu_memstep_t bad = { 9, 200, 2 };
        CHECK(wubu_memspec_schedule(&bad, 1, 100, 0, &peak_aware) == -1);
    }

    if (failures) {
        fprintf(stderr, "test_eg64_memspec: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg64_memspec: PASS\n");
    return 0;
}
