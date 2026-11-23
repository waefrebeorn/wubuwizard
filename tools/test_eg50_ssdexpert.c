/* test_eg50_ssdexpert.c -- EG-50 gate: concurrent SSD reads + async H2D. */
#include <stdio.h>
#include <string.h>

#include "wubu_ssd_expert.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_ssd_task_t tasks[8];
    wubu_ssd_result_t out[8];
    int overlap, ticks, seq = 0;

    for (int i = 0; i < 8; i++) {
        tasks[i] = (wubu_ssd_task_t){ i + 100, 4, 2, i & 3 };
        seq += tasks[i].read_ticks + tasks[i].h2d_ticks;
    }
    memset(out, 0, sizeof(out));
    CHECK(wubu_ssd_expert_run(tasks, 8, 4, 2, out, 8, &overlap, &ticks) == 0);
    printf("EG-50 sequential=%d ticks concurrent=%d ticks max_overlap=%d\n",
           seq, ticks, overlap);
    CHECK(overlap >= 5);
    CHECK(ticks < seq / 2);

    for (int i = 0; i < 8; i++) {
        int found = 0;
        for (int j = 0; j < 8; j++)
            if (out[j].expert == tasks[i].expert) {
                found = 1;
                CHECK(out[j].decoded == tasks[i].sym);
                CHECK(out[j].done_tick >= 0 && out[j].done_tick < ticks);
            }
        CHECK(found);
    }
    printf("EG-50 decoded all 8 expert payloads in VRAM-local path\n");

    CHECK(wubu_ssd_expert_run(tasks, 8, 4, 2, out, 7, &overlap, &ticks) == -1);
    CHECK(wubu_ssd_expert_run(tasks, 8, 0, 2, out, 8, &overlap, &ticks) == -1);
    CHECK(wubu_ssd_expert_run(tasks, 0, 4, 2, out, 8, &overlap, &ticks) == -1);
    {
        wubu_ssd_task_t bad = tasks[0];
        bad.sym = 4;
        CHECK(wubu_ssd_expert_run(&bad, 1, 1, 1, out, 1, &overlap, &ticks) == -1);
    }

    if (failures) {
        fprintf(stderr, "test_eg50_ssdexpert: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg50_ssdexpert: PASS\n");
    return 0;
}
