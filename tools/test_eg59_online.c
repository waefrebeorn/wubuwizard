/* test_eg59_online.c -- EG-59 gate: acceptance drift under online updates. */
#include <stdio.h>

#include "wubu_online_sd.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define STEPS 4000
#define V 8

int main(void) {
    float target[V];
    wubu_online_t online, control;
    int i, o_first = 0, o_last = 0, c_first = 0, c_last = 0;

    {
        float sum = 0.0f;
        for (i = 0; i < V; i++) { target[i] = 1.0f / (float)(i + 1); sum += target[i]; }
        for (i = 0; i < V; i++) target[i] /= sum;
    }
    CHECK(wubu_online_init(&online, V, 16, 0.5f, 12345ULL) == 0);
    CHECK(wubu_online_init(&control, V, 1000000, 0.0f, 12345ULL) == 0);
    for (i = 0; i < STEPS; i++) {
        int a = wubu_online_step(&online, target);
        int b = wubu_online_step(&control, target);
        CHECK(a == 0 || a == 1);
        CHECK(b == 0 || b == 1);
        if (i < STEPS / 4) { o_first += a; c_first += b; }
        if (i >= 3 * STEPS / 4) { o_last += a; c_last += b; }
    }
    printf("EG-59 acceptance online first=%.4f last=%.4f | static first=%.4f last=%.4f\n",
           (double)o_first / (STEPS / 4), (double)o_last / (STEPS / 4),
           (double)c_first / (STEPS / 4), (double)c_last / (STEPS / 4));
    CHECK((double)o_last / (STEPS / 4) > (double)o_first / (STEPS / 4) + 0.05);
    CHECK((double)o_last / (STEPS / 4) > (double)c_last / (STEPS / 4) + 0.05);

    CHECK(wubu_online_init(&online, 0, 16, 0.5f, 1ULL) == -1);
    CHECK(wubu_online_step(NULL, target) == -1);
    wubu_online_free(&online);
    wubu_online_free(&control);

    if (failures) {
        fprintf(stderr, "test_eg59_online: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg59_online: PASS\n");
    return 0;
}
