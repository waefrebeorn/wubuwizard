/* test_eg40_invalidkv.c -- EG-40 gate: invalidated-KV draft verification. */
#include <stdio.h>
#include <math.h>

#include "wubu_invalid_kv.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    int draft[4] = { 1, 2, 3, 4 };
    int fresh[4] = { 1, 2, 9, 4 };
    int same[4] = { 1, 2, 3, 4 };
    int accepted, preempt;
    float rate;

    CHECK(wubu_invalid_kv_verify(draft, fresh, 4, 0.5f,
                                 &accepted, &rate, &preempt) == 0);
    printf("EG-40 edited accept=%d rate=%.2f preempt=%d\n",
           accepted, rate, preempt);
    CHECK(accepted == 2 && fabsf(rate - 0.5f) < 1e-6f && preempt == 0);

    CHECK(wubu_invalid_kv_verify(draft, fresh, 4, 0.75f,
                                 &accepted, &rate, &preempt) == 0);
    printf("EG-40 preempt accept=%d rate=%.2f preempt=%d\n",
           accepted, rate, preempt);
    CHECK(accepted == 0 && preempt == 1);

    CHECK(wubu_invalid_kv_verify(same, same, 4, 0.75f,
                                 &accepted, &rate, &preempt) == 0);
    printf("EG-40 identical accept=%d rate=%.2f preempt=%d\n",
           accepted, rate, preempt);
    CHECK(accepted == 4 && rate == 1.0f && preempt == 0);

    CHECK(wubu_invalid_kv_verify(NULL, fresh, 4, 0.5f,
                                 &accepted, &rate, &preempt) == -1);
    CHECK(wubu_invalid_kv_verify(draft, fresh, 0, 0.5f,
                                 &accepted, &rate, &preempt) == -1);
    CHECK(wubu_invalid_kv_verify(draft, fresh, 4, 1.5f,
                                 &accepted, &rate, &preempt) == -1);
    CHECK(wubu_invalid_kv_verify(draft, fresh, 4, 0.5f,
                                 NULL, &rate, &preempt) == -1);

    if (failures) {
        fprintf(stderr, "test_eg40_invalidkv: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg40_invalidkv: PASS\n");
    return 0;
}
