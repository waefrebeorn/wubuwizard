/* test_eg32_compact_session.c -- EG-32b gate: long-lived compact session. */
#include <stdio.h>

#include "wubu_compact_session.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_compact_session_t *p = wubu_compact_session_create(4, 1);
    int slots[4] = {-1, -1, -1, -1};
    int compact = -1, reason = -1, n;
    CHECK(p != NULL);
    CHECK(wubu_compact_enqueue(p, 1, WUBU_COMPACT_PROMPT, 4) == 0);
    CHECK(wubu_compact_enqueue(p, 2, WUBU_COMPACT_DECODE, 1) == 0);
    CHECK(wubu_compact_enqueue(p, 3, WUBU_COMPACT_PROMPT, 2) == 0);

    n = wubu_compact_step(p, slots, 4);
    CHECK(n == 3);
    CHECK(slots[0] == 1 && slots[1] == 3 && slots[2] == 2);
    CHECK(slots[3] == -1);                       /* no padded holes */
    CHECK(wubu_compact_resident_count(p) == 3);
    wubu_compact_capability(p, &compact, &reason);
    CHECK(compact == 1 && reason == 0);
    printf("EG-32b step=%d packed=[%d %d %d] empty=%d compact=1 reason=0\n",
           n, slots[0], slots[1], slots[2], slots[3]);

    /* A queued request is admitted into the remaining free slot. */
    CHECK(wubu_compact_enqueue(p, 4, WUBU_COMPACT_DECODE, 1) == 0);
    CHECK(wubu_compact_step(p, slots, 4) == 4);
    CHECK(slots[0] == 1 && slots[1] == 3 && slots[2] == 2 && slots[3] == 4);
    CHECK(wubu_compact_resident_count(p) == 4);
    CHECK(wubu_compact_pending_count(p) == 0);

    /* Disabled compact steps report a capability reason; a full session
       keeps extra commands pending instead of dropping or padding them. */
    {
        wubu_compact_session_t *q = wubu_compact_session_create(2, 0);
        wubu_compact_session_t *r = wubu_compact_session_create(2, 1);
        int slots2[2] = {-1, -1}, compact2 = -1, reason2 = -1;
        CHECK(q != NULL && r != NULL);
        wubu_compact_capability(q, &compact2, &reason2);
        CHECK(compact2 == 0 && reason2 == 1);
        CHECK(wubu_compact_step(q, slots2, 2) == -1);
        CHECK(wubu_compact_enqueue(r, 1, WUBU_COMPACT_PROMPT, 2) == 0);
        CHECK(wubu_compact_enqueue(r, 2, WUBU_COMPACT_DECODE, 1) == 0);
        CHECK(wubu_compact_enqueue(r, 3, WUBU_COMPACT_DECODE, 1) == 0);
        CHECK(wubu_compact_step(r, slots2, 2) == 2);
        CHECK(wubu_compact_pending_count(r) == 1);
        CHECK(wubu_compact_step(r, slots2, 2) == 2);
        CHECK(wubu_compact_pending_count(r) == 1);
        printf("EG-32b disabled reason=1; full session holds 1 pending\n");
        CHECK(wubu_compact_enqueue(r, 1, WUBU_COMPACT_DECODE, 1) == -1);
        CHECK(wubu_compact_enqueue(r, 9, 99, 1) == -1);
        CHECK(wubu_compact_enqueue(r, 9, WUBU_COMPACT_PROMPT, 0) == -1);
        CHECK(wubu_compact_session_create(0, 1) == NULL);
        CHECK(wubu_compact_step(NULL, slots2, 2) == -1);
        CHECK(wubu_compact_resident_count(NULL) == -1);
        wubu_compact_session_free(q);
        wubu_compact_session_free(r);
    }

    wubu_compact_session_free(p);
    if (failures) { fprintf(stderr, "test_eg32_compact_session: FAIL (%d)\n", failures); return 1; }
    printf("test_eg32_compact_session: PASS\n");
    return 0;
}
