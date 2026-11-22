/* test_eg10_admit.c -- EG-10 gate: one shared admission validator with a
 * token budget, queue cap and timeout expiry. Prints: test_eg10_admit: PASS. */
#include <stdio.h>

#include "wubu_admission.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_admission_t *a = wubu_admission_create(100, 3, 2);
    CHECK(a != NULL);
    if (!a) return 1;

    CHECK(wubu_admission_validate(a, 1, 40, 0) == WUBU_ADMIT_OK);
    CHECK(wubu_admission_validate(a, 2, 40, 0) == WUBU_ADMIT_OK);
    CHECK(wubu_admission_reserved(a) == 80);
    CHECK(wubu_admission_queue_len(a) == 2);
    printf("EG-10 reserved=%llu queue=%d\n",
           (unsigned long long)wubu_admission_reserved(a),
           wubu_admission_queue_len(a));

    CHECK(wubu_admission_validate(a, 3, 30, 0) == WUBU_ADMIT_OVER_BUDGET);
    CHECK(wubu_admission_validate(a, 1, 10, 0) == WUBU_ADMIT_DUPLICATE);
    CHECK(wubu_admission_validate(NULL, 9, 5, 0) == WUBU_ADMIT_ERR_ARG);
    CHECK(wubu_admission_validate(a, 4, 0, 0) == WUBU_ADMIT_ERR_ARG);
    CHECK(wubu_admission_reserved(a) == 80); /* rejections reserve nothing */

    /* Timeout boundary: > timeout expires, == timeout does not. */
    CHECK(wubu_admission_step(a, 2) == 0);
    CHECK(wubu_admission_step(a, 3) == 2);
    CHECK(wubu_admission_reserved(a) == 0);
    CHECK(wubu_admission_queue_len(a) == 0);
    printf("EG-10 expired=%llu reserved=%llu\n",
           (unsigned long long)wubu_admission_expired(a),
           (unsigned long long)wubu_admission_reserved(a));

    /* Queue cap is enforced after timeout frees the slots. */
    CHECK(wubu_admission_validate(a, 5, 60, 3) == WUBU_ADMIT_OK);
    CHECK(wubu_admission_validate(a, 6, 20, 3) == WUBU_ADMIT_OK);
    CHECK(wubu_admission_validate(a, 7, 1, 3) == WUBU_ADMIT_OK);
    CHECK(wubu_admission_validate(a, 8, 1, 3) == WUBU_ADMIT_QUEUE_FULL);
    CHECK(wubu_admission_queue_len(a) == 3);
    CHECK(wubu_admission_release(a, 5) == 0);
    CHECK(wubu_admission_release(a, 5) == -1);
    CHECK(wubu_admission_reserved(a) == 21);

    printf("EG-10 accepted=%llu rejected=%llu expired=%llu\n",
           (unsigned long long)wubu_admission_accepted(a),
           (unsigned long long)wubu_admission_rejected(a),
           (unsigned long long)wubu_admission_expired(a));
    CHECK(wubu_admission_accepted(a) == 5);
    CHECK(wubu_admission_rejected(a) == 3);
    CHECK(wubu_admission_expired(a) == 2);

    CHECK(wubu_admission_create(0, 3, 2) == NULL);
    CHECK(wubu_admission_create(10, 0, 2) == NULL);
    CHECK(wubu_admission_create(10, 3, -1) == NULL);
    CHECK(wubu_admission_release(NULL, 1) == -1);

    wubu_admission_free(a);
    if (failures) {
        fprintf(stderr, "test_eg10_admit: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg10_admit: PASS\n");
    return 0;
}
