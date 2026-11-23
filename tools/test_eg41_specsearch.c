/* test_eg41_specsearch.c -- EG-41 gate: memory-aware config search. */
#include <stdio.h>

#include "wubu_spec_search.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define MiB(x) ((size_t)(x) * 1024u * 1024u)

int main(void) {
    wubu_spec_candidate_t c[4] = {
        { 1, 2, 1, MiB(100), 0.7f },
        { 2, 4, 1, MiB(100), 0.8f },
        { 3, 6, 2, MiB(300), 0.9f },
        { 4, 8, 4, MiB(600), 0.95f }
    };
    wubu_spec_choice_t choice;

    CHECK(wubu_spec_search(c, 4, 0.1, 1.0, MiB(400), &choice) == 0);
    printf("EG-41 400MiB best=%d feasible=%d rate=%.3f\n",
           choice.best_id, choice.n_feasible, choice.throughput);
    CHECK(choice.n_feasible == 3 && choice.best_id == 3);
    CHECK(choice.throughput > 2.5);

    CHECK(wubu_spec_search(c, 4, 0.1, 1.0, MiB(150), &choice) == 0);
    printf("EG-41 150MiB best=%d feasible=%d rate=%.3f\n",
           choice.best_id, choice.n_feasible, choice.throughput);
    CHECK(choice.n_feasible == 2 && choice.best_id == 2);

    CHECK(wubu_spec_search(c, 4, 0.1, 1.0, MiB(50), &choice) == 0);
    CHECK(choice.n_feasible == 0 && choice.best_id == -1);

    CHECK(wubu_spec_search(NULL, 4, 0.1, 1.0, MiB(400), &choice) == -1);
    CHECK(wubu_spec_search(c, 0, 0.1, 1.0, MiB(400), &choice) == -1);
    CHECK(wubu_spec_search(c, 4, 0.1, 0.0, MiB(400), &choice) == -1);
    CHECK(wubu_spec_search(c, 4, 0.1, 1.0, MiB(400), NULL) == -1);

    if (failures) {
        fprintf(stderr, "test_eg41_specsearch: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg41_specsearch: PASS\n");
    return 0;
}
