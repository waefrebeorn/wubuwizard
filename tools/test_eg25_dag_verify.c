/* test_eg25_dag_verify.c -- EG-25c gate: DAG verification of shared prefixes. */
#include <stdio.h>

#include "wubu_dag_verify.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_dag_verify_t *p = wubu_dag_verify_create(16, 8);
    int a[3] = {1, 2, 3}, b[3] = {1, 2, 4};
    int order[16];
    int n, saved;
    CHECK(p != NULL);

    CHECK(wubu_dag_verify_add_path(p, a, 3) == 0);
    CHECK(wubu_dag_verify_add_path(p, b, 3) == 0);
    CHECK(wubu_dag_verify_node_count(p) == 5);   /* root + 1,2,3,4 */

    n = wubu_dag_verify_plan(p, order, 16);
    CHECK(n == 5);
    CHECK(order[0] == -1);                       /* synthetic root */
    CHECK(order[1] == 1 && order[2] == 2);
    CHECK(order[3] == 3 && order[4] == 4);

    saved = wubu_dag_verify_checks_saved(p, 3 + 3);
    CHECK(saved == 2);                           /* shared 2-token prefix */
    printf("EG-25c nodes=5 plan=%d checks_saved=%d of 6\n", n, saved);

    /* Disjoint paths share no verification work; duplicate paths add no nodes. */
    {
        wubu_dag_verify_t *q = wubu_dag_verify_create(16, 4);
        int c[3] = {5, 6, 7}, d[3] = {8, 9, 10};
        CHECK(q != NULL);
        CHECK(wubu_dag_verify_add_path(q, c, 3) == 0);
        CHECK(wubu_dag_verify_add_path(q, d, 3) == 0);
        CHECK(wubu_dag_verify_node_count(q) == 7);
        CHECK(wubu_dag_verify_add_path(q, c, 3) == 0);
        CHECK(wubu_dag_verify_node_count(q) == 7);
        CHECK(wubu_dag_verify_path_count(q) == 3);
        CHECK(wubu_dag_verify_checks_saved(q, 6) == 0);
        CHECK(wubu_dag_verify_plan(q, order, 6) == -1);
        printf("EG-25c disjoint paths save 0 checks; duplicate adds no nodes\n");
        CHECK(wubu_dag_verify_create(1, 4) == NULL);
        CHECK(wubu_dag_verify_add_path(q, NULL, 3) == -1);
        CHECK(wubu_dag_verify_add_path(q, c, 0) == -1);
        CHECK(wubu_dag_verify_node_count(NULL) == -1);
        wubu_dag_verify_free(q);
    }

    wubu_dag_verify_free(p);
    if (failures) { fprintf(stderr, "test_eg25_dag_verify: FAIL (%d)\n", failures); return 1; }
    printf("test_eg25_dag_verify: PASS\n");
    return 0;
}
