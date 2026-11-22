/* test_eg01_vecdot.c -- EG-01 gate: runtime per-type vec_dot dispatch and
 * multi-row GEMV. Verifies the table routes every known quant type, refuses
 * unknown types, reproduces a hand-computed Q8_0 dot, and reuses one dispatch
 * across a 4-row batch. Prints: test_eg01_vecdot: PASS. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gguf_reader.h"
#include "wubu_vecdot_dispatch.h"

#define QK8 32
typedef struct { uint16_t d; int8_t qs[QK8]; } q8_block;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

/* fp16 1.0f */
#define FP16_ONE 0x3C00u

static void fill_row(q8_block *row, int nblocks, int seed) {
    srand(seed);
    for (int b = 0; b < nblocks; b++) {
        row[b].d = FP16_ONE;
        for (int j = 0; j < QK8; j++)
            row[b].qs[j] = (int8_t)((rand() % 15) - 7);
    }
}

static float reference_dot(const q8_block *w, const q8_block *x, int nblocks) {
    double acc = 0.0;
    for (int b = 0; b < nblocks; b++)
        for (int j = 0; j < QK8; j++)
            acc += (double)w[b].qs[j] * (double)x[b].qs[j];
    return (float)acc; /* d == 1.0 for both sides */
}

int main(void) {
    enum { NB = 4, NR = 4 };
    int n = NB * QK8;

    /* 1. Table completeness and fail-closed behaviour. */
    int count = wubu_vecdot_table_count();
    printf("EG-01 table entries: %d\n", count);
    CHECK(count >= 7);
    const int types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                          GGML_TYPE_Q8_0, GGML_TYPE_IQ2_XXS,
                          GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS };
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        const wubu_vecdot_entry_t *e = wubu_vecdot_lookup(types[i]);
        CHECK(e != NULL);
        CHECK(e && e->fn != NULL);
        CHECK(wubu_vecdot_supported(types[i]));
        if (e) printf("  type=%2d %-8s -> %s\n", e->ggml_type, e->name,
                      e->fn ? "kernel" : "MISSING");
    }
    CHECK(wubu_vecdot_lookup(9999) == NULL);
    CHECK(wubu_vecdot_supported(9999) == 0);
    float dummy = 0.0f;
    int w[4] = {0};
    CHECK(wubu_vecdot_one(9999, n, w, w, &dummy) == -1);

    /* 2. Q8_0 single-row parity against an independent scalar reference. */
    q8_block *weight = (q8_block *)malloc(sizeof(q8_block) * NB);
    q8_block **rows = (q8_block **)malloc(sizeof(q8_block *) * NR);
    float single[NR];
    CHECK(weight != NULL && rows != NULL);
    if (!weight || !rows) return 1;
    fill_row(weight, NB, 12345);
    for (int r = 0; r < NR; r++) {
        rows[r] = (q8_block *)malloc(sizeof(q8_block) * NB);
        CHECK(rows[r] != NULL);
        if (!rows[r]) return 1;
        fill_row(rows[r], NB, 700 + r);
        CHECK(wubu_vecdot_one(GGML_TYPE_Q8_0, n, weight, rows[r], &single[r]) == 0);
        float ref = reference_dot(weight, rows[r], NB);
        float err = fabsf(single[r] - ref);
        printf("EG-01 row %d: dispatch=%.1f ref=%.1f err=%.3f\n",
               r, single[r], ref, err);
        CHECK(err < 0.001f);
    }

    /* 3. Multi-row path: one dispatch for the whole batch, identical outputs. */
    float multi[NR];
    int lookups = 0;
    CHECK(wubu_vecdot_rows(GGML_TYPE_Q8_0, n, weight,
                           (const void *const *)rows, NR, multi, &lookups) == 0);
    printf("EG-01 multi-row: rows=%d lookups=%d\n", NR, lookups);
    CHECK(lookups == 1);
    for (int r = 0; r < NR; r++) {
        CHECK(fabsf(multi[r] - single[r]) < 1e-6f);
    }
    CHECK(wubu_vecdot_rows(9999, n, weight, (const void *const *)rows,
                           NR, multi, &lookups) == -1);

    for (int r = 0; r < NR; r++) free(rows[r]);
    free(rows);
    free(weight);

    if (failures) {
        fprintf(stderr, "test_eg01_vecdot: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg01_vecdot: PASS\n");
    return 0;
}
