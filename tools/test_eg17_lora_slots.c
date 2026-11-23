/* test_eg17_lora_slots.c -- EG-17 gate: multi-adapter LoRA slots. */
#include <stdio.h>
#include <math.h>

#include "wubu_lora_slots.h"
#include "wubu_lora.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)
#define NEAR(a,b) (fabsf((a)-(b)) < 1e-5f)

int main(void) {
    wubu_lora_slots_t *s = wubu_lora_slots_create(4);
    float A0[2] = { 1.0f, 0.0f }, B0[2] = { 2.0f, 0.0f };
    float A1[2] = { 0.0f, 1.0f }, B1[2] = { 0.0f, 5.0f };
    float A2[3] = { 1.0f, 0.0f, 0.0f }, B2[2] = { 1.0f, 1.0f };
    float x[2] = { 3.0f, 4.0f }, y[2];
    int sel[2];

    CHECK(s != NULL);
    if (!s) return 1;
    CHECK(wubu_lora_slots_add(s, 0, 1, 1.0f, 2, 2, A0, B0) == 0);
    CHECK(wubu_lora_slots_add(s, 1, 1, 1.0f, 2, 2, A1, B1) == 0);
    CHECK(wubu_lora_slots_add(s, 2, 1, 1.0f, 3, 2, A2, B2) == 0);
    CHECK(wubu_lora_slots_add(s, 4, 1, 1.0f, 2, 2, A0, B0) == -1);
    CHECK(wubu_lora_slots_count(s) == 3);

    sel[0] = 0;
    CHECK(wubu_lora_slots_apply_grouped(s, sel, 1, x, y, 2) == 0);
    printf("EG-17 adapter0=[%.1f %.1f]\n", y[0], y[1]);
    CHECK(NEAR(y[0], 6.0f) && NEAR(y[1], 0.0f));

    sel[0] = 0; sel[1] = 1;
    CHECK(wubu_lora_slots_apply_grouped(s, sel, 2, x, y, 2) == 0);
    printf("EG-17 grouped=[%.1f %.1f]\n", y[0], y[1]);
    CHECK(NEAR(y[0], 6.0f) && NEAR(y[1], 20.0f));

    /* Per-request isolation: previous grouped call must not leak. */
    sel[0] = 1;
    CHECK(wubu_lora_slots_apply_grouped(s, sel, 1, x, y, 2) == 0);
    printf("EG-17 isolated=[%.1f %.1f]\n", y[0], y[1]);
    CHECK(NEAR(y[0], 0.0f) && NEAR(y[1], 20.0f));

    CHECK(wubu_lora_slots_apply_grouped(s, NULL, 0, x, y, 2) == 0);
    CHECK(NEAR(y[0], 0.0f) && NEAR(y[1], 0.0f));

    /* Reject duplicates, unknown ids and dimension mismatches. */
    sel[0] = 0; sel[1] = 0;
    CHECK(wubu_lora_slots_apply_grouped(s, sel, 2, x, y, 2) == -1);
    sel[0] = 0; sel[1] = 9;
    CHECK(wubu_lora_slots_apply_grouped(s, sel, 2, x, y, 2) == -1);
    sel[0] = 0; sel[1] = 2;
    CHECK(wubu_lora_slots_apply_grouped(s, sel, 2, x, y, 2) == -1);

    CHECK(wubu_lora_slots_remove(s, 1) == 0);
    CHECK(wubu_lora_slots_count(s) == 2);
    sel[0] = 1;
    CHECK(wubu_lora_slots_apply_grouped(s, sel, 1, x, y, 2) == -1);
    CHECK(wubu_lora_slots_remove(s, 1) == -1);

    wubu_lora_slots_free(s);
    if (failures) {
        fprintf(stderr, "test_eg17_lora_slots: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg17_lora_slots: PASS\n");
    return 0;
}
