/* test_eg66_aot_pack.c -- EG-66c gate: AOT kernel pack selection. */
#include <stdio.h>

#include "wubu_aot_pack.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_aot_pack_entry_t packs[3] = {
        {0, 1, 8, 11}, {0, 9, 64, 12}, {1, 1, 16, 21}
    };
    wubu_aot_pack_t *p = wubu_aot_pack_create(packs, 3);
    int variant = -1;
    CHECK(p != NULL);
    CHECK(wubu_aot_pack_count(p) == 3);

    CHECK(wubu_aot_pack_select(p, 0, 4, &variant) == 0 && variant == 11);
    CHECK(wubu_aot_pack_select(p, 0, 32, &variant) == 0 && variant == 12);
    CHECK(wubu_aot_pack_select(p, 1, 8, &variant) == 0 && variant == 21);
    CHECK(wubu_aot_pack_select(p, 1, 32, &variant) == -1);
    printf("EG-66c dtype0 n4->11 n32->12 dtype1 n8->21 n32->unsupported\n");

    CHECK(wubu_aot_pack_create(NULL, 3) == NULL);
    CHECK(wubu_aot_pack_create(packs, 0) == NULL);
    CHECK(wubu_aot_pack_select(p, 0, 0, &variant) == -1);
    CHECK(wubu_aot_pack_select(p, 0, 4, NULL) == -1);
    CHECK(wubu_aot_pack_select(NULL, 0, 4, &variant) == -1);
    {
        wubu_aot_pack_entry_t overlap[2] = {{0, 1, 16, 1}, {0, 8, 32, 2}};
        CHECK(wubu_aot_pack_create(overlap, 2) == NULL);
    }
    printf("EG-66c overlapping packs rejected\n");

    wubu_aot_pack_free(p);
    if (failures) { fprintf(stderr, "test_eg66_aot_pack: FAIL (%d)\n", failures); return 1; }
    printf("test_eg66_aot_pack: PASS\n");
    return 0;
}
