/* test_eg99_conv_blocks.c -- EG-99 gate: content-addressed conversation blocks. */
#include <stdio.h>
#include <string.h>

#include "wubu_conv_store.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_conv_store_t *p = wubu_conv_store_create(8, 3);
    unsigned char a[24], b[24];
    int n1, n2, n3;
    CHECK(p != NULL);
    for (int i = 0; i < 24; ++i) { a[i] = (unsigned char)i; b[i] = (unsigned char)(i + 1); }

    /* 24 bytes at 8-byte blocks = 3 physical files. */
    n1 = wubu_conv_store_write(p, 42, a, sizeof(a));
    CHECK(n1 == 3);
    CHECK(wubu_conv_store_file_count(p) == 3);
    CHECK(wubu_conv_store_has_block(p, 42, a, 8) == 1);

    /* Same prefix and bytes are deduplicated across conversations. */
    n2 = wubu_conv_store_write(p, 42, a, sizeof(a));
    CHECK(n2 == 0);
    CHECK(wubu_conv_store_file_count(p) == 3);
    printf("EG-99 shared prefix+bytes: first=%d new, second=%d new, files=%d\n",
           n1, n2, wubu_conv_store_file_count(p));

    /* A different prefix or changed bytes writes new files. */
    n3 = wubu_conv_store_write(p, 43, a, sizeof(a));
    CHECK(n3 == 3);
    CHECK(wubu_conv_store_write(p, 42, b, sizeof(b)) == 3);
    CHECK(wubu_conv_store_file_count(p) == 9);
    printf("EG-99 different prefix=%d new, changed bytes=%d new, files=%d\n",
           n3, wubu_conv_store_file_count(p) - 6, wubu_conv_store_file_count(p));

    /* Checkpoints are bounded to the root plus three turn ends. */
    {
        wubu_conv_store_t *q = wubu_conv_store_create(8, 3);
        CHECK(q != NULL);
        CHECK(wubu_conv_store_checkpoint(q, 1, 0) == 0);
        for (int i = 1; i <= 4; ++i)
            CHECK(wubu_conv_store_checkpoint(q, 0, i * 10) == 0);
        CHECK(wubu_conv_store_checkpoint_count(q) == 4);
        CHECK(wubu_conv_store_checkpoint(q, 1, 0) == 0);
        CHECK(wubu_conv_store_checkpoint_count(q) == 4);
        printf("EG-99 checkpoints root + last 3 turns (count=%d)\n",
               wubu_conv_store_checkpoint_count(q));
        CHECK(wubu_conv_store_create(0, 3) == NULL);
        CHECK(wubu_conv_store_create(8, 0) == NULL);
        CHECK(wubu_conv_store_write(q, 1, NULL, 8) == -1);
        CHECK(wubu_conv_store_has_block(q, 1, NULL, 8) == -1);
        CHECK(wubu_conv_store_checkpoint(NULL, 1, 0) == -1);
        CHECK(wubu_conv_store_file_count(NULL) == -1);
        wubu_conv_store_free(q);
    }

    wubu_conv_store_free(p);
    if (failures) { fprintf(stderr, "test_eg99_conv_blocks: FAIL (%d)\n", failures); return 1; }
    printf("test_eg99_conv_blocks: PASS\n");
    return 0;
}
