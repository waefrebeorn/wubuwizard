/* test_eg66_tensorfold.c -- EG-66 gate: disk prefix snapshot round-trip. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_tensorfold.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define NT 16
#define D 8

int main(void) {
    const char *dir = getenv("TMPDIR");
    char path[512];
    int tokens[NT], got[NT], i;
    float kv[NT*D], back[NT*D], serial[NT*D];
    FILE *f;

    if (!dir) dir = ".";
    snprintf(path, sizeof(path), "%s/eg66_prefix.snap", dir);
    for (i = 0; i < NT; i++) tokens[i] = i * 3;
    wubu_tensorfold_serial_kv(tokens, NT, D, kv);

    CHECK(wubu_tensorfold_snapshot(path, tokens, NT, kv, D) == 0);
    CHECK(wubu_tensorfold_restore(path, got, NT, back, D) == 0);
    CHECK(memcmp(tokens, got, sizeof(tokens)) == 0);
    CHECK(memcmp(kv, back, sizeof(kv)) == 0);
    printf("EG-66 round-trip %d tokens x %d dims byte-identical (%zu bytes payload)\n",
           NT, D, (size_t)NT * (sizeof(int) + D * sizeof(float)));

    wubu_tensorfold_serial_kv(got, NT, D, serial);
    CHECK(memcmp(serial, back, sizeof(serial)) == 0);
    printf("EG-66 serial-equivalence: restored KV == serial recompute\n");

    f = fopen(path, "rb+");
    CHECK(f != NULL);
    if (f) {
        CHECK(fseek(f, 24 + 6, SEEK_SET) == 0);
        CHECK(fputc(0x7f, f) != EOF);
        fclose(f);
    }
    CHECK(wubu_tensorfold_restore(path, got, NT, back, D) == -1);
    printf("EG-66 corrupted snapshot rejected\n");

    CHECK(wubu_tensorfold_snapshot(path, tokens, NT, kv, 0) == -1);
    CHECK(wubu_tensorfold_restore(path, got, NT + 1, back, D) == -1);
    CHECK(wubu_tensorfold_restore("/nonexistent/eg66.snap", got, NT, back, D) == -1);
    remove(path);

    if (failures) {
        fprintf(stderr, "test_eg66_tensorfold: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg66_tensorfold: PASS\n");
    return 0;
}
