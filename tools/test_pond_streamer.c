/* test_pond_streamer.c — research-pond intake into the KV namespace (AN22 #4)
 *
 * Verifies:
 *   T1 create + poll streams NEW files into /kv/in/<pond>/<relpath>
 *   T2 regions resolve through the KV embedding bridge
 *   T3 a CHANGED file is re-encoded by the next poll
 *   T4 an UNCHANGED file is skipped (poll returns 0)
 *   T5 file count + cumulative stream ledger
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_pond_streamer.h"
#include "wubu_kv_embedding.h"
#include "wubu_kvfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

static int fail = 0;

static void chk(const char *label, int cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", label);
    if (!cond) fail++;
}

static void mkf(const char *path, const char *content) {
    FILE *f = fopen(path, "wb");
    fwrite(content, 1, strlen(content), f);
    fclose(f);
}

int main(void) {
    const char *root = "/tmp/wubu_pond_test/pond-x";
    system("rm -rf /tmp/wubu_pond_test && mkdir -p /tmp/wubu_pond_test/pond-x/sub");
    mkf("/tmp/wubu_pond_test/pond-x/a.txt", "attention kv-cache serving");
    mkf("/tmp/wubu_pond_test/pond-x/sub/b.txt", "gated delta net");

    const char *roots[] = { root };
    wubu_kvfs_t *fs = wubu_kvfs_create(256, 256);
    wubu_kv_embedding_t *kv = wubu_kv_embedding_create(fs, 256);
    wubu_pond_streamer_t *ps = wubu_pond_streamer_create(kv, roots, 1);
    chk("pond_streamer_create returns handle", ps != NULL);

    int n = -1;
    printf("[T1] initial poll\n");
    chk("poll returns 0", wubu_pond_streamer_poll(ps, &n) == 0);
    chk("poll streamed 2 new files", n == 2);

    printf("[T2] regions resolve in the KV namespace\n");
    uint32_t blk; size_t off, nf;
    chk("a.txt region resolves",
        wubu_kv_embedding_region(kv, "/kv/in/pond-x/a.txt", &blk, &off, &nf) == 0);
    printf("       (a.txt n_floats=%zu)\n", nf);
    chk("a.txt n_floats == 26", nf == 26);
    chk("sub/b.txt region resolves",
        wubu_kv_embedding_region(kv, "/kv/in/pond-x/sub/b.txt", &blk, &off, &nf) == 0);
    printf("       (b.txt n_floats=%zu)\n", nf);
    chk("sub/b.txt n_floats == 15", nf == 15);

    printf("[T3] changed file is re-encoded\n");
    usleep(1100000); /* let mtime tick past */
    mkf("/tmp/wubu_pond_test/pond-x/a.txt", "attention kv-cache serving robin moa");
    chk("re-poll returns 0", wubu_pond_streamer_poll(ps, &n) == 0);
    chk("re-poll streamed 1 changed file", n == 1);
    chk("a.txt still resolves",
        wubu_kv_embedding_region(kv, "/kv/in/pond-x/a.txt", &blk, &off, &nf) == 0);
    printf("       (a.txt n_floats=%zu)\n", nf);
    chk("a.txt n_floats == 36 after change", nf == 36);

    printf("[T4] unchanged poll is a no-op\n");
    chk("third poll returns 0", wubu_pond_streamer_poll(ps, &n) == 0);
    chk("unchanged poll streamed 0", n == 0);

    printf("[T5] ledger counts\n");
    chk("tracked files == 2", wubu_pond_streamer_count(ps) == 2);
    chk("cumulative streamed == 3",
        wubu_pond_streamer_total_streamed(ps) == 3);

    wubu_pond_streamer_free(ps);
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);
    system("rm -rf /tmp/wubu_pond_test");

    printf("test_pond_streamer: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
