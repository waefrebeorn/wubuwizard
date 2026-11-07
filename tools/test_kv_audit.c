/* test_kv_audit.c — immune audit hooks over KV-FS seams (AN22 #10)
 *
 * Verifies:
 *   T1 clean LOAD->MUTATE->SAVE lifecycle flags nothing
 *   T2 MUTATE before LOAD -> MUTATE_BEFORE_LOAD (spurious write)
 *   T3 on a clean path: MUTATE then LOAD -> DIRTY_LOAD; SAVE clears it
 *   T4 SAVE without MUTATE -> SAVE_WITHOUT_MUTATE
 *   T5 checksum advances with ops; heal() re-baselines + resets state
 *
 * The anomaly COUNT is the cumulative ledger — each flag is +1.
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_kv_audit.h"
#include "wubu_kv_embedding.h"
#include "wubu_kvfs.h"
#include <stdio.h>
#include <stdlib.h>

static int fail = 0;
static void chk(const char *label, int cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", label);
    if (!cond) fail++;
}

int main(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(256, 256);
    wubu_kv_embedding_t *kv = wubu_kv_embedding_create(fs, 256);
    wubu_kv_audit_t *a = wubu_kv_audit_create(kv, NULL);
    chk("kv_audit_create returns handle", a != NULL);
    chk("starts with 0 anomalies", wubu_kv_audit_anomalies(a) == 0);

    const char *A = "/kv/in/a.txt", *B = "/kv/in/b.txt";
    const char *C = "/kv/in/c.txt", *D = "/kv/in/d.txt";
    wubu_kv_audit_code_t r;

    printf("[T1] clean lifecycle (path A)\n");
    chk("LOAD ok", wubu_kv_audit_record(a, WUBU_KV_AUDIT_LOAD, A) == WUBU_KV_AUDIT_OK);
    chk("MUTATE ok", wubu_kv_audit_record(a, WUBU_KV_AUDIT_MUTATE, A) == WUBU_KV_AUDIT_OK);
    chk("SAVE ok", wubu_kv_audit_record(a, WUBU_KV_AUDIT_SAVE, A) == WUBU_KV_AUDIT_OK);
    chk("no anomalies on clean life", wubu_kv_audit_anomalies(a) == 0);

    printf("[T2] mutate before load (path B)\n");
    r = wubu_kv_audit_record(a, WUBU_KV_AUDIT_MUTATE, B);
    chk("MUTATE on fresh path flags MUTATE_BEFORE_LOAD",
        r == WUBU_KV_AUDIT_MUTATE_BEFORE_LOAD);
    chk("anomalies == 1", wubu_kv_audit_anomalies(a) == 1);
    chk("last-path checksum nonzero", wubu_kv_audit_last_path_checksum(a) != 0);

    printf("[T3] dirty read on a clean path (path D)\n");
    chk("LOAD D ok", wubu_kv_audit_record(a, WUBU_KV_AUDIT_LOAD, D) == WUBU_KV_AUDIT_OK);
    chk("MUTATE D ok", wubu_kv_audit_record(a, WUBU_KV_AUDIT_MUTATE, D) == WUBU_KV_AUDIT_OK);
    r = wubu_kv_audit_record(a, WUBU_KV_AUDIT_LOAD, D);
    chk("LOAD of unsaved mutation flags DIRTY_LOAD",
        r == WUBU_KV_AUDIT_DIRTY_LOAD);
    chk("anomalies == 2", wubu_kv_audit_anomalies(a) == 2);
    chk("SAVE D ok", wubu_kv_audit_record(a, WUBU_KV_AUDIT_SAVE, D) == WUBU_KV_AUDIT_OK);
    chk("LOAD D ok again (saved)", wubu_kv_audit_record(a, WUBU_KV_AUDIT_LOAD, D) == WUBU_KV_AUDIT_OK);
    chk("anomalies still 2", wubu_kv_audit_anomalies(a) == 2);

    printf("[T4] save without mutate (path C)\n");
    chk("LOAD C ok", wubu_kv_audit_record(a, WUBU_KV_AUDIT_LOAD, C) == WUBU_KV_AUDIT_OK);
    r = wubu_kv_audit_record(a, WUBU_KV_AUDIT_SAVE, C);
    chk("SAVE of unchanged path flags SAVE_WITHOUT_MUTATE",
        r == WUBU_KV_AUDIT_SAVE_WITHOUT_MUTATE);
    chk("anomalies == 3", wubu_kv_audit_anomalies(a) == 3);

    printf("[T5] checksum + heal\n");
    uint32_t before = wubu_kv_audit_checksum(a, A);
    chk("A path checksum advanced", before != 0);
    chk("heal returns 0", wubu_kv_audit_heal(a, A) == 0);
    uint32_t after = wubu_kv_audit_checksum(a, A);
    chk("heal re-baselined the checksum", after != before);
    /* heal resets state: A is unloaded again */
    r = wubu_kv_audit_record(a, WUBU_KV_AUDIT_MUTATE, A);
    chk("post-heal MUTATE flags MUTATE_BEFORE_LOAD (state reset)",
        r == WUBU_KV_AUDIT_MUTATE_BEFORE_LOAD);
    chk("anomalies == 4", wubu_kv_audit_anomalies(a) == 4);

    wubu_kv_audit_free(a);
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    printf("test_kv_audit: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
