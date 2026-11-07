/* wubu_kv_audit.c — immune audit over the KV-FS seams (5+1: detect+isolate)
 *
 * Per-path operation ledger with rule checking. An op that violates a
 * lifecycle rule is reported as an anomaly; the path is the isolate.
 * heal() re-baselines so the 5+1 recovery can re-validate after a fix.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, module #10 (AN22).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_kv_audit.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    char     path[512];
    int      loaded;
    int      mutated;
    int      dirty;        /* mutated && not (saved since) */
    uint32_t checksum;     /* FNV-1a over path + op sequence */
} au_path_t;

struct wubu_kv_audit {
    wubu_kv_embedding_t *kv;
    wubu_kv_audit_cfg_t  cfg;
    au_path_t  *paths;
    size_t      n_paths;
    size_t      cap_paths;
    unsigned long anomalies;
    uint32_t    last_anomaly_checksum;
};

/* FNV-1a 32-bit */
static uint32_t fnv1a_step(uint32_t h, unsigned char c) {
    h ^= c;
    h *= 0x01000193u;
    return h;
}

static uint32_t fnv1a(const char *path, unsigned opbyte) {
    uint32_t h = 0x811c9dc5u;
    for (const char *p = path; *p; p++) h = fnv1a_step(h, (unsigned char)*p);
    h = fnv1a_step(h, opbyte);
    return h;
}

wubu_kv_audit_cfg_t wubu_kv_audit_default_cfg(void) {
    wubu_kv_audit_cfg_t c;
    c.detect_mutate_before_load  = 1;
    c.detect_save_without_mutate = 1;
    c.detect_dirty_load          = 1;
    return c;
}

static int au_ensure_cap(wubu_kv_audit_t *a) {
    if (a->n_paths < a->cap_paths) return 0;
    size_t newcap = a->cap_paths ? a->cap_paths * 2 : 16;
    au_path_t *p = (au_path_t *)realloc(a->paths, newcap * sizeof(*p));
    if (!p) return -1;
    a->paths = p;
    a->cap_paths = newcap;
    return 0;
}

static au_path_t *au_get(wubu_kv_audit_t *a, const char *path, int create) {
    for (size_t i = 0; i < a->n_paths; i++)
        if (strcmp(a->paths[i].path, path) == 0) return &a->paths[i];
    if (!create) return NULL;
    if (au_ensure_cap(a) != 0) return NULL;
    au_path_t *p = &a->paths[a->n_paths++];
    strncpy(p->path, path, sizeof(p->path) - 1);
    p->path[sizeof(p->path) - 1] = '\0';
    p->loaded = p->mutated = p->dirty = 0;
    p->checksum = 0x811c9dc5u; /* FNV offset — empty sequence */
    return p;
}

static void au_flag(wubu_kv_audit_t *a, au_path_t *p, unsigned opbyte) {
    a->anomalies++;
    a->last_anomaly_checksum = fnv1a(p->path, opbyte);
}

wubu_kv_audit_t *wubu_kv_audit_create(wubu_kv_embedding_t *kv,
                                      const wubu_kv_audit_cfg_t *cfg) {
    if (!kv) return NULL;
    wubu_kv_audit_t *a = (wubu_kv_audit_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->kv = kv;
    a->cfg = cfg ? *cfg : wubu_kv_audit_default_cfg();
    return a;
}

wubu_kv_audit_code_t wubu_kv_audit_record(wubu_kv_audit_t *a,
                                          wubu_kv_audit_op_t op,
                                          const char *path) {
    if (!a || !path) return WUBU_KV_AUDIT_OK;
    if (op != WUBU_KV_AUDIT_LOAD && op != WUBU_KV_AUDIT_SAVE &&
        op != WUBU_KV_AUDIT_FORWARD && op != WUBU_KV_AUDIT_MUTATE)
        return WUBU_KV_AUDIT_OK;
    au_path_t *p = au_get(a, path, 1);
    if (!p) return WUBU_KV_AUDIT_OK;

    unsigned ob = (unsigned)op;
    wubu_kv_audit_code_t code = WUBU_KV_AUDIT_OK;

    switch (op) {
    case WUBU_KV_AUDIT_MUTATE:
        if (a->cfg.detect_mutate_before_load && !p->loaded)
            code = WUBU_KV_AUDIT_MUTATE_BEFORE_LOAD;
        p->mutated = 1;
        p->dirty = 1;   /* mutated and not yet saved */
        break;
    case WUBU_KV_AUDIT_SAVE:
        if (a->cfg.detect_save_without_mutate && !p->mutated)
            code = WUBU_KV_AUDIT_SAVE_WITHOUT_MUTATE;
        p->mutated = 0;
        p->dirty = 0;   /* saved — clean again */
        break;
    case WUBU_KV_AUDIT_LOAD:
        if (a->cfg.detect_dirty_load && p->dirty)
            code = WUBU_KV_AUDIT_DIRTY_LOAD;
        p->loaded = 1;
        break;
    case WUBU_KV_AUDIT_FORWARD:
    default:
        break;          /* no rule; still advances the checksum */
    }

    p->checksum = fnv1a_step(p->checksum, ob);
    if (code != WUBU_KV_AUDIT_OK)
        au_flag(a, p, ob);
    return code;
}

unsigned long wubu_kv_audit_anomalies(const wubu_kv_audit_t *a) {
    return a ? a->anomalies : 0;
}

uint32_t wubu_kv_audit_last_path_checksum(const wubu_kv_audit_t *a) {
    return a ? a->last_anomaly_checksum : 0;
}

uint32_t wubu_kv_audit_checksum(const wubu_kv_audit_t *a, const char *path) {
    if (!a || !path) return 0;
    au_path_t *p = au_get((wubu_kv_audit_t *)a, path, 0);
    return p ? p->checksum : 0;
}

int wubu_kv_audit_heal(wubu_kv_audit_t *a, const char *path) {
    if (!a || !path) return -1;
    au_path_t *p = au_get(a, path, 0);
    if (!p) return -1;
    p->loaded = p->mutated = p->dirty = 0;
    p->checksum = 0x811c9dc5u; /* re-baseline: empty sequence */
    return 0;
}

void wubu_kv_audit_free(wubu_kv_audit_t *a) {
    if (!a) return;
    free(a->paths);
    free(a);
}
