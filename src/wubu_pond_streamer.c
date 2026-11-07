/* wubu_pond_streamer.c — research-pond intake into the KV namespace
 *
 * Incremental on-disk watcher: rescans watched pond roots on each poll and
 * streams NEW/CHANGED regular files into /kv/in/<pond>/<relpath> through
 * the KV embedding bridge. An mtime+size ledger skips untouched files.
 *
 * Design: docs/wubu1-improvement-plan.md module #4 (AN22), research/047.
 * WaefreBeorn Umbrella License v3.0
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_pond_streamer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>

/* One watched file in the ledger. */
typedef struct {
    char      fullpath[4096]; /* absolute disk path for mtime stat */
    char      kvpath[512];    /* /kv/in/<pond>/<relpath> namespace key */
    time_t    mtime;
    off_t     size;
} ps_file_t;

/* One pond root. */
typedef struct {
    char  root[4096];    /* absolute disk root */
    char  name[256];     /* basename -> pond name in namespace */
} ps_pond_t;

struct wubu_pond_streamer {
    wubu_kv_embedding_t *kv;
    ps_pond_t           *ponds;
    size_t               n_ponds;
    ps_file_t           *files;
    size_t               n_files;
    size_t               cap_files;
    unsigned long        total_streamed;
};

static int ps_ensure_cap(wubu_pond_streamer_t *ps) {
    if (ps->n_files < ps->cap_files) return 0;
    size_t newcap = ps->cap_files ? ps->cap_files * 2 : 32;
    ps_file_t *f = (ps_file_t *)realloc(ps->files, newcap * sizeof(*f));
    if (!f) return -1;
    ps->files = f;
    ps->cap_files = newcap;
    return 0;
}

/* Find a ledger entry by full path, or -1. */
static int ps_find(const wubu_pond_streamer_t *ps, const char *fullpath) {
    for (size_t i = 0; i < ps->n_files; i++)
        if (strcmp(ps->files[i].fullpath, fullpath) == 0) return (int)i;
    return -1;
}

/* Read an entire regular file into a heap buffer. Caller frees *out.
 * Returns 0 on success; *out_len holds byte count. */
static int ps_read_file(const char *path, char **out, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size < 0) { fclose(f); return -1; }
    char *buf = (char *)malloc((size_t)st.st_size ? (size_t)st.st_size : 1);
    if (!buf) { fclose(f); return -1; }
    size_t got = fread(buf, 1, (size_t)st.st_size, f);
    fclose(f);
    if (got != (size_t)st.st_size) { free(buf); return -1; }
    *out = buf;
    *out_len = got;
    return 0;
}

/* Recursively stream one pond root. Returns 0 on success, -1 on IO error. */
static int ps_walk(wubu_pond_streamer_t *ps, ps_pond_t *pond,
                   const char *dirpath, const char *relpath) {
    DIR *dir = opendir(dirpath);
    if (!dir) return -1;
    struct dirent *entry;
    struct stat st;
    int rc = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.' && (entry->d_name[1] == 0 ||
            (entry->d_name[1] == '.' && entry->d_name[2] == 0)))
            continue;
        char fullpath[4096];
        char rel[512];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);
        if (strlen(relpath) > 0)
            snprintf(rel, sizeof(rel), "%s/%s", relpath, entry->d_name);
        else
            snprintf(rel, sizeof(rel), "%s", entry->d_name);
        if (stat(fullpath, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            rc |= ps_walk(ps, pond, fullpath, rel);
        } else if (S_ISREG(st.st_mode)) {
            int idx = ps_find(ps, fullpath);
            if (idx >= 0 &&
                ps->files[idx].mtime == st.st_mtime &&
                ps->files[idx].size == st.st_size)
                continue; /* unchanged — skip */
            char kvpath[512];
            snprintf(kvpath, sizeof(kvpath), "%s/%s", pond->name, rel);
            if (idx >= 0) {
                /* Changed existing file: kvfs_mount refuses a second mount
                 * of an already-mounted path, so replace the old region. */
                char mount_path[512];
                snprintf(mount_path, sizeof(mount_path), "/kv/in/%s", kvpath);
                wubu_kvfs_unmount(wubu_kv_embedding_get_fs(ps->kv), mount_path);
            }
            /* Read and stream through the KV embedding bridge */
            char *bytes = NULL; size_t n = 0;
            if (ps_read_file(fullpath, &bytes, &n) != 0) { rc |= -1; continue; }
            /* Encode into /kv/in/<pond>/<rel>. Re-encode replaces in place. */
            size_t nt = 0;
            if (wubu_kv_embedding_encode(ps->kv, kvpath, bytes, n, &nt) != 0) {
                free(bytes); rc |= -1; continue;
            }
            free(bytes);
            ps->total_streamed++;
            /* Upsert the ledger row */
            if (idx < 0) {
                if (ps_ensure_cap(ps) != 0) { rc |= -1; continue; }
                idx = (int)ps->n_files++;
                strncpy(ps->files[idx].fullpath, fullpath, sizeof(ps->files[idx].fullpath) - 1);
                ps->files[idx].fullpath[sizeof(ps->files[idx].fullpath) - 1] = '\0';
            }
            strncpy(ps->files[idx].kvpath, kvpath, sizeof(ps->files[idx].kvpath) - 1);
            ps->files[idx].kvpath[sizeof(ps->files[idx].kvpath) - 1] = '\0';
            ps->files[idx].mtime = st.st_mtime;
            ps->files[idx].size = st.st_size;
        }
    }
    closedir(dir);
    return rc;
}

wubu_pond_streamer_t *wubu_pond_streamer_create(wubu_kv_embedding_t *kv,
                                                const char **pond_roots,
                                                size_t n_roots) {
    if (!kv || !pond_roots || n_roots == 0) return NULL;
    wubu_pond_streamer_t *ps = (wubu_pond_streamer_t *)calloc(1, sizeof(*ps));
    if (!ps) return NULL;
    ps->kv = kv;
    ps->ponds = (ps_pond_t *)calloc(n_roots, sizeof(ps_pond_t));
    if (!ps->ponds) { free(ps); return NULL; }
    ps->n_ponds = n_roots;
    for (size_t i = 0; i < n_roots; i++) {
        if (!pond_roots[i]) { free(ps->ponds); free(ps); return NULL; }
        strncpy(ps->ponds[i].root, pond_roots[i], sizeof(ps->ponds[i].root) - 1);
        ps->ponds[i].root[sizeof(ps->ponds[i].root) - 1] = '\0';
        /* pond name = basename of the root path */
        const char *p = strrchr(ps->ponds[i].root, '/');
        const char *base = p ? p + 1 : ps->ponds[i].root;
        strncpy(ps->ponds[i].name, base, sizeof(ps->ponds[i].name) - 1);
        ps->ponds[i].name[sizeof(ps->ponds[i].name) - 1] = '\0';
    }
    return ps;
}

int wubu_pond_streamer_poll(wubu_pond_streamer_t *ps, int *out_new) {
    if (!ps) return -1;
    int before = (int)ps->total_streamed;
    int rc = 0;
    for (size_t i = 0; i < ps->n_ponds; i++)
        rc |= ps_walk(ps, &ps->ponds[i], ps->ponds[i].root, "");
    if (out_new) *out_new = (int)(ps->total_streamed - (unsigned long)before);
    return rc;
}

int wubu_pond_streamer_count(const wubu_pond_streamer_t *ps) {
    if (!ps) return 0;
    return (int)ps->n_files;
}

unsigned long wubu_pond_streamer_total_streamed(const wubu_pond_streamer_t *ps) {
    return ps ? ps->total_streamed : 0;
}

void wubu_pond_streamer_free(wubu_pond_streamer_t *ps) {
    if (!ps) return;
    free(ps->files);
    free(ps->ponds);
    free(ps);
}
