#include <dirent.h>
/*
 * wubu_safetensors_shard.c -- multi-shard safetensors loader.
 * See wubu_safetensors_shard.h. Self-contained; uses safetensors_reader.
 */
#include "wubu_safetensors_shard.h"
#include "safetensors_reader.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define SHARD_MAX 1024

struct wubu_shard_ctx {
    st_ctx *shards[SHARD_MAX];
    int     n;
};

/* Scan a directory for model-NNNNN-of-NNNNN.safetensors files. */
#include <fnmatch.h>
static int scan_shards(const char *dir, char paths[SHARD_MAX][1024]) {
    /* dependency-free directory scan — opendir/readdir + fnmatch filter */
    DIR *d = opendir(dir);
    if (!d) return 0;
    int n = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && n < SHARD_MAX) {
        if (fnmatch("model-*-of-*.safetensors", ent->d_name, 0) == 0) {
            snprintf(paths[n], 1024, "%s/%s", dir, ent->d_name);
            n++;
        }
    }
    closedir(d);
    if (n <= 1) return n;
    /* sort for stable load order */
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (strcmp(paths[i], paths[j]) > 0) {
                /* memcpy of the actual length, not strncpy(.., 1023). strncpy does not stop at the
                 * terminator: it pads the whole 1023-byte field with zeros, so this inner loop used to
                 * zero ~3 KB per comparison, i.e. ~3 KB * n^2/2 before copying a single useful byte.
                 * Shard filenames are ~40 bytes, so the real copy is ~2% of that work. */
                char tmp[1024];
                const size_t li = strlen(paths[i]) + 1, lj = strlen(paths[j]) + 1;
                memcpy(tmp,      paths[i], li);
                memcpy(paths[i], paths[j], lj);
                memcpy(paths[j], tmp,      li);
            }
    return n;
}

/* Derive the directory + base name from a single shard path. */
static void dir_of(const char *path, char *dir, size_t dsize) {
    const char *sl = strrchr(path, '/');
    if (sl) { size_t n = (size_t)(sl - path); if (n >= dsize) n = dsize-1;
              memcpy(dir, path, n); dir[n]=0; }
    else { dir[0]=0; }
}

wubu_shard_ctx_t *wubu_shard_open(const char *path_or_dir) {
    char dir[1024]; dir_of(path_or_dir, dir, sizeof(dir));
    if (dir[0]==0) snprintf(dir, sizeof(dir), ".");

    char paths[SHARD_MAX][1024];
    int n = 0;
    /* If the given path is itself a shard, scan its directory. */
    if (strstr(path_or_dir, "model-") && strstr(path_or_dir, "-of-")) {
        n = scan_shards(dir, paths);
    } else {
        /* Treat as directory; also check for a lone file. */
        n = scan_shards(path_or_dir, paths);
    }

    /* Fallback: a single (non-sharded) .safetensors file. This is the
     * common case for small models and single-shard checkpoints. */
    if (n == 0 && strstr(path_or_dir, ".safetensors")) {
        st_ctx *s = st_open(path_or_dir);
        if (s) {
            wubu_shard_ctx_t *sc = (wubu_shard_ctx_t *)calloc(1, sizeof(*sc));
            if (sc) { sc->shards[0] = s; sc->n = 1; return sc; }
            st_close(s);
        }
    }

    if (n == 0) return NULL;

    wubu_shard_ctx_t *sc = (wubu_shard_ctx_t *)calloc(1, sizeof(*sc));
    if (!sc) return NULL;
    sc->n = 0;
    for (int i = 0; i < n && i < SHARD_MAX; i++) {
        st_ctx *s = st_open(paths[i]);
        if (s) sc->shards[sc->n++] = s;
    }
    if (sc->n == 0) { free(sc); return NULL; }
    return sc;
}

int wubu_shard_count(const wubu_shard_ctx_t *sc) { return sc ? sc->n : 0; }

int64_t wubu_shard_n_tensors(const wubu_shard_ctx_t *sc) {
    if (!sc) return 0;
    int64_t t = 0;
    for (int i = 0; i < sc->n; i++) t += st_n_tensors(sc->shards[i]);
    return t;
}

static const st_tensor_info *find_across(const wubu_shard_ctx_t *sc,
                                         const char *name, int *shard_idx) {
    for (int i = 0; i < sc->n; i++) {
        const st_tensor_info *t = st_find_tensor(sc->shards[i], name);
        if (t) { if (shard_idx) *shard_idx = i; return t; }
    }
    return NULL;
}

float *wubu_shard_load_f32(const wubu_shard_ctx_t *sc, const char *name,
                           int64_t *n_elems_out) {
    if (!sc) return NULL;
    int si = 0;
    const st_tensor_info *t = find_across(sc, name, &si);
    if (!t) return NULL;
    int64_t n = t->n_elems;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return NULL;
    if (st_read_tensor_f32(sc->shards[si], t, buf, n) != n) { free(buf); return NULL; }
    if (n_elems_out) *n_elems_out = n;
    return buf;
}

float *wubu_shard_load_f32_t(const wubu_shard_ctx_t *sc, const char *name,
                             int rows, int cols) {
    if (!sc) return NULL;
    int si = 0;
    const st_tensor_info *t = find_across(sc, name, &si);
    if (!t) return NULL;
    int64_t need = (int64_t)rows * cols;
    if (t->n_elems < need) return NULL;
    float *tmp = (float *)malloc((size_t)need * sizeof(float));
    if (!tmp) return NULL;
    if (st_read_tensor_f32(sc->shards[si], t, tmp, need) != need) { free(tmp); return NULL; }
    float *out = (float *)malloc((size_t)need * sizeof(float));
    if (!out) { free(tmp); return NULL; }
    /* transpose [rows,cols] -> [cols,rows] */
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            out[(size_t)c * rows + r] = tmp[(size_t)r * cols + c];
    free(tmp);
    return out;
}

const float *wubu_shard_data_f32(const wubu_shard_ctx_t *sc, const char *name,
                                 int64_t *n_elems_out) {
    if (!sc) return NULL;
    int si = 0;
    const st_tensor_info *t = find_across(sc, name, &si);
    if (!t) return NULL;
    /* st_read_tensor_f32 writes into a caller buffer; to avoid a copy we
     * re-read into a persistent (malloc'd) buffer owned by this call's
     * caller is not possible here, so we return NULL and require the
     * load_f32 variants. Kept for API completeness. */
    (void)n_elems_out;
    return NULL;
}

void wubu_shard_close(wubu_shard_ctx_t *sc) {
    if (!sc) return;
    for (int i = 0; i < sc->n; i++) st_close(sc->shards[i]);
    free(sc);
}

int wubu_shard_dimof(const wubu_shard_ctx_t *sc, const char *name, int i) {
    int si = 0;
    const st_tensor_info *t = find_across(sc, name, &si);
    return (t && i < t->n_dims) ? (int)t->dims[i] : -1;
}

int wubu_shard_has(const wubu_shard_ctx_t *sc, const char *name) {
    return find_across(sc, name, NULL) != NULL;
}

const uint8_t *wubu_shard_raw(const wubu_shard_ctx_t *sc, const char *name,
                             int *out_dtype, int64_t *out_row) {
    if (!sc) return NULL;
    int si = 0;
    const st_tensor_info *t = find_across(sc, name, &si);
    if (!t) return NULL;
    if (out_dtype) *out_dtype = (int)t->dtype;
    int64_t row = 0;
    /* row length = product of all dims except the first (outer) one */
    for (int d = 1; d < t->n_dims; d++) row *= t->dims[d];
    if (row == 0 && t->n_dims >= 1) row = t->n_elems / t->dims[0];
    if (out_row) *out_row = row;
    return st_tensor_raw_ptr(sc->shards[si], t);
}
