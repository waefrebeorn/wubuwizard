/* wubu_kv_embedding.c — the KV-FS embedding bridge
 *
 * The file system IS the model's context. This module bridges files
 * into the KV namespace and measures coherence (whether the model
 * "understands" a file from its attention patterns).
 *
 * Design: docs/wubu1-hive-mind-plan.md §4 Phase 1 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_kv_embedding.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* Internal: the path registry maps /kv/in/<path> and /kv/synth/<path>
 * to their token counts and KV tensor offset. The token data itself
 * is written to the KV tensor by the executor at runtime; this module
 * owns the namespace bookkeeping and the coherence computation. */
typedef struct {
    char    path[256];
    int64_t n_tokens;     /* tokens this file occupies in the namespace */
    int64_t kv_offset;    /* float offset in the flat KV tensor */
} kv_path_record_t;

struct wubu_kv_embedding {
    wubu_kvfs_t *fs;          /* the namespace (not owned) */
    uint32_t     block_size;  /* floats per block */
    kv_path_record_t *paths;  /* path registry */
    size_t          n_paths;
    size_t          cap_paths;
    uint32_t        next_block; /* first free block for new mounts */
};

/* Simple byte-level tokenizer: each byte maps to a token id.
 * Byte 0 → token 0 + vocab_base, byte 1 → token 1 + vocab_base, etc.
 * We use vocab_base = 16384 - 256 so the 256 byte-tokens land at the
 * top of the 16384 vocab (they won't collide with BPE merges).
 * Each byte is one uint16_t token. */
#define BYTE_VOCAB_BASE  (16384 - 256)

/* Grow the path registry */
static int kv_emb_ensure_cap(wubu_kv_embedding_t *kv) {
    if (kv->n_paths < kv->cap_paths) return 0;
    size_t newcap = kv->cap_paths ? kv->cap_paths * 2 : 16;
    kv_path_record_t *p = (kv_path_record_t *)realloc(
        kv->paths, newcap * sizeof(kv_path_record_t));
    if (!p) return -1;
    kv->paths = p;
    kv->cap_paths = newcap;
    return 0;
}

wubu_kv_embedding_t *wubu_kv_embedding_create(wubu_kvfs_t *fs,
                                               uint32_t block_size) {
    if (!fs || block_size == 0) return NULL;
    wubu_kv_embedding_t *kv = (wubu_kv_embedding_t *)calloc(1, sizeof(*kv));
    if (!kv) return NULL;
    kv->fs = fs;
    kv->block_size = block_size;
    kv->next_block = 1; /* block 0 reserved for system */
    return kv;
}

void wubu_kv_embedding_free(wubu_kv_embedding_t *kv) {
    if (!kv) return;
    free(kv->paths);
    free(kv);
}

size_t wubu_kv_embedding_file_count(const wubu_kv_embedding_t *kv) {
    if (!kv) return 0;
    return kv->n_paths;
}

const char *wubu_kv_embedding_get_path(const wubu_kv_embedding_t *kv, size_t i) {
    if (!kv || i >= kv->n_paths) return NULL;
    /* Return pointer to the stored path inside the record */
    return kv->paths[i].path;
}

size_t wubu_kv_embedding_get_n_tokens(const wubu_kv_embedding_t *kv, size_t i) {
    if (!kv || i >= kv->n_paths) return 0;
    return (size_t)kv->paths[i].n_tokens;
}

wubu_kvfs_t *wubu_kv_embedding_get_fs(wubu_kv_embedding_t *kv) {
    if (!kv) return NULL;
    return kv->fs;
}

/* ENCODE: file bytes → /kv/in/<path>
 * Each byte becomes one token (uint16_t). The file is mounted into
 * the KV namespace; the executor writes the token floats into the
 * actual KV tensor at runtime using the resolved offset.
 * Returns 0 on success, -1 on failure. Sets *out_n_tokens. */
int wubu_kv_embedding_encode(wubu_kv_embedding_t *kv,
                              const char *path,
                              const void *content, size_t content_bytes,
                              size_t *out_n_tokens) {
    if (!kv || !path) return -1;
    if (content_bytes == 0) {
        if (out_n_tokens) *out_n_tokens = 0;
        return 0;
    }
    if (!content) return -1;

    size_t n_tokens = content_bytes; /* 1 byte = 1 token */
    if (out_n_tokens) *out_n_tokens = n_tokens;

    /* Build the KV path: /kv/in/<path> */
    char kv_path[320];
    snprintf(kv_path, sizeof(kv_path), "/kv/in/%s", path);

    /* Convert byte tokens to uint16_t */
    const uint8_t *bytes = (const uint8_t *)content;
    uint16_t *tokens = (uint16_t *)malloc(n_tokens * sizeof(uint16_t));
    if (!tokens) return -1;
    for (size_t i = 0; i < n_tokens; i++)
        tokens[i] = (uint16_t)(BYTE_VOCAB_BASE + bytes[i]);

    /* Encode the tokens */
    int rc = wubu_kv_embedding_encode_tokens(kv, kv_path, tokens, n_tokens);
    free(tokens);
    return rc;
}

int wubu_kv_embedding_encode_tokens(wubu_kv_embedding_t *kv,
                                     const char *path,
                                     const uint16_t *tokens, size_t n_tokens) {
    if (!kv || !path || !tokens || n_tokens == 0) return -1;

    /* Re-encoding an existing path REPLACES it (KV-FS write semantics):
     * find a prior record, drop its namespace mount + record, then remount. */
    int replace_idx = -1;
    for (size_t i = 0; i < kv->n_paths; i++)
        if (strcmp(kv->paths[i].path, path) == 0) { replace_idx = (int)i; break; }
    if (replace_idx >= 0)
        wubu_kvfs_unmount(kv->fs, path);

    /* Compute blocks needed: each token is 1 float in the KV tensor */
    int64_t n_floats = (int64_t)n_tokens;
    uint32_t n_blocks = (uint32_t)((n_floats + kv->block_size - 1) / kv->block_size);

    /* Allocate from the freelist (amoeba-like: stable pointers) */
    uint32_t start_block = kv->next_block;
    kv->next_block += n_blocks;

    /* Mount /kv/in/<path> → [start_block, n_blocks) in the namespace */
    int rc = wubu_kvfs_mount(kv->fs, path, start_block, n_blocks);
    if (rc != 0) {
        kv->next_block -= n_blocks; /* rollback */
        return -1;
    }

    /* Resolve the absolute offset via the handle API */
    wubu_kvfs_handle_t *h = wubu_kvfs_open(kv->fs, path);
    if (!h) {
        wubu_kvfs_unmount(kv->fs, path);
        kv->next_block -= n_blocks;
        return -1;
    }
    size_t abs_offset = wubu_kvfs_handle_offset(h);
    wubu_kvfs_handle_close(h);

    /* Register (or refresh) the path for coherence lookup */
    if (replace_idx < 0) {
        if (kv_emb_ensure_cap(kv) != 0) {
            wubu_kvfs_unmount(kv->fs, path);
            kv->next_block -= n_blocks;
            return -1;
        }
        replace_idx = (int)kv->n_paths;
        kv->n_paths++;
    }
    kv_path_record_t *rec = &kv->paths[replace_idx];
    strncpy(rec->path, path, sizeof(rec->path) - 1);
    rec->path[sizeof(rec->path) - 1] = '\0';
    rec->n_tokens = (int64_t)n_tokens;
    rec->kv_offset = (int64_t)abs_offset;

    /* The caller (executor) writes the actual token floats into the KV
     * tensor at `offset` using wubu_kvfs_open + wubu_kvfs_handle_write.
     * This module owns the namespace bookkeeping only. */
    return 0;
}

int wubu_kv_embedding_decode(wubu_kv_embedding_t *kv,
                              const char *path,
                              const uint16_t *tokens, size_t n_tokens) {
    if (!kv || !path || !tokens || n_tokens == 0) return -1;

    /* Build the KV path: /kv/synth/<path> */
    char kv_path[320];
    snprintf(kv_path, sizeof(kv_path), "/kv/synth/%s", path);

    /* Mount the synth region */
    int64_t n_floats = (int64_t)n_tokens;
    uint32_t n_blocks = (uint32_t)((n_floats + kv->block_size - 1) / kv->block_size);

    uint32_t start_block = kv->next_block;
    kv->next_block += n_blocks;

    int rc = wubu_kvfs_mount(kv->fs, kv_path, start_block, n_blocks);
    if (rc != 0) {
        kv->next_block -= n_blocks;
        return -1;
    }

    /* Resolve offset and register */
    wubu_kvfs_handle_t *dh = wubu_kvfs_open(kv->fs, kv_path);
    if (dh) {
        size_t abs_offset = wubu_kvfs_handle_offset(dh);
        wubu_kvfs_handle_close(dh);
        if (kv_emb_ensure_cap(kv) == 0) {
            kv_path_record_t *rec = &kv->paths[kv->n_paths];
            strncpy(rec->path, kv_path, sizeof(rec->path) - 1);
            rec->path[sizeof(rec->path) - 1] = '\0';
            rec->n_tokens = (int64_t)n_tokens;
            rec->kv_offset = (int64_t)abs_offset;
            kv->n_paths++;
        }
    }

    return 0;
}

int wubu_kv_embedding_region(const wubu_kv_embedding_t *kv,
                              const char *path,
                              uint32_t *out_block, size_t *out_offset,
                              size_t *out_n_floats) {
    if (!kv || !path) return -1;
    for (size_t i = 0; i < kv->n_paths; i++) {
        if (strcmp(kv->paths[i].path, path) == 0) {
            if (out_block) *out_block = (uint32_t)(kv->paths[i].kv_offset / kv->block_size);
            if (out_offset) *out_offset = (size_t)kv->paths[i].kv_offset;
            if (out_n_floats) *out_n_floats = (size_t)kv->paths[i].n_tokens;
            return 0;
        }
    }
    return -1;
}

/* COHERENCE: compute the coherence score from attention weights.
 *
 * attention_weights is a flattened [n_query_tokens][n_context_tokens]
 * matrix (row-major). Query tokens attend over context tokens.
 * context_start/context_len locate the file's region in context.
 * query_start/query_len locate the query tokens that read the file.
 *
 * Metrics:
 *   attention_mass = fraction of attention (from query tokens) that
 *     lands within the file's context region. High = focused.
 *   attention_entropy = mean Shannon entropy of the query→context
 *     attention distribution. Low = focused.
 *   consistency = 1.0 when entropy is below threshold; 0.0 otherwise.
 *     (Full consistency requires multiple forward passes — handled by
 *      the trainer. This single-pass version is a binary proxy.)
 */
int wubu_kv_embedding_coherence(const wubu_kv_embedding_t *kv,
                                 const char *path,
                                 const float *attention_weights,
                                 size_t n_query_tokens,
                                 size_t n_context_tokens,
                                 size_t context_start, size_t context_len,
                                 size_t query_start, size_t query_len,
                                 wubu_coherence_t *out) {
    if (!kv || !path || !attention_weights || !out ||
        n_query_tokens == 0 || n_context_tokens == 0) return -1;

    /* Validate the file region exists in context */
    if (context_start + context_len > n_context_tokens) return -1;
    if (query_start + query_len > n_query_tokens) return -1;

    memset(out, 0, sizeof(*out));
    out->n_tokens = (int)context_len;

    /* Find the file's path record — the path must have been encoded */
    int found = 0;
    for (size_t i = 0; i < kv->n_paths; i++) {
        if (strcmp(kv->paths[i].path, path) == 0) {
            found = 1;
            break;
        }
    }
    if (!found) return -1;

    /* Compute attention_mass: fraction of attention (from query tokens)
     * that lands within the file's context region. */
    double total_attn = 0.0, file_mass = 0.0;
    for (size_t q = query_start; q < query_start + query_len; q++) {
        const float *row = attention_weights + q * n_context_tokens;
        for (size_t c = 0; c < n_context_tokens; c++) {
            float a = row[c];
            total_attn += a;
            if (c >= context_start && c < context_start + context_len)
                file_mass += a;
        }
    }

    out->attention_mass = total_attn > 0 ? (float)(file_mass / total_attn) : 0.0f;

    /* Compute attention_entropy: mean Shannon entropy across query rows.
     * Low entropy = the model is focused (confident about where to look).
     * Single-pass, no redundant inner-loop sum recomputation. */
    double total_entropy = 0.0;
    double max_entropy = log2((double)n_context_tokens);
    for (size_t q = query_start; q < query_start + query_len; q++) {
        const float *row = attention_weights + q * n_context_tokens;
        double row_entropy = 0.0;
        /* First pass: compute row sum for normalization */
        double sum = 0.0;
        for (size_t c = 0; c < n_context_tokens; c++)
            sum += (double)row[c];
        if (sum <= 0.0) sum = 1.0;
        /* Second pass: Shannon entropy */
        for (size_t c = 0; c < n_context_tokens; c++) {
            double p = (double)row[c] / sum;
            if (p > 1e-12) row_entropy -= p * log2(p);
        }
        total_entropy += row_entropy;
    }
    double mean_entropy = total_entropy / (double)query_len;
    out->attention_entropy = (float)mean_entropy;

    /* Consistency proxy: high when entropy is low (focused attention).
     * True consistency is measured across query reformulations by the
     * trainer; here we report 1.0 for focused, 0.0 for diffuse. */
    out->consistency = (float)(mean_entropy < 2.0 ? 1.0 : 0.0);

    /* Composite score: 0.4*mass + 0.3*focus + 0.3*consistency */
    double focus = max_entropy > 0.0 ? (1.0 - mean_entropy / max_entropy) : 0.0;
    if (focus < 0.0) focus = 0.0;
    out->score = (float)(0.4 * out->attention_mass +
                         0.3 * focus +
                         0.3 * out->consistency);

    return 0;
}
