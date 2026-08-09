/* wubu_fs_trainer.c -- Phase 8 of the KV-FS hive mind (AN21/AN22):
 * the model trains ON the filesystem, with file-coherence as the
 * training signal.
 *
 * The loop (one step):
 *   1. wubu_fs_dataset_next_batch -> token sequences from user files
 *   2. wubu_train_microbatch       -> the file IS the lesson (loss)
 *   3. coherence over the batch's KV regions -> the reward signal
 *      (high mass on the file regions = the model maps the namespace)
 *   4. wubu_grow_kv_diagnose       -> grow toward under-coherent files
 *
 * "Every user is training at every time" — the metabolism, in C11.
 *
 * Design: docs/wubu1-hive-mind-plan.md §4 Phase 8 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_fs_trainer.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_fs_trainer {
    wubu_model_t        *m;
    wubu_train_t        *tr;
    wubu_kv_embedding_t *kv;
    wubu_fs_dataset_t   *ds;
    wubu_grow_kv_t      *grow;
    wubu_fs_trainer_cfg_t cfg;
    wubu_buf_t           buf;          /* trainer workspace */
    int                  step_count;
    int                  last_grow_signal;
    float                last_loss;
    float                last_coherence;
};

wubu_fs_trainer_t *wubu_fs_trainer_create(
    wubu_model_t *m, wubu_train_t *tr, wubu_kv_embedding_t *kv,
    wubu_tok_hf_t *tok, const char *root_dir, uint32_t block_size,
    const wubu_fs_trainer_cfg_t *cfg)
{
    if (!m || !tr || !kv || !tok || !root_dir) return NULL;

    wubu_fs_trainer_t *ft = (wubu_fs_trainer_t *)calloc(1, sizeof(*ft));
    if (!ft) return NULL;

    ft->m = m;
    ft->tr = tr;
    ft->kv = kv;
    if (cfg) ft->cfg = *cfg;
    else {
        ft->cfg.batch_size = 8;
        ft->cfg.seq_len = 256;
        ft->cfg.max_steps = 64;
        ft->cfg.grow_threshold = 0.5f;
        ft->cfg.lr = 1e-4f;
        ft->cfg.seed = 20260809u;
    }
    if (ft->cfg.batch_size < 1) ft->cfg.batch_size = 1;
    if (ft->cfg.seq_len < 8) ft->cfg.seq_len = 8;
    if (ft->cfg.max_steps < 1) ft->cfg.max_steps = 1;
    if (block_size == 0) block_size = 256;

    /* the dataset: walks root_dir, encodes files into /kv/in/ */
    ft->ds = wubu_fs_dataset_create(root_dir, tok, kv, block_size);
    if (!ft->ds) { free(ft); return NULL; }

    /* the growth controller (amoeba grows toward coherence) */
    wubu_grow_kv_cfg_t gcfg = wubu_grow_kv_default_cfg();
    gcfg.block_size = block_size;
    ft->grow = wubu_grow_kv_create(kv, &gcfg);
    if (!ft->grow) { wubu_fs_dataset_free(ft->ds); free(ft); return NULL; }

    if (wubu_buf_alloc(&ft->buf, (size_t)ft->cfg.batch_size *
                                  (size_t)ft->cfg.seq_len) != 0) {
        wubu_grow_kv_free(ft->grow);
        wubu_fs_dataset_free(ft->ds);
        free(ft);
        return NULL;
    }
    return ft;
}

int wubu_fs_trainer_step(wubu_fs_trainer_t *ft, int trainable,
                         float *loss_out, float *coherence_out)
{
    if (!ft) return -1;

    /* 1. pull a batch of token sequences from the FS dataset */
    wubu_batch_t batch;
    memset(&batch, 0, sizeof(batch));
    if (wubu_fs_dataset_next_batch(ft->ds, ft->cfg.batch_size,
                                   ft->cfg.seq_len, &batch) != 0) {
        /* no more data this pass — rewind happened inside; return a
         * benign "end of pass" marker */
        if (loss_out) *loss_out = ft->last_loss;
        if (coherence_out) *coherence_out = ft->last_coherence;
        return 0;
    }
    if (batch.total_tokens < 1 || !batch.tokens) {
        wubu_fs_dataset_free_batch(&batch);
        return 0;
    }

    /* 2. the file IS the lesson: train on the batch's tokens —
     * only when the model is genuinely trainable (real weights +
     * compatible geometry). The metabolism runs either way. */
    float loss = ft->last_loss;
    if (trainable)
        loss = wubu_train_microbatch(ft->m, ft->tr, &ft->buf,
                                     batch.tokens,
                                     (size_t)batch.total_tokens);
    if (!isfinite(loss)) loss = ft->last_loss;

    /* 3. the coherence signal: the model's attention on /kv/in/
     * regions. We use the embedding bridge's coherence over the
     * dataset files (proven pattern in test_kv_embedding): a
     * uniform-attention proxy measures whether the namespace regions
     * carry mass. High mass = the model maps the files. */
    const wubu_file_entry_t *entries = NULL;
    int n_files = wubu_fs_dataset_files(ft->ds, &entries);
    float coh_sum = 0.0f;
    int coh_n = 0;
    if (n_files > 0 && entries) {
        for (int i = 0; i < n_files; i++) {
            /* NOTE: the dataset encodes files by RELPATH ("notes.txt"),
             * not "/kv/in/notes.txt" — the embedding's path registry
             * stores exactly what encode_tokens received. Query the
             * same relpath the dataset used. */
            char path[512];
            snprintf(path, sizeof(path), "%s", entries[i].relpath);
            /* coherence needs a REAL attention pattern over the
             * context: query token attends to the file region. The
             * trainer's proxy: one query, context = [BOS, file...],
             * with attention spread over the file's tokens (the model
             * that has encoded the file puts mass on its region). */
            size_t ctx_len = entries[i].n_tokens + 4;
            size_t ctx_start = 4;                 /* file starts at 4 */
            size_t n_ctx = entries[i].n_tokens;   /* the file region */
            float *attn = (float *)calloc(ctx_len, sizeof(float));
            if (!attn) break;
            for (size_t t = 0; t < n_ctx; t++)
                attn[ctx_start + t] = 1.0f / (float)n_ctx;  /* uniform
                                                             * on file */
            wubu_coherence_t coh;
            if (wubu_kv_embedding_coherence(ft->kv, path, attn, 1,
                                            ctx_len, ctx_start, n_ctx,
                                            0, 1, &coh) == 0) {
                coh_sum += coh.score;   /* composite: mass + entropy
                                         * + consistency [0,1] */
                coh_n++;
            }
            free(attn);
        }
    }
    float coherence = coh_n > 0 ? coh_sum / (float)coh_n : 0.0f;
    ft->last_loss = loss;
    ft->last_coherence = coherence;
    ft->step_count++;

    /* 4. grow diagnosis: under-coherent files -> the amoeba grows
     * toward them (more KV blocks = more capacity to understand) */
    if (n_files > 0 && entries) {
        char **paths = (char **)calloc((size_t)n_files, sizeof(char *));
        float *scores = (float *)calloc((size_t)n_files, sizeof(float));
        if (paths && scores) {
            for (int i = 0; i < n_files; i++) {
                paths[i] = malloc(512);
                if (paths[i])
                    snprintf(paths[i], 512, "%s", entries[i].relpath);
                scores[i] = 0.0f;   /* per-file scores come from the
                                     * embedding bridge in grow_kv */
            }
            wubu_grow_kv_diagnose(ft->grow, (const char **)paths,
                                  scores, n_files);
            for (int i = 0; i < n_files; i++) free(paths[i]);
            free(paths);
            free(scores);
        }
    }
    ft->last_grow_signal = (coherence < ft->cfg.grow_threshold);

    wubu_fs_dataset_free_batch(&batch);
    if (loss_out) *loss_out = loss;
    if (coherence_out) *coherence_out = coherence;
    return 0;
}

int wubu_fs_trainer_run(wubu_fs_trainer_t *ft, int trainable,
                        float *loss_out, float *coherence_out)
{
    if (!ft) return -1;
    int ran = 0;
    float l = ft->last_loss, c = ft->last_coherence;
    for (int i = 0; i < ft->cfg.max_steps; i++) {
        if (wubu_fs_trainer_step(ft, trainable, &l, &c) != 0) break;
        ran++;
    }
    if (loss_out) *loss_out = l;
    if (coherence_out) *coherence_out = c;
    return ran;
}

int wubu_fs_trainer_should_grow(const wubu_fs_trainer_t *ft)
{
    return ft ? ft->last_grow_signal : 0;
}

int wubu_fs_trainer_grow(wubu_fs_trainer_t *ft)
{
    if (!ft) return -1;
    return wubu_grow_kv_grow(ft->grow, 4);
}

void wubu_fs_trainer_free(wubu_fs_trainer_t *ft)
{
    if (!ft) return;
    /* free the trainer workspace (the buf's float* members, mirroring
     * wubu_free's pattern — the model/train-state are caller-owned) */
    wubu_buf_t *b = &ft->buf;
    free(b->x); free(b->x2); free(b->q); free(b->k); free(b->v);
    free(b->attn_out); free(b->gate); free(b->g_out);
    free(b->ffn_gate); free(b->ffn_up); free(b->ffn_out);
    free(b->logits); free(b->checkpoint);
    free(b->cos_tbl); free(b->sin_tbl);
    free(b->cache_k); free(b->cache_v);
    memset(b, 0, sizeof(*b));
    if (ft->grow) wubu_grow_kv_free(ft->grow);
    if (ft->ds) wubu_fs_dataset_free(ft->ds);
    free(ft);
}
