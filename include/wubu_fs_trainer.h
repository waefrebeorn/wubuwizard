/* wubu_fs_trainer.h -- Phase 8 of the KV-FS hive mind (AN21/AN22):
 * the model trains ON the filesystem, with file-coherence as the
 * training signal.
 *
 * The user's directive: "every user is training at every time."
 * This is the metabolism that makes it real:
 *
 *   files -> wubu_fs_dataset -> token batches -> wubu_train_microbatch
 *         -> model forward/backward (the file IS the lesson)
 *   coherence -> wubu_coherence_reward (the model's understanding of
 *         /kv/ files IS the reward: high mass on known paths = the
 *         model maps the namespace; low mass = it doesn't)
 *   diagnosis -> wubu_grow_kv (the amoeba grows toward coherence:
 *         under-coherent files -> more KV blocks -> more capacity)
 *
 * The loop is a single op: wubu_fs_trainer_run() pulls a batch,
 * trains on it, computes the coherence reward, and returns both —
 * the caller (or a live agent loop) decides grow/shrink/stop.
 *
 * Design: docs/wubu1-hive-mind-plan.md §4 Phase 8 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_FS_TRAINER_H
#define WUBU_FS_TRAINER_H

#include <stdint.h>
#include <stddef.h>
#include "wubu_train.h"
#include "wubu_fs_dataset.h"
#include "wubu_kv_embedding.h"
#include "wubu_grow_kv.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct wubu_fs_trainer wubu_fs_trainer_t;

/* The trainer's configuration */
typedef struct {
    int          batch_size;    /* sequences per step (default 8) */
    int          seq_len;       /* tokens per sequence (default 256) */
    int          max_steps;     /* run() ceiling per call (default 64) */
    float        grow_threshold;/* coherence below this -> grow signal */
    float        lr;            /* learning rate (default 1e-4) */
    uint32_t     seed;          /* deterministic batches */
} wubu_fs_trainer_cfg_t;

/* Create the trainer: owns the dataset + growth controller; borrows
 * model, train state, and KV embedding (caller-owned). block_size is
 * the KV block geometry (floats per block, passed to the dataset).
 * Returns NULL on failure. */
wubu_fs_trainer_t *wubu_fs_trainer_create(
    wubu_model_t *m, wubu_train_t *tr, wubu_kv_embedding_t *kv,
    wubu_tok_hf_t *tok, const char *root_dir, uint32_t block_size,
    const wubu_fs_trainer_cfg_t *cfg);

/* One training step: pull a batch from the FS dataset, train on it,
 * compute the coherence reward over /kv/in/, and update the growth
 * controller's diagnosis. Returns 0 on success (or when the dataset
 * rewound), -1 on fatal error.
 *
 * trainable: when 0, the step SKIPS the microbatch (no model weights
 * or incompatible geometry) and still runs the dataset -> coherence
 * -> grow metabolism. The model needs real weights for the loss. */
int wubu_fs_trainer_step(wubu_fs_trainer_t *ft, int trainable,
                         float *loss_out, float *coherence_out);

/* Run up to cfg.max_steps steps. Returns the number of steps actually
 * run; sets loss_out/coherence_out to the LAST step's values.
 * trainable: 0 skips the microbatch (see step). */
int wubu_fs_trainer_run(wubu_fs_trainer_t *ft, int trainable,
                        float *loss_out, float *coherence_out);

/* The growth diagnosis from the last step: should the amoeba grow?
 * (coherence below threshold). Returns 0/1. */
int wubu_fs_trainer_should_grow(const wubu_fs_trainer_t *ft);

/* Grow the KV blocks toward coherence (call when should_grow). */
int wubu_fs_trainer_grow(wubu_fs_trainer_t *ft);

/* Free the trainer (frees the dataset + growth controller; NOT the
 * model/train-state/kv, those are caller-owned). */
void wubu_fs_trainer_free(wubu_fs_trainer_t *ft);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_FS_TRAINER_H */
