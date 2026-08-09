/*
 * wubu_runtime_dims.h -- runtime dimensions for the WuBu-35M mustard-seed engine.
 *
 * The Revolver Doctrine (THEORY/06): no fixed geometry. The WuBu-35M
 * engine was compiled with #define constants for every dimension
 * (WUBU_DIM=448, WUBU_LAYERS=12, ...). That is a soldered cartridge:
 * the engine cannot grow, shrink, or load a checkpoint with different
 * geometry without a recompile.
 *
 * Theory/08 (Aligned Rewrite): the runtime dims are now aligned so that
 * every block width (QK_K=256, AVX-512=64, 2:4-sparse=4, VNNI=16)
 * divides evenly — no remainder handling, no fallback to scalar vec_dot.
 * The probe (wubu_runtime_dims_probe) still reads the on-disk checkpoint,
 * but aligns the result: dim rounds UP to the nearest 256-multiple so
 * that 448 → 512, keeping every trick tileable.
 *
 * This module makes the geometry RUNTIME, mirroring the Colonel engine's
 * wubu_dims.h pattern: the loader probes the real checkpoint tensor
 * shapes, fills a wubu_runtime_dims_t, and every macro reads from the runtime
 * global. The old #define values become the DEFAULTS (wubu_runtime_dims_default)
 * so a legacy load without explicit geometry still works.
 *
 * C11, opaque-ish: only the struct + macro aliases live here.
 */
#ifndef WUBU_RUNTIME_DIMS_H
#define WUBU_RUNTIME_DIMS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Every geometry knob of the WuBu-35M engine. */
typedef struct {
    int vocab;          /* WUBU_VOCAB       16384  */
    int dim;            /* WUBU_DIM         448    */
    int layers;         /* WUBU_LAYERS      12     */
    int heads;          /* WUBU_HEADS       7      */
    int kv_heads;       /* WUBU_KV_HEADS    1      */
    int head_dim;       /* WUBU_HEAD_DIM    64     */
    int rope_dim;       /* WUBU_ROPE_DIM    32     */
    int ffn_dim;        /* WUBU_FFN_DIM     1228   */
    int max_seq;        /* WUBU_MAX_SEQ     2048   */
    int local_win;      /* WUBU_LOCAL_WIN   256    */
    int full_every;     /* WUBU_FULL_EVERY  4      */
    int select_every;   /* WUBU_SELECT_EVERY 4     */
    float clip;         /* WUBU_CLIP        10.0f  */
    float eps;          /* WUBU_EPS         1e-6f  */
    int selectors;      /* WUBU_SELECTORS   3      */
    float rope_theta;   /* RoPE base freq   10000.0f (was hardcoded) */
    long  params;       /* WUBU_PARAMS      35072768 */
    int ckpt_dim;       /* native checkpoint dim (pre-alignment; e.g. 448) */
    int ckpt_ffn_dim;  /* native checkpoint ffn (pre-alignment; e.g. 1228) */
    int ckpt_head_dim; /* native checkpoint head_dim (pre-alignment; e.g. 64) */
    int ckpt_heads;   /* native checkpoint heads count (pre-alignment) */
    int ckpt_kv_heads;/* native checkpoint kv_heads count (pre-alignment) */
} wubu_runtime_dims_t;

/* The single active 35M dimension set. The loader probes the checkpoint
 * and calls wubu_runtime_dims_set() before running. */
extern wubu_runtime_dims_t WUBU_RUNTIME_DIMS;

/* Seed the runtime global with the released WuBu-35M geometry. */
void wubu_runtime_dims_default(void);

/* Set explicitly (loader probes the real checkpoint tensor shapes). */
void wubu_runtime_dims_set(const wubu_runtime_dims_t *d);

/* Probe a safetensors checkpoint and fill *d from the REAL tensor shapes.
 * Returns 0 on success, -1 if the file can't be read. Falls back to
 * defaults for any field the probe cannot determine. */
int wubu_runtime_dims_probe(const char *safetensors_path, wubu_runtime_dims_t *d);

#ifdef __cplusplus
}
#endif

/* ---- Macro aliases ----
 * Principle (mirrors wubu_dims.h): the forward bodies keep using the
 * WUBU_* names — the macros now read from the runtime global, so zero
 * edits inside the 700-line forward. Only the struct layout in wubu.h
 * must stay compile-time (C arrays need constant bounds), so the fixed
 * arrays keep the #define MAX as their physical bound while the ACTIVE
 * count is m->n_layers (the revolver cylinder: fixed chambers, active
 * rotation at runtime). */
#define WUBU_VOCAB       WUBU_RUNTIME_DIMS.vocab
#define WUBU_DIM         WUBU_RUNTIME_DIMS.dim
#define WUBU_LAYERS_MAX  12   /* physical array bound (cylinder chambers) */
#define WUBU_LAYERS      WUBU_RUNTIME_DIMS.layers   /* ACTIVE layers (rotation) */
#define WUBU_HEADS       WUBU_RUNTIME_DIMS.heads
#define WUBU_KV_HEADS    WUBU_RUNTIME_DIMS.kv_heads
#define WUBU_HEAD_DIM    WUBU_RUNTIME_DIMS.head_dim
#define WUBU_ROPE_DIM    WUBU_RUNTIME_DIMS.rope_dim
#define WUBU_FFN_DIM     WUBU_RUNTIME_DIMS.ffn_dim
#define WUBU_MAX_SEQ     WUBU_RUNTIME_DIMS.max_seq
#define WUBU_LOCAL_WIN   WUBU_RUNTIME_DIMS.local_win
#define WUBU_FULL_EVERY  WUBU_RUNTIME_DIMS.full_every
#define WUBU_SELECT_EVERY WUBU_RUNTIME_DIMS.select_every
#define WUBU_CLIP        WUBU_RUNTIME_DIMS.clip
#define WUBU_EPS         WUBU_RUNTIME_DIMS.eps
#define WUBU_SELECTORS   WUBU_RUNTIME_DIMS.selectors
#define WUBU_PARAMS      WUBU_RUNTIME_DIMS.params

/* the rope theta was a hardcoded literal in wubu.c — route via dims */
#define WUBU_ROPE_THETA  WUBU_RUNTIME_DIMS.rope_theta

#endif /* WUBU_RUNTIME_DIMS_H */
