#ifndef WUBU_VRAM_PLAN_H
#define WUBU_VRAM_PLAN_H

/*
 * wubu_vram_plan.h -- EG-03: metadata-first VRAM planner (T1 = 6 GB).
 *
 * wubu_resource.c estimates fit from a parameter count at Q4; this module
 * plans from measured bytes: device VRAM, UMA-corrected free memory, the
 * actual GGUF blob size and the metadata-derived KV bytes per token.
 *
 * Rules: usable = 80% of corrected free memory; prefer full context, then
 * shrink context (floor 64), then stream weights if allowed, else refuse.
 * A near-empty device recommends a recovery wait instead of a false FIT.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WUBU_VRAM_PLAN_FIT = 0,
    WUBU_VRAM_PLAN_SHRINK,
    WUBU_VRAM_PLAN_STREAM,
    WUBU_VRAM_PLAN_REFUSE
} wubu_vram_plan_kind_t;

typedef struct {
    uint64_t device_vram_bytes;
    uint64_t measured_free_bytes;  /* already UMA/page-cache corrected */
    uint64_t usable_bytes;         /* 80% of measured free */
    uint64_t weight_bytes;         /* GGUF blob size */
    uint64_t kv_bytes_per_token;   /* from GGUF metadata */
    int      requested_ctx;
    int      allowed_ctx;          /* 0 on REFUSE */
    int      allow_stream;
    int      wait_ms;              /* >0 when free memory is near empty */
    wubu_vram_plan_kind_t decision;
} wubu_vram_plan_t;

/* Fill out a plan. Returns 0 on success, -1 on bad arguments. */
int wubu_vram_plan_make(uint64_t device_vram, uint64_t measured_free,
                        uint64_t weight_bytes, uint64_t kv_bytes_per_token,
                        int requested_ctx, int allow_stream,
                        wubu_vram_plan_t *out);

const char *wubu_vram_plan_name(wubu_vram_plan_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_VRAM_PLAN_H */
