/*
 * wubu_vram_plan.c -- EG-03 metadata-first VRAM planner. Pure C11, no OS calls:
 * the caller supplies measured (UMA-corrected) free bytes and GGUF metadata.
 */
#include "wubu_vram_plan.h"
#include <string.h>

#define WUBU_VRAM_MIN_CTX 64

const char *wubu_vram_plan_name(wubu_vram_plan_kind_t kind) {
    switch (kind) {
        case WUBU_VRAM_PLAN_FIT:    return "FIT";
        case WUBU_VRAM_PLAN_SHRINK: return "SHRINK";
        case WUBU_VRAM_PLAN_STREAM: return "STREAM";
        case WUBU_VRAM_PLAN_REFUSE: return "REFUSE";
        default:                    return "UNKNOWN";
    }
}

int wubu_vram_plan_make(uint64_t device_vram, uint64_t measured_free,
                        uint64_t weight_bytes, uint64_t kv_bytes_per_token,
                        int requested_ctx, int allow_stream,
                        wubu_vram_plan_t *out) {
    if (!out || device_vram == 0 || kv_bytes_per_token == 0 ||
        requested_ctx <= 0)
        return -1;

    wubu_vram_plan_t p;
    memset(&p, 0, sizeof(p));
    p.device_vram_bytes = device_vram;
    p.measured_free_bytes = measured_free > device_vram ? device_vram : measured_free;
    p.weight_bytes = weight_bytes;
    p.kv_bytes_per_token = kv_bytes_per_token;
    p.requested_ctx = requested_ctx;
    p.allow_stream = allow_stream ? 1 : 0;

    /* 80% rule: never plan against the last fifth of device memory. */
    p.usable_bytes = (p.measured_free_bytes / 10) * 8;

    /* Recovery wait: a nearly empty device is transient contention, not a
     * permanent NOFIT. Recommend waiting instead of reporting a false fit. */
    if (p.measured_free_bytes < device_vram / 20)
        p.wait_ms = 100;

    if (weight_bytes > p.usable_bytes) {
        if (p.allow_stream) {
            p.decision = WUBU_VRAM_PLAN_STREAM;
            uint64_t ctx = p.usable_bytes / kv_bytes_per_token;
            p.allowed_ctx = ctx > (uint64_t)requested_ctx ? requested_ctx : (int)ctx;
            if (p.allowed_ctx < WUBU_VRAM_MIN_CTX)
                p.decision = WUBU_VRAM_PLAN_REFUSE;
        } else {
            p.decision = WUBU_VRAM_PLAN_REFUSE;
            p.allowed_ctx = 0;
        }
        *out = p;
        return 0;
    }

    uint64_t remaining = p.usable_bytes - weight_bytes;
    uint64_t max_ctx = remaining / kv_bytes_per_token;

    if (max_ctx >= (uint64_t)requested_ctx) {
        p.decision = WUBU_VRAM_PLAN_FIT;
        p.allowed_ctx = requested_ctx;
    } else if (max_ctx >= WUBU_VRAM_MIN_CTX) {
        p.decision = WUBU_VRAM_PLAN_SHRINK;
        p.allowed_ctx = (int)max_ctx;
    } else if (p.allow_stream) {
        p.decision = WUBU_VRAM_PLAN_STREAM;
        uint64_t ctx = p.usable_bytes / kv_bytes_per_token;
        p.allowed_ctx = ctx > (uint64_t)requested_ctx ? requested_ctx : (int)ctx;
        if (p.allowed_ctx < WUBU_VRAM_MIN_CTX)
            p.decision = WUBU_VRAM_PLAN_REFUSE;
    } else {
        p.decision = WUBU_VRAM_PLAN_REFUSE;
        p.allowed_ctx = 0;
    }
    if (p.decision == WUBU_VRAM_PLAN_REFUSE) p.allowed_ctx = 0;

    *out = p;
    return 0;
}
