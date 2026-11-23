#ifndef WUBU_ISQ_PLAN_H
#define WUBU_ISQ_PLAN_H

/*
 * wubu_isq_plan.h -- EG-27: in-situ quantization (ISQ) decision planner
 * (mistral.rs ISQ lineage), reimplemented in C11.
 *
 * Decide whether to quantize a weight block on the host before transferring
 * it to the device. ISQ is forced when the raw block cannot fit the VRAM
 * budget, and otherwise chosen when host quantization plus the smaller
 * transfer beats transferring raw bytes and quantizing on the device.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct {
    int isq;                  /* 1 = quantize on CPU before load */
    int forced_by_budget;
    double host_total_ms;
    double device_total_ms;
    size_t bytes_to_device;
} wubu_isq_decision_t;

wubu_isq_decision_t wubu_isq_decide(size_t raw_bytes, size_t quant_bytes,
                                    double host_quant_ms, double dev_quant_ms,
                                    double link_bandwidth_bps,
                                    size_t vram_free);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ISQ_PLAN_H */
