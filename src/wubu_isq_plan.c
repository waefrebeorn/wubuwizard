/*
 * wubu_isq_plan.c -- EG-27 ISQ decision planner. Pure C11.
 */
#include "wubu_isq_plan.h"

wubu_isq_decision_t wubu_isq_decide(size_t raw_bytes, size_t quant_bytes,
                                    double host_quant_ms, double dev_quant_ms,
                                    double link_bandwidth_bps,
                                    size_t vram_free) {
    wubu_isq_decision_t d;
    double raw_xfer, q_xfer;
    d.isq = 0;
    d.forced_by_budget = 0;
    d.host_total_ms = 0.0;
    d.device_total_ms = 0.0;
    d.bytes_to_device = raw_bytes;
    if (raw_bytes == 0 || quant_bytes == 0 || link_bandwidth_bps <= 0.0)
        return d;

    raw_xfer = (double)raw_bytes / link_bandwidth_bps * 1000.0;
    q_xfer = (double)quant_bytes / link_bandwidth_bps * 1000.0;
    d.host_total_ms = host_quant_ms + q_xfer;
    d.device_total_ms = raw_xfer + dev_quant_ms;

    if (raw_bytes > vram_free) {
        d.isq = 1;
        d.forced_by_budget = 1;
    } else if (quant_bytes <= vram_free && d.host_total_ms < d.device_total_ms) {
        d.isq = 1;
    }
    if (d.isq) d.bytes_to_device = quant_bytes;
    return d;
}
